from __future__ import annotations

import argparse
import logging
import os
import tempfile
from datetime import date as _date

import pandas as pd

from flock import PINDER_VERSION
from flock.aws import download_file_from_s3
from flock.aws import upload_file_to_s3
from flock.logging_utils import setup_logging
from flock.paths import FLOCK_S3
from flock.paths import get_pinder_raw_path
from flock.paths import make_dated_filename
from flock.provenance import write_csv_with_provenance

# UniProt accessions for commonly co-crystallised engineered tags that are not
# real biological binding partners for the proteins they appear with in PDB
# entries. Any PINDER pair involving one of these accessions is dropped during
# compilation. See notebooks/pinder_eda.ipynb for the analysis motivating this
# list and the pairs it removes.
KNOWN_TAGS: dict[str, str] = {
    'P42212': 'GFP (Aequorea victoria)',
    'Q9U6Y8': 'DsRed / RFP (Discosoma)',
    'P08515': 'GST (Schistosoma japonicum)',
    'P0AEX9': 'MBP / malE (E. coli)',
    'P40035': 'SUMO / Smt3 (S. cerevisiae)',
    'P63165': 'SUMO-1 (human)',
    'P61956': 'SUMO-2 (human)',
    'P55854': 'SUMO-3 (human)',
}


def load_pinder(version: str = PINDER_VERSION, date: str | None = None) -> pd.DataFrame:
    """Download the raw PINDER metadata parquet from S3.

    Args:
        version: Version string for the PINDER raw file, e.g. 'v1'.
        date: Date the file was downloaded (YYYY-MM-DD). Defaults to the latest
            dated file for the given version in S3.

    Returns:
        DataFrame with all PINDER columns.
    """
    raw_path = get_pinder_raw_path(version=version, date=date)
    folder, filename = raw_path.rsplit('/', 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, filename)
        download_file_from_s3(folder + '/', filename, local_path)
        return pd.read_parquet(local_path)


def filter_pinder(
    df: pd.DataFrame,
    max_resolution: float = 4.5,
    min_buried_sasa: float = 200.0,
    min_intermolecular_contacts: int = 4,
) -> pd.DataFrame:
    """Apply quality filters to the raw PINDER metadata.

    Keeps only biologically relevant interfaces (label == BIO), removes
    structures with resolution worse than max_resolution angstroms, and applies
    interface-quality filters that drop systems whose interface is too small or
    barely in contact. The latter remove questionable pairs where two chains
    barely touch and are held together by a third chain in a larger assembly
    (e.g. GNB1/OPRK1, where the receptor is held near Gβ by Gα). Filtering is
    per-system: because a UniProt pair maps to many PINDER systems and is
    deduplicated downstream, a pair survives if any of its systems passes.
    NaN values fail the >= comparison and are therefore dropped.

    Args:
        df: Raw PINDER metadata DataFrame.
        max_resolution: Maximum allowable resolution in angstroms.
        min_buried_sasa: Minimum buried surface area of the interface in Ų.
        min_intermolecular_contacts: Minimum number of intermolecular contacts
            across the interface.

    Returns:
        Filtered DataFrame.
    """
    df = df[df['label'] == 'BIO']
    df = df[df['resolution'] <= max_resolution]
    df = df[df['buried_sasa'] >= min_buried_sasa]
    df = df[df['intermolecular_contacts'] >= min_intermolecular_contacts]
    return df


def extract_uniprot_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Parse UniProt IDs from the PINDER id column and return a pairs DataFrame.

    The id column format is: {pdb}__{chain}_{uniprot}--{pdb}__{chain}_{uniprot}

    Drops any entries where either UniProt is UNDEFINED.

    Args:
        df: Filtered PINDER DataFrame with an id column.

    Returns:
        DataFrame with columns uniprot_a and uniprot_b.
    """
    def _parse(ppi_id: str) -> tuple[str, str]:
        parts = ppi_id.split('--')
        uniprot_a, uniprot_b = (
            part.split(
                '__',
            )[1].split('_')[1] for part in parts
        )
        return uniprot_a, uniprot_b

    pairs = df['id'].apply(_parse).tolist()
    result = pd.DataFrame(pairs, columns=['uniprot_a', 'uniprot_b'])
    defined = (result['uniprot_a'] != 'UNDEFINED') & (
        result['uniprot_b'] != 'UNDEFINED'
    )
    result = result[defined]
    return result.reset_index(drop=True)


def deduplicate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise pair order and remove duplicate rows.

    Sorts each pair so uniprot_a <= uniprot_b, ensuring that (A, B) and (B, A)
    are treated as the same pair.

    Args:
        df: DataFrame with columns uniprot_a and uniprot_b.

    Returns:
        Deduplicated DataFrame with the same columns and index reset.
    """
    mask = df['uniprot_a'] > df['uniprot_b']
    df = df.copy()
    df.loc[
        mask, ['uniprot_a', 'uniprot_b'],
    ] = df.loc[mask, ['uniprot_b', 'uniprot_a']].values
    return df.drop_duplicates().reset_index(drop=True)


def filter_tag_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pairs where either protein is a known engineered tag.

    Args:
        df: Pairs DataFrame with columns uniprot_a and uniprot_b.

    Returns:
        DataFrame with any row containing a UniProt in KNOWN_TAGS removed.
    """
    tag_accessions = set(KNOWN_TAGS)
    tag_a = df['uniprot_a'].isin(tag_accessions)
    tag_b = df['uniprot_b'].isin(tag_accessions)
    keep = ~(tag_a | tag_b)
    return df[keep].reset_index(drop=True)


def count_partners_per_protein(df: pd.DataFrame) -> pd.Series:
    """Count the number of unique binding partners each UniProt ID has.

    Args:
        df: Deduplicated pairs DataFrame with columns uniprot_a and uniprot_b.

    Returns:
        Series indexed by UniProt ID with partner counts, sorted descending.
    """
    return pd.concat([df['uniprot_a'], df['uniprot_b']], ignore_index=True).value_counts()


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compile PINDER metadata into positive UniProt pairs for Flock.',
    )
    parser.add_argument(
        '--min-partners', type=int, default=1,
        help=(
            'Minimum unique binding partners for a protein to appear as a '
            'Target. Defaults to 1, which admits every pair.'
        ),
    )
    parser.add_argument(
        '--version', default=PINDER_VERSION,
        help='Version string for the PINDER raw file.',
    )
    parser.add_argument(
        '--date', default=None,
        help='Date of the raw file (YYYY-MM-DD). Defaults to the latest in S3.',
    )
    parser.add_argument(
        '--no-upload', action='store_true',
        help='Write the output locally without uploading to S3.',
    )
    return parser


def main(
        min_partners: int = 1,
        version: str = PINDER_VERSION,
        date: str | None = None,
        upload: bool = True,
) -> None:
    """Compile the PINDER positive PPI dataset and upload it to S3.

    Downloads the raw PINDER metadata, applies quality filters, deduplicates
    pairs, and writes a TSV of directed (Target, Partner) pairs to FLOCK_PINDER.
    Only proteins with at least min_partners binding partners are included as
    Targets.

    Args:
        min_partners: Minimum number of unique binding partners a protein must
            have to be included as a Target in the output. Defaults to 1, which
            admits every pair and makes the output a symmetric edge list. Flock
            is a training set and sets no minimum pair count on either label;
            the only minimum left in the project is the leakage-free
            benchmark's, and that one counts clean partners after leakage
            filtering rather than before it, so filtering here can only starve
            it.
        version: Version string for the PINDER raw file, e.g. 'v1'.
        date: Date the raw file was downloaded (YYYY-MM-DD). Defaults to the
            latest dated file for the given version in S3.
        upload: Whether to upload the compiled table to S3.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # Resolve raw path upfront so the date is pinned and available for provenance
    raw_path = get_pinder_raw_path(version=version, date=date)
    pinder_date = raw_path.rsplit('_', 1)[-1].removesuffix('.parquet')
    pinder_filename = raw_path.rsplit('/', 1)[-1]

    logger.info('Loading PINDER metadata...')
    raw = load_pinder(version, date=pinder_date)
    logger.info('Loaded %d rows', len(raw))

    filtered = filter_pinder(raw)
    logger.info(
        '%d rows after quality filters (BIO label, resolution ≤4.5 Å, '
        'buried_sasa ≥200 Å², intermolecular_contacts ≥4)', len(filtered),
    )

    pairs = extract_uniprot_pairs(filtered)
    logger.info('%d rows with both UniProts defined', len(pairs))

    pairs = deduplicate_pairs(pairs)
    logger.info('%d unique pairs after deduplication', len(pairs))

    pairs = filter_tag_pairs(pairs)
    logger.info(
        '%d unique pairs after removing known-tag pairs (%d tags blocklisted)',
        len(pairs), len(KNOWN_TAGS),
    )

    partner_counts = count_partners_per_protein(pairs)
    qualifying = set(partner_counts[partner_counts >= min_partners].index)
    logger.info(
        '%d proteins with >= %d binding partners',
        len(qualifying), min_partners,
    )

    # Expand each undirected pair into two directed rows, keep rows where Target qualifies
    pairs_ab = pairs.rename(
        columns={'uniprot_a': 'Target', 'uniprot_b': 'Partner'},
    )
    pairs_ba = pairs.rename(columns={'uniprot_b': 'Target', 'uniprot_a': 'Partner'})[
        ['Target', 'Partner']
    ]
    directed = pd.concat(
        [pairs_ab, pairs_ba], ignore_index=True,
    ).drop_duplicates()

    output = directed[
        directed['Target'].isin(
            qualifying,
        )
    ].reset_index(drop=True)
    logger.info('%d directed pairs with qualifying Targets', len(output))

    output_filename = make_dated_filename(
        'pinder_pairs', PINDER_VERSION, '.csv', _date.today().isoformat(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        local_out = os.path.join(tmpdir, output_filename)
        write_csv_with_provenance(
            output, local_out, sources={'pinder_raw_source': pinder_filename},
        )
        if upload:
            logger.info('Uploading to %s', FLOCK_S3 + output_filename)
            upload_file_to_s3(local_out, FLOCK_S3)
        else:
            logger.info('Wrote %s (upload skipped)', output_filename)

    logger.info('Done.')


if __name__ == '__main__':
    parser = parse_args()
    cli_args = parser.parse_args()
    main(
        min_partners=cli_args.min_partners, version=cli_args.version,
        date=cli_args.date, upload=not cli_args.no_upload,
    )
