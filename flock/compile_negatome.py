from __future__ import annotations

import argparse
import logging
import os
import tempfile
from datetime import date as _date

import pandas as pd

from flock import NEGATOME_PDB_VERSION
from flock.aws import download_file_from_s3
from flock.aws import upload_file_to_s3
from flock.intact_filter import filter_against_intact
from flock.intact_filter import load_intact_pairs
from flock.logging_utils import setup_logging
from flock.paths import FLOCK_S3
from flock.paths import get_intact_pairs_path
from flock.paths import get_negatome_lit_path
from flock.paths import get_negatome_literature_negatives_path
from flock.paths import get_negatome_pdb_path
from flock.paths import make_dated_filename
from flock.paths import NEGATOME_S3
from flock.provenance import write_csv_with_provenance

# The published Negatome 2.0 manually curated source this pipeline still reads.
# Named here rather than tracked by NEGATOME_LIT_VERSION, which now points at the
# in-house literature Negatome v3 under its own S3 prefix.
NEGATOME_MANUAL_VERSION = 'v2'

# The three sources, as they are labelled in the output's source column. Manual
# is literature-derived too - it is Blohm et al.'s hand curation - so the label
# that distinguishes the in-house mined set names the pipeline, not the medium.
SOURCE_MANUAL = 'Manual'
SOURCE_PDB = 'Pdb'
SOURCE_LITERATURE = 'Lit'
ALL_SOURCES = (SOURCE_MANUAL, SOURCE_PDB, SOURCE_LITERATURE)

# The published Manual v2 set is deliberately not compiled in: the in-house
# literature source supersedes it rather than extending it, so including both
# would mix a superseded dataset into the benchmark. It stays reachable through
# `include` for the version comparison the EDA notebooks make.
DEFAULT_SOURCES = (SOURCE_PDB, SOURCE_LITERATURE)


def load_negatome(
        version: str = NEGATOME_PDB_VERSION,
        pdb_date: str | None = None,
        lit_date: str | None = None,
        literature_date: str | None = None,
        include: tuple[str, ...] = DEFAULT_SOURCES,
) -> pd.DataFrame:
    """Download and concatenate the Negatome source files.

    The sources differ in layout: the published Negatome 2.0 Manual file is a
    bare TSV with no header, while the two in-house files carry a header row -
    ProteinA / ProteinB / PDB_Code from flock.negatome_v3.pdb.build, and
    uniprot_a / uniprot_b from flock.negatome_v3.literature.negatives. Each
    source therefore declares its own header row index, so a header is discarded
    rather than being read as a pair. Provenance comment lines are skipped for
    all three; only the literature file carries any.

    include exists for the literature source's own build, which needs the pairs
    the Negatome already held before its output joined them.

    Args:
        version: Version string for the PDB negatome source, e.g. 'v3'.
        pdb_date: Date override for the PDB source (YYYY-MM-DD). Defaults to
            the latest dated file for version in S3.
        lit_date: Date override for the Manual source (YYYY-MM-DD). Defaults to
            the latest dated file for NEGATOME_MANUAL_VERSION in S3.
        literature_date: Date override for the in-house literature source
            (YYYY-MM-DD). Defaults to the latest dated file in S3.
        include: Which sources to read, by their source-column label.

    Returns:
        DataFrame with columns uniprot_a, uniprot_b, and source, one row per
        raw pair.
    """
    # Resolved lazily, inside the include check: the literature source's own
    # build reads this function to find what the Negatome already held, and at
    # that point the file it is about to write does not exist yet.
    sources = [
        (
            lambda: get_negatome_lit_path(
                NEGATOME_MANUAL_VERSION, date=lit_date,
            ),
            SOURCE_MANUAL, None,
        ),
        (
            lambda: get_negatome_pdb_path(version=version, date=pdb_date),
            SOURCE_PDB, 0,
        ),
        (
            lambda: get_negatome_literature_negatives_path(
                date=literature_date,
            ),
            SOURCE_LITERATURE, 0,
        ),
    ]
    dfs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for resolve_path, source_label, header_row in sources:
            if source_label not in include:
                continue
            s3_folder, filename = resolve_path().rsplit('/', 1)
            s3_folder += '/'
            local_path = os.path.join(tmpdir, filename)
            download_file_from_s3(s3_folder, filename, local_path)
            df = pd.read_csv(
                local_path, sep='\t', usecols=range(
                    2,
                ), names=['uniprot_a', 'uniprot_b'], header=header_row,
                comment='#',
            )
            df['source'] = source_label
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def deduplicate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise pair order and remove duplicate rows, aggregating sources.

    Sorts each pair so uniprot_a <= uniprot_b, ensuring that (A, B) and (B, A)
    are treated as the same pair. Pairs appearing in both sources are collapsed
    into a single row with source set to "Manual,Pdb".

    Args:
        df: DataFrame with columns uniprot_a, uniprot_b, and source.

    Returns:
        DataFrame with the same columns, duplicates removed and index reset.
    """
    mask = df['uniprot_a'] > df['uniprot_b']
    df.loc[
        mask, ['uniprot_a', 'uniprot_b'],
    ] = df.loc[mask, ['uniprot_b', 'uniprot_a']].values
    return (
        df.groupby(['uniprot_a', 'uniprot_b'], sort=False)['source']
        .agg(lambda sources: ','.join(sorted(set(sources))))
        .reset_index()
    )


def count_negatives_per_protein(df: pd.DataFrame) -> pd.Series:
    """Count how many negative partners each UniProt ID has across all pairs.

    Has no caller in the pipeline since the minimum was removed. Kept for the
    EDA notebooks, which read the distribution of negatives per protein, and
    because this is the honest place to compute it.

    Args:
        df: Deduplicated pairs DataFrame with columns uniprot_a and uniprot_b.

    Returns:
        Series indexed by UniProt ID with counts of negative partners, sorted
        descending.
    """
    return (
        pd.concat([df['uniprot_a'], df['uniprot_b']], ignore_index=True)
        .value_counts()
    )


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compile the in-house Negatome sources into Flock negative pairs.',
    )
    parser.add_argument(
        '--version', default=NEGATOME_PDB_VERSION,
        help='Version string for the PDB Negatome source.',
    )
    parser.add_argument(
        '--date', default=None,
        help=(
            'Pin the PDB Negatome source (YYYY-MM-DD). Defaults to the latest '
            'in S3. Pins only that source; use --lit-date for the other.'
        ),
    )
    parser.add_argument(
        '--lit-date', default=None,
        help=(
            'Pin the literature Negatome source (YYYY-MM-DD). Defaults to the '
            'latest in S3.'
        ),
    )
    return parser


def main(
        version: str = NEGATOME_PDB_VERSION,
        date: str | None = None,
        lit_date: str | None = None,
) -> None:
    """Compile the Negatome dataset and upload it to S3.

    Downloads the two in-house Negatome sources - the PDB-derived set and the
    literature-mined set - deduplicates pairs, filters against IntAct positive
    interactions (following Blohm et al. 2014), and writes a CSV of directed
    (Target, Negative) pairs to FLOCK_NEGATOME. Every pair is expanded into
    both directions, so the output is a symmetric edge list and Target carries
    no information beyond which way a row happens to be written.

    There is no minimum number of negatives per protein. Flock is a training
    set and admits every pair its sources supply; the only remaining minimum in
    the project belongs to the leakage-free benchmark, which counts clean
    partners after leakage filtering rather than before it.

    The published Negatome 2.0 Manual set is not compiled in: the literature
    source supersedes it. Pass include=ALL_SOURCES to load_negatome to get it
    back for a version comparison.

    The literature source has already been filtered against the same IntAct
    table by the stage that wrote it, so the IntAct pass here is a no-op over
    its rows unless the two runs resolved different snapshots.

    Args:
        version: Version string for the PDB negatome source, e.g. 'v3'.
        date: Date of the PDB source file (YYYY-MM-DD). Pins that source only;
            the literature source has its own argument, because the two are
            published by different stages and do not move together.
        lit_date: Date of the literature source file (YYYY-MM-DD). Defaults to
            the latest in S3.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # Resolve all source paths upfront so dates are pinned and available for provenance
    pdb_path = get_negatome_pdb_path(version=version, date=date)
    pdb_filename = pdb_path.rsplit('/', 1)[-1]
    pdb_stem = pdb_filename.removesuffix('.txt')
    pdb_date = pdb_stem.rsplit('_', 1)[-1]
    pdb_stringent_filename = f'pdb_stringent_{version}_{pdb_date}.txt'

    literature_path = get_negatome_literature_negatives_path(date=lit_date)
    literature_filename = literature_path.rsplit('/', 1)[-1]
    literature_date = literature_filename.removesuffix(
        '.txt',
    ).rsplit('_', 1)[-1]

    intact_path = get_intact_pairs_path()
    intact_filename = intact_path.rsplit('/', 1)[-1]

    logger.info('Loading Negatome sources...')
    raw = load_negatome(
        version, pdb_date=pdb_date, literature_date=literature_date,
    )
    logger.info('Loaded %d rows before deduplication', len(raw))

    pairs = deduplicate_pairs(raw)
    logger.info('%d unique pairs after normalization', len(pairs))

    intact_pairs = load_intact_pairs(path=intact_path)

    # Save IntAct-filtered PDB-only pairs as a separate source file for comparison.
    # The stringent filename mirrors the PDB source filename with _stringent appended.

    pdb_only = pairs[pairs['source'].str.contains('Pdb')].copy()
    logger.info('%d unique PDB pairs before IntAct filter', len(pdb_only))
    pdb_filtered, pdb_removed = filter_against_intact(pdb_only, intact_pairs)
    logger.info(
        'PDB IntAct filter: removed %d pairs (%d remain)',
        pdb_removed, len(pdb_filtered),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        local_out = os.path.join(tmpdir, pdb_stringent_filename)
        write_csv_with_provenance(
            pdb_filtered[['uniprot_a', 'uniprot_b']],
            local_out,
            sources={
                'negatome_pdb_source': pdb_filename,
                'intact_source': intact_filename,
            },
            sep='\t',
            header=False,
        )
        logger.info(
            'Uploading PDB stringent to %s',
            NEGATOME_S3 + pdb_stringent_filename,
        )
        upload_file_to_s3(local_out, NEGATOME_S3)

    pairs, n_removed = filter_against_intact(pairs, intact_pairs)
    logger.info(
        'IntAct filter: removed %d pairs (%d remain)',
        n_removed, len(pairs),
    )

    # Expand each undirected pair into two directed rows
    pairs_ab = pairs.rename(
        columns={'uniprot_a': 'Target', 'uniprot_b': 'Negative'},
    )
    pairs_ba = pairs.rename(columns={'uniprot_b': 'Target', 'uniprot_a': 'Negative'})[
        ['Target', 'Negative', 'source']
    ]
    filtered = pd.concat(
        [pairs_ab, pairs_ba], ignore_index=True,
    ).drop_duplicates().reset_index(drop=True)
    logger.info('%d directed pairs', len(filtered))

    output_filename = make_dated_filename(
        'negatome_pairs', version, '.csv', _date.today().isoformat(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        local_out = os.path.join(tmpdir, output_filename)
        write_csv_with_provenance(
            filtered,
            local_out,
            sources={
                'negatome_pdb_source': pdb_filename,
                'negatome_literature_v3_source': literature_filename,
                'intact_source': intact_filename,
            },
        )
        logger.info('Uploading to %s', FLOCK_S3 + output_filename)
        upload_file_to_s3(local_out, FLOCK_S3)

    logger.info('Done.')


if __name__ == '__main__':
    parser = parse_args()
    cli_args = parser.parse_args()
    main(
        version=cli_args.version, date=cli_args.date,
        lit_date=cli_args.lit_date,
    )
