from __future__ import annotations

import argparse
import logging
import os
import tempfile
from datetime import date as _date

import pandas as pd

from flock import PPI3D_VERSION
from flock.aws import download_file_from_s3
from flock.aws import download_folder_contents
from flock.aws import upload_file_to_s3
from flock.cofolding_benchmark.ppi3d_clusters import interface_cluster
from flock.compile_pinder import count_partners_per_protein
from flock.compile_pinder import deduplicate_pairs
from flock.compile_pinder import filter_tag_pairs
from flock.compile_pinder import KNOWN_TAGS
from flock.logging_utils import setup_logging
from flock.negatome_v3.pdb.metadata import get_pdb_chain_uniprot_map
from flock.paths import FLOCK_S3
from flock.paths import get_flock_ppi3d_interfaces_path
from flock.paths import get_ppi3d_prefix
from flock.paths import make_dated_filename
from flock.pdb_metadata import PdbUniprotData
from flock.ppi3d import read_interfaces
from flock.prodigy import classify_interfaces
from flock.prodigy import download_interfaces
from flock.prodigy import interface_filename
from flock.provenance import write_csv_with_provenance


# Columns carried into the interface output. Everything the cofolding benchmark
# needs (pair identity, interface cluster, release date, BIO call) plus the
# quality metrics that justify each row.
INTERFACE_OUTPUT_COLUMNS = [
    'pdb_id', 'biounit_no', 'release_date', 'subunit_1', 'subunit_2',
    'uniprot_1', 'uniprot_2', 'pair', 'area', 'number_of_contacts',
    'resolution', 'cluster_data_40', 'prodigy_class', 'prodigy_prob_bio',
]


def load_ppi3d(cache_dir: str, version: str = PPI3D_VERSION, date: str | None = None) -> pd.DataFrame:
    """Load a PPI3D pull, downloading it from S3 if the local cache is empty.

    Args:
        cache_dir: Local directory holding (or to receive) the pull's files.
        version: Version string for the S3 prefix.
        date: Pull date. Defaults to the latest pull in S3.

    Returns:
        Concatenated interface table.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(cache_dir, exist_ok=True)
    cached = sorted(
        os.path.join(cache_dir, name) for name in os.listdir(cache_dir)
        if name.endswith('.csv.gz')
    )
    if not cached:
        prefix = get_ppi3d_prefix(version=version, date=date)
        logger.info('Cache empty; downloading %s', prefix)
        download_folder_contents(prefix, cache_dir)
        cached = sorted(
            os.path.join(cache_dir, name) for name in os.listdir(cache_dir)
            if name.endswith('.csv.gz')
        )
    logger.info('Reading %d cached windows from %s', len(cached), cache_dir)
    frame = read_interfaces(cached)
    frame['pdb_id'] = frame['pdb_id'].str.lower()
    return frame


def filter_ppi3d(
    df: pd.DataFrame,
    max_resolution: float = 4.0,
    min_area: float = 100.0,
    min_intermolecular_contacts: int = 4,
) -> pd.DataFrame:
    """Apply quality filters to raw PPI3D interfaces.

    Keeps hetero-interactions only and applies interface-quality thresholds
    equivalent to compile_pinder.filter_pinder. PPI3D reports a Voronoi contact
    area, which measures the dividing surface once where buried SASA counts it on
    both partners: the phase 0 spike measured a median area/buried_sasa ratio of
    0.481 at Spearman 0.997, so min_area of 100 A^2 corresponds to PINDER's
    200 A^2 buried-SASA threshold. Both defaults sit at PPI3D's own ingestion
    floors, so in practice these filters remove almost nothing — that is the
    finding, not an oversight.

    Args:
        df: Raw PPI3D interface table.
        max_resolution: Maximum allowable resolution in angstroms. PPI3D only
            ingests structures better than 4 A, so higher values do not widen it.
        min_area: Minimum Voronoi contact area in A^2.
        min_intermolecular_contacts: Minimum number of interface contacts.

    Returns:
        Filtered DataFrame.
    """
    df = df[df['homo'] == 0]
    df = df[df['resolution'] <= max_resolution]
    df = df[df['area'] >= min_area]
    df = df[df['number_of_contacts'] >= min_intermolecular_contacts]
    return df


def map_subunits_to_uniprot(df: pd.DataFrame) -> pd.DataFrame:
    """Add uniprot_1 / uniprot_2 columns by mapping PPI3D subunits through SIFTS.

    PPI3D identifies subunits as '<pdb_id>_<chain>' and carries no UniProt
    accession, so the mapping comes from PdbUniprotData. Chains resolving to
    several accessions (chimeras, fusion constructs) are treated as unmapped
    rather than picking one arbitrarily. Roughly a sixth of interfaces drop out
    here, dominated by antibody and nanobody chains that have no accession to map
    to — PINDER loses the same constructs to its UNDEFINED label.

    Args:
        df: PPI3D interface table with subunit_1 and subunit_2 columns.

    Returns:
        DataFrame with chain_1, chain_2, uniprot_1 and uniprot_2 added, keeping
        only rows where both chains mapped to a single distinct accession.
    """
    logger = logging.getLogger(__name__)
    chain_map = get_pdb_chain_uniprot_map(
        sorted(df['pdb_id'].unique()), PdbUniprotData().data,
    )

    df = df.copy()
    df['chain_1'] = df['subunit_1'].str.split('_', n=1).str[1]
    df['chain_2'] = df['subunit_2'].str.split('_', n=1).str[1]
    for index in ('1', '2'):
        accessions = []
        for pdb_id, chain in zip(df['pdb_id'], df[f'chain_{index}']):
            candidates = chain_map.get(pdb_id, {}).get(chain, set())
            accessions.append(
                next(iter(candidates)) if len(candidates) == 1 else pd.NA,
            )
        df[f'uniprot_{index}'] = pd.Series(
            accessions, index=df.index, dtype='object',
        )

    mapped = df['uniprot_1'].notna() & df['uniprot_2'].notna()
    logger.info(
        '%d of %d interfaces mapped both chains to UniProt (%.1f%%)',
        mapped.sum(), len(df), 100 * mapped.mean(),
    )
    df = df[mapped]
    return df[df['uniprot_1'] != df['uniprot_2']]


def add_pair_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add an order-independent 'pair' key over the two UniProt accessions.

    Args:
        df: DataFrame with uniprot_1 and uniprot_2 columns.

    Returns:
        DataFrame with a 'pair' column of the form 'ACC1|ACC2', sorted.
    """
    both = df[['uniprot_1', 'uniprot_2']]
    df = df.copy()
    df['pair'] = both.min(axis=1) + '|' + both.max(axis=1)
    return df


def select_representative_interfaces(df: pd.DataFrame, per_pair: int = 1) -> pd.DataFrame:
    """Pick the interfaces that will be sent to PRODIGY-cryst for each pair.

    Classification needs one structure download per interface, and a UniProt pair
    maps to a mean of ~3 interfaces, so classifying all of them roughly triples
    the cost for little gain. Following compile_pinder's convention that a pair is
    judged by its best structure, the largest-area interfaces are used as
    representatives.

    Args:
        df: Interface table carrying a 'pair' column.
        per_pair: How many interfaces per pair to classify. Raising this makes a
            pair harder to lose to a single unlucky call, at proportional cost.

    Returns:
        The selected subset of df.
    """
    ordered = df.sort_values('area', ascending=False)
    return ordered.groupby('pair', sort=False).head(per_pair)


def classify_representatives(df: pd.DataFrame, pdb_dir: str) -> pd.DataFrame:
    """Download and PRODIGY-cryst classify a set of representative interfaces.

    Args:
        df: Representative interfaces, carrying download_url.
        pdb_dir: Directory to cache interface coordinate files in.

    Returns:
        df with prodigy_class and prodigy_prob_bio columns added. Interfaces that
        could not be downloaded or classified carry NA.
    """
    logger = logging.getLogger(__name__)
    paths = download_interfaces(df['download_url'].tolist(), pdb_dir)
    logger.info('Downloaded %d of %d interface files', len(paths), len(df))

    classified = classify_interfaces(paths)
    classified['fname'] = classified['pdb_path'].str.rsplit('/', n=1).str[-1]

    df = df.copy()
    df['fname'] = df['download_url'].map(interface_filename)
    merged = df.merge(
        classified[['fname', 'prodigy_class', 'prodigy_prob_bio']],
        on='fname', how='left',
    )
    return merged


def attach_classifications(
    all_interfaces: pd.DataFrame,
    classified: pd.DataFrame,
) -> pd.DataFrame:
    """Broadcast the representatives' BIO calls back onto every mapped interface.

    Only one interface per pair is classified, but the published table must
    carry them all. The leakage checks read source structures and cluster
    membership from it, and both are per-pair facts that a single representative
    cannot express: 39% of pairs have more than one source PDB, and a pair whose
    largest-area interface happens to be recent would otherwise read as
    post-cutoff even when the same pair sits in a decade-old entry.

    prodigy_class is left NA on unclassified rows rather than copied across the
    pair, so consumers scoping to BIO interfaces (the homolog check's query side)
    still see exactly the interfaces that were actually classified.

    Args:
        all_interfaces: Every mapped, tag-filtered interface, carrying 'pair'.
        classified: Representatives with prodigy_class and prodigy_prob_bio.

    Returns:
        all_interfaces with the two prodigy columns added.
    """
    calls = classified[['fname', 'prodigy_class', 'prodigy_prob_bio']]
    work = all_interfaces.copy()
    work['fname'] = work['download_url'].map(interface_filename)
    return work.merge(calls, on='fname', how='left')


def build_cluster_dates(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Earliest release date per PPI3D interface cluster, over the whole pull.

    Deliberately computed before any Flock-specific filtering — UniProt mapping,
    tag removal, deduplication. Cluster membership answers "did a model see a
    homolog of this interface", and a model saw every pre-cutoff structure in a
    cluster regardless of whether we could map its chains to accessions.

    Args:
        raw_df: The full PPI3D pull, with cluster_data_40 and release_date.

    Returns:
        DataFrame with cluster and min_release_date columns.
    """
    work = raw_df[['cluster_data_40', 'release_date']].dropna()
    clusters = [interface_cluster(value) for value in work['cluster_data_40']]
    dates = pd.to_datetime(
        work['release_date'], errors='coerce',
    ).dt.strftime('%Y-%m-%d')
    frame = pd.DataFrame({'cluster': clusters, 'release_date': dates}).dropna()
    grouped = frame.groupby('cluster', as_index=False)['release_date'].min()
    return grouped.rename(columns={'release_date': 'min_release_date'})


def bio_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce classified interfaces to the pairs with a biological interface.

    A pair survives if any of its classified representatives is BIO, mirroring
    compile_pinder's rule that a pair is judged by its best structure.

    Args:
        df: Classified interfaces carrying 'pair' and 'prodigy_class'.

    Returns:
        DataFrame with uniprot_a and uniprot_b for the surviving pairs.
    """
    survivors = sorted(set(df.loc[df['prodigy_class'] == 'BIO', 'pair']))
    split = [pair.split('|') for pair in survivors]
    return pd.DataFrame(split, columns=['uniprot_a', 'uniprot_b'])


def to_directed_pairs(pairs: pd.DataFrame, min_partners: int = 1) -> pd.DataFrame:
    """Expand undirected pairs into directed (Target, Partner) rows.

    Mirrors compile_pinder's output contract so the two positive sources can be
    unioned downstream without reshaping.

    Args:
        pairs: DataFrame with uniprot_a and uniprot_b.
        min_partners: Minimum unique binding partners for a protein to appear as
            a Target. Defaults to 1, which admits every pair and makes the
            output symmetric; see compile_pinder.main for why the minimum is
            gone.

    Returns:
        DataFrame with Target and Partner columns.
    """
    logger = logging.getLogger(__name__)
    partner_counts = count_partners_per_protein(pairs)
    qualifying = set(partner_counts[partner_counts >= min_partners].index)
    logger.info(
        '%d proteins with >= %d binding partners', len(
            qualifying,
        ), min_partners,
    )

    forward = pairs.rename(
        columns={'uniprot_a': 'Target', 'uniprot_b': 'Partner'},
    )
    reverse = pairs.rename(
        columns={'uniprot_b': 'Target', 'uniprot_a': 'Partner'},
    )[['Target', 'Partner']]
    directed = pd.concat(
        [forward, reverse],
        ignore_index=True,
    ).drop_duplicates()
    return directed[directed['Target'].isin(qualifying)].reset_index(drop=True)


def write_pair_table(
        pairs: pd.DataFrame,
        cache_dir: str,
        min_partners: int,
        sources: dict[str, str],
) -> tuple[pd.DataFrame, str]:
    """Normalise, expand and write the directed pair table.

    The tail both publishing routes share — main after its PRODIGY-cryst pass,
    and rebuild_pairs_from_interfaces after reading a published interfaces
    table. Kept in one place so the two cannot drift on the steps between
    bio_pairs and the written file, which is what happened to the
    deduplicate_pairs normalisation.

    Args:
        pairs: Undirected pairs from bio_pairs.
        cache_dir: Directory to write the dated CSV into.
        min_partners: Minimum unique binding partners to appear as a Target.
        sources: Provenance entries for the written file's header.

    Returns:
        Tuple of (directed pair table, path written).
    """
    logger = logging.getLogger(__name__)
    output = to_directed_pairs(deduplicate_pairs(pairs), min_partners)
    logger.info('%d directed pairs with qualifying Targets', len(output))
    output_filename = make_dated_filename(
        'ppi3d_pairs', PPI3D_VERSION, '.csv', _date.today().isoformat(),
    )
    local_out = os.path.join(cache_dir, output_filename)
    write_csv_with_provenance(output, local_out, sources=sources)
    return output, local_out


def rebuild_pairs_from_interfaces(
        cache_dir: str,
        min_partners: int = 1,
        date: str | None = None,
        upload: bool = True,
) -> pd.DataFrame:
    """Republish the pair table from an already-published interfaces table.

    Exists because min_partners shapes only the directed expansion at the very
    end of main, while everything upstream of it — the pull, the quality
    filters, the UniProt mapping and above all the PRODIGY-cryst pass over
    ~39k structures — is unchanged by it. The published interfaces table
    carries `pair` and `prodigy_class`, which is the entire input bio_pairs
    reads, so this route reproduces main's pair table exactly without
    reclassifying anything. Verified against the 2026-08-04 tables: rebuilding
    at min_partners=2 returns the published 27,505 directed rows identically.

    Args:
        cache_dir: Local directory to download into and write the output to.
        min_partners: Minimum unique binding partners to appear as a Target.
        date: Interfaces table date to rebuild from (YYYY-MM-DD). Defaults to
            the latest in S3.
        upload: Whether to upload the rebuilt table to S3.

    Returns:
        The directed pair table that was written.
    """
    logger = logging.getLogger(__name__)
    s3_path = get_flock_ppi3d_interfaces_path(date=date)
    folder, interfaces_filename = s3_path.rsplit('/', 1)
    os.makedirs(cache_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_interfaces = os.path.join(tmpdir, interfaces_filename)
        download_file_from_s3(
            folder + '/', interfaces_filename, local_interfaces,
        )
        interfaces = pd.read_csv(
            local_interfaces, comment='#', low_memory=False,
        )
    logger.info(
        '%d interfaces read from %s', len(interfaces), interfaces_filename,
    )

    pairs = bio_pairs(interfaces)
    logger.info('%d pairs with a biological interface', len(pairs))
    output, local_out = write_pair_table(
        pairs, cache_dir, min_partners,
        sources={
            'ppi3d_interfaces_source': interfaces_filename,
            'min_partners': str(min_partners),
        },
    )
    if upload:
        logger.info('Uploading to %s', FLOCK_S3 + os.path.basename(local_out))
        upload_file_to_s3(local_out, FLOCK_S3)
    else:
        logger.info('Wrote %s (upload skipped)', local_out)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compile PPI3D interfaces into positive UniProt pairs for Flock.',
    )
    parser.add_argument(
        '--cache-dir', default='ppi3d_cache',
        help='Local directory holding the PPI3D pull; downloaded from S3 if empty.',
    )
    parser.add_argument(
        '--pdb-dir', default='ppi3d_interfaces',
        help='Directory to cache interface coordinate files in.',
    )
    parser.add_argument(
        '--pull-date', default=None,
        help='PPI3D pull date to compile (YYYY-MM-DD). Defaults to the latest in S3.',
    )
    parser.add_argument(
        '--per-pair', type=int, default=1,
        help='Interfaces per pair to classify with PRODIGY-cryst.',
    )
    parser.add_argument(
        '--min-partners', type=int, default=1,
        help='Minimum unique binding partners for a protein to appear as a Target.',
    )
    parser.add_argument(
        '--from-interfaces', nargs='?', const='latest', default=None,
        metavar='DATE',
        help='Republish the pair table from a published interfaces table '
             'instead of re-running the pull and PRODIGY-cryst. Optionally '
             'takes the interfaces date (YYYY-MM-DD); defaults to the latest.',
    )
    parser.add_argument(
        '--no-upload', action='store_true',
        help='Write the output locally without uploading to S3.',
    )
    return parser


def main() -> None:
    """Compile the PPI3D positive PPI dataset and upload it to S3.

    Mirrors compile_pinder so the two positive sources produce the same
    (Target, Partner) contract and can be unioned by assemble_flock.

    The pipeline is ordered to keep PRODIGY-cryst last. PPI3D ships no
    biological-vs-crystallographic label, and one has to be computed by
    downloading each interface's coordinates and running the classifier, so every
    cheap filter — hetero-only, interface quality, UniProt mapping, engineered-tag
    removal, deduplication to unique pairs — runs first. That takes the number of
    structures to fetch from ~152k interfaces to ~39k representative ones.

    Interface-quality thresholds are equivalent to PINDER's rather than
    calibrated separately: PPI3D's Voronoi contact area is half buried SASA
    (Spearman 0.997), so its own ingestion floors already encode them. See
    the PPI3D section of the README and notebooks/ppi3d_vs_pinder.ipynb.

    Requires the separate `prodigy` conda env; see the plan for setup.
    --from-interfaces skips all of it and republishes the pair table alone from
    an interfaces table already in S3; use it when only min_partners moved.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = parse_args()
    args = parser.parse_args()

    if args.from_interfaces is not None:
        interfaces_date = (
            None if args.from_interfaces == 'latest' else args.from_interfaces
        )
        rebuild_pairs_from_interfaces(
            args.cache_dir, min_partners=args.min_partners,
            date=interfaces_date, upload=not args.no_upload,
        )
        logger.info('Done.')
        return

    prefix = get_ppi3d_prefix(date=args.pull_date)
    pull_date = prefix.rstrip('/').rsplit('_', 1)[-1]

    raw = load_ppi3d(args.cache_dir, date=args.pull_date)
    logger.info('Loaded %d interfaces', len(raw))

    filtered = filter_ppi3d(raw)
    logger.info(
        '%d interfaces after quality filters (hetero, resolution, area, contacts)',
        len(filtered),
    )

    mapped = map_subunits_to_uniprot(filtered)
    logger.info('%d interfaces with two distinct mapped accessions', len(mapped))

    keyed = add_pair_key(mapped)
    tag_free = filter_tag_pairs(
        keyed.rename(
            columns={
                'uniprot_1': 'uniprot_a',
                'uniprot_2': 'uniprot_b',
            },
        ),
    ).rename(columns={'uniprot_a': 'uniprot_1', 'uniprot_b': 'uniprot_2'})
    logger.info(
        '%d interfaces after removing known-tag pairs (%d tags blocklisted)',
        len(tag_free), len(KNOWN_TAGS),
    )
    logger.info('%d unique UniProt pairs', tag_free['pair'].nunique())

    representatives = select_representative_interfaces(tag_free, args.per_pair)
    logger.info(
        '%d representative interfaces to classify (%d per pair)',
        len(representatives), args.per_pair,
    )

    classified = classify_representatives(representatives, args.pdb_dir)
    counts = classified['prodigy_class'].value_counts(dropna=False)
    logger.info('PRODIGY-cryst classes: %s', counts.to_dict())

    pairs = bio_pairs(classified)
    logger.info('%d pairs with a biological interface', len(pairs))

    # Two artifacts beyond the pair list, both required by the cofolding
    # benchmark and neither reconstructible without another classification pass.
    #
    # The interface table carries EVERY mapped interface, not just the
    # classified representatives. The leakage checks read a pair's source
    # structures from it, and 39% of pairs have more than one source PDB — a
    # single representative would make a pair look as recent as whichever
    # interface happened to have the largest area.
    #
    # The cluster-date table is computed over the raw pull, before any
    # Flock-specific filtering, because cluster membership asks what a model
    # saw rather than what we could map.
    provenance = {
        'ppi3d_pull': prefix.rstrip('/').rsplit('/', 1)[-1],
        'bio_classifier': 'prodigy-cryst 1.0.1',
    }
    today = _date.today().isoformat()

    all_interfaces = attach_classifications(tag_free, classified)
    logger.info(
        '%d interfaces published (%d carrying a BIO/XTAL call)',
        len(all_interfaces), int(
            all_interfaces['prodigy_class'].notna().sum(),
        ),
    )
    interfaces_filename = make_dated_filename(
        'ppi3d_interfaces', PPI3D_VERSION, '.csv', today,
    )
    interfaces_out = os.path.join(args.cache_dir, interfaces_filename)
    write_csv_with_provenance(
        all_interfaces[INTERFACE_OUTPUT_COLUMNS], interfaces_out,
        sources=provenance,
    )

    cluster_dates = build_cluster_dates(raw)
    logger.info(
        '%d interface clusters with a release date',
        len(cluster_dates),
    )
    clusters_filename = make_dated_filename(
        'ppi3d_cluster_dates', PPI3D_VERSION, '.csv', today,
    )
    clusters_out = os.path.join(args.cache_dir, clusters_filename)
    write_csv_with_provenance(cluster_dates, clusters_out, sources=provenance)

    output, local_out = write_pair_table(
        pairs, args.cache_dir, args.min_partners,
        sources={
            'ppi3d_pull': prefix.rstrip('/').rsplit('/', 1)[-1],
            'ppi3d_snapshot_date': pull_date,
            'bio_classifier': 'prodigy-cryst 1.0.1',
            'interfaces_per_pair': str(args.per_pair),
            'min_partners': str(args.min_partners),
        },
    )
    if args.no_upload:
        logger.info(
            'Wrote %s and %s (upload skipped)',
            local_out, interfaces_out,
        )
    else:
        for path in (local_out, interfaces_out, clusters_out):
            logger.info('Uploading to %s', FLOCK_S3 + os.path.basename(path))
            upload_file_to_s3(path, FLOCK_S3)
    logger.info('Done.')


if __name__ == '__main__':
    main()
