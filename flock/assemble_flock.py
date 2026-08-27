from __future__ import annotations

import argparse
import logging
import os
import tempfile
from datetime import date as _date

import pandas as pd

from flock import FLOCK_VERSION
from flock.aws import download_file_from_s3
from flock.aws import upload_file_to_s3
from flock.logging_utils import setup_logging
from flock.paths import FLOCK_S3
from flock.paths import get_flock_negatome_path
from flock.paths import get_flock_pinder_path
from flock.paths import get_flock_ppi3d_path
from flock.paths import make_dated_filename
from flock.provenance import write_csv_with_provenance
from flock.uniprot import add_gene_name_columns


def union_positive_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine directed positive pairs from several sources, recording provenance.

    A pair supported by more than one source is kept once, with Positive_source
    listing every source that supports it (comma-joined, sorted). Nothing in the
    pipeline branches on that column — the leakage checks run every source over
    every positive and reconcile the results — but it is what makes the
    benchmark's composition auditable, e.g. reporting how many surviving
    positives came from a source PINDER never had.

    Args:
        sources: Mapping of source label -> directed pairs with Target and
            Partner columns.

    Returns:
        DataFrame with Target, Partner and Positive_source.
    """
    labelled = [
        frame[['Target', 'Partner']].assign(Positive_source=label)
        for label, frame in sources.items() if frame is not None
    ]
    combined = pd.concat(labelled, ignore_index=True)
    return (
        combined
        .groupby(['Target', 'Partner'], as_index=False)['Positive_source']
        .agg(lambda values: ','.join(sorted(set(values))))
    )


def assemble_flock(
    negatome_df: pd.DataFrame,
    pinder_df: pd.DataFrame,
    ppi3d_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the Flock dataset from the negatome and the union of positive sources.

    Every pair from every source is kept. There is no intersection on shared
    targets and no minimum pair count: Flock is a model training set, and the pairs of
    targets carrying enough clean partners for evaluation are selected later, by
    the leakage-free benchmark build.

    Pairs that appear as both a Negatome negative and a positive are resolved by
    dropping the negative. This errs on the side of strictness: when a source
    lists the pair as a biological interface in any PDB, we don't claim it as a
    non-interaction regardless of what the PDB-derived negatome's Cβ-Cβ ≤ 8 Å
    rule emitted. The conflict rule applies to the union, so a pair that only
    PPI3D calls positive still removes the corresponding negative. The negatome
    source files (negatome_pairs and pdb_v3) remain untouched — only the
    assembled flock dataset is filtered.

    Homodimers are dropped here rather than in the positive compilers, so that
    one line governs the whole dataset and each source table stays faithful to
    its upstream. They are all positives; the negative sources emit none.

    The Target/Partner column names are kept for continuity with the
    leakage-free build and the cofolding packages, but Target is now a labelling
    artifact rather than a filter: both labels are symmetric edge lists, so
    grouping by Target does not give a classification problem.

    Args:
        negatome_df: Negatome directed pairs with Target and Negative columns.
        pinder_df: PINDER directed pairs with Target and Partner columns.
        ppi3d_df: Optional PPI3D directed pairs with Target and Partner columns.
            Omitting it reproduces the PINDER-only build.

    Returns:
        DataFrame with columns Target, Partner, Type (Positive/Negative) and
        Positive_source, sorted by Target.
    """
    logger = logging.getLogger(__name__)
    all_positives = union_positive_sources(
        {'pinder': pinder_df, 'ppi3d': ppi3d_df},
    )

    negatives = (
        negatome_df[['Target', 'Negative']]
        .rename(columns={'Negative': 'Partner'})
    )
    positive_a = all_positives[['Target', 'Partner']].min(axis=1)
    positive_b = all_positives[['Target', 'Partner']].max(axis=1)
    positive_pair_set = set(zip(positive_a, positive_b))

    negative_a = negatives[['Target', 'Partner']].min(axis=1)
    negative_b = negatives[['Target', 'Partner']].max(axis=1)
    conflict_mask = pd.MultiIndex.from_arrays(
        [negative_a, negative_b],
    ).isin(positive_pair_set)

    n_conflict_rows = int(conflict_mask.sum())
    conflict_pairs = set(
        zip(negative_a[conflict_mask], negative_b[conflict_mask]),
    )
    n_conflict_pairs = len(conflict_pairs)
    logger.info(
        'Dropped %d negative rows (%d unique undirected pairs) that overlap with positives',
        n_conflict_rows, n_conflict_pairs,
    )

    negatives = negatives[~conflict_mask]

    combined = pd.concat(
        [
            negatives.assign(Type='Negative', Positive_source=pd.NA),
            all_positives.assign(Type='Positive'),
        ],
        ignore_index=True,
    )

    homodimers = combined['Target'] == combined['Partner']
    logger.info(
        'Dropped %d homodimer rows (%s)', int(homodimers.sum()),
        combined.loc[homodimers, 'Type'].value_counts().to_dict(),
    )

    return (
        combined[~homodimers]
        .sort_values(['Target', 'Type', 'Partner'])
        .reset_index(drop=True)
    )


def _read_s3_csv(path: str, tmpdir: str) -> pd.DataFrame:
    """Download a provenance-headed CSV from S3 and read it.

    Args:
        path: Full s3:// path.
        tmpdir: Directory to download into.

    Returns:
        Parsed DataFrame with the # provenance lines skipped.
    """
    folder, filename = path.rsplit('/', 1)
    local_path = os.path.join(tmpdir, filename)
    download_file_from_s3(folder + '/', filename, local_path)
    return pd.read_csv(local_path, comment='#')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Assemble the Flock benchmark from the negatome and positive sources.',
    )
    parser.add_argument(
        '--no-ppi3d', action='store_true',
        help='Build from PINDER positives alone, reproducing the pre-PPI3D dataset.',
    )
    return parser


def main() -> None:
    """Assemble the Flock dataset and upload it to S3.

    Loads the compiled negatome and positive-source CSVs, combines positive and
    negative pairs, drops homodimers, adds gene names from UniProt, and writes
    the result to S3. Every pair from every source is kept: Flock is a training
    set, and the evaluation subset is selected downstream by the leakage-free
    build.

    Positives come from PINDER and PPI3D unioned rather than PINDER alone.
    PINDER's snapshot stops at 2024-02-07, so it cannot see structures released
    since; PPI3D tracks current PDB holdings. They are unioned rather than
    swapped because PPI3D recovers only ~70% of PINDER's pairs (measured in
    notebooks/ppi3d_vs_pinder.ipynb), so dropping PINDER would lose positives.
    Each positive records which source(s) support it, which the cofolding
    benchmark's homolog check needs.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = parse_args()
    args = parser.parse_args()

    negatome_path = get_flock_negatome_path()
    negatome_folder, negatome_filename = negatome_path.rsplit('/', 1)
    logger.info('Loading negatome pairs from %s...', negatome_path)

    pinder_path = get_flock_pinder_path()
    pinder_filename = pinder_path.rsplit('/', 1)[-1]
    logger.info('Loading PINDER pairs from %s...', pinder_path)

    ppi3d_path = None if args.no_ppi3d else get_flock_ppi3d_path()
    ppi3d_filename = None if ppi3d_path is None else ppi3d_path.rsplit(
        '/', 1,
    )[-1]
    if ppi3d_path is not None:
        logger.info('Loading PPI3D pairs from %s...', ppi3d_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        negatome_df = _read_s3_csv(negatome_path, tmpdir)
        pinder_df = _read_s3_csv(pinder_path, tmpdir)
        ppi3d_df = None if ppi3d_path is None else _read_s3_csv(
            ppi3d_path, tmpdir,
        )

    logger.info(
        '%d negatome directed pairs, %d unique targets',
        len(negatome_df), negatome_df['Target'].nunique(),
    )
    logger.info(
        '%d PINDER directed pairs, %d unique targets',
        len(pinder_df), pinder_df['Target'].nunique(),
    )

    if ppi3d_df is not None:
        logger.info(
            '%d PPI3D directed pairs, %d unique targets',
            len(ppi3d_df), ppi3d_df['Target'].nunique(),
        )

    output = assemble_flock(negatome_df, pinder_df, ppi3d_df)
    logger.info('%d total rows in Flock v1', len(output))
    positives = output[output['Type'] == 'Positive']
    logger.info(
        'positive rows by source: %s',
        positives['Positive_source'].value_counts().to_dict(),
    )

    logger.info('Fetching gene names from UniProt...')
    output = add_gene_name_columns(
        output,
        uniprot_cols=['Target', 'Partner'],
        gene_name_cols=['Target_gene', 'Partner_gene'],
    )

    output_filename = make_dated_filename(
        'flock', FLOCK_VERSION, '.csv', _date.today().isoformat(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        local_out = os.path.join(tmpdir, output_filename)
        write_csv_with_provenance(
            output,
            local_out,
            sources={
                'negatome_source': negatome_filename,
                'pinder_source': pinder_filename,
                'ppi3d_source': ppi3d_filename or 'excluded',
            },
        )
        logger.info('Uploading to %s', FLOCK_S3 + output_filename)
        upload_file_to_s3(local_out, FLOCK_S3)

    logger.info('Done.')


if __name__ == '__main__':
    main()
