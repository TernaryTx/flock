from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
from collections import defaultdict
from datetime import date as _date

import numpy as np
import pandas as pd

from flock import FLOCK_VERSION
from flock.aws import upload_file_to_s3
from flock.cofolding_benchmark.annotation.afdb_coverage import compute_coverage
from flock.cofolding_benchmark.date_flags import build_negatome_pdb_index
from flock.cofolding_benchmark.date_flags import build_pinder_pair_index
from flock.cofolding_benchmark.date_flags import build_ppi3d_pair_index
from flock.cofolding_benchmark.date_flags import merge_pair_indexes
from flock.cofolding_benchmark.date_flags import undirected_pair
from flock.cofolding_benchmark.leakage_free_set import _download
from flock.cofolding_benchmark.leakage_free_set import negative_source_pdb_index
from flock.logging_utils import setup_logging
from flock.paths import COFOLDING_BENCHMARK_S3
from flock.paths import get_cofolding_benchmark_path
from flock.paths import get_flock_ppi3d_interfaces_path
from flock.paths import get_negatome_pdb_path
from flock.paths import get_pinder_raw_path
from flock.paths import make_dated_filename
from flock.provenance import write_csv_with_provenance


def _build_acc_to_source_pdbs(
    pairs_df: pd.DataFrame,
    positive_pair_index: dict[tuple[str, str], set[str]],
    negatome_pair_index: dict[tuple[str, str], list[str]],
    literature_pair_index: dict[tuple[str, str], set[str]] | None = None,
) -> dict[str, set[str]]:
    """Map every UniProt accession in the leakage-free set to its source PDB IDs.

    For positive pairs, source PDB IDs come from the positive-source pair index
    (PINDER entry_id and PPI3D pdb_id, unioned). For negative pairs they come
    from the negatome pair index (PDB_Code column), plus the literature
    co-presence index for pairs the PDB-derived source never named. Each
    accession accumulates PDB IDs across all pairs it appears in.

    Both indexes must span their whole label. Coverage is computed over the
    residues observed in a pair's source structures, so a pair whose sources
    are missing resolves to no observed residues and a meaningless coverage
    figure — most of the leakage-free positives are PPI3D-only, and every
    literature-mined negative is absent from the PDB-derived negatome index by
    construction, since its evidence is a sentence rather than a structure.

    Args:
        pairs_df: Leakage-free pairs CSV with target, partner, type columns.
        positive_pair_index: Merged positive-source index, e.g.
            merge_pair_indexes(pinder_index, ppi3d_index).
        negatome_pair_index: Output of build_negatome_pdb_index.
        literature_pair_index: Optional pair -> shared PDB entries for
            literature-mined negatives. Without it those pairs return no source
            PDBs, no SIFTS-observed residues and NaN AFDB coverage.

    Returns:
        Dict mapping UniProt accession to a set of source PDB entry IDs.
    """
    acc_to_pdbs: dict[str, set[str]] = defaultdict(set)
    for target, partner, row_type in zip(
        pairs_df['target'], pairs_df['partner'], pairs_df['type'],
    ):
        pair = undirected_pair(target, partner)
        if row_type == 'Positive':
            pdbs: set[str] | list[str] = positive_pair_index.get(pair, set())
        else:
            pdbs = set(negatome_pair_index.get(pair, []))
            if literature_pair_index is not None:
                pdbs = pdbs | literature_pair_index.get(pair, set())
        for acc in (target, partner):
            acc_to_pdbs[acc].update(pdbs)
    return dict(acc_to_pdbs)


def _nan_mean(values: list[float]) -> float:
    """Return the mean of non-NaN values, or NaN if all are NaN."""
    clean = [v for v in values if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float('nan')


def build_annotated_pairs(
    pairs_df: pd.DataFrame,
    coverage: dict[str, float],
) -> pd.DataFrame:
    """Add AFDB pLDDT>=65 coverage columns to the leakage-free pairs DataFrame.

    Args:
        pairs_df: Leakage-free pairs with target, partner columns.
        coverage: Per-accession AFDB coverage from compute_coverage.

    Returns:
        Copy of pairs_df with target_afdb_plddt65_obs and
        partner_afdb_plddt65_obs columns appended.
    """
    out = pairs_df.copy()
    out['target_afdb_plddt65_obs'] = out['target'].map(coverage)
    out['partner_afdb_plddt65_obs'] = out['partner'].map(coverage)
    return out


def build_annotated_targets(
    targets_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    coverage: dict[str, float],
) -> pd.DataFrame:
    """Add AFDB pLDDT>=65 coverage columns to the leakage-free targets DataFrame.

    Adds the target's own coverage, plus mean coverage of its positive and
    negative partners (aggregated from the pairs DataFrame).

    Args:
        targets_df: Leakage-free targets with target column.
        pairs_df: Leakage-free pairs with target, partner, type columns.
        coverage: Per-accession AFDB coverage from compute_coverage.

    Returns:
        Copy of targets_df with target_afdb_plddt65_obs,
        pos_partner_afdb_plddt65_obs_mean and neg_partner_afdb_plddt65_obs_mean
        columns appended.
    """
    pos_partner_coverage: dict[str, list[float]] = defaultdict(list)
    neg_partner_coverage: dict[str, list[float]] = defaultdict(list)
    for target, partner, row_type in zip(
        pairs_df['target'], pairs_df['partner'], pairs_df['type'],
    ):
        partner_cov = coverage.get(partner, float('nan'))
        if row_type == 'Positive':
            pos_partner_coverage[target].append(partner_cov)
        else:
            neg_partner_coverage[target].append(partner_cov)

    out = targets_df.copy()
    out['target_afdb_plddt65_obs'] = out['target'].map(coverage)
    out['pos_partner_afdb_plddt65_obs_mean'] = out['target'].map(
        lambda tgt: _nan_mean(pos_partner_coverage.get(tgt, [])),
    )
    out['neg_partner_afdb_plddt65_obs_mean'] = out['target'].map(
        lambda tgt: _nan_mean(neg_partner_coverage.get(tgt, [])),
    )
    return out


def main() -> None:
    """Annotate a benchmark arm with AFDB pLDDT>=65 observed-residue coverage.

    Reads one arm's published pairs and targets CSVs from S3 - the leakage-free
    set by default, or either arm of the leakage comparison via
    --benchmark-set - computes AFDB pLDDT>=65 coverage for every UniProt
    accession in it (restricted to residues observed in the source PDB
    structures via SIFTS), and writes two new annotated CSVs under the same
    cofolding_benchmark S3 prefix:
      - <set>_targets_annotated_<version>_<date>.csv: one row per target, with
        target_afdb_plddt65_obs, pos_partner_afdb_plddt65_obs_mean and
        neg_partner_afdb_plddt65_obs_mean columns added.
      - <set>_pairs_annotated_<version>_<date>.csv: one row per pair, with
        target_afdb_plddt65_obs and partner_afdb_plddt65_obs added.

    Coverage is the fraction of SIFTS-observed UniProt residues (union over all
    source PDB structures for that accession in the arm) that have AFDB
    pLDDT >= 65. NaN where AFDB is unavailable or no SIFTS residues found.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = parse_args()
    args = parser.parse_args()

    pairs_path = get_cofolding_benchmark_path(f'{args.benchmark_set}_pairs')
    pairs_filename = os.path.basename(pairs_path)
    targets_path = get_cofolding_benchmark_path(
        f'{args.benchmark_set}_targets',
    )
    targets_filename = os.path.basename(targets_path)
    pinder_path = get_pinder_raw_path()
    pinder_filename = os.path.basename(pinder_path)
    negatome_path = get_negatome_pdb_path()
    negatome_filename = os.path.basename(negatome_path)

    with tempfile.TemporaryDirectory() as download_dir:
        logger.info(
            'Loading %s pairs %s...',
            args.benchmark_set, pairs_filename,
        )
        pairs_df = pd.read_csv(
            _download(pairs_path, download_dir), comment='#',
        )

        logger.info(
            'Loading %s targets %s...', args.benchmark_set, targets_filename,
        )
        targets_df = pd.read_csv(
            _download(targets_path, download_dir), comment='#',
        )

        logger.info('Loading PINDER metadata (id + entry_id only)...')
        pinder_df = pd.read_parquet(
            _download(pinder_path, download_dir),
            columns=['id', 'entry_id'],
        )

        logger.info('Loading PDB negatome...')
        negatome_df = pd.read_csv(
            _download(negatome_path, download_dir), sep='\t',
        )

        ppi3d_interfaces_df = None
        if not args.no_ppi3d:
            ppi3d_interfaces_path = get_flock_ppi3d_interfaces_path()
            ppi3d_interfaces_filename = os.path.basename(ppi3d_interfaces_path)
            logger.info(
                'Loading classified PPI3D interfaces %s...',
                ppi3d_interfaces_filename,
            )
            ppi3d_interfaces_df = pd.read_csv(
                _download(ppi3d_interfaces_path, download_dir), comment='#',
            )
        else:
            ppi3d_interfaces_filename = 'excluded'

    logger.info('Building source-PDB indexes...')
    positive_pair_index = build_pinder_pair_index(pinder_df)
    if ppi3d_interfaces_df is not None:
        positive_pair_index = merge_pair_indexes(
            positive_pair_index, build_ppi3d_pair_index(ppi3d_interfaces_df),
        )
    negatome_pair_index = build_negatome_pdb_index(negatome_df)
    literature_pair_index = (
        negative_source_pdb_index(pd.read_parquet(args.negative_leakage))
        if args.negative_leakage is not None else None
    )
    if literature_pair_index is not None:
        logger.info(
            '%d negative pairs carry a shared PDB entry',
            len(literature_pair_index),
        )
    acc_to_source_pdbs = _build_acc_to_source_pdbs(
        pairs_df, positive_pair_index, negatome_pair_index,
        literature_pair_index=literature_pair_index,
    )
    logger.info('Unique accessions: %d', len(acc_to_source_pdbs))

    coverage = compute_coverage(acc_to_source_pdbs, n_workers=args.workers)
    nan_count = sum(1 for v in coverage.values() if math.isnan(v))
    logger.info(
        'Coverage computed: %d/%d NaN', nan_count, len(coverage),
    )

    logger.info('Building annotated pairs CSV...')
    annotated_pairs = build_annotated_pairs(pairs_df, coverage)

    logger.info('Building annotated targets CSV...')
    annotated_targets = build_annotated_targets(targets_df, pairs_df, coverage)

    sources = {
        'pairs_source': pairs_filename,
        'targets_source': targets_filename,
        'pinder_source': pinder_filename,
        'ppi3d_interfaces_source': ppi3d_interfaces_filename,
        'pdb_negatome_source': negatome_filename,
        'coverage_method': (
            'AFDB pLDDT>=65 fraction over SIFTS-observed UniProt residues '
            '(union of source PDBs per accession)'
        ),
    }
    today = _date.today().isoformat()
    targets_annotated_filename = make_dated_filename(
        f'{args.benchmark_set}_targets_annotated', FLOCK_VERSION, '.csv', today,
    )
    pairs_annotated_filename = make_dated_filename(
        f'{args.benchmark_set}_pairs_annotated', FLOCK_VERSION, '.csv', today,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        targets_out = os.path.join(temp_dir, targets_annotated_filename)
        pairs_out = os.path.join(temp_dir, pairs_annotated_filename)
        write_csv_with_provenance(
            annotated_targets, targets_out, sources=sources,
        )
        write_csv_with_provenance(annotated_pairs, pairs_out, sources=sources)
        logger.info(
            'Wrote %s (%d targets)',
            targets_out, len(annotated_targets),
        )
        logger.info('Wrote %s (%d pairs)', pairs_out, len(annotated_pairs))
        if not args.no_upload:
            logger.info('Uploading to %s', COFOLDING_BENCHMARK_S3)
            upload_file_to_s3(targets_out, COFOLDING_BENCHMARK_S3)
            upload_file_to_s3(pairs_out, COFOLDING_BENCHMARK_S3)
    logger.info('Done.')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Annotate the leakage-free cofolding benchmark set with AFDB '
            'pLDDT>=65 coverage over observed (SIFTS) residues.'
        ),
    )
    parser.add_argument(
        '--no-upload',
        action='store_true',
        help='Write the annotated CSVs locally only; do not upload to S3.',
    )
    parser.add_argument(
        '--benchmark-set',
        default='leakage_free',
        help='Which arm to annotate, as the shared stem of its targets and '
             'pairs files: leakage_free, leaked or leakage_free_capped. The '
             'outputs take the same stem plus _annotated '
             '(default: %(default)s).',
    )
    parser.add_argument(
        '--no-ppi3d',
        action='store_true',
        help=(
            'Take positive source structures from PINDER only. Matches the '
            'leakage_free_set --no-ppi3d build; on a union build it would leave '
            'most positives with no source structures.'
        ),
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help=(
            'Parallel threads for SIFTS/AFDB lookups (default: 4). At 16 the '
            'EBI SIFTS endpoint throttled hard enough to fail 20% of fetches '
            'against 8% at lower concurrency, and a failed fetch costs an '
            'accession its coverage figure.'
        ),
    )
    parser.add_argument(
        '--negative-leakage',
        default=None,
        help=(
            'Per-pair negative leakage table from '
            'flock.cofolding_benchmark.sequence_homology. Supplies source PDB '
            'entries for literature-mined negatives, which the PDB-derived '
            'negatome index cannot resolve; without it those accessions get no '
            'observed residues and NaN coverage.'
        ),
    )
    return parser


if __name__ == '__main__':
    main()
