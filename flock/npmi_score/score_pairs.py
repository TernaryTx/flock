from __future__ import annotations

import argparse
import logging
import os

from flock import REPO_ROOT
from flock.logging_utils import setup_logging
from flock.npmi_score import EXTRACTION_FEATURE_SPACE_KEYS
from flock.npmi_score.io import load_config
from flock.npmi_score.io import parquet_provenance
from flock.npmi_score.io import read_pairs
from flock.npmi_score.io import report_table_provenance
from flock.npmi_score.io import resolve_feature_files
from flock.npmi_score.query import load_query_features
from flock.npmi_score.query import missing_proteins
from flock.npmi_score.score import check_feature_space
from flock.npmi_score.score import FEATURE_SPACE_MATCH_KEYS
from flock.npmi_score.score import load_npmi_lookup
from flock.npmi_score.score import score_pair_set

DEFAULT_CONFIG = str(REPO_ROOT / 'configs' / 'npmi.yaml')

# Keys the config must supply, checked up front so a renamed key gives a clear message
# rather than a bare KeyError once the table is loaded.
REQUIRED_SCORING_KEYS = ('top_k', 'rho', 't_max')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Score query protein pairs against a published NPMI table.',
    )
    parser.add_argument(
        '--pairs', required=True,
        help='Pair list: a CSV carrying id_a and id_b, optionally pair_id and label.',
    )
    parser.add_argument(
        '--features', required=True, nargs='+',
        help='One or more Parquet files, directories, or globs holding the query SAE '
             'features. See FEATURES.md for the column contract.',
    )
    parser.add_argument(
        '--npmi-table', required=True,
        help='Path to the published NPMI table CSV.',
    )
    parser.add_argument(
        '--config', default=DEFAULT_CONFIG,
        help='Path to the NPMI config holding top_k, rho and t_max.',
    )
    parser.add_argument(
        '--min-count', type=float, default=0.0,
        help='Drop table pairs whose accumulated count is not above this, as a floor on '
             'rare features. Default 0.0, which is the paper\'s method as stated.',
    )
    parser.add_argument(
        '--output', default='pair_scores.csv',
        help='Path to write the scores to.',
    )
    return parser


def main() -> None:
    """Score query protein pairs against a precomputed NPMI lookup table.

    For each residue pair, the best NPMI across the top-k active features of each side,
    then the mean over the top-T residue pairs (eqs 17-18 of the paper). Nothing is
    trained and no structures or contacts are involved: those are a build-time input to
    the table, which is why scoring needs only sequences.

    The query features must come from the same feature space the table was counted over,
    or feature ids do not mean what the table holds and every lookup is silently wrong
    rather than an error. Where the query Parquet records its own extraction provenance
    that is checked here; where it does not, the table's expected feature space is logged
    and checking it is the caller's responsibility. FEATURES.md states the contract.

    Pairs naming a protein the features do not hold are scored NaN and say so in a status
    column, rather than being dropped or scored zero, which would rank them as confident
    negatives.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = parse_args()
    args = parser.parse_args()

    config = load_config(args.config, REQUIRED_SCORING_KEYS)
    pairs = read_pairs(args.pairs)
    named = sorted(set(pairs['id_a']) | set(pairs['id_b']))
    logger.info('%d pairs over %d proteins', len(pairs), len(named))

    lookup = load_npmi_lookup(args.npmi_table, min_count=args.min_count)
    feature_files = resolve_feature_files(args.features)
    logger.info('%d feature file(s)', len(feature_files))

    # Only when the query side actually recorded something. An extraction run outside this
    # package need not, and comparing against absent keys would refuse every such run.
    query_space = parquet_provenance(feature_files[0])
    if any(key in query_space for key in FEATURE_SPACE_MATCH_KEYS):
        check_feature_space(lookup, query_space)
    else:
        logger.warning(
            'The query features record no extraction provenance, so they cannot be '
            'checked against the table\'s feature space. Confirm the extraction matches '
            'what FEATURES.md states; a mismatch scores every pair silently wrong.',
        )
        report_table_provenance(
            lookup.extraction, EXTRACTION_FEATURE_SPACE_KEYS,
        )

    features = load_query_features(feature_files, wanted=set(named))
    absent = missing_proteins(features, named)
    if absent:
        logger.warning(
            '%d of %d proteins have no features, e.g. %s',
            len(absent), len(named), ', '.join(absent[:5]),
        )

    scores = score_pair_set(
        pairs, features.top, lookup,
        top_k=int(config['top_k']),
        rho=float(config['rho']),
        t_max=int(config['t_max']),
    )
    directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(directory, exist_ok=True)
    scores.to_csv(args.output, index=False)
    logger.info('Wrote %s', args.output)


if __name__ == '__main__':
    main()
