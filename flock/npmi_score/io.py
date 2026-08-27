from __future__ import annotations

import glob
import logging
import os
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from flock.npmi_score.schema import PAIR_COLUMNS

# Local file reading only. The corpus build resolved its features out of a distributed
# extraction run, staging shards against a plan and its sentinels; a release takes the
# Parquet files it is handed, so none of that applies here.

# Recognised extensions for a pair list.
_CSV_SUFFIXES = ('.csv',)

# Ids named per category in a log line. The counts themselves are exact.
REPORT_EXAMPLES = 5


def load_config(path: str, required: tuple[str, ...] = ()) -> dict[str, Any]:
    """Load a YAML config file and check the keys the caller depends on are present.

    Args:
        path: Path to the YAML config.
        required: Keys that must be present.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If the file is not a mapping or a required key is missing.
    """
    with open(path) as file_in:
        config = yaml.safe_load(file_in)
    if not isinstance(config, dict):
        raise ValueError(f'{path} must contain a mapping at the top level')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f'{path} is missing required keys: {", ".join(missing)}',
        )
    return config


def read_pairs(path: str) -> pd.DataFrame:
    """Read a pair list from CSV.

    Args:
        path: Local path to a CSV carrying id_a and id_b, and optionally pair_id and
            label.

    Returns:
        The pair list. pair_id is generated positionally where the input has none, since
        deriving it from the two ids would collide if a pair repeats.

    Raises:
        ValueError: If the file is not a CSV, lacks the two id columns, holds no pairs,
            or repeats a pair_id.
    """
    # On the suffix, so another format says so rather than being parsed as CSV into one
    # nonsense column and reported as missing the ids it actually carries.
    if not path.endswith(_CSV_SUFFIXES):
        raise ValueError(
            f'{path} is not a CSV; a pair list is a CSV carrying id_a and id_b',
        )
    frame = pd.read_csv(path)
    required = [name for name in PAIR_COLUMNS if name != 'pair_id']
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            f'{path} lacks columns the pair list needs: {", ".join(missing)}',
        )
    if frame.empty:
        raise ValueError(f'{path} holds no pairs')
    if 'pair_id' not in frame.columns:
        frame['pair_id'] = [f'pair_{index}' for index in range(len(frame))]
    frame['pair_id'] = frame['pair_id'].astype(str)
    repeated = frame['pair_id'][frame['pair_id'].duplicated()].unique()
    if repeated.size:
        raise ValueError(
            f'{path} carries {repeated.size} pair_id(s) more than once, e.g. '
            f'{", ".join(repeated[:REPORT_EXAMPLES])}; scores would not be joinable '
            f'back to their input row',
        )
    for column in ('id_a', 'id_b'):
        frame[column] = frame[column].astype(str)
    return frame


def resolve_feature_files(features: list[str]) -> list[str]:
    """Expand the feature arguments into a sorted list of Parquet files.

    Args:
        features: Any mix of Parquet files, directories to search recursively, and glob
            patterns.

    Returns:
        The matching paths, sorted and deduplicated.

    Raises:
        ValueError: If nothing matched, which would otherwise surface as every pair
            scoring NaN for missing features.
    """
    found: set[str] = set()
    for entry in features:
        if os.path.isdir(entry):
            found.update(
                glob.glob(
                    os.path.join(entry, '**', '*.parquet'),
                    recursive=True,
                ),
            )
        elif os.path.isfile(entry):
            found.add(entry)
        else:
            found.update(glob.glob(entry, recursive=True))
    if not found:
        raise ValueError(
            f'no Parquet files matched {features}; every pair would score NaN for '
            f'missing features',
        )
    return sorted(found)


def parquet_provenance(path: str) -> dict[str, str]:
    """Read whatever extraction provenance a feature file carries in its Parquet metadata.

    Optional by design: a query extracted outside this package need not record anything,
    and the scorer only compares the keys that are actually present.

    Args:
        path: Local path to a features Parquet file.

    Returns:
        The key-value metadata decoded to str, empty if the file carries none.
    """
    metadata = pq.read_schema(path).metadata or {}
    return {key.decode(): value.decode() for key, value in metadata.items()}


def report_table_provenance(record: dict[str, str], keys: tuple[str, ...]) -> None:
    """Log the feature space an NPMI table was counted over.

    The released scorer cannot verify the query was extracted the same way, because the
    query extraction is the user's own. Printing what the table expects is what replaces
    that check; see FEATURES.md.

    Args:
        record: The table's provenance header.
        keys: The keys that fix the feature space.
    """
    logger = logging.getLogger(__name__)
    if not record:
        logger.warning(
            'The NPMI table carries no provenance header, so nothing states which '
            'feature space it was counted over.',
        )
        return
    named = {key: record[key] for key in keys if key in record}
    if named:
        logger.info(
            'Table feature space: %s',
            ', '.join(f'{key}={value}' for key, value in named.items()),
        )
