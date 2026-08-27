from __future__ import annotations

import json
from typing import Any

import numpy as np
import pyarrow as pa

# Every column contract the scorer depends on, gathered into one module because a release
# has no corpus-build side to keep them apart from.

# The NPMI table, as the table builder writes it. Declared rather than left to pandas so
# that an empty table matches a populated one: building the empty frame from bare lists
# gives float64 feature indices, and a consumer concatenating the two would then get float
# indices whose set_index lookups silently fail to match integers.
TABLE_DTYPES = {
    'feature_i': np.int64,
    'feature_j': np.int64,
    'count': np.float64,
    'npmi': np.float64,
}
TABLE_COLUMNS = tuple(TABLE_DTYPES)

# A pair list's required columns, and the optional label an
# evaluation set carries.
PAIR_COLUMNS = ('pair_id', 'id_a', 'id_b')
LABEL_COLUMN = 'label'

# Columns of the emitted scores table. n_positive travels because a score whose selected
# residue pairs were mostly absent from the table is otherwise indistinguishable from one
# where every pair carried a real value.
SCORE_COLUMNS = (
    'pair_id', 'id_a', 'id_b', 'score', 'n_active_a', 'n_active_b',
    'n_selected', 'n_positive', 'status',
)

# Why a pair has no score. A protein with no features covers both the case where extraction
# produced nothing for it and the case where it was never extracted at all.
STATUS_OK = 'ok'
STATUS_MISSING_A = 'missing_features_a'
STATUS_MISSING_B = 'missing_features_b'

# One row per residue with list columns, not one row per
# active feature: at about three active features per residue the long form would repeat
# interface_id three times per residue. int32 covers a 16384-entry codebook and any
# sequence position.
#
# This is the contract a query extraction must satisfy. interface_id carries the protein
# id, magnitudes the IDF-NORMALISED activation (not the raw one) -- see FEATURES.md.
FEATURE_FIELDS = (
    ('interface_id', pa.string()),
    ('side', pa.string()),
    ('position', pa.int32()),
    ('feature_ids', pa.list_(pa.int32())),
    ('magnitudes', pa.list_(pa.float32())),
)


def features_schema(provenance: dict[str, Any]) -> pa.Schema:
    """Build the feature table's schema, carrying the run's provenance.

    Embedded rather than written as a sidecar, matching write_csv_with_provenance: a
    sidecar is separated from the data by copying just the parquet.

    Args:
        provenance: The extraction provenance record.

    Returns:
        The schema, with provenance in its key-value metadata.
    """
    metadata = {
        key: value if isinstance(value, str) else json.dumps(value)
        for key, value in provenance.items()
    }
    return pa.schema(list(FEATURE_FIELDS)).with_metadata(metadata)


def build_table(
        interface_ids: np.ndarray,
        sides: np.ndarray,
        positions: np.ndarray,
        offsets: np.ndarray,
        feature_ids: np.ndarray,
        magnitudes: np.ndarray,
        schema: pa.Schema,
) -> pa.Table:
    """Build rows of the feature table from one batch's active features.

    Args:
        interface_ids: Interface id per active residue.
        sides: Side per active residue.
        positions: Untrimmed position per active residue.
        offsets: Delimits each residue's run within feature_ids and magnitudes, so it
            is one longer than the number of residues.
        feature_ids: Active feature ids, grouped by residue.
        magnitudes: Their normalised magnitudes, in the same order.
        schema: The table's schema.

    Returns:
        One row per residue.
    """
    return pa.Table.from_pydict(
        {
            'interface_id': pa.array(interface_ids, type=pa.string()),
            'side': pa.array(sides, type=pa.string()),
            'position': pa.array(positions, type=pa.int32()),
            'feature_ids': pa.ListArray.from_arrays(
                pa.array(offsets, type=pa.int32()),
                pa.array(feature_ids, type=pa.int32()),
            ),
            'magnitudes': pa.ListArray.from_arrays(
                pa.array(offsets, type=pa.int32()),
                pa.array(magnitudes, type=pa.float32()),
            ),
        },
        schema=schema,
    )
