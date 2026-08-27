from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from flock.npmi_score.query import load_query_features
from flock.npmi_score.query import missing_proteins
from flock.npmi_score.schema import build_table
from flock.npmi_score.schema import features_schema

# Rows deliberately out of protein and position order, the way a batched extraction
# leaves them: protein A's two residues are split by protein B's, and A's later position comes
# first. Grouping and position ordering both have to be done by the reader.
INTERFACE_IDS = ('A', 'B', 'A')
SIDES = ('s1', 's1', 's1')
POSITIONS = (5, 1, 3)
OFFSETS = (0, 2, 3, 5)
FEATURE_IDS = (0, 3, 1, 2, 7)
MAGNITUDES = (0.4, 0.8, 0.5, 0.9, 0.1)


def write_shard(path, provenance=None):
    """Write a feature shard with the real schema the container writes.

    Args:
        path: Where to write the parquet.
        provenance: Extraction provenance to embed, or None for a minimal record.

    Returns:
        The path written.
    """
    schema = features_schema(provenance or {'codebook_size': 8})
    table = build_table(
        np.array(INTERFACE_IDS, dtype=object),
        np.array(SIDES, dtype=object),
        np.array(POSITIONS, dtype=np.int32),
        np.array(OFFSETS, dtype=np.int32),
        np.array(FEATURE_IDS, dtype=np.int32),
        np.array(MAGNITUDES, dtype=np.float32),
        schema,
    )
    pq.write_table(table, str(path))
    return str(path)


def test_load_query_features_groups_by_protein(tmp_path):
    store = load_query_features(
        [write_shard(tmp_path / 'features.parquet')], max_top_k=2,
    )
    assert sorted(store.top) == ['A', 'B']
    assert store.n_shards == 1
    assert store.n_residues == 3


def test_load_query_features_orders_residues_by_position(tmp_path):
    store = load_query_features(
        [write_shard(tmp_path / 'features.parquet')], max_top_k=2,
    )
    # A's rows arrive as position 5 then 3; the store puts them in sequence order, which
    # position 3's features coming out first is what shows.
    assert store.top['A'].tolist() == [[2, 7], [3, 0]]


def test_load_query_features_pads_a_residue_with_one_feature(tmp_path):
    store = load_query_features(
        [write_shard(tmp_path / 'features.parquet')], max_top_k=2,
    )
    # Repeating the only active feature leaves eq 17's max unchanged.
    assert store.top['B'].tolist() == [[1, 1]]


def test_load_query_features_counts_active_residues(tmp_path):
    store = load_query_features(
        [write_shard(tmp_path / 'features.parquet')], max_top_k=2,
    )
    # The |A_active| of eq 18 is exactly the stored row count, which is what score_pair
    # reads off the array, because extraction omits residues with nothing active rather
    # than writing them empty.
    assert store.top['A'].shape[0] == 2
    assert store.top['B'].shape[0] == 1
    assert 'ABSENT' not in store.top


def test_load_query_features_honours_wanted(tmp_path):
    store = load_query_features(
        [write_shard(tmp_path / 'features.parquet')],
        max_top_k=2, wanted={'B'},
    )
    assert list(store.top) == ['B']
    assert store.n_residues == 1


def test_load_query_features_rejects_a_protein_in_two_shards(tmp_path):
    first = write_shard(tmp_path / 'one.parquet')
    second = write_shard(tmp_path / 'two.parquet')
    with pytest.raises(ValueError, match='more than one feature shard'):
        load_query_features([first, second], max_top_k=2)


def test_load_query_features_stores_the_sweep_ceiling(tmp_path):
    wide = load_query_features(
        [write_shard(tmp_path / 'features.parquet')], max_top_k=4,
    )
    narrow = load_query_features(
        [write_shard(tmp_path / 'features.parquet')], max_top_k=2,
    )
    assert wide.top['A'].shape == (2, 4)
    assert np.array_equal(narrow.top['A'], wide.top['A'][:, :2])


def test_missing_proteins_names_what_the_features_lack(tmp_path):
    store = load_query_features(
        [write_shard(tmp_path / 'features.parquet')], max_top_k=2,
    )
    assert missing_proteins(store, ['A', 'B', 'C', 'C']) == ['C']
