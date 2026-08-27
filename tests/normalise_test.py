from __future__ import annotations

import pickle

import numpy as np
import pytest

from flock.npmi_score.normalise import active_features
from flock.npmi_score.normalise import load_statistics


def write_statistics(tmp_path, max_per_feature, idf_per_feature, name='stats.pkl'):
    """Write a statistics pickle in Biohub's format and return its path."""
    path = tmp_path / name
    with open(path, 'wb') as file_out:
        pickle.dump(
            {
                'max_per_feature': np.asarray(max_per_feature, dtype=np.float32),
                'idf_per_feature': np.asarray(idf_per_feature, dtype=np.float32),
            },
            file_out,
        )
    return str(path)


def test_load_statistics_returns_both_vectors_as_float32(tmp_path):
    path = write_statistics(tmp_path, [10.0, 20.0], [2.0, 4.0])
    max_per_feature, idf_per_feature, _ = load_statistics(path)
    assert max_per_feature.dtype == np.float32
    assert idf_per_feature.dtype == np.float32
    assert max_per_feature.tolist() == [10.0, 20.0]


def test_load_statistics_downcasts_float64_idf(tmp_path):
    # The published log10 file ships idf as float64, and the SAE's own buffers are
    # float32, so the downcast happens either way.
    path = tmp_path / 'stats.pkl'
    with open(path, 'wb') as file_out:
        pickle.dump(
            {
                'max_per_feature': np.array([10.0], dtype=np.float32),
                'idf_per_feature': np.array([2.0], dtype=np.float64),
            },
            file_out,
        )
    _, idf_per_feature, _ = load_statistics(str(path))
    assert idf_per_feature.dtype == np.float32


def test_load_statistics_md5_is_stable_and_distinguishes_files(tmp_path):
    first = write_statistics(tmp_path, [10.0], [2.0], name='a.pkl')
    same = write_statistics(tmp_path, [10.0], [2.0], name='b.pkl')
    other = write_statistics(tmp_path, [10.0], [3.0], name='c.pkl')
    assert load_statistics(first)[2] == load_statistics(same)[2]
    assert load_statistics(first)[2] != load_statistics(other)[2]


def test_load_statistics_rejects_all_ones_idf(tmp_path):
    # Exactly what the SAE checkpoint itself ships, which would make normalisation a
    # silent no-op and the 0.5 threshold keep essentially everything.
    path = write_statistics(tmp_path, [10.0, 20.0], [1.0, 1.0])
    with pytest.raises(ValueError, match='idf_per_feature is all ones'):
        load_statistics(path)


def test_load_statistics_rejects_all_ones_max(tmp_path):
    path = write_statistics(tmp_path, [1.0, 1.0], [2.0, 4.0])
    with pytest.raises(ValueError, match='max_per_feature is all ones'):
        load_statistics(path)


def test_load_statistics_rejects_a_missing_key(tmp_path):
    path = tmp_path / 'stats.pkl'
    with open(path, 'wb') as file_out:
        pickle.dump({'max_per_feature': np.array([1.0])}, file_out)
    with pytest.raises(ValueError, match='missing'):
        load_statistics(str(path))


def test_load_statistics_rejects_vectors_of_different_length(tmp_path):
    path = write_statistics(tmp_path, [10.0, 20.0], [2.0])
    with pytest.raises(ValueError, match='disagree in length'):
        load_statistics(path)


def test_load_statistics_rejects_a_zero_max(tmp_path):
    # A zero divides the raw activation to inf, which clears the threshold and reaches
    # Parquet rather than raising anywhere.
    path = write_statistics(tmp_path, [10.0, 0.0], [2.0, 4.0])
    with pytest.raises(ValueError, match='finite and positive'):
        load_statistics(path)


def test_load_statistics_rejects_a_non_finite_max(tmp_path):
    path = write_statistics(tmp_path, [10.0, np.inf], [2.0, 4.0])
    with pytest.raises(ValueError, match='finite and positive'):
        load_statistics(path)


def test_load_statistics_rejects_a_non_finite_idf(tmp_path):
    path = write_statistics(tmp_path, [10.0, 20.0], [2.0, np.nan])
    with pytest.raises(ValueError, match='idf_per_feature must be finite'):
        load_statistics(path)


def test_load_statistics_accepts_a_zero_idf(tmp_path):
    # Legitimate: a feature active in every protein has idf log(N / N).
    path = write_statistics(tmp_path, [10.0, 20.0], [0.0, 4.0])
    _, idf_per_feature, _ = load_statistics(path)
    assert idf_per_feature.tolist() == [0.0, 4.0]


def test_active_features_keeps_only_what_clears_the_threshold():
    # max 10 and idf 2 for every feature, so the multiplier is 0.2 and a raw activation
    # clears 0.5 above 2.5.
    max_per_feature = np.full(4, 10.0, dtype=np.float32)
    idf_per_feature = np.full(4, 2.0, dtype=np.float32)
    rows = np.array([0, 0, 1])
    ids = np.array([0, 1, 2])
    raw = np.array([10.0, 1.0, 5.0], dtype=np.float32)

    residues, offsets, feature_ids, magnitudes = active_features(
        rows, ids, raw, max_per_feature, idf_per_feature,
    )
    assert residues.tolist() == [0, 1]
    assert offsets.tolist() == [0, 1, 2]
    assert feature_ids.tolist() == [0, 2]
    assert magnitudes.tolist() == pytest.approx([2.0, 1.0])


def test_active_features_groups_by_residue_with_aligned_offsets():
    max_per_feature = np.full(5, 1.0, dtype=np.float32)
    idf_per_feature = np.full(5, 1.0, dtype=np.float32)
    rows = np.array([0, 0, 0, 2, 2])
    ids = np.array([4, 3, 1, 2, 0])
    raw = np.ones(5, dtype=np.float32)

    residues, offsets, feature_ids, _ = active_features(
        rows, ids, raw, max_per_feature, idf_per_feature,
    )
    assert residues.tolist() == [0, 2]
    # Residue 0 owns the first three, residue 2 the last two, in input order.
    assert offsets.tolist() == [0, 3, 5]
    assert feature_ids[offsets[0]:offsets[1]].tolist() == [4, 3, 1]
    assert feature_ids[offsets[1]:offsets[2]].tolist() == [2, 0]


def test_active_features_uses_the_per_feature_multiplier():
    # Feature 0 is ubiquitous with idf 0 so can never be active; feature 1 is rare.
    max_per_feature = np.array([10.0, 10.0], dtype=np.float32)
    idf_per_feature = np.array([0.0, 5.0], dtype=np.float32)
    rows = np.array([0, 0])
    ids = np.array([0, 1])
    raw = np.array([10.0, 2.0], dtype=np.float32)

    residues, _, feature_ids, _ = active_features(
        rows, ids, raw, max_per_feature, idf_per_feature,
    )
    assert residues.tolist() == [0]
    assert feature_ids.tolist() == [1]


def test_active_features_returns_empty_when_nothing_clears():
    max_per_feature = np.full(2, 100.0, dtype=np.float32)
    idf_per_feature = np.full(2, 1.0, dtype=np.float32)
    residues, offsets, feature_ids, magnitudes = active_features(
        np.array([0, 1]), np.array([0, 1]), np.ones(2, dtype=np.float32),
        max_per_feature, idf_per_feature,
    )
    assert residues.size == 0
    assert feature_ids.size == 0
    assert magnitudes.size == 0
    assert offsets.tolist() == [0]


def test_active_features_rejects_descending_rows():
    # np.unique sorts the residues while the feature ids keep input order, so this would
    # otherwise hand every residue another residue's features.
    max_per_feature = np.full(3, 1.0, dtype=np.float32)
    idf_per_feature = np.full(3, 1.0, dtype=np.float32)
    with pytest.raises(RuntimeError, match='descending rows'):
        active_features(
            np.array([2, 0]), np.array([0, 1]), np.ones(2, dtype=np.float32),
            max_per_feature, idf_per_feature,
        )


def test_active_features_respects_an_explicit_threshold():
    max_per_feature = np.full(2, 1.0, dtype=np.float32)
    idf_per_feature = np.full(2, 1.0, dtype=np.float32)
    rows = np.array([0, 1])
    ids = np.array([0, 1])
    raw = np.array([0.4, 0.6], dtype=np.float32)

    residues, _, _, _ = active_features(
        rows, ids, raw, max_per_feature, idf_per_feature, threshold=0.3,
    )
    assert residues.tolist() == [0, 1]
