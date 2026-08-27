from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from flock.npmi_score.score import check_feature_space
from flock.npmi_score.score import load_npmi_lookup
from flock.npmi_score.score import MAX_TOP_K
from flock.npmi_score.score import NpmiLookup
from flock.npmi_score.score import residue_pair_npmi
from flock.npmi_score.score import score_pair
from flock.npmi_score.score import score_pair_set
from flock.npmi_score.score import selected_pair_count
from flock.npmi_score.score import STATUS_MISSING_A
from flock.npmi_score.score import STATUS_MISSING_B
from flock.npmi_score.score import STATUS_OK
from flock.npmi_score.score import top_features
from flock.provenance import write_csv_with_provenance

# A six-feature codebook with five unordered pairs, one of them on the diagonal, so the
# symmetrisation can be checked to mirror the off-diagonal and not double the diagonal.
PAIRS = (
    (0, 1, 4.0, 0.9),
    (0, 2, 3.0, 0.5),
    (1, 2, 2.0, 0.2),
    (3, 3, 5.0, 0.7),
    (2, 4, 1.0, 0.1),
)
CODEBOOK_SIZE = 6

# The features of the worked example. R is computed by hand in the docstring of
# test_residue_pair_npmi_worked_example.
TOP_A = np.array([[0, 3], [2, 5]], dtype=np.int32)
TOP_B = np.array([[1, 4], [3, 5]], dtype=np.int32)
EXPECTED_R = np.array([[0.9, 0.7], [0.2, 0.0]], dtype=np.float32)

FEATURE_SPACE = {
    'esmc_repo': 'biohub/ESMC-6B',
    'esmc_revision': 'aaaa',
    'sae_repo': 'biohub/ESMC-6B-sae-layer60-k64-codebook16384',
    'sae_revision': 'bbbb',
    'transformers_revision': 'cccc',
    'esm_revision': 'dddd',
    'fused_kernels': 'none',
    'statistics_md5': '9c01182d1d22b2c76673610ed77d358b',
    'image_commit': '4f71554',
    'layer': '60',
    'codebook_size': '6',
    'top_k': '64',
    'threshold': '0.5',
}


def write_table(
        path, pairs=PAIRS, codebook_size=CODEBOOK_SIZE, extraction=None,
        min_npmi=None,
):
    """Write a small NPMI table with a provenance header, as the build does.

    Args:
        path: Where to write it.
        pairs: (feature_i, feature_j, count, npmi) rows.
        codebook_size: Value to record in the provenance, or None to omit it.
        extraction: Feature-space record to embed, or None to omit it.
        min_npmi: Threshold to record in the provenance, or None to omit it.

    Returns:
        The path written.
    """
    frame = pd.DataFrame(
        list(pairs), columns=['feature_i', 'feature_j', 'count', 'npmi'],
    )
    # Every value written as a string, and the nested extraction record JSON-encoded,
    # since write_csv_with_provenance writes each value verbatim.
    record = {'variant': 'no_nucleic', 'cluster_scheme': 'contacts'}
    if codebook_size is not None:
        record['codebook_size'] = str(codebook_size)
    if extraction is not None:
        record['extraction'] = json.dumps(dict(extraction), sort_keys=True)
    if min_npmi is not None:
        record['min_npmi'] = str(min_npmi)
    write_csv_with_provenance(frame, str(path), record)
    return str(path)


def build_lookup(pairs=PAIRS, codebook_size=CODEBOOK_SIZE):
    """Build a lookup directly, so most tests need no file on disk.

    Args:
        pairs: (feature_i, feature_j, count, npmi) rows.
        codebook_size: Side length of the matrix.

    Returns:
        The lookup, symmetrised the way load_npmi_lookup symmetrises.
    """
    feature_i = np.array([pair[0] for pair in pairs])
    feature_j = np.array([pair[1] for pair in pairs])
    npmi = np.array([pair[3] for pair in pairs], dtype=np.float32)
    off_diagonal = feature_i != feature_j
    matrix = csr_matrix(
        (
            np.concatenate((npmi, npmi[off_diagonal])),
            (
                np.concatenate((feature_i, feature_j[off_diagonal])),
                np.concatenate((feature_j, feature_i[off_diagonal])),
            ),
        ),
        shape=(codebook_size, codebook_size), dtype=np.float32,
    )
    return NpmiLookup(
        matrix=matrix, provenance={}, codebook_size=codebook_size,
        n_pairs=len(pairs),
    )


def test_load_npmi_lookup_mirrors_off_diagonal(tmp_path):
    lookup = load_npmi_lookup(write_table(tmp_path / 'table.csv'))
    dense = lookup.matrix.toarray()
    assert dense[0, 1] == pytest.approx(0.9)
    assert dense[1, 0] == pytest.approx(0.9)
    assert np.array_equal(dense, dense.T)


def test_load_npmi_lookup_does_not_double_the_diagonal(tmp_path):
    # The table emits one row per unordered pair, so a naive counts + counts.T would
    # give 1.4 here and inflate every self-pair's contribution to eq 17.
    lookup = load_npmi_lookup(write_table(tmp_path / 'table.csv'))
    assert lookup.matrix.toarray()[3, 3] == pytest.approx(0.7)


def test_load_npmi_lookup_needs_a_codebook_size(tmp_path):
    path = write_table(tmp_path / 'table.csv', codebook_size=None)
    with pytest.raises(ValueError, match='codebook_size'):
        load_npmi_lookup(path)


def test_load_npmi_lookup_rejects_ids_outside_the_codebook(tmp_path):
    path = write_table(
        tmp_path / 'table.csv', pairs=((0, 9, 1.0, 0.5),), codebook_size=6,
    )
    with pytest.raises(ValueError, match='outside the codebook'):
        load_npmi_lookup(path)


def test_load_npmi_lookup_min_count_drops_rare_pairs(tmp_path):
    path = write_table(tmp_path / 'table.csv')
    lookup = load_npmi_lookup(path, min_count=2.5)
    # Counts of 4, 3 and 5 survive; 2 and 1 do not.
    assert lookup.n_pairs == 3
    assert lookup.matrix.toarray()[1, 2] == pytest.approx(0.0)


def test_top_features_sorts_by_descending_magnitude():
    offsets = np.array([0, 3, 5])
    feature_ids = np.array([7, 2, 5, 9, 4])
    magnitudes = np.array([0.3, 0.9, 0.5, 0.2, 0.8], dtype=np.float32)
    top = top_features(offsets, feature_ids, magnitudes, top_k=3)
    assert top.tolist() == [[2, 5, 7], [4, 9, 9]]


def test_top_features_is_a_prefix_of_a_wider_k():
    offsets = np.array([0, 3, 5])
    feature_ids = np.array([7, 2, 5, 9, 4])
    magnitudes = np.array([0.3, 0.9, 0.5, 0.2, 0.8], dtype=np.float32)
    wide = top_features(offsets, feature_ids, magnitudes, top_k=3)
    narrow = top_features(offsets, feature_ids, magnitudes, top_k=2)
    # The property the whole sweep design rests on: one stored array serves every topk.
    assert np.array_equal(narrow, wide[:, :2])


def test_top_features_breaks_ties_on_feature_id():
    offsets = np.array([0, 2])
    feature_ids = np.array([9, 3])
    magnitudes = np.array([0.5, 0.5], dtype=np.float32)
    top = top_features(offsets, feature_ids, magnitudes, top_k=2)
    assert top.tolist() == [[3, 9]]


def test_top_features_rejects_a_residue_with_nothing_active():
    offsets = np.array([0, 0, 2])
    feature_ids = np.array([1, 2])
    magnitudes = np.array([0.9, 0.8], dtype=np.float32)
    with pytest.raises(ValueError, match='no active features'):
        top_features(offsets, feature_ids, magnitudes, top_k=2)


def test_top_features_rejects_offsets_that_do_not_describe_the_arrays():
    with pytest.raises(ValueError, match='describe different data'):
        top_features(
            np.array([0, 3]), np.array([1, 2]),
            np.array([0.5, 0.5], dtype=np.float32),
        )


def test_residue_pair_npmi_worked_example():
    """R(i, j) = max NPMI over the k*k feature combinations (eq 17).

    A residue 0 holds features {0, 3} and residue 1 {2, 5}; B residue 0 holds {1, 4} and
    residue 1 {3, 5}. Against the table above:
        R(0,0) = max(M[0,1], M[0,4], M[3,1], M[3,4]) = max(0.9, 0, 0, 0) = 0.9
        R(0,1) = max(M[0,3], M[0,5], M[3,3], M[3,5]) = max(0, 0, 0.7, 0) = 0.7
        R(1,0) = max(M[2,1], M[2,4], M[5,1], M[5,4]) = max(0.2, 0.1, 0, 0) = 0.2
        R(1,1) = max(M[2,3], M[2,5], M[5,3], M[5,5]) = 0
    """
    result = residue_pair_npmi(TOP_A, TOP_B, build_lookup())
    assert result == pytest.approx(EXPECTED_R, abs=1e-6)


def test_residue_pair_npmi_is_unaffected_by_the_row_block():
    lookup = build_lookup()
    whole = residue_pair_npmi(TOP_A, TOP_B, lookup, row_block=64)
    split = residue_pair_npmi(TOP_A, TOP_B, lookup, row_block=1)
    assert np.array_equal(whole, split)


def test_residue_pair_npmi_transposes_when_the_proteins_swap():
    lookup = build_lookup()
    forward = residue_pair_npmi(TOP_A, TOP_B, lookup)
    reverse = residue_pair_npmi(TOP_B, TOP_A, lookup)
    assert np.array_equal(forward, reverse.T)


def test_residue_pair_npmi_rejects_mismatched_k():
    with pytest.raises(ValueError, match='one k for both'):
        residue_pair_npmi(TOP_A, TOP_B[:, :1], build_lookup())


def test_selected_pair_count_floors_and_keeps_at_least_one():
    # 0.25 * 2 floors to zero, which would select nothing at all.
    assert selected_pair_count(2, 2, rho=0.25, t_max=70) == 1
    assert selected_pair_count(100, 300, rho=0.25, t_max=70) == 25
    assert selected_pair_count(400, 300, rho=0.25, t_max=70) == 70


def test_selected_pair_count_uses_the_smaller_protein():
    assert selected_pair_count(40, 4000, rho=0.5, t_max=70) == 20


def test_score_pair_takes_the_mean_of_the_top_t():
    # rho = 1.0 makes T = 2 on this pair, selecting 0.9 and 0.7.
    result = score_pair(TOP_A, TOP_B, build_lookup(), rho=1.0, t_max=70)
    assert result.score == pytest.approx(0.8, abs=1e-6)
    assert result.n_selected == 2
    assert result.n_positive == 2
    assert result.status == STATUS_OK


def test_score_pair_counts_selected_pairs_absent_from_the_table():
    # T = 4 selects every residue pair, one of which no feature combination reaches.
    result = score_pair(TOP_A, TOP_B, build_lookup(), rho=2.0, t_max=70)
    assert result.n_selected == 4
    assert result.n_positive == 3
    assert result.score == pytest.approx((0.9 + 0.7 + 0.2) / 4, abs=1e-6)


def test_score_pair_is_symmetric():
    lookup = build_lookup()
    forward = score_pair(TOP_A, TOP_B, lookup, rho=1.0)
    reverse = score_pair(TOP_B, TOP_A, lookup, rho=1.0)
    assert forward.score == pytest.approx(reverse.score)
    assert forward.n_selected == reverse.n_selected


def test_score_pair_reports_a_protein_with_no_active_residues():
    empty = np.empty((0, 2), dtype=np.int32)
    result = score_pair(empty, TOP_B, build_lookup())
    assert np.isnan(result.score)
    assert result.status == STATUS_MISSING_A
    result = score_pair(TOP_A, empty, build_lookup())
    assert result.status == STATUS_MISSING_B


def test_score_pair_set_scores_every_row_and_keeps_labels():
    pairs = pd.DataFrame({
        'pair_id': ['p0', 'p1'],
        'id_a': ['A', 'A'],
        'id_b': ['B', 'ABSENT'],
        'label': [1, 0],
    })
    features = {'A': TOP_A, 'B': TOP_B}
    scores = score_pair_set(
        pairs, features, build_lookup(), top_k=2, rho=1.0,
    )
    assert list(scores['pair_id']) == ['p0', 'p1']
    assert scores.loc[0, 'score'] == pytest.approx(0.8, abs=1e-6)
    assert scores.loc[0, 'status'] == STATUS_OK
    assert np.isnan(scores.loc[1, 'score'])
    assert scores.loc[1, 'status'] == STATUS_MISSING_B
    assert list(scores['label']) == [1, 0]


def test_score_pair_set_slices_the_stored_features_to_top_k():
    """Stored at k = 2, scored at k = 1, so only each residue's strongest feature counts.

    A reduces to [[0], [2]] and B to [[1], [3]], leaving M[0,1] = 0.9 and M[2,1] = 0.2
    reachable. R(0,1) = M[0,3] and R(1,1) = M[2,3] are both absent from the table, so the
    0.7 that the diagonal pair contributed at k = 2 is gone.
    """
    pairs = pd.DataFrame({'pair_id': ['p0'], 'id_a': ['A'], 'id_b': ['B']})
    scores = score_pair_set(
        pairs, {'A': TOP_A, 'B': TOP_B}, build_lookup(), top_k=1, rho=2.0,
    )
    assert scores.loc[0, 'n_positive'] == 2
    assert scores.loc[0, 'score'] == pytest.approx(1.1 / 4, abs=1e-6)


def test_score_pair_set_rejects_a_top_k_wider_than_the_store():
    pairs = pd.DataFrame({'pair_id': ['p0'], 'id_a': ['A'], 'id_b': ['B']})
    with pytest.raises(ValueError, match='stored at 2'):
        score_pair_set(
            pairs, {'A': TOP_A, 'B': TOP_B}, build_lookup(), top_k=3,
        )


def test_score_pair_set_needs_the_id_columns():
    pairs = pd.DataFrame({'pair_id': ['p0'], 'id_a': ['A']})
    with pytest.raises(ValueError, match='id_b'):
        score_pair_set(pairs, {}, build_lookup())


def test_check_feature_space_accepts_a_matching_run(tmp_path):
    path = write_table(tmp_path / 'table.csv', extraction=FEATURE_SPACE)
    lookup = load_npmi_lookup(path)
    check_feature_space(lookup, FEATURE_SPACE)


def test_check_feature_space_rejects_a_different_statistics_file(tmp_path):
    path = write_table(tmp_path / 'table.csv', extraction=FEATURE_SPACE)
    lookup = load_npmi_lookup(path)
    query = dict(FEATURE_SPACE, statistics_md5='different')
    with pytest.raises(ValueError, match='statistics_md5'):
        check_feature_space(lookup, query)


def test_check_feature_space_only_warns_on_a_rebuilt_image(tmp_path):
    # A rebuild changes image_commit even when it touches nothing the activations depend
    # on, so blocking on it would refuse every run after any rebuild.
    path = write_table(tmp_path / 'table.csv', extraction=FEATURE_SPACE)
    lookup = load_npmi_lookup(path)
    check_feature_space(lookup, dict(FEATURE_SPACE, image_commit='81e98e0'))


def test_check_feature_space_tolerates_a_table_without_the_record(tmp_path):
    path = write_table(tmp_path / 'table.csv')
    lookup = load_npmi_lookup(path)
    check_feature_space(lookup, FEATURE_SPACE)


def test_check_feature_space_accepts_a_table_predating_a_key(tmp_path):
    # A key added to the fingerprint after a table was built is absent from it, which is
    # not disagreement: reading it as one would refuse every table built before the key.
    older = {
        key: value for key, value in FEATURE_SPACE.items() if key != 'esm_revision'
    }
    path = write_table(tmp_path / 'table.csv', extraction=older)
    lookup = load_npmi_lookup(path)
    check_feature_space(lookup, dict(FEATURE_SPACE, esm_revision='esmsha'))


def test_check_feature_space_still_compares_the_keys_a_table_does_carry(tmp_path):
    # The tolerance above must not become a way past the whole check.
    older = {
        key: value for key, value in FEATURE_SPACE.items() if key != 'esm_revision'
    }
    path = write_table(tmp_path / 'table.csv', extraction=older)
    lookup = load_npmi_lookup(path)
    query = dict(
        FEATURE_SPACE, esm_revision='esmsha',
        fused_kernels='xformers',
    )
    with pytest.raises(ValueError, match='fused_kernels'):
        check_feature_space(lookup, query)


def test_max_top_k_covers_the_papers_sweep():
    # The sweep runs topk over [3, 5, 7, 9, 11, 13, 15], and the store is what makes each
    # of those a prefix slice rather than a re-read.
    assert MAX_TOP_K == 15


def test_load_npmi_lookup_refuses_a_table_holding_negative_npmi(tmp_path):
    # eq 17 floors an absent pair at zero, so a table keeping negatives would rank a pair
    # it does hold below one it does not. min_npmi is a config value, not a constant.
    path = write_table(tmp_path / 'npmi_table.csv', min_npmi=-0.5)
    with pytest.raises(ValueError, match='negative NPMI'):
        load_npmi_lookup(path)


def test_load_npmi_lookup_accepts_the_paper_threshold(tmp_path):
    path = write_table(tmp_path / 'npmi_table.csv', min_npmi=0.0)
    assert load_npmi_lookup(path).codebook_size == CODEBOOK_SIZE
