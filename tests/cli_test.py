from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

from flock.npmi_score.io import parquet_provenance
from flock.npmi_score.io import resolve_feature_files
from flock.npmi_score.schema import STATUS_MISSING_A
from flock.npmi_score.schema import STATUS_MISSING_B
from flock.npmi_score.schema import STATUS_OK
from flock.npmi_score.score_pairs import main
from tests.conftest import EXTRACTION

# The CLI's own behaviour. The scoring maths is covered by score_test.py; what is checked
# here is the file handling around it: which files are found, which mismatches are
# refused, and that a refusal happens before any output is written.


def run_cli(monkeypatch, tmp_path, pairs, features, table, **extra):
    """Run the CLI in-process and return the scores it wrote."""
    output = str(tmp_path / 'scores.csv')
    argv = [
        'score_pairs', '--pairs', pairs, '--npmi-table', table, '--output', output,
        '--features', *([features] if isinstance(features, str) else features),
    ]
    for key, value in extra.items():
        argv += [f'--{key.replace("_", "-")}', str(value)]
    monkeypatch.setattr(sys, 'argv', argv)
    main()
    return output


def test_scores_a_pair_end_to_end(
    monkeypatch, tmp_path, write_pairs, write_features,
    write_npmi_table,
):
    output = run_cli(
        monkeypatch, tmp_path, write_pairs(), write_features(), write_npmi_table(),
    )
    frame = pd.read_csv(output)
    assert list(frame['pair_id']) == ['p0']
    assert frame.loc[0, 'status'] == STATUS_OK
    assert frame.loc[0, 'n_active_a'] == 2
    assert frame.loc[0, 'n_active_b'] == 2
    # T floors rho * min(2, 2) to zero, and the floor of one applies.
    assert frame.loc[0, 'n_selected'] == 1


def test_a_pair_naming_an_absent_protein_scores_nan(
    monkeypatch, tmp_path, write_pairs,
    write_features, write_npmi_table,
):
    pairs = write_pairs(
        rows=(
            ('p0', 'A', 'B'), ('p1', 'A', 'ABSENT'),
            ('p2', 'ABSENT', 'B'),
        ),
    )
    output = run_cli(
        monkeypatch, tmp_path, pairs, write_features(), write_npmi_table(),
    )
    frame = pd.read_csv(output).set_index('pair_id')
    assert frame.loc['p0', 'status'] == STATUS_OK
    assert frame.loc['p1', 'status'] == STATUS_MISSING_B
    assert frame.loc['p2', 'status'] == STATUS_MISSING_A
    # Named and NaN rather than dropped: a dropped row cannot be joined back to its input.
    assert pd.isna(frame.loc['p1', 'score'])
    assert pd.isna(frame.loc['p2', 'score'])


def test_features_may_be_a_directory_a_glob_or_a_file(tmp_path, write_features):
    first = write_features(name='shard_0.parquet')
    second = write_features(
        name='shard_1.parquet',
        proteins={'C': {0: [(1, 1.0)]}},
    )
    directory = os.path.dirname(first)
    assert resolve_feature_files([directory]) == sorted([first, second])
    assert resolve_feature_files([os.path.join(directory, '*.parquet')]) == sorted(
        [first, second],
    )
    assert resolve_feature_files([first]) == [first]


def test_no_matching_feature_files_is_refused(tmp_path):
    # Rather than every pair scoring NaN, which reads as a run that worked.
    with pytest.raises(ValueError, match='no Parquet files matched'):
        resolve_feature_files([str(tmp_path / 'nothing_here')])


def test_a_protein_in_two_files_is_refused(
    monkeypatch, tmp_path, write_pairs,
    write_features, write_npmi_table,
):
    write_features(name='shard_0.parquet')
    write_features(name='shard_1.parquet')
    directory = str(tmp_path / 'features')
    with pytest.raises(ValueError, match='more than one feature shard'):
        run_cli(
            monkeypatch, tmp_path, write_pairs(), directory, write_npmi_table(),
        )


def test_a_different_statistics_file_is_refused(
    monkeypatch, tmp_path, write_pairs,
    write_features, write_npmi_table,
):
    # The one failure that would otherwise yield plausible numbers instead of an error.
    other = dict(EXTRACTION, statistics_md5='9c01182d1d22b2c76673610ed77d358b')
    features = write_features(extraction=EXTRACTION)
    table = write_npmi_table(extraction=other)
    with pytest.raises(ValueError, match='not in the feature space'):
        run_cli(monkeypatch, tmp_path, write_pairs(), features, table)
    assert not os.path.exists(tmp_path / 'scores.csv')


def test_unstamped_features_are_scored_with_a_warning(
    monkeypatch, tmp_path, caplog,
    write_pairs, write_features,
    write_npmi_table,
):
    # An extraction outside this package records nothing, so the check cannot run. That
    # must warn and proceed, not refuse, or no external query could ever be scored.
    features = write_features(extraction={})
    assert parquet_provenance(features) == {}
    output = run_cli(
        monkeypatch, tmp_path, write_pairs(), features, write_npmi_table(),
    )
    assert os.path.exists(output)
    assert 'record no extraction provenance' in caplog.text


def test_a_table_without_codebook_size_is_refused(
    monkeypatch, tmp_path, write_pairs,
    write_features, write_npmi_table,
):
    table = write_npmi_table()
    stripped = str(tmp_path / 'no_size.csv')
    with open(table) as file_in, open(stripped, 'w') as file_out:
        for line in file_in:
            if not line.startswith('# codebook_size'):
                file_out.write(line)
    with pytest.raises(ValueError, match='carries no codebook_size'):
        run_cli(
            monkeypatch, tmp_path, write_pairs(),
            write_features(), stripped,
        )


def test_a_table_holding_negative_npmi_is_refused(
    monkeypatch, tmp_path, write_pairs,
    write_features, write_npmi_table,
):
    # Scoring floors an absent pair at zero, which only reads as not-positive while the
    # table holds nothing below zero.
    table = write_npmi_table(min_npmi=-1.0)
    with pytest.raises(ValueError, match='holds negative NPMI'):
        run_cli(monkeypatch, tmp_path, write_pairs(), write_features(), table)


def test_a_pair_list_missing_an_id_column_is_refused(
    monkeypatch, tmp_path, write_pairs,
    write_features, write_npmi_table,
):
    pairs = write_pairs(rows=(('p0', 'A'),), columns=('pair_id', 'id_a'))
    with pytest.raises(ValueError, match='lacks columns the pair list needs'):
        run_cli(
            monkeypatch, tmp_path, pairs,
            write_features(), write_npmi_table(),
        )


def test_a_repeated_pair_id_is_refused(
    monkeypatch, tmp_path, write_pairs, write_features,
    write_npmi_table,
):
    pairs = write_pairs(rows=(('p0', 'A', 'B'), ('p0', 'B', 'A')))
    with pytest.raises(ValueError, match='more than once'):
        run_cli(
            monkeypatch, tmp_path, pairs,
            write_features(), write_npmi_table(),
        )


def test_the_config_must_carry_the_scoring_keys(
    monkeypatch, tmp_path, write_pairs,
    write_features, write_npmi_table,
):
    config = tmp_path / 'partial.yaml'
    config.write_text('top_k: 3\nrho: 0.25\n')
    with pytest.raises(ValueError, match='missing required keys: t_max'):
        run_cli(
            monkeypatch, tmp_path, write_pairs(), write_features(), write_npmi_table(),
            config=str(config),
        )
