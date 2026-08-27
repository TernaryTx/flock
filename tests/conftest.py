from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from flock.npmi_score.schema import build_table
from flock.npmi_score.schema import features_schema
from flock.provenance import write_csv_with_provenance

# Fixtures shared by the CLI tests. Built with the package's own writers rather than by
# hand, so a change to the contract breaks these rather than leaving them describing a
# format nothing produces any more.
#
# write_csv_with_provenance writes each value verbatim, so the nested extraction record is
# JSON-encoded here and every value is a string. NpmiLookup.extraction json.loads it back.

CODEBOOK = 32

# A feature space the table and the query features can be made to agree or disagree on.
EXTRACTION = {
    'esmc_repo': 'EvolutionaryScale/esmc-6b-2024-12',
    'esmc_revision': 'aaa1111',
    'sae_repo': 'biohub/ESMC-6B-sae-layer60-k64-codebook16384',
    'sae_revision': 'bbb2222',
    'transformers_revision': 'ccc3333',
    'esm_revision': 'ddd4444',
    'fused_kernels': 'none',
    'statistics_md5': 'b75317aab86ba3eecd696b8d8ff2ba2e',
    'image_commit': 'eee5555',
    'layer': '60',
    'codebook_size': str(CODEBOOK),
    'top_k': '64',
    'threshold': '0.5',
}


@pytest.fixture
def write_npmi_table(tmp_path):
    """Return a writer for an NPMI table with a provenance header."""
    def write(
        name='npmi_table.csv', extraction=EXTRACTION, codebook_size=CODEBOOK,
        min_npmi=0.0, pairs=((0, 1, 10.0, 0.8), (2, 3, 5.0, 0.6), (1, 1, 4.0, 0.4)),
    ):
        frame = pd.DataFrame(
            list(pairs), columns=['feature_i', 'feature_j', 'count', 'npmi'],
        )
        record = {
            'codebook_size': str(codebook_size), 'min_npmi': str(min_npmi),
        }
        if extraction is not None:
            record['extraction'] = json.dumps(dict(extraction), sort_keys=True)
        path = tmp_path / name
        write_csv_with_provenance(frame, str(path), record)
        return str(path)
    return write


@pytest.fixture
def write_features(tmp_path):
    """Return a writer for a features Parquet file in the FEATURE_FIELDS contract."""
    def write(name='features.parquet', proteins=None, extraction=None, subdir='features'):
        proteins = proteins or {
            'A': {0: [(0, 2.0), (2, 1.0)], 1: [(1, 3.0)]},
            'B': {0: [(1, 2.5)], 1: [(3, 1.5), (0, 1.0)]},
        }
        interface_ids, sides, positions, offsets, ids, magnitudes = [], [], [], [0], [], []
        for name_, residues in proteins.items():
            for position in sorted(residues):
                active = residues[position]
                interface_ids.append(name_)
                sides.append('a')
                positions.append(position)
                ids.extend(feature for feature, _ in active)
                magnitudes.extend(magnitude for _, magnitude in active)
                offsets.append(len(ids))
        schema = features_schema(extraction or {})
        table = build_table(
            np.array(interface_ids), np.array(
                sides,
            ), np.array(positions, dtype=np.int32),
            np.array(offsets, dtype=np.int32), np.array(ids, dtype=np.int32),
            np.array(magnitudes, dtype=np.float32), schema,
        )
        directory = tmp_path / subdir
        directory.mkdir(exist_ok=True)
        path = directory / name
        pq.write_table(table, path)
        return str(path)
    return write


@pytest.fixture
def write_pairs(tmp_path):
    """Return a writer for a pair list."""
    def write(name='pairs.csv', rows=(('p0', 'A', 'B'),), columns=('pair_id', 'id_a', 'id_b')):
        path = tmp_path / name
        pd.DataFrame(
            list(rows), columns=list(
                columns,
            ),
        ).to_csv(path, index=False)
        return str(path)
    return write
