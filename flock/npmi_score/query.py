from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from flock.npmi_score.schema import FEATURE_FIELDS
from flock.npmi_score.score import MAX_TOP_K
from flock.npmi_score.score import top_features

# The query side of the feature table, keyed by protein rather than by interface. The
# NPMI count drops magnitudes; scoring is the consumer that needs them, since it takes
# the top-k active features per residue by activation magnitude. side is left unread: a
# query protein has no interface frame, so extraction pins it to one value.
QUERY_FEATURE_COLUMNS = tuple(
    name for name, _ in FEATURE_FIELDS if name != 'side'
)

# Ids named per category in a log line. The counts themselves are exact.
REPORT_EXAMPLES = 5


@dataclass
class QueryFeatures:
    """Per-protein top-k active features, ready for scoring.

    Attributes:
        top: Protein id to its (n_active_residues, stored_k) int32 feature ids, each row
            sorted by descending magnitude so a smaller top-k is a prefix slice.
        stored_k: Features kept per residue, the ceiling any topk may be swept to.
        n_shards: Feature shards read.
        n_residues: Active residues across every protein loaded.
    """

    top: dict[str, np.ndarray] = field(default_factory=dict)
    stored_k: int = MAX_TOP_K
    n_shards: int = 0
    n_residues: int = 0

    def __len__(self) -> int:
        """Return how many proteins the store holds."""
        return len(self.top)


def _shard_proteins(
        path: str,
        max_top_k: int,
        wanted: set[str] | None,
) -> dict[str, np.ndarray]:
    """Read one feature shard into per-protein top-k features.

    Args:
        path: Local path to one shard's features.parquet.
        max_top_k: Features to keep per residue.
        wanted: Only load these protein ids, None for every protein in the shard.

    Returns:
        The shard's top-k features, keyed by protein id.
    """
    table = pq.read_table(
        path, columns=list(
            QUERY_FEATURE_COLUMNS,
        ),
    ).combine_chunks()
    proteins = pc.dictionary_encode(table.column('interface_id').chunk(0))
    codes = proteins.indices.to_numpy(zero_copy_only=False)
    names = proteins.dictionary.to_pylist()

    positions = table.column('position').chunk(0).to_numpy()
    id_lists = table.column('feature_ids').chunk(0)
    magnitude_lists = table.column('magnitudes').chunk(0)
    flat_ids = id_lists.flatten().to_numpy(zero_copy_only=False)
    flat_magnitudes = magnitude_lists.flatten().to_numpy(zero_copy_only=False)
    lengths = id_lists.value_lengths().to_numpy(zero_copy_only=False)
    bounds = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)

    # A stable argsort on the dictionary codes, so each protein's rows are one contiguous
    # run and no protein id is ever compared as a string. Rows arrive in extraction
    # order, not protein order, so grouping is needed either way.
    order = np.argsort(codes, kind='stable')
    ordered = codes[order]
    keys = np.arange(len(names))
    starts = np.searchsorted(ordered, keys, side='left')
    stops = np.searchsorted(ordered, keys, side='right')
    del table, proteins, codes, ordered, id_lists, magnitude_lists

    top: dict[str, np.ndarray] = {}
    for key, name in enumerate(names):
        protein_id = str(name)
        if wanted is not None and protein_id not in wanted:
            continue
        rows = order[starts[key]:stops[key]]
        if not rows.size:
            continue
        # Position order rather than the shard's row order, so the residue axis of the
        # score is reproducible and traceable back to the sequence.
        rows = rows[np.argsort(positions[rows], kind='stable')]
        offsets = np.concatenate(
            ([0], np.cumsum(bounds[rows + 1] - bounds[rows])),
        ).astype(np.int64)
        picks = np.concatenate([
            np.arange(bounds[row], bounds[row + 1]) for row in rows
        ])
        top[protein_id] = top_features(
            offsets, flat_ids[picks], flat_magnitudes[picks], max_top_k,
        )
    return top


def load_query_features(
        shard_paths: Iterable[str],
        max_top_k: int = MAX_TOP_K,
        wanted: set[str] | None = None,
) -> QueryFeatures:
    """Load a query extraction run into per-protein top-k features.

    Stored at max_top_k rather than the topk a run scores with, so a sweep over the
    paper's grid is a prefix slice rather than a second pass.

    Args:
        shard_paths: Local paths to each shard's features.parquet, consumed once.
        max_top_k: Features to keep per residue.
        wanted: Only load these protein ids, None for every protein present.

    Returns:
        The store, keyed by the protein id extraction wrote as interface_id.

    Raises:
        ValueError: If one protein appears in more than one shard, which would mean the
            shards come from two different runs and the second silently wins.
    """
    logger = logging.getLogger(__name__)
    store = QueryFeatures(stored_k=max_top_k)
    for path in shard_paths:
        top = _shard_proteins(path, max_top_k, wanted)
        repeated = sorted(set(top) & set(store.top))
        if repeated:
            raise ValueError(
                f'{len(repeated)} protein(s) appear in more than one feature shard, '
                f'most recently {path}, e.g. '
                f'{", ".join(repeated[:REPORT_EXAMPLES])}; the shards are from two '
                f'different runs',
            )
        store.top.update(top)
        store.n_shards += 1
        store.n_residues += sum(int(array.shape[0]) for array in top.values())
    logger.info(
        '%d proteins and %d active residues from %d shard(s), top %d per residue',
        len(store), store.n_residues, store.n_shards, max_top_k,
    )
    return store


def missing_proteins(
        store: QueryFeatures,
        protein_ids: Iterable[str],
) -> list[str]:
    """Return the proteins a pair list names that the features do not hold.

    Reported before scoring, so a run that lost a whole protein is visible up front.

    Args:
        store: The loaded features.
        protein_ids: Every protein the pair list names.

    Returns:
        The absent ids, sorted and deduplicated.
    """
    return sorted({
        str(protein_id) for protein_id in protein_ids
        if str(protein_id) not in store.top
    })
