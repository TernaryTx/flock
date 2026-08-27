from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

from flock.npmi_score import EXTRACTION_FEATURE_SPACE_KEYS
from flock.npmi_score.schema import LABEL_COLUMN
from flock.npmi_score.schema import PAIR_COLUMNS
from flock.npmi_score.schema import SCORE_COLUMNS
from flock.npmi_score.schema import STATUS_MISSING_A
from flock.npmi_score.schema import STATUS_MISSING_B
from flock.npmi_score.schema import STATUS_OK
from flock.npmi_score.schema import TABLE_DTYPES
from flock.provenance import csv_provenance_lines
from flock.provenance import read_csv_provenance

# Query-pair scoring: for each residue pair the best NPMI across the top-k active features
# of each side, then the mean over the top-T residue pairs. These are equations 17 and 18
# of the paper, which is the reference for the wording below; the README's query pair
# scoring section restates them.

# The paper's tuned scoring parameters. configs/npmi.yaml is what a run reads; these are
# the library defaults.
DEFAULT_TOP_K = 3
DEFAULT_RHO = 0.25
DEFAULT_T_MAX = 70

# Ceiling of the paper's topk sweep, so a smaller topk is a prefix slice of one array.
MAX_TOP_K = 15

# Residues of A gathered at once, bounding the sparse intermediate that grows with length.
DEFAULT_ROW_BLOCK = 512

# What must agree between the features a query was extracted into and the ones the table
# was counted over. image_commit changes on every rebuild, so it warns rather than fails.
FEATURE_SPACE_MATCH_KEYS = tuple(
    key for key in EXTRACTION_FEATURE_SPACE_KEYS if key != 'image_commit'
)


@dataclass
class NpmiLookup:
    """The NPMI table as a symmetric sparse matrix, with what identifies it.

    Attributes:
        matrix: (codebook_size, codebook_size) float32 CSR of NPMI values, symmetric so
            that a lookup can take a from either protein. A pair the table does not
            hold is absent, which the residue-pair max reads as zero; see score_pair.
        provenance: The table's own provenance header, carried into the scores file.
        codebook_size: Side length of the matrix, from the table's provenance.
        n_pairs: Unordered pairs the table held, after any min_count filter.
    """

    matrix: csr_matrix
    provenance: dict[str, str]
    codebook_size: int
    n_pairs: int

    @property
    def extraction(self) -> dict[str, str]:
        """Return the extraction fingerprint of the corpus the table was counted over.

        Returns:
            The feature-space keys the table builder recorded, empty for a table
            written before it recorded them.
        """
        recorded = self.provenance.get('extraction')
        if not recorded:
            return {}
        parsed = json.loads(recorded)
        return {key: str(value) for key, value in parsed.items()}


@dataclass
class PairScore:
    """One query pair's interaction score and what produced it.

    Attributes:
        score: Mean NPMI of the selected residue pairs, NaN where undefined.
        n_active_a: Residues of A with at least one active feature, its |A_active|.
        n_active_b: The same for B.
        n_selected: T, the residue pairs the mean was taken over.
        n_positive: How many of those T carried a positive NPMI. The rest were pairs the
            table does not hold, counted as zero.
        status: STATUS_OK, or why the pair has no score.
    """

    score: float
    n_active_a: int
    n_active_b: int
    n_selected: int
    n_positive: int
    status: str


def load_npmi_lookup(path: str, min_count: float = 0.0) -> NpmiLookup:
    """Load a built NPMI table into a symmetric sparse matrix for random access.

    CSV is the archive format, not the query format: scoring one protein pair is
    millions of random lookups.

    Args:
        path: Local path to a published NPMI table.
        min_count: Drop pairs whose accumulated count is not above this, as a floor on
            rare features. Zero keeps every pair, the paper's method as stated.

    Returns:
        The lookup, carrying the table's provenance so a score can be traced to it.

    Raises:
        ValueError: If the table has no codebook_size in its provenance, lacks a column,
            holds a feature id outside the codebook, or was built keeping negative NPMI.
    """
    logger = logging.getLogger(__name__)
    record = read_csv_provenance(path)
    declared = record.get('codebook_size')
    if declared is None:
        raise ValueError(
            f'{path} carries no codebook_size in its provenance header, so the lookup '
            f'cannot be sized. It was not written by the NPMI table builder.',
        )
    codebook_size = int(declared)

    # Scoring floors an absent pair at zero, which only reads as not-positive while the
    # table holds nothing below zero. min_npmi is a config value, so it is checked.
    built_above = float(record.get('min_npmi', 0.0))
    if built_above < 0:
        raise ValueError(
            f'{path} was built with min_npmi={built_above}, so it holds negative NPMI. '
            f'Scoring reads a pair the table lacks as zero, which would rank those '
            f'below an absent pair rather than above it.',
        )

    skiprows = csv_provenance_lines(path)
    present = pd.read_csv(path, nrows=0, skiprows=skiprows).columns
    missing = [name for name in TABLE_DTYPES if name not in present]
    if missing:
        raise ValueError(
            f'{path} lacks columns the scorer needs: {", ".join(missing)}',
        )
    frame = pd.read_csv(
        path, skiprows=skiprows, usecols=list(TABLE_DTYPES), dtype=TABLE_DTYPES,
    )
    if min_count > 0:
        kept = frame['count'] > min_count
        logger.info(
            '%d of %d pairs kept above count %.3f',
            int(kept.sum()), len(frame), min_count,
        )
        frame = frame[kept]

    feature_i = frame['feature_i'].to_numpy()
    feature_j = frame['feature_j'].to_numpy()
    npmi = frame['npmi'].to_numpy(dtype=np.float32)
    if feature_i.size:
        highest = max(int(feature_i.max()), int(feature_j.max()))
        lowest = min(int(feature_i.min()), int(feature_j.min()))
        if lowest < 0 or highest >= codebook_size:
            raise ValueError(
                f'{path} holds feature ids in [{lowest}, {highest}], outside the '
                f'codebook [0, {codebook_size}) its provenance declares',
            )

    # The table holds one row per unordered pair as i <= j, so the mirrored half excludes
    # the diagonal: adding a self-pair twice would double its NPMI rather than symmetrise.
    off_diagonal = feature_i != feature_j
    rows = np.concatenate((feature_i, feature_j[off_diagonal]))
    cols = np.concatenate((feature_j, feature_i[off_diagonal]))
    values = np.concatenate((npmi, npmi[off_diagonal]))
    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(codebook_size, codebook_size), dtype=np.float32,
    ).tocsr()
    logger.info(
        '%d unordered pairs over a %d codebook, %d stored entries',
        len(frame), codebook_size, matrix.nnz,
    )
    return NpmiLookup(
        matrix=matrix,
        provenance=record,
        codebook_size=codebook_size,
        n_pairs=len(frame),
    )


def check_feature_space(
        lookup: NpmiLookup,
        extraction: Mapping[str, str],
) -> None:
    """Refuse query features extracted into a different feature space than the table's.

    The one failure mode in scoring that yields plausible numbers instead of an error:
    different weights or statistics give feature ids that do not mean what the table
    counted, so every lookup is silently wrong.

    Args:
        lookup: The loaded table, carrying the corpus extraction's fingerprint.
        extraction: The query run's extraction provenance, as io.parquet_provenance read
            it from the feature files' Parquet key-value metadata.

    Raises:
        ValueError: If any key that fixes the feature space disagrees.
    """
    logger = logging.getLogger(__name__)
    recorded = lookup.extraction
    if not recorded:
        logger.warning(
            'The NPMI table records no extraction provenance, so the query features '
            'cannot be checked against the corpus they will be looked up in.',
        )
        return
    # A key the table does not carry cannot be compared. That means a table built before
    # the key existed, not a matching feature space, so it is warned about rather than
    # read as agreement - and never treated as a difference, which would refuse every
    # older table outright.
    absent = [key for key in FEATURE_SPACE_MATCH_KEYS if key not in recorded]
    if absent:
        logger.warning(
            'The NPMI table predates %s in the feature-space fingerprint, so %s '
            'cannot be checked against it. Every key the table does carry is compared.',
            'them' if len(absent) > 1 else 'it', ', '.join(absent),
        )
    differences = [
        f'{key}: table {recorded.get(key)!r}, query {str(extraction.get(key))!r}'
        for key in FEATURE_SPACE_MATCH_KEYS
        if key in recorded and recorded[key] != str(extraction.get(key))
    ]
    if differences:
        raise ValueError(
            f'the query features are not in the feature space the NPMI table was '
            f'counted over: {"; ".join(differences)}. Feature ids would not mean what '
            f'the table holds, so every lookup would be silently wrong.',
        )
    table_image = recorded.get('image_commit')
    query_image = str(extraction.get('image_commit'))
    if table_image != query_image:
        logger.warning(
            'The query features were extracted by image %s against the table\'s %s. '
            'Every key that fixes the feature space agrees, so this is a rebuild rather '
            'than a different feature space, but it is not a bit-identical container.',
            query_image, table_image,
        )


def top_features(
        offsets: np.ndarray,
        feature_ids: np.ndarray,
        magnitudes: np.ndarray,
        top_k: int = MAX_TOP_K,
) -> np.ndarray:
    """Return each residue's strongest active features, padded to a fixed width.

    Rows come out sorted by descending magnitude, which makes any smaller top-k a prefix
    slice of the result rather than a second pass over the features.

    Args:
        offsets: Delimits each residue's run within feature_ids and magnitudes, so it is
            one longer than the number of residues.
        feature_ids: Active feature ids, grouped by residue.
        magnitudes: Their IDF-normalised magnitudes, in the same order.
        top_k: Features to keep per residue.

    Returns:
        An (n_residues, top_k) int32 array. A residue with fewer than top_k active
        features repeats its weakest, which the max over combinations reads identically.

    Raises:
        ValueError: If top_k is below one, if offsets does not describe the arrays given,
            or if a residue has no active features at all.
    """
    if top_k < 1:
        raise ValueError(f'top_k must be at least 1, got {top_k}')
    if offsets.size < 1 or int(offsets[-1]) != feature_ids.size:
        raise ValueError(
            f'offsets end at {int(offsets[-1]) if offsets.size else None} but '
            f'{feature_ids.size} feature ids were given; they describe different data',
        )
    if feature_ids.size != magnitudes.size:
        raise ValueError(
            f'{feature_ids.size} feature ids against {magnitudes.size} magnitudes',
        )
    counts = np.diff(offsets)
    if counts.size and int(counts.min()) < 1:
        raise ValueError(
            'a residue was given with no active features; extraction omits those '
            'rather than writing them empty, so this is a malformed feature table',
        )

    starts = offsets[:-1]
    # Descending magnitude within each residue, feature id breaking ties, so that a run
    # is reproducible: argpartition is not stable and a sweep compares runs to each other.
    residue_of = np.repeat(np.arange(counts.size), counts)
    order = np.lexsort((feature_ids, -magnitudes, residue_of))
    picks = np.minimum(
        starts[:, None] + np.arange(top_k)[None, :],
        (starts + counts - 1)[:, None],
    )
    return feature_ids[order[picks]].astype(np.int32)


def residue_pair_npmi(
        top_a: np.ndarray,
        top_b: np.ndarray,
        lookup: NpmiLookup,
        row_block: int = DEFAULT_ROW_BLOCK,
) -> np.ndarray:
    """Score every residue pair by the best NPMI among their features.

    Args:
        top_a: (n_a, k) top-k feature ids per active residue of protein A.
        top_b: (n_b, k) the same for protein B.
        lookup: The loaded NPMI table.
        row_block: Residues of A gathered at once.

    Returns:
        An (n_a, n_b) float32 array of R(i, j). A residue pair none of whose k*k feature
        combinations is in the table scores zero, since the table holds only NPMI above
        its threshold and an absent pair is therefore not-positive rather than known.

    Raises:
        ValueError: If the two arrays disagree on k, or row_block is below one.
    """
    if top_a.ndim != 2 or top_b.ndim != 2:
        raise ValueError(
            f'expected two 2-D arrays, got {top_a.ndim}-D and {top_b.ndim}-D',
        )
    if top_a.shape[1] != top_b.shape[1]:
        raise ValueError(
            f'top_a keeps {top_a.shape[1]} features per residue and top_b '
            f'{top_b.shape[1]}; the max is taken over one k for both',
        )
    if row_block < 1:
        raise ValueError(f'row_block must be at least 1, got {row_block}')

    top_k = top_a.shape[1]
    n_b = top_b.shape[0]
    columns = top_b.ravel()
    # Scattered into a flat buffer rather than a 2-D one: np.maximum.at resolves a
    # multi-dimensional index per element, measured at up to 1.9x one int64 index.
    result = np.zeros(top_a.shape[0] * n_b, dtype=np.float32)
    for start in range(0, top_a.shape[0], row_block):
        block = top_a[start:start + row_block]
        # One gather for the block's whole feature grid, folded down to residues by
        # integer division: floor(index / k) is the residue that feature slot belongs to.
        pairs = lookup.matrix[block.ravel()][:, columns].tocoo()
        if pairs.nnz:
            rows = (pairs.row // top_k).astype(np.int64) + start
            np.maximum.at(result, rows * n_b + pairs.col // top_k, pairs.data)
    return result.reshape(top_a.shape[0], n_b)


def selected_pair_count(
        n_active_a: int,
        n_active_b: int,
        rho: float = DEFAULT_RHO,
        t_max: int = DEFAULT_T_MAX,
) -> int:
    """Return T, the number of residue pairs a score is the mean of.

    Args:
        n_active_a: Residues of A with at least one active feature.
        n_active_b: The same for B.
        rho: Fraction of the smaller protein's active residues to score.
        t_max: Cap on the number of residue pairs.

    Returns:
        T, at least one. Eq 18 gives a real number and does not say how to make it an
        integer; this floors it, which decides most pairs rather than a few.
    """
    scaled = int(rho * min(n_active_a, n_active_b))
    return max(1, min(t_max, scaled))


def score_pair(
        top_a: np.ndarray,
        top_b: np.ndarray,
        lookup: NpmiLookup,
        rho: float = DEFAULT_RHO,
        t_max: int = DEFAULT_T_MAX,
        row_block: int = DEFAULT_ROW_BLOCK,
) -> PairScore:
    """Score one query pair as the mean NPMI of its top-T residue pairs.

    The score is symmetric by construction: swapping the arguments transposes R, and
    both T and the mean of the T largest values are invariant under that.

    Args:
        top_a: (n_a, k) top-k feature ids per active residue of protein A.
        top_b: (n_b, k) the same for protein B.
        lookup: The loaded NPMI table.
        rho: Fraction of the smaller protein's active residues to score.
        t_max: Cap on the number of residue pairs.
        row_block: Residues of A gathered at once.

    Returns:
        The score and the counts behind it. A protein with no active residues scores NaN
        rather than zero, which would rank it as a confident negative.
    """
    n_active_a = int(top_a.shape[0])
    n_active_b = int(top_b.shape[0])
    if not n_active_a or not n_active_b:
        return PairScore(
            score=float('nan'),
            n_active_a=n_active_a,
            n_active_b=n_active_b,
            n_selected=0,
            n_positive=0,
            status=STATUS_MISSING_A if not n_active_a else STATUS_MISSING_B,
        )

    n_selected = selected_pair_count(n_active_a, n_active_b, rho, t_max)
    residue_pairs = residue_pair_npmi(top_a, top_b, lookup, row_block).ravel()
    if n_selected >= residue_pairs.size:
        selected = residue_pairs
    else:
        selected = np.partition(residue_pairs, -n_selected)[-n_selected:]
    return PairScore(
        score=float(selected.mean()),
        n_active_a=n_active_a,
        n_active_b=n_active_b,
        n_selected=int(selected.size),
        n_positive=int(np.count_nonzero(selected > 0)),
        status=STATUS_OK,
    )


def score_pair_set(
        pairs: pd.DataFrame,
        features: Mapping[str, np.ndarray],
        lookup: NpmiLookup,
        top_k: int = DEFAULT_TOP_K,
        rho: float = DEFAULT_RHO,
        t_max: int = DEFAULT_T_MAX,
        row_block: int = DEFAULT_ROW_BLOCK,
) -> pd.DataFrame:
    """Score every pair in a pair list against the table.

    Takes a plain mapping rather than reading the shards itself, so this module stays
    free of pyarrow.

    Args:
        pairs: The pair list, carrying PAIR_COLUMNS and optionally LABEL_COLUMN.
        features: Protein id to its (n_active, stored_k) top-k feature ids, sliced to
            top_k here, which is why the store keeps the sweep ceiling.
        lookup: The loaded NPMI table.
        top_k: Features per residue the max is taken over.
        rho: Fraction of the smaller protein's active residues to score.
        t_max: Cap on the number of residue pairs.
        row_block: Residues of A gathered at once.

    Returns:
        One row per input pair, in input order, with SCORE_COLUMNS plus LABEL_COLUMN
        where the input carried it. A pair naming a protein absent from the features
        scores NaN and says so, rather than being dropped.

    Raises:
        ValueError: If the pair list lacks a required column, or top_k exceeds the width
            the features were stored at.
    """
    logger = logging.getLogger(__name__)
    missing = [name for name in PAIR_COLUMNS if name not in pairs.columns]
    if missing:
        raise ValueError(
            f'the pair list lacks columns the scorer needs: {", ".join(missing)}',
        )
    stored = {array.shape[1] for array in features.values()}
    if stored and top_k > min(stored):
        raise ValueError(
            f'top_k is {top_k} but the features were stored at {min(stored)} per '
            f'residue; re-load them with a larger max_top_k',
        )

    rows = []
    for pair in pairs.itertuples(index=False):
        id_a, id_b = str(pair.id_a), str(pair.id_b)
        top_a = features.get(id_a)
        top_b = features.get(id_b)
        if top_a is None or top_b is None:
            result = PairScore(
                score=float('nan'),
                n_active_a=0 if top_a is None else int(top_a.shape[0]),
                n_active_b=0 if top_b is None else int(top_b.shape[0]),
                n_selected=0,
                n_positive=0,
                status=(
                    STATUS_MISSING_A if top_a is None else STATUS_MISSING_B
                ),
            )
        else:
            result = score_pair(
                top_a[:, :top_k], top_b[:, :top_k], lookup, rho, t_max, row_block,
            )
        row = {
            'pair_id': str(pair.pair_id),
            'id_a': id_a,
            'id_b': id_b,
            'score': result.score,
            'n_active_a': result.n_active_a,
            'n_active_b': result.n_active_b,
            'n_selected': result.n_selected,
            'n_positive': result.n_positive,
            'status': result.status,
        }
        if LABEL_COLUMN in pairs.columns:
            row[LABEL_COLUMN] = getattr(pair, LABEL_COLUMN)
        rows.append(row)

    columns = list(SCORE_COLUMNS)
    if LABEL_COLUMN in pairs.columns:
        columns.append(LABEL_COLUMN)
    frame = pd.DataFrame(rows, columns=columns)
    unscored = frame['status'] != STATUS_OK
    if bool(unscored.any()):
        logger.warning(
            '%d of %d pairs could not be scored: %s',
            int(unscored.sum()), len(frame),
            frame.loc[unscored, 'status'].value_counts().to_dict(),
        )
    return frame
