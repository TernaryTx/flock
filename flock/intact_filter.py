"""Filter negatome pairs against known positive interactions from IntAct."""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pandas as pd

from flock.aws import download_file_from_s3
from flock.paths import get_intact_pairs_path


def load_intact_pairs(path: str | None = None) -> set[tuple[str, str]]:
    """Load the pre-processed IntAct positive pair lookup from S3.

    Resolves the latest dated intact_positive_pairs file and reads it,
    skipping provenance comment lines.

    Args:
        path: Optional explicit S3 path. Defaults to the latest dated file.

    Returns:
        Set of normalised (uniprot_a, uniprot_b) tuples where a <= b.
    """
    logger = logging.getLogger(__name__)
    folder, filename = (path or get_intact_pairs_path()).rsplit('/', 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, filename)
        download_file_from_s3(folder + '/', filename, local_path)
        df = pd.read_csv(local_path, comment='#')
    pairs = set(zip(df['uniprot_a'], df['uniprot_b']))
    logger.info(
        'Loaded %d IntAct positive pairs from %s',
        len(pairs), filename,
    )
    return pairs


def pair_membership(
    pairs: pd.DataFrame,
    reference_pairs: set[tuple[str, str]],
) -> np.ndarray:
    """Flag which pairs appear in a reference pair set.

    Expects the input DataFrame to have normalised pair order (uniprot_a <= uniprot_b),
    as produced by compile_negatome.deduplicate_pairs(). The lookup is an exact
    tuple match, so an unsorted row silently misses rather than raising, which is
    why the ordering is the caller's job and is stated here rather than assumed.

    Args:
        pairs: DataFrame with columns uniprot_a and uniprot_b.
        reference_pairs: Set of (uniprot_a, uniprot_b) tuples to test against —
            IntAct positives here, structural positives and the existing
            Negatome for the literature source's filters.

    Returns:
        Boolean array, True where the pair is in reference_pairs.
    """
    pair_index = pd.MultiIndex.from_arrays(
        [pairs['uniprot_a'], pairs['uniprot_b']],
    )
    return pair_index.isin(reference_pairs)


def filter_against_intact(
    pairs: pd.DataFrame,
    intact_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, int]:
    """Remove pairs that appear as known positive interactions in IntAct.

    Expects the input DataFrame to have normalised pair order (uniprot_a <= uniprot_b),
    as produced by compile_negatome.deduplicate_pairs().

    Args:
        pairs: DataFrame with columns uniprot_a and uniprot_b.
        intact_pairs: Set of (uniprot_a, uniprot_b) tuples from IntAct.

    Returns:
        Tuple of (filtered DataFrame with IntAct positives removed, count of
        pairs removed).
    """
    before = len(pairs)
    positive = pair_membership(pairs, intact_pairs)
    filtered = pairs[~positive].reset_index(drop=True)
    n_removed = before - len(filtered)
    return filtered, n_removed
