from __future__ import annotations

import hashlib
import pickle

import numpy as np

# IDF normalisation and the active-feature cut. Torch-free so both are testable.
#
# The statistics are NOT in the SAE checkpoint - its idf and max buffers are all ones,
# making transformers' normalize_sae a silent no-op. They are published separately:
#
#   aws s3 cp s3://esm-protein-atlas/v1/normalization/max_idf_log10.pkl . --no-sign-request
#
# Two files are published and they are not interchangeable: max_idf.pkl is natural log
# and max_idf_log10.pkl a separate base-10 pass, giving about 12 active features per
# residue against 3. Use one file's max and idf together, never mixed. Both were counted over
# the corpus to check: log10 gave the closer pair count and activation distribution, so it is
# what every table is built with.

STATISTICS_KEYS = ('max_per_feature', 'idf_per_feature')


def load_statistics(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Load the published per-feature normalisation statistics.

    Args:
        path: Local path to Biohub's pickled statistics.

    Returns:
        The per-feature maximum activation, the per-feature IDF, and the file's md5,
        which pins which of the two published files a run used.

    Raises:
        ValueError: If the file is missing a key, the vectors disagree in length, either
            is all ones and so would make normalisation a no-op, or a value is
            non-finite or a non-positive max.
    """
    with open(path, 'rb') as file_in:
        raw = file_in.read()
    statistics = pickle.loads(raw)
    missing = [key for key in STATISTICS_KEYS if key not in statistics]
    if missing:
        raise ValueError(f'{path}: statistics file is missing {missing}')

    # float32 for both. The log10 file ships idf as float64, where the downcast costs
    # 5.9e-8 relative, far below anything that moves a value across the threshold.
    max_per_feature = np.asarray(
        statistics['max_per_feature'], dtype=np.float32,
    )
    idf_per_feature = np.asarray(
        statistics['idf_per_feature'], dtype=np.float32,
    )
    if max_per_feature.shape != idf_per_feature.shape:
        raise ValueError(
            f'{path}: max and idf disagree in length, '
            f'{max_per_feature.shape} against {idf_per_feature.shape}',
        )
    for name, values in zip(STATISTICS_KEYS, (max_per_feature, idf_per_feature)):
        if bool(np.all(values == 1.0)):
            raise ValueError(
                f'{path}: {name} is all ones, so normalisation would be a no-op. This '
                'is what the SAE checkpoint itself ships; fetch the published '
                'statistics from s3://esm-protein-atlas/v1/normalization/',
            )
    # max divides the raw activation, so a zero writes inf to the feature table and inf
    # clears any threshold.
    if not bool(np.all(np.isfinite(max_per_feature) & (max_per_feature > 0))):
        raise ValueError(
            f'{path}: max_per_feature must be finite and positive',
        )
    # Zero idf is legitimate - a feature active in every protein has log(N/N) - so only
    # the non-finite values are rejected here.
    if not bool(np.all(np.isfinite(idf_per_feature))):
        raise ValueError(
            f'{path}: idf_per_feature must be finite',
        )
    return max_per_feature, idf_per_feature, hashlib.md5(raw).hexdigest()


def active_features(
        rows: np.ndarray,
        feature_ids: np.ndarray,
        magnitudes: np.ndarray,
        max_per_feature: np.ndarray,
        idf_per_feature: np.ndarray,
        threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalise raw activations and keep the ones the paper counts as active.

    Args:
        rows: Residue index of each non-zero, ascending.
        feature_ids: Feature id of each non-zero.
        magnitudes: Raw activation of each non-zero.
        max_per_feature: Per-feature maximum activation over UniRef90.
        idf_per_feature: Per-feature IDF over UniRef90.
        threshold: Minimum normalised activation to count as active. The paper's cut,
            and it applies to the NORMALISED quantity; against a raw activation it
            keeps everything.

    Returns:
        The residues that have at least one active feature, the offsets delimiting each
        residue's run, and the active feature ids and normalised magnitudes, ordered so
        that residue k occupies offsets[k] to offsets[k + 1].

    Raises:
        RuntimeError: If rows is not ascending.
    """
    # np.unique sorts the residues while feature_ids keeps input order, so unsorted rows
    # would hand every residue another residue's features with nothing to show for it.
    if rows.size and bool(np.any(np.diff(rows) < 0)):
        raise RuntimeError(
            'active_features was given descending rows; the caller must pass the '
            "coalesced sparse tensor's indices",
        )
    scale = idf_per_feature[feature_ids] / max_per_feature[feature_ids]
    normalised = magnitudes * scale
    keep = normalised > threshold
    kept_rows = rows[keep]
    # Residues with nothing active are omitted rather than written empty; the loader's
    # contract allows either, and measured it is 0.07% of them.
    residues, counts = np.unique(kept_rows, return_counts=True)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return (
        residues,
        offsets,
        feature_ids[keep].astype(np.int32),
        normalised[keep].astype(np.float32),
    )
