from __future__ import annotations

import re

import pandas as pd

from flock.cofolding_benchmark import CUTOFF_DATE
from flock.cofolding_benchmark.date_flags import undirected_pair
from flock.compile_pinder import filter_pinder

# Sort key larger than any real YYYY-MM-DD date and the same shape as the dates
# it is compared against, so a cluster with no known member release date never
# counts as pre-cutoff (uncertain -> survivor).
_NEVER = '9999-99-99'

# A PINDER cluster_id is 'cluster_{R}_{L}'; a component equal to the -1 sentinel
# marks that side unclustered. Such systems are excluded from the homolog test
# so they cannot falsely leak. Match -1 as a whole '_'-delimited token (not a
# bare substring), so a multi-digit index like cluster_312_5 is never mis-flagged.
_UNCLUSTERED_SENTINEL = '-1'
_UNCLUSTERED_PATTERN = rf'(?:^|_){re.escape(_UNCLUSTERED_SENTINEL)}(?:_|$)'


def _is_unclustered(cluster_id: object) -> bool:
    """Whether a PINDER cluster_id has an unclustered (-1) component."""
    return re.search(_UNCLUSTERED_PATTERN, str(cluster_id)) is not None


def build_cluster_min_release_date(
    index_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> dict[str, str]:
    """Map each PINDER interface cluster to its earliest member release date.

    Joins the index (id -> cluster_id) to PINDER metadata (id -> release_date),
    drops unclustered systems (a cluster_id with a -1 component) and rows
    with no release date, then takes the per-cluster minimum. The cluster_id
    encodes BOTH monomer clusters (cluster_{R}_{L}), so the both-sides homolog
    criterion is baked in.

    Args:
        index_df: PINDER index with id and cluster_id columns.
        metadata_df: PINDER metadata with id and release_date (YYYY-MM-DD; the
            caller coerces it to the date prefix).

    Returns:
        Dict mapping cluster_id -> earliest member release_date (YYYY-MM-DD).
    """
    merged = index_df[['id', 'cluster_id']].merge(
        metadata_df[['id', 'release_date']], on='id', how='left',
    )
    clustered = merged[
        ~merged['cluster_id'].astype(str).str.contains(
            _UNCLUSTERED_PATTERN, regex=True,
        )
    ].dropna(subset=['release_date'])
    return clustered.groupby('cluster_id', observed=True)['release_date'].min().to_dict()


def build_pair_to_clusters(
    index_df: pd.DataFrame,
    pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], set[str]]:
    """Map each requested undirected pair to its interface cluster IDs.

    Restricts the index to rows whose undirected (uniprot_R, uniprot_L) pair is
    in pairs, then aggregates cluster_id into a set per pair.

    The caller is expected to pass an index already restricted to the systems
    that define the pair as a positive (see annotate_positive_homolog_flags):
    the question this answers is which clusters the pair's *biological*
    interface belongs to, so a crystal-packing contact of the same two proteins
    must not drag in the clusters it happens to fall into.

    Args:
        index_df: PINDER index with uniprot_R, uniprot_L and cluster_id columns.
        pairs: Undirected UniProt pairs to look up (e.g. the clean positives).

    Returns:
        Dict mapping each pair to the set of interface cluster IDs it appears in.
    """
    pair_column = [
        undirected_pair(uniprot_r, uniprot_l)
        for uniprot_r, uniprot_l in zip(
            index_df['uniprot_R'], index_df['uniprot_L'],
        )
    ]
    work = index_df.assign(pair=pair_column)
    work = work[work['pair'].isin(pairs)]
    return work.groupby('pair')['cluster_id'].agg(set).to_dict()


def positive_homolog_in_training_set(
    pair: tuple[str, str],
    pair_to_clusters: dict[tuple[str, str], set[str]],
    cluster_min_date: dict[str, str],
    cutoff: str = CUTOFF_DATE,
) -> bool:
    """Decide whether a positive pair is homolog-leaked by the interface-cluster check.

    The pair is leaked iff any of its interface clusters (excluding the '-1'
    unclustered sentinel) has a member released on or before the cutoff.

    Uncertain cases default to NOT leaked (survivor) — the deliberate OPPOSITE
    of the date flag's uncertain->leaked default: a pair with no clustered
    membership, or a cluster with no known member date, means PINDER's 2024-02
    snapshot saw no pre-cutoff homolog, so the pair survives. Do not "fix" this
    to leaked.

    Args:
        pair: Undirected UniProt pair.
        pair_to_clusters: Output of build_pair_to_clusters.
        cluster_min_date: Output of build_cluster_min_release_date.
        cutoff: Release-date cutoff (YYYY-MM-DD), inclusive.

    Returns:
        True if a pre-cutoff cluster member exists, else False.
    """
    clusters = {
        cluster
        for cluster in pair_to_clusters.get(pair, set())
        if not _is_unclustered(cluster)
    }
    return any(
        cluster_min_date.get(cluster, _NEVER) <= cutoff for cluster in clusters
    )


def annotate_positive_homolog_flags(
    annotated_df: pd.DataFrame,
    index_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cutoff: str = CUTOFF_DATE,
) -> pd.DataFrame:
    """Populate homolog_in_training_set for positives via the interface-cluster check.

    Operates on the date-annotated benchmark (output of
    date_flags.annotate_date_flags). The flag is computed for every Positive
    undirected pair (not just the date-clean ones) and broadcast to both
    directed rows, so the refreshed benchmark carries the homolog flag
    independently of the date flag. Negative rows are left as NA: the
    interface-cluster check applies only to positives, which have a complex.

    The two halves of the check are scoped differently, deliberately:

    - Which clusters the query pair belongs to is taken from the pair's
      quality-passing systems only (compile_pinder.filter_pinder — the same
      BIO / resolution / buried-SASA / contact thresholds that made the pair a
      positive). A pair is a positive because of its biological interface, so
      that is the interface whose homologs we look for.
    - Which cluster members count as training data is left unscoped: a model
      trained on the PDB saw every pre-cutoff structure in a cluster, whether
      or not PINDER labelled that particular system biological.

    Args:
        annotated_df: Benchmark with Target, Partner, Type, in_training_set,
            homolog_in_training_set columns.
        index_df: PINDER index with id, cluster_id, uniprot_R, uniprot_L.
        metadata_df: PINDER metadata with id, release_date (YYYY-MM-DD) and the
            columns filter_pinder needs (label, resolution, buried_sasa,
            intermolecular_contacts).
        cutoff: Release-date cutoff (YYYY-MM-DD).

    Returns:
        Copy with homolog_in_training_set populated for Positive rows (boolean
        dtype); negative rows stay NA.
    """
    annotated = annotated_df.copy()
    positive_pairs = {
        undirected_pair(target, partner)
        for target, partner, row_type in zip(
            annotated['Target'], annotated['Partner'], annotated['Type'],
        )
        if row_type == 'Positive'
    }
    cluster_min_date = build_cluster_min_release_date(index_df, metadata_df)
    quality_ids = set(filter_pinder(metadata_df)['id'])
    quality_index_df = index_df[index_df['id'].isin(quality_ids)]
    pair_to_clusters = build_pair_to_clusters(quality_index_df, positive_pairs)
    pair_flag = {
        pair: positive_homolog_in_training_set(
            pair, pair_to_clusters, cluster_min_date, cutoff,
        )
        for pair in positive_pairs
    }
    flags: list[bool | None] = [
        pair_flag[undirected_pair(target, partner)]
        if row_type == 'Positive' else pd.NA
        for target, partner, row_type in zip(
            annotated['Target'], annotated['Partner'], annotated['Type'],
        )
    ]
    annotated['homolog_in_training_set'] = pd.array(flags, dtype='boolean')
    return annotated
