from __future__ import annotations

import pandas as pd

from flock.cofolding_benchmark import CUTOFF_DATE
from flock.cofolding_benchmark.date_flags import undirected_pair

# Sort key larger than any real YYYY-MM-DD date and the same shape as the dates
# it is compared against, so a cluster with no known member release date never
# counts as pre-cutoff (uncertain -> survivor). Mirrors pinder_clusters.
_NEVER = '9999-99-99'

# PPI3D interface clustering at 40% sequence identity plus 50% interface-contact
# similarity. This is the level that corresponds to PINDER's interface
# cluster_id: both subunits are clustered, so the both-sides homolog criterion is
# built in, and 40% identity is the conventional remote-homology boundary.
CLUSTER_COLUMN = 'cluster_data_40'


def interface_cluster(cluster_data: object, mode_specific: bool = False) -> str | None:
    """Reduce a PPI3D cluster_data value to a cluster identity.

    PPI3D writes '<type>_<sequence_cluster>_<mode>', e.g. '0_43_2'. The trailing
    field is an alternative-interaction-mode index *within* the sequence cluster,
    not part of the cluster's identity: PPI3D clusters on sequence and interface
    contacts jointly so that alternative binding modes survive deduplication.

    Dropping it is both the PINDER-matching and the conservative choice. PINDER's
    cluster_id encodes the two monomer sequence clusters with no mode
    subdivision, and keeping the mode would shrink mean cluster membership from
    3.88 to 1.90, systematically *under*-detecting homolog leakage. Treating a
    homologous complex as leakage even when it was captured in a different
    binding mode is the safer error for a leakage-free benchmark.

    Args:
        cluster_data: Raw value from a PPI3D cluster_data_* column.
        mode_specific: Keep the interaction-mode field, making clusters narrower
            and leakage detection more permissive.

    Returns:
        Cluster identity string, e.g. '0_43', or None when the value is missing.
        Callers must skip None rather than grouping on it: str(NaN) is 'nan',
        which would collapse every unclustered interface into one pseudo-cluster
        and let a single pre-cutoff member flag all of them as leaked.
    """
    if cluster_data is None or (
        isinstance(cluster_data, float) and cluster_data != cluster_data
    ):
        return None
    text = str(cluster_data)
    if not text or text.lower() == 'nan':
        return None
    if mode_specific:
        return text
    return text.rsplit('_', 1)[0]


def build_cluster_min_release_date(
    ppi3d_df: pd.DataFrame,
    cluster_column: str = CLUSTER_COLUMN,
    mode_specific: bool = False,
) -> dict[str, str]:
    """Map each PPI3D interface cluster to its earliest member release date.

    Membership is deliberately unscoped — every interface PPI3D indexes counts,
    biological or not. A model trained on the PDB saw every pre-cutoff structure
    in a cluster regardless of how any classifier later labelled it.

    Args:
        ppi3d_df: PPI3D interface table with a cluster column and release_date.
        cluster_column: Which cluster_data_* column to use.
        mode_specific: Passed through to interface_cluster.

    Returns:
        Dict mapping cluster identity -> earliest member release_date (YYYY-MM-DD).
    """
    work = ppi3d_df[[cluster_column, 'release_date']].dropna()
    clusters = [
        interface_cluster(value, mode_specific) for value in work[cluster_column]
    ]
    dates = pd.Series(
        pd.to_datetime(work['release_date'], errors='coerce'),
    ).dt.strftime('%Y-%m-%d')
    frame = pd.DataFrame({'cluster': clusters, 'release_date': dates}).dropna()
    return frame.groupby('cluster')['release_date'].min().to_dict()


def build_pair_to_clusters(
    ppi3d_df: pd.DataFrame,
    pairs: set[tuple[str, str]],
    cluster_column: str = CLUSTER_COLUMN,
    mode_specific: bool = False,
) -> dict[tuple[str, str], set[str]]:
    """Map each requested undirected pair to its interface cluster identities.

    The caller is expected to pass an interface table already restricted to the
    systems that make the pair a positive — for PPI3D that means the
    quality-passing interfaces PRODIGY-cryst called BIO. The question this
    answers is which clusters the pair's *biological* interface belongs to, so a
    crystal-packing contact between the same two proteins must not drag in
    whatever clusters it happens to fall into.

    Args:
        ppi3d_df: PPI3D interfaces with uniprot_1, uniprot_2 and a cluster column.
        pairs: Undirected UniProt pairs to look up.
        cluster_column: Which cluster_data_* column to use.
        mode_specific: Passed through to interface_cluster.

    Returns:
        Dict mapping each pair to the set of interface cluster identities it
        appears in.
    """
    pair_column = [
        undirected_pair(uniprot_1, uniprot_2)
        for uniprot_1, uniprot_2 in zip(
            ppi3d_df['uniprot_1'], ppi3d_df['uniprot_2'],
        )
    ]
    work = ppi3d_df.assign(pair=pair_column)
    work = work[work['pair'].isin(pairs)]
    work = work.assign(
        cluster=[
            interface_cluster(value, mode_specific)
            for value in work[cluster_column]
        ],
    ).dropna(subset=['cluster'])
    return work.groupby('pair')['cluster'].agg(set).to_dict()


def positive_homolog_in_training_set(
    pair: tuple[str, str],
    pair_to_clusters: dict[tuple[str, str], set[str]],
    cluster_min_date: dict[str, str],
    cutoff: str = CUTOFF_DATE,
) -> bool | None:
    """Decide whether a positive pair is homolog-leaked by the interface-cluster check.

    The pair is leaked iff any of its interface clusters has a member released on
    or before the cutoff.

    Returns None — not False — for a pair PPI3D has no interfaces for. This
    deliberately DIVERGES from pinder_clusters, which treats absence as
    "survivor", and the difference is not cosmetic.

    That convention is safe in the PINDER implementation because PINDER is also
    the source of the positives: every positive pair is in PINDER by
    construction, so absence from its index genuinely means unclustered. PPI3D
    is a different resource from the pairs it is asked about. Measured against
    the current Flock positives, 23.8% of pairs have no PPI3D interfaces at all,
    and treating those as clean accounted for 94% of the cases where this check
    disagreed with PINDER's — silently admitting pairs PINDER knows are leaked.
    Absence of evidence is not evidence of cleanliness, so it is reported as
    unknown and clean_pair_masks (which treats NA as not-clean) excludes it.

    A pair that IS present but whose clusters have no pre-cutoff member is a
    real negative and returns False.

    Args:
        pair: Undirected UniProt pair.
        pair_to_clusters: Output of build_pair_to_clusters.
        cluster_min_date: Output of build_cluster_min_release_date.
        cutoff: Release-date cutoff (YYYY-MM-DD), inclusive.

    Returns:
        True if a pre-cutoff cluster member exists, False if the pair is present
        with none, None if PPI3D has no interfaces for the pair.
    """
    clusters = pair_to_clusters.get(pair)
    if not clusters:
        return None
    return any(
        cluster_min_date.get(cluster, _NEVER) <= cutoff for cluster in clusters
    )


def annotate_positive_homolog_flags(
    annotated_df: pd.DataFrame,
    ppi3d_df: pd.DataFrame,
    bio_df: pd.DataFrame | None = None,
    cutoff: str = CUTOFF_DATE,
    cluster_column: str = CLUSTER_COLUMN,
    mode_specific: bool = False,
    cluster_min_date: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Populate homolog_in_training_set for positives via PPI3D interface clusters.

    Drop-in replacement for pinder_clusters.annotate_positive_homolog_flags,
    reading PPI3D's cluster columns instead of PINDER's index.parquet. Negative
    rows are left as NA: the interface-cluster check applies only to positives,
    which have a complex.

    The two halves of the check are scoped differently, deliberately, exactly as
    in the PINDER implementation:

    - Which clusters the query pair belongs to comes from bio_df, the
      quality-passing interfaces PRODIGY-cryst called BIO — the same interfaces
      that made the pair a positive in the first place.
    - Which cluster members count as training data comes from ppi3d_df,
      unscoped: a model trained on the PDB saw every pre-cutoff structure in a
      cluster, biological or not.

    Args:
        annotated_df: Benchmark with Target, Partner, Type, in_training_set,
            homolog_in_training_set columns.
        ppi3d_df: Full PPI3D interface table, for cluster membership dates.
        bio_df: Quality-passing BIO interfaces, for locating the query pair's
            clusters. Defaults to ppi3d_df, which widens the query scope and is
            the more permissive choice — pass the classified subset to match the
            PINDER semantics.
        cutoff: Release-date cutoff (YYYY-MM-DD).
        cluster_column: Which cluster_data_* column to use.
        mode_specific: Passed through to interface_cluster.
        cluster_min_date: Authoritative cluster -> earliest member release date,
            from compile_ppi3d's ppi3d_cluster_dates table. Computed from
            ppi3d_df when omitted, which understates membership.

    Returns:
        Copy with homolog_in_training_set populated for Positive rows (boolean
        dtype). Negative rows stay NA, as do positives PPI3D has no interfaces
        for — see positive_homolog_in_training_set on why those two NAs mean
        different things but are both correctly excluded downstream.
    """
    annotated = annotated_df.copy()
    positive_pairs = {
        undirected_pair(target, partner)
        for target, partner, row_type in zip(
            annotated['Target'], annotated['Partner'], annotated['Type'],
        )
        if row_type == 'Positive'
    }
    # Prefer the precomputed table: it is built over the raw pull, before any
    # Flock-specific filtering, so cluster membership reflects what a model saw
    # rather than what this pipeline could map to accessions. Falling back to
    # ppi3d_df understates membership and therefore under-detects leakage.
    if cluster_min_date is None:
        cluster_min_date = build_cluster_min_release_date(
            ppi3d_df, cluster_column, mode_specific,
        )
    query_df = ppi3d_df if bio_df is None else bio_df
    pair_to_clusters = build_pair_to_clusters(
        query_df, positive_pairs, cluster_column, mode_specific,
    )
    pair_flag = {
        pair: positive_homolog_in_training_set(
            pair, pair_to_clusters, cluster_min_date, cutoff,
        )
        for pair in positive_pairs
    }
    flags: list[bool | None] = [
        pair_flag[undirected_pair(target, partner)]
        if row_type == 'Positive' else None
        for target, partner, row_type in zip(
            annotated['Target'], annotated['Partner'], annotated['Type'],
        )
    ]
    annotated['homolog_in_training_set'] = pd.array(flags, dtype='boolean')
    return annotated
