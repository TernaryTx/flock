from __future__ import annotations

import argparse
import logging
import os
import tempfile
from collections import defaultdict
from datetime import date as _date
from typing import NamedTuple

import pandas as pd

from flock import FLOCK_VERSION
from flock.aws import download_file_from_s3
from flock.aws import upload_file_to_s3
from flock.cofolding_benchmark.date_flags import annotate_date_flags
from flock.cofolding_benchmark.date_flags import build_negatome_pdb_index
from flock.cofolding_benchmark.date_flags import build_pinder_pair_index
from flock.cofolding_benchmark.date_flags import build_ppi3d_pair_index
from flock.cofolding_benchmark.date_flags import merge_pair_indexes
from flock.cofolding_benchmark.date_flags import parse_pinder_uniprot_pair
from flock.cofolding_benchmark.date_flags import undirected_pair
from flock.cofolding_benchmark.pinder_clusters import annotate_positive_homolog_flags
from flock.cofolding_benchmark.ppi3d_clusters import annotate_positive_homolog_flags as annotate_ppi3d_homolog_flags
from flock.logging_utils import setup_logging
from flock.paths import COFOLDING_BENCHMARK_S3
from flock.paths import get_flock_ppi3d_cluster_dates_path
from flock.paths import get_flock_ppi3d_interfaces_path
from flock.paths import get_flock_v1_path
from flock.paths import get_negatome_pdb_path
from flock.paths import get_pinder_index_path
from flock.paths import get_pinder_raw_path
from flock.paths import make_dated_filename
from flock.pdb_release_dates import PdbReleaseDateData
from flock.provenance import write_csv_with_provenance
from flock.rcsb import fetch_assembly_chain_counts
from flock.uniprot import fetch_gene_names


def _download(path: str, dest_dir: str | None = None) -> str:
    folder, filename = path.rsplit('/', 1)
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp()
    local_path = os.path.join(dest_dir, filename)
    download_file_from_s3(folder + '/', filename, local_path)
    return local_path


class PairMetadata(NamedTuple):
    """Objective PINDER metadata aggregated over a clean positive pair's systems."""
    min_oligomeric_count: int
    max_buried_sasa: int
    example_pdb: str


def combine_homolog_flags(
    flags: list[pd.Series],
) -> pd.Series:
    """Reduce per-source homolog flags to one, taking the strictest evidence.

    Each positive source can only speak to pairs it actually contains: PINDER's
    interface clusters say nothing about a PPI3D-only pair, and vice versa, so
    each source's check returns NA outside its own coverage. The reduction is:

    - any source says leaked   -> leaked
    - else any source says clean (with evidence) -> clean
    - else                     -> NA (no source could judge the pair)

    Leaked wins over clean rather than voting, because the cost of admitting a
    memorised complex into a leakage-free benchmark is much higher than the cost
    of dropping a usable pair.

    Args:
        flags: One nullable-boolean Series per source, all identically indexed.

    Returns:
        Combined nullable-boolean Series.
    """
    combined = pd.concat(flags, axis=1)
    any_leaked = combined.eq(True).any(axis=1)
    any_clean = combined.eq(False).any(axis=1)
    result = pd.Series(pd.NA, index=combined.index, dtype='boolean')
    result[any_clean] = False
    result[any_leaked] = True
    return result


def negative_source_pdb_index(
    negative_leakage_df: pd.DataFrame | None,
) -> dict[tuple[str, str], set[str]] | None:
    """Map each negative pair to every PDB entry holding both of its sides.

    Reads copresence_entries, which is UNDATED on purpose. The leakage flags
    care only about pre-cutoff entries, but the caller here is annotate, which
    wants source structures to take SIFTS-observed residues from, and that
    question has nothing to do with the cutoff. Reading the date-filtered
    copresence_entry instead would return nothing for every pair that reaches
    the benchmark, since a pair is only in the benchmark if it is clean, and
    clean means it has no pre-cutoff entry.

    Args:
        negative_leakage_df: Table from sequence_homology, or None.

    Returns:
        Pair -> the PDB entry ids holding both sides, or None when no table
        was supplied. Pairs sharing no entry are absent.
    """
    if negative_leakage_df is None:
        return None
    if 'copresence_entries' not in negative_leakage_df.columns:
        raise ValueError(
            'Negative leakage table predates the copresence_entries column; '
            'rebuild it with flock.cofolding_benchmark.sequence_homology.',
        )
    index = {}
    for side_a, side_b, entries in zip(
        negative_leakage_df['uniprot_a'],
        negative_leakage_df['uniprot_b'],
        negative_leakage_df['copresence_entries'],
    ):
        if not isinstance(entries, str) or not entries:
            continue
        index[undirected_pair(side_a, side_b)] = set(entries.split(','))
    return index


def assert_negative_leakage_coverage(
    flock_df: pd.DataFrame,
    negative_leakage_df: pd.DataFrame,
) -> None:
    """Refuse a leakage table that does not score every Flock negative.

    A negative absent from the table keeps NA on the homolog flag, and
    clean_pair_masks treats a negative's NA there as admissible, so an
    unscored pair is published as clean rather than rejected. That is the
    right default for a pair the rule genuinely could not reach, and the wrong
    one for a pair the table simply never saw, and nothing downstream can tell
    the two apart. Checking coverage here is what keeps them distinguishable.

    Args:
        flock_df: Assembled Flock, carrying Target, Partner and Type.
        negative_leakage_df: Table from sequence_homology.

    Raises:
        ValueError: If any Flock negative pair is absent from the table.
    """
    negatives = flock_df[flock_df['Type'] == 'Negative']
    wanted = set(
        zip(
            negatives[['Target', 'Partner']].min(axis=1),
            negatives[['Target', 'Partner']].max(axis=1),
        ),
    )
    scored = set(
        zip(
            negative_leakage_df['uniprot_a'], negative_leakage_df['uniprot_b'],
        ),
    )
    missing = wanted - scored
    if missing:
        raise ValueError(
            f'Negative leakage table scores {len(scored)} pairs and is missing '
            f'{len(missing)} of Flock\'s {len(wanted)} negatives, e.g. '
            f'{sorted(missing)[:3]}. Rebuild it against this Flock build with '
            'flock.cofolding_benchmark.sequence_homology.',
        )


def apply_negative_copresence_flags(
    benchmark_df: pd.DataFrame,
    negative_leakage_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Write a definite date verdict onto negatives the leakage table scored.

    Exists because "no pre-cutoff PDB entry" and "never checked" are both NA in
    annotate_date_flags, and only the first of the two is clean.
    pair_in_training_set returns None for an empty source set as well as for a
    missing one, so a negative whose accessions genuinely share no pre-cutoff
    entry cannot be expressed through the source index at all - it comes back
    NA, and clean_pair_masks reads NA as not-clean. Every literature-mined
    negative is in exactly that position, since its evidence is a sentence
    rather than a structure, so without this step the whole source is silently
    excluded from the clean pool.

    Only rows the table actually scored are touched, and only in the safe
    direction: NA becomes the table's verdict, False is upgraded to True when
    co-presence found a pre-cutoff PDB entry the pair's own sources missed, and a
    True from the pair's own source structures is never weakened.

    Args:
        benchmark_df: Benchmark carrying Target, Partner, Type, in_training_set.
        negative_leakage_df: Table from sequence_homology, or None.

    Returns:
        Copy of benchmark_df with negatives' date flag completed.
    """
    if negative_leakage_df is None:
        return benchmark_df
    annotated = benchmark_df.copy()
    verdicts = pd.Series(
        negative_leakage_df['copresence_in_training_set'].to_numpy(dtype=bool),
        index=pd.MultiIndex.from_arrays([
            negative_leakage_df['uniprot_a'], negative_leakage_df['uniprot_b'],
        ]),
    )
    # undirected_pair is min/max, so the row's own accessions reorder to the
    # table's key without a Python-level call per row.
    sides = annotated[['Target', 'Partner']].to_numpy()
    scored = verdicts.reindex(
        pd.MultiIndex.from_arrays([
            sides.min(axis=1), sides.max(axis=1),
        ]),
    ).to_numpy()
    verdict = pd.Series(scored, index=annotated.index, dtype='boolean')
    verdict[annotated['Type'] != 'Negative'] = pd.NA

    # Only in the safe direction: NA takes the verdict, False rises to True
    # where co-presence found a PDB entry the pair's own sources missed, and a
    # True already there is never weakened.
    column = annotated['in_training_set']
    annotated['in_training_set'] = column.mask(
        verdict.notna(), column.fillna(False) | verdict.fillna(False),
    ).astype('boolean')
    return annotated


def apply_negative_homolog_flags(
    benchmark_df: pd.DataFrame,
    negative_leakage_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Write the sequence-identity homolog flag onto the negative rows.

    Must run AFTER combine_homolog_flags — see annotate_benchmark. Positives are
    left untouched, so the interface-cluster verdict stays theirs alone.

    A negative absent from the leakage table keeps NA rather than being set
    False. NA here means the rule found no such PDB entry, which clean_pair_masks
    treats as permissive; writing False would claim the pair was checked and
    cleared, which is a stronger statement than the rule supports.

    Args:
        benchmark_df: Benchmark carrying Target, Partner, Type and the flags.
        negative_leakage_df: Table from sequence_homology, or None.

    Returns:
        Copy of benchmark_df with negatives' homolog flag written.
    """
    if negative_leakage_df is None:
        return benchmark_df
    flagged = {
        undirected_pair(side_a, side_b)
        for side_a, side_b, is_homolog in zip(
            negative_leakage_df['uniprot_a'],
            negative_leakage_df['uniprot_b'],
            negative_leakage_df['homolog_in_training_set'],
        )
        if bool(is_homolog)
    }
    annotated = benchmark_df.copy()
    is_negative = annotated['Type'] == 'Negative'
    hits = pd.Series(
        [
            undirected_pair(target, partner) in flagged
            for target, partner in zip(
                annotated['Target'], annotated['Partner'],
            )
        ],
        index=annotated.index,
    )
    column = annotated['homolog_in_training_set'].copy()
    column[is_negative & hits] = True
    annotated['homolog_in_training_set'] = column.astype('boolean')
    return annotated


def annotate_benchmark(
    flock_df: pd.DataFrame,
    pinder_metadata_df: pd.DataFrame,
    pinder_index_df: pd.DataFrame,
    negatome_df: pd.DataFrame,
    date_lookup: dict[str, str],
    ppi3d_interfaces_df: pd.DataFrame | None = None,
    ppi3d_cluster_dates_df: pd.DataFrame | None = None,
    negative_leakage_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Annotate every Flock pair with both leakage flags, in-memory.

    Applies in_training_set (PINDER complex / co-presence release date) to all
    rows, then homolog_in_training_set (interface-cluster) to positives and
    (sequence identity) to negatives. No intermediate CSV is persisted — this is
    the single in-memory annotation that feeds the leakage-free derivation.

    When PPI3D interfaces are supplied, the homolog check runs once per source
    and the results are combined by combine_homolog_flags. Running both is not
    redundant: Flock's positives are the union of PINDER and PPI3D pairs, and
    neither source's clusters cover the other's exclusive pairs.

    ORDER MATTERS AND FAILS SILENTLY. Both cluster steps rebuild
    homolog_in_training_set wholesale, overwriting every non-Positive row with
    NA. The negative flags therefore have to be written after
    combine_homolog_flags, not before, or they vanish into a correctly-typed
    column full of plausible NAs that nothing downstream can distinguish from
    "never checked".

    Args:
        flock_df: Assembled Flock benchmark (Target, Partner, Type, *_gene).
        pinder_metadata_df: PINDER metadata with id, entry_id, release_date.
        pinder_index_df: PINDER index with id, cluster_id, uniprot_R, uniprot_L.
        negatome_df: Raw PDB negatome (ProteinA, ProteinB, PDB_Code).
        date_lookup: Uppercase entry_id -> release_date map.
        ppi3d_interfaces_df: Optional PPI3D interfaces with uniprot_1,
            uniprot_2, pdb_id, release_date, cluster_data_40 and prodigy_class.
            Must be every mapped interface, not one per pair: the date flag
            reads a pair's source structures from it. Omitting it reproduces the
            PINDER-only annotation.
        ppi3d_cluster_dates_df: Optional cluster -> min_release_date table from
            compile_ppi3d, giving unscoped cluster membership for the homolog
            check.
        negative_leakage_df: Optional per-pair negative leakage table from
            sequence_homology, carrying uniprot_a, uniprot_b,
            copresence_in_training_set and homolog_in_training_set. Omitting it
            leaves every literature-mined negative's date flag NA and every
            negative's homolog flag NA.

    Returns:
        Copy of flock_df with in_training_set and homolog_in_training_set
        populated for both labels.
    """
    # Positives' source structures must span every positive source. A pair
    # missing from this index gets NA and is dropped downstream, so building it
    # from PINDER alone would silently discard every PPI3D-only positive — the
    # pairs the union exists to add.
    positive_index = build_pinder_pair_index(pinder_metadata_df)
    if ppi3d_interfaces_df is not None:
        positive_index = merge_pair_indexes(
            positive_index, build_ppi3d_pair_index(ppi3d_interfaces_df),
        )
    negatome_index = build_negatome_pdb_index(negatome_df)
    annotated = annotate_date_flags(
        flock_df, positive_index, negatome_index, date_lookup,
    )
    annotated = apply_negative_copresence_flags(annotated, negative_leakage_df)
    from_pinder = annotate_positive_homolog_flags(
        annotated, pinder_index_df, pinder_metadata_df,
    )
    if ppi3d_interfaces_df is None:
        return apply_negative_homolog_flags(from_pinder, negative_leakage_df)

    # Query clusters come from the BIO interfaces only — the ones that made the
    # pair a positive — while cluster membership stays unscoped. Same split as
    # the PINDER path; see ppi3d_clusters.annotate_positive_homolog_flags.
    bio_df = ppi3d_interfaces_df[ppi3d_interfaces_df['prodigy_class'] == 'BIO']
    cluster_min_date = None
    if ppi3d_cluster_dates_df is not None:
        cluster_min_date = dict(
            zip(
                ppi3d_cluster_dates_df['cluster'],
                ppi3d_cluster_dates_df['min_release_date'],
            ),
        )
    from_ppi3d = annotate_ppi3d_homolog_flags(
        annotated, ppi3d_interfaces_df, bio_df=bio_df,
        cluster_min_date=cluster_min_date,
    )
    combined = from_pinder.copy()
    combined['homolog_in_training_set'] = combine_homolog_flags([
        from_pinder['homolog_in_training_set'],
        from_ppi3d['homolog_in_training_set'],
    ])
    return apply_negative_homolog_flags(combined, negative_leakage_df)


def clean_pair_masks(benchmark_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return row masks flagging clean (leakage-free) positive and negative pairs.

    The two labels treat an unknown flag differently, and deliberately.

    Positives use .eq(False).fillna(False) on both columns, so NA is NOT clean.
    A positive's evidence is its own deposited complex; if we cannot date it,
    we cannot claim it is unseen.

    Negatives use .eq(False).fillna(False) on the date flag but
    .ne(True).fillna(True) on the homolog flag. The homolog rule is evidence
    FOR leakage, not a property every negative can be scored on: a PDB negative
    whose sides no MMseqs2 hit reaches has no homolog evidence either way, and
    demanding an explicit False there would empty the clean negative pool. A
    positive homolog hit still disqualifies. This is the only asymmetry between
    the two labels' tests, and it exists because NA means "no such PDB entry"
    on one side and "cannot be checked" on the other.

    Args:
        benchmark_df: Benchmark with Type, in_training_set,
            homolog_in_training_set columns.

    Returns:
        (clean_positive_mask, clean_negative_mask), each aligned to benchmark_df
        and disjoint (a row is at most one of positive/negative).
    """
    is_positive = benchmark_df['Type'] == 'Positive'
    is_negative = benchmark_df['Type'] == 'Negative'
    date_clean = benchmark_df['in_training_set'].eq(False).fillna(False)
    homolog_clean = benchmark_df['homolog_in_training_set'].eq(
        False,
    ).fillna(False)
    homolog_not_leaked = benchmark_df['homolog_in_training_set'].ne(
        True,
    ).fillna(True)
    clean_positive_mask = is_positive & date_clean & homolog_clean
    clean_negative_mask = is_negative & date_clean & homolog_not_leaked
    return clean_positive_mask, clean_negative_mask


def clean_partner_graphs(
    benchmark_df: pd.DataFrame,
    clean_positive_mask: pd.Series,
    clean_negative_mask: pd.Series,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build per-target clean positive and clean negative partner sets.

    Positive partners come from the UNDIRECTED clean-positive graph (both
    endpoints of every clean positive pair gain each other), so a protein that
    appears only as a Partner still accrues positive partners. Negative partners
    are the distinct clean-negative partners per Target taken from the directed
    clean-negative rows (the benchmark emits both directions for qualifying
    targets, so directed-Target grouping equals the undirected neighbourhood).

    Args:
        benchmark_df: Benchmark with Target, Partner, Type and the leakage flags.
        clean_positive_mask: Clean positive row mask (from clean_pair_masks).
        clean_negative_mask: Clean negative row mask (from clean_pair_masks).

    Returns:
        (positive_partners, negative_partners), each mapping a UniProt accession
        to its set of clean partner accessions.
    """
    positive = benchmark_df.loc[clean_positive_mask]
    negative = benchmark_df.loc[clean_negative_mask]
    positive_partners: dict[str, set[str]] = defaultdict(set)
    for target, partner in zip(positive['Target'], positive['Partner']):
        positive_partners[target].add(partner)
        positive_partners[partner].add(target)
    negative_partners: dict[str, set[str]] = defaultdict(set)
    for target, partner in zip(negative['Target'], negative['Partner']):
        negative_partners[target].add(partner)
    return positive_partners, negative_partners


def compute_leakage_free_targets(
    positive_partners: dict[str, set[str]],
    negative_partners: dict[str, set[str]],
    min_clean_positives: int = 2,
    min_clean_negatives: int = 20,
) -> dict[str, dict[str, int]]:
    """Select the leakage-free target set from the clean partner graphs.

    Partner counts are taken from the CLEAN partner graphs (built on the
    leakage-filtered subset), because removing leaked pairs drops some targets
    below the original per-source thresholds. A target qualifies when it has at
    least min_clean_positives clean positive partners AND at least
    min_clean_negatives clean negative partners (undirected counts).

    Args:
        positive_partners: Clean positive partner sets (from clean_partner_graphs).
        negative_partners: Clean negative partner sets (from clean_partner_graphs).
        min_clean_positives: Minimum distinct clean positive partners.
        min_clean_negatives: Minimum distinct clean negative partners.

    Returns:
        Dict target -> {'n_clean_pos': int, 'n_clean_neg': int} for qualifiers.
    """
    targets: dict[str, dict[str, int]] = {}
    for target, partners in positive_partners.items():
        if len(partners) < min_clean_positives:
            continue
        n_clean_neg = len(negative_partners.get(target, set()))
        if n_clean_neg < min_clean_negatives:
            continue
        targets[target] = {
            'n_clean_pos': len(partners),
            'n_clean_neg': n_clean_neg,
        }
    return targets


def build_pinder_pair_metadata(
    metadata_df: pd.DataFrame,
    survivor_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], PairMetadata]:
    """Aggregate objective PINDER metadata per clean positive pair.

    For each clean positive pair, aggregates over all its PINDER source systems:
    the minimum oligomeric_count (the smallest source assembly bounds task
    tractability) and the maximum buried_sasa (the largest observed interface).
    Records one example source entry_id. Missing oligomeric_count/buried_sasa
    are treated as 0.

    Args:
        metadata_df: PINDER metadata with id, entry_id, oligomeric_count and
            buried_sasa columns.
        survivor_pairs: Clean positive undirected pairs to aggregate.

    Returns:
        Dict pair -> {'min_oligomeric_count', 'max_buried_sasa', 'example_pdb'}.
    """
    work = metadata_df.copy()
    work['pair'] = [
        parse_pinder_uniprot_pair(
            system_id,
        ) for system_id in work['id']
    ]
    work = work[work['pair'].isin(survivor_pairs)]
    work['oligomeric_count'] = work['oligomeric_count'].fillna(0)
    work['buried_sasa'] = work['buried_sasa'].fillna(0)
    grouped = work.groupby('pair')
    min_oligomeric = grouped['oligomeric_count'].min()
    max_buried_sasa = grouped['buried_sasa'].max()
    example_pdb = grouped['entry_id'].first()
    # The three Series share the same group index; zip their aligned values
    # rather than indexing each by a tuple key (ambiguous on a non-MultiIndex).
    return {
        pair: PairMetadata(
            min_oligomeric_count=int(min_count),
            max_buried_sasa=int(max_sasa),
            example_pdb=str(pdb),
        )
        for pair, min_count, max_sasa, pdb in zip(
            min_oligomeric.index, min_oligomeric, max_buried_sasa, example_pdb,
        )
    }


def build_ppi3d_pair_metadata(
    ppi3d_interfaces_df: pd.DataFrame,
    survivor_pairs: set[tuple[str, str]],
    sasa_per_area: float = 0.481,
) -> dict[tuple[str, str], PairMetadata]:
    """Aggregate the PINDER-equivalent metadata per clean positive pair from PPI3D.

    Mirrors build_pinder_pair_metadata for pairs PINDER does not carry. Only the
    BIO interfaces contribute, since those are what make the pair a positive.

    Two of the three fields need translating rather than reading off:

    - buried SASA: PPI3D reports a Voronoi contact area, which measures the
      dividing surface once where buried SASA counts it on both partners. The
      phase 0 spike measured the ratio at 0.481 (Spearman 0.997), so area is
      divided by it to put the two sources on one scale.
    - assembly chain count: PPI3D publishes no subunit count, so it is fetched
      from RCSB per (pdb_id, biounit_no) and passed in by the caller. A pair
      whose assemblies RCSB does not resolve gets 0, matching the missing-value
      convention in build_pinder_pair_metadata.

    Args:
        ppi3d_interfaces_df: Classified PPI3D interfaces with pair, pdb_id,
            biounit_no, area, prodigy_class and a chain_count column.
        survivor_pairs: Clean positive undirected pairs to aggregate.
        sasa_per_area: Measured Voronoi-area to buried-SASA ratio.

    Returns:
        Dict pair -> PairMetadata.
    """
    work = ppi3d_interfaces_df[ppi3d_interfaces_df['prodigy_class'] == 'BIO'].copy(
    )
    work['pair_tuple'] = [
        undirected_pair(uniprot_1, uniprot_2)
        for uniprot_1, uniprot_2 in zip(work['uniprot_1'], work['uniprot_2'])
    ]
    work = work[work['pair_tuple'].isin(survivor_pairs)]
    if work.empty:
        return {}
    work['buried_sasa'] = (work['area'].fillna(0) / sasa_per_area)
    work['chain_count'] = work['chain_count'].fillna(0)
    grouped = work.groupby('pair_tuple')
    min_chains = grouped['chain_count'].min()
    max_buried_sasa = grouped['buried_sasa'].max()
    example_pdb = grouped['pdb_id'].first()
    return {
        pair: PairMetadata(
            min_oligomeric_count=int(min_count),
            max_buried_sasa=int(max_sasa),
            example_pdb=str(pdb).upper(),
        )
        for pair, min_count, max_sasa, pdb in zip(
            min_chains.index, min_chains, max_buried_sasa, example_pdb,
        )
    }


def add_ppi3d_chain_counts(ppi3d_interfaces_df: pd.DataFrame) -> pd.DataFrame:
    """Attach RCSB assembly chain counts to PPI3D interfaces.

    PPI3D identifies the source assembly as (pdb_id, biounit_no) but publishes
    no chain count, and that count bounds task tractability in the curated
    benchmark. RCSB's GraphQL API resolves them in batches; assemblies it does
    not recognise are left as NA.

    Args:
        ppi3d_interfaces_df: Classified PPI3D interfaces with pdb_id and
            biounit_no columns.

    Returns:
        Copy with an added chain_count column.
    """
    logger = logging.getLogger(__name__)
    work = ppi3d_interfaces_df.copy()
    pdb_part = work['pdb_id'].astype(str).str.upper()
    biounit_part = work['biounit_no'].astype('Int64').astype(str)
    work['assembly_id'] = pdb_part + '-' + biounit_part
    counts = fetch_assembly_chain_counts(work['assembly_id'].unique().tolist())
    work['chain_count'] = work['assembly_id'].map(counts).astype('Int64')
    logger.info(
        '%d of %d PPI3D interfaces have an RCSB chain count',
        int(work['chain_count'].notna().sum()), len(work),
    )
    return work


def build_positive_pair_metadata(
    metadata_df: pd.DataFrame,
    ppi3d_interfaces_df: pd.DataFrame | None,
    survivor_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], PairMetadata]:
    """Aggregate the objective source-assembly metadata for a positive pair set.

    PINDER is preferred where it has the pair, so the metadata of pairs both
    sources carry stays byte-identical to previous builds; PPI3D only fills the
    gaps. The RCSB chain-count lookup is therefore restricted to the pairs that
    actually need it, and to BIO interfaces, since querying RCSB for XTAL
    assemblies would be wasted calls.

    Args:
        metadata_df: PINDER metadata with id, entry_id, oligomeric_count and
            buried_sasa columns.
        ppi3d_interfaces_df: Classified PPI3D interfaces, or None to take
            metadata from PINDER alone.
        survivor_pairs: The selected positive undirected pairs to describe.

    Returns:
        Dict pair -> PairMetadata, missing any pair neither source covers.
    """
    logger = logging.getLogger(__name__)
    pair_metadata = build_pinder_pair_metadata(metadata_df, survivor_pairs)
    if ppi3d_interfaces_df is None:
        return pair_metadata
    uncovered = survivor_pairs - set(pair_metadata)
    logger.info(
        '%d positive pairs need PPI3D metadata (no PINDER row)', len(
            uncovered,
        ),
    )
    if not uncovered:
        return pair_metadata
    needed = ppi3d_interfaces_df.copy()
    needed['pair_tuple'] = [
        undirected_pair(uniprot_1, uniprot_2)
        for uniprot_1, uniprot_2 in zip(
            needed['uniprot_1'], needed['uniprot_2'],
        )
    ]
    wanted_pair = needed['pair_tuple'].isin(uncovered)
    is_bio = needed['prodigy_class'] == 'BIO'
    needed = needed[wanted_pair & is_bio]
    pair_metadata.update(
        build_ppi3d_pair_metadata(add_ppi3d_chain_counts(needed), uncovered),
    )
    return pair_metadata


def build_target_view(
    targets: dict[str, dict[str, int]],
    positive_partners: dict[str, set[str]],
    pair_metadata: dict[tuple[str, str], PairMetadata],
    gene_lookup: dict[str, str],
) -> pd.DataFrame:
    """Build the one-row-per-target leakage-free selection view.

    Aggregates each target's clean positive pairs into source-assembly
    chain-count and buried_sasa ranges (objective metadata only — no biological
    category or quality tier). min_num_chains/max_num_chains are the PINDER
    oligomeric_count of the source biological assemblies the positive pairs were
    extracted from.

    Args:
        targets: Output of compute_leakage_free_targets.
        positive_partners: Clean positive partner sets (from clean_partner_graphs).
        pair_metadata: Output of build_pinder_pair_metadata.
        gene_lookup: UniProt accession -> gene name.

    Returns:
        DataFrame sorted by min_num_chains then gene, one row per target.
    """
    rows = []
    for target, counts in targets.items():
        oligomeric: list[int] = []
        buried_sasa: list[int] = []
        example_pdb = ''
        # Sorted, not set order: example_pdb is whichever partner is visited
        # first, so iterating the set makes the column vary between processes
        # under string hash randomisation and the file irreproducible.
        for partner in sorted(positive_partners[target]):
            metadata = pair_metadata.get(undirected_pair(target, partner))
            if metadata is None:
                continue
            oligomeric.append(metadata.min_oligomeric_count)
            buried_sasa.append(metadata.max_buried_sasa)
            if not example_pdb:
                example_pdb = metadata.example_pdb
        rows.append({
            'target': target,
            'target_gene': gene_lookup.get(target, ''),
            'n_clean_pos': counts['n_clean_pos'],
            'n_clean_neg': counts['n_clean_neg'],
            'min_num_chains': min(oligomeric) if oligomeric else 0,
            'max_num_chains': max(oligomeric) if oligomeric else 0,
            'min_buried_sasa': min(buried_sasa) if buried_sasa else 0,
            'max_buried_sasa': max(buried_sasa) if buried_sasa else 0,
            'example_pdb': example_pdb,
        })
    return pd.DataFrame(rows).sort_values(
        ['min_num_chains', 'target_gene'],
    ).reset_index(drop=True)


def build_pair_subset(
    benchmark_df: pd.DataFrame,
    target_set: set[str],
    gene_lookup: dict[str, str],
    clean_positive_mask: pd.Series,
    clean_negative_mask: pd.Series,
) -> pd.DataFrame:
    """Build the pair-level leakage-free subset for the qualifying targets.

    Keeps every clean positive and clean negative row whose Target is a
    qualifier, copying the leakage provenance from the benchmark (flags are not
    recomputed here). Keeps the benchmark's directed convention, so an undirected
    pair appears twice when both endpoints are qualifying targets.

    Args:
        benchmark_df: Benchmark with the leakage flags populated.
        target_set: The leakage-free target accessions.
        gene_lookup: UniProt accession -> gene name.
        clean_positive_mask: Clean positive row mask (from clean_pair_masks).
        clean_negative_mask: Clean negative row mask (from clean_pair_masks).

    Returns:
        DataFrame with columns target, target_gene, partner, partner_gene, type,
        in_training_set, homolog_in_training_set.
    """
    clean_mask = clean_positive_mask | clean_negative_mask
    keep = clean_mask & benchmark_df['Target'].isin(target_set)
    subset = benchmark_df.loc[
        keep,
        ['Target', 'Partner', 'Type', 'in_training_set', 'homolog_in_training_set'],
    ].copy()
    subset = subset.rename(
        columns={'Target': 'target', 'Partner': 'partner', 'Type': 'type'},
    )
    subset['target_gene'] = subset['target'].map(gene_lookup)
    subset['partner_gene'] = subset['partner'].map(gene_lookup)
    return subset[[
        'target', 'target_gene', 'partner', 'partner_gene', 'type',
        'in_training_set', 'homolog_in_training_set',
    ]].reset_index(drop=True)


def build_gene_lookup(
    benchmark_df: pd.DataFrame,
    targets: set[str],
) -> dict[str, str]:
    """Build a UniProt -> gene-name map, backfilling missing target genes.

    Seeds from the benchmark's own Target_gene/Partner_gene columns, then
    backfills any qualifying target with no gene there via the UniProt API.

    Args:
        benchmark_df: Benchmark with Target/Partner and *_gene columns.
        targets: The leakage-free target accessions to ensure are resolved.

    Returns:
        Dict UniProt accession -> gene name.
    """
    gene_lookup: dict[str, str] = {}
    for uniprot_col, gene_col in [('Target', 'Target_gene'), ('Partner', 'Partner_gene')]:
        for accession, gene in zip(benchmark_df[uniprot_col], benchmark_df[gene_col]):
            if isinstance(gene, str) and gene and gene != '?' and accession not in gene_lookup:
                gene_lookup[accession] = gene
    missing = [target for target in targets if not gene_lookup.get(target)]
    if missing:
        # Best-effort: the benchmark already carries gene names from the Flock
        # build, and this only backfills targets it could not resolve. A UniProt
        # outage should leave a few gene cells blank, not fail the rebuild.
        for accession, gene in fetch_gene_names(missing, strict=False).items():
            if gene:
                gene_lookup[accession] = gene
    return gene_lookup


class BenchmarkInputs(NamedTuple):
    """The eight inputs the leakage annotation needs, plus their filenames.

    The filenames are carried alongside the frames because the provenance
    header names every input file, and the resolvers pick the latest date in S3
    rather than being pinned - so the caller cannot re-derive them.
    """
    flock_df: pd.DataFrame
    pinder_metadata_df: pd.DataFrame
    pinder_index_df: pd.DataFrame
    negatome_df: pd.DataFrame
    date_lookup: dict[str, str]
    ppi3d_interfaces_df: pd.DataFrame | None
    ppi3d_cluster_dates_df: pd.DataFrame | None
    negative_leakage_df: pd.DataFrame | None
    flock_filename: str
    pinder_filename: str
    pinder_index_filename: str
    negatome_filename: str
    ppi3d_interfaces_filename: str


def load_benchmark_inputs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> BenchmarkInputs:
    """Load every input annotate_benchmark needs, resolving the latest in S3.

    Shared by the leakage-free build and the leaked comparison build so the two
    arms are annotated from the same eight inputs rather than from two copies of
    this block that can drift apart. Only the Flock build is pinnable
    (--flock-date); the rest follow the newest date in S3.

    Args:
        args: Parsed arguments carrying flock_date, no_ppi3d, negative_leakage
            and no_upload.
        parser: The parser, used to error out when --negative-leakage is
            missing on an uploading run.

    Returns:
        BenchmarkInputs with every frame and the filenames for the provenance
        header.
    """
    logger = logging.getLogger(__name__)

    flock_path = get_flock_v1_path(args.flock_date)
    flock_filename = flock_path.rsplit('/', 1)[-1]
    pinder_path = get_pinder_raw_path()
    pinder_filename = pinder_path.rsplit('/', 1)[-1]
    index_path = get_pinder_index_path()
    index_filename = index_path.rsplit('/', 1)[-1]
    negatome_path = get_negatome_pdb_path()
    negatome_filename = negatome_path.rsplit('/', 1)[-1]

    logger.info('Loading assembled Flock benchmark %s...', flock_filename)
    flock_df = pd.read_csv(_download(flock_path), comment='#')

    logger.info('Loading PINDER metadata...')
    # label / resolution / intermolecular_contacts are needed alongside
    # buried_sasa so pinder_clusters can re-apply compile_pinder.filter_pinder
    # and scope the homolog check to quality-passing systems.
    metadata_df = pd.read_parquet(
        _download(pinder_path),
        columns=[
            'id', 'entry_id', 'release_date',
            'oligomeric_count', 'buried_sasa',
            'label', 'resolution', 'intermolecular_contacts',
        ],
    )
    metadata_df['release_date'] = metadata_df['release_date'].astype(
        str,
    ).str[:10]

    logger.info('Loading PINDER interface index...')
    index_df = pd.read_parquet(
        _download(index_path),
        columns=['id', 'cluster_id', 'uniprot_R', 'uniprot_L'],
    )

    logger.info('Loading PDB negatome co-presence sources...')
    negatome_df = pd.read_csv(_download(negatome_path), sep='\t')

    logger.info('Loading whole-PDB release-date table...')
    date_lookup = PdbReleaseDateData().get_release_date_dict()

    ppi3d_interfaces_df = None
    if not args.no_ppi3d:
        ppi3d_interfaces_path = get_flock_ppi3d_interfaces_path()
        ppi3d_interfaces_filename = ppi3d_interfaces_path.rsplit('/', 1)[-1]
        logger.info(
            'Loading classified PPI3D interfaces %s...', ppi3d_interfaces_filename,
        )
        ppi3d_interfaces_df = pd.read_csv(
            _download(ppi3d_interfaces_path), comment='#',
        )
        ppi3d_cluster_dates_path = get_flock_ppi3d_cluster_dates_path()
        logger.info(
            'Loading PPI3D cluster dates %s...',
            ppi3d_cluster_dates_path.rsplit('/', 1)[-1],
        )
        ppi3d_cluster_dates_df = pd.read_csv(
            _download(ppi3d_cluster_dates_path), comment='#',
        )
    else:
        ppi3d_interfaces_filename = 'excluded'
        ppi3d_cluster_dates_df = None

    negative_leakage_df = None
    if args.negative_leakage is not None:
        negative_leakage_df = pd.read_parquet(args.negative_leakage)
        logger.info(
            'Loaded %d negative leakage rows (%d co-presence, %d homolog)',
            len(negative_leakage_df),
            int(negative_leakage_df['copresence_in_training_set'].sum()),
            int(negative_leakage_df['homolog_in_training_set'].sum()),
        )
        assert_negative_leakage_coverage(flock_df, negative_leakage_df)
    elif not args.no_upload:
        # Without the table the negative side has no homology rule at all and
        # the published set is a different dataset - 446 targets rather than
        # 335 - carrying no marker that says so. Fine to build locally, never
        # fine to publish under the same name.
        parser.error(
            '--negative-leakage is required to upload. Build the table with '
            'flock.cofolding_benchmark.sequence_homology, or pass --no-upload '
            'to write a date-only build locally.',
        )

    return BenchmarkInputs(
        flock_df=flock_df,
        pinder_metadata_df=metadata_df,
        pinder_index_df=index_df,
        negatome_df=negatome_df,
        date_lookup=date_lookup,
        ppi3d_interfaces_df=ppi3d_interfaces_df,
        ppi3d_cluster_dates_df=ppi3d_cluster_dates_df,
        negative_leakage_df=negative_leakage_df,
        flock_filename=flock_filename,
        pinder_filename=pinder_filename,
        pinder_index_filename=index_filename,
        negatome_filename=negatome_filename,
        ppi3d_interfaces_filename=ppi3d_interfaces_filename,
    )


def main() -> None:
    """Derive and persist the leakage-free target set to S3.

    The single S3-writing entrypoint for the co-folding benchmark. Loads the
    assembled Flock benchmark, PINDER metadata + interface index, the raw PDB
    negatome and the whole-PDB release-date table, annotates both leakage flags
    in-memory (in_training_set + interface-cluster homolog_in_training_set; no
    intermediate CSV is persisted), then derives the leakage-free target set:
    targets keeping at least 2 clean positive partners AND at least 20 clean
    negative partners after leakage filtering. Negatives carry both flags too,
    from the --negative-leakage table: co-presence for the date flag and
    sequence identity for the homolog one, since the interface-cluster check
    needs a real complex and so applies only to positives. Without that table
    the negative side is date-only and both flags stay NA for every
    literature-mined negative, which drops the whole source.

    Writes two versioned artifacts under the cofolding_benchmark S3 prefix and
    uploads them unless --no-upload is given:
      - leakage_free_targets_<version>_<date>.csv: one row per target with gene,
        clean partner counts, source-assembly chain-count range (num_chains) and
        buried_sasa range.
      - leakage_free_pairs_<version>_<date>.csv: every clean positive and clean
        negative pair for those targets, with leakage provenance.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = parse_args()
    args = parser.parse_args()

    inputs = load_benchmark_inputs(args, parser)
    flock_df = inputs.flock_df
    metadata_df = inputs.pinder_metadata_df
    index_df = inputs.pinder_index_df
    negatome_df = inputs.negatome_df
    date_lookup = inputs.date_lookup
    ppi3d_interfaces_df = inputs.ppi3d_interfaces_df
    ppi3d_cluster_dates_df = inputs.ppi3d_cluster_dates_df
    negative_leakage_df = inputs.negative_leakage_df

    logger.info('Annotating leakage flags (date + homology)...')
    benchmark_df = annotate_benchmark(
        flock_df, metadata_df, index_df, negatome_df, date_lookup,
        ppi3d_interfaces_df, ppi3d_cluster_dates_df,
        negative_leakage_df=negative_leakage_df,
    )

    logger.info('Deriving the leakage-free target set...')
    min_clean_positives = 2
    # The only minimum on pair counts left in the project. Flock itself sets
    # none, so this floor is met from the full negatome rather than from a set
    # already filtered to the same threshold upstream.
    min_clean_negatives = 20
    clean_positive_mask, clean_negative_mask = clean_pair_masks(benchmark_df)
    positive_partners, negative_partners = clean_partner_graphs(
        benchmark_df, clean_positive_mask, clean_negative_mask,
    )
    targets = compute_leakage_free_targets(
        positive_partners, negative_partners,
        min_clean_positives, min_clean_negatives,
    )
    survivor_pairs = {
        undirected_pair(target, partner)
        for target, partner in zip(
            benchmark_df.loc[clean_positive_mask, 'Target'],
            benchmark_df.loc[clean_positive_mask, 'Partner'],
        )
    }
    logger.info('Leakage-free targets: %d', len(targets))

    pair_metadata = build_positive_pair_metadata(
        metadata_df, ppi3d_interfaces_df, survivor_pairs,
    )
    gene_lookup = build_gene_lookup(benchmark_df, set(targets))

    target_view = build_target_view(
        targets, positive_partners, pair_metadata, gene_lookup,
    )
    pair_subset = build_pair_subset(
        benchmark_df, set(
            targets,
        ), gene_lookup, clean_positive_mask, clean_negative_mask,
    )

    sources = {
        'flock_source': inputs.flock_filename,
        'pinder_source': inputs.pinder_filename,
        'pinder_index_source': inputs.pinder_index_filename,
        'pdb_negatome_source': inputs.negatome_filename,
        'ppi3d_interfaces_source': inputs.ppi3d_interfaces_filename,
        'min_clean_positives': str(min_clean_positives),
        'min_clean_negatives': str(min_clean_negatives),
        'leakage_method': (
            'positives: source-union complex date + per-source '
            'interface-cluster homolog (strictest evidence wins); '
            'negatives: co-presence date + sequence-identity homolog'
        ),
        # Without the leakage table the negative side is date-only and the two
        # builds differ in size, so the record has to say which one this is.
        'negative_leakage_source': (
            os.path.basename(args.negative_leakage)
            if args.negative_leakage is not None else 'none (negatives date-only)'
        ),
        'negative_homolog_status': (
            'sequence identity vs pre-cutoff PDB chains'
            if negative_leakage_df is not None
            else 'not computed (no --negative-leakage table; NA)'
        ),
    }
    today = _date.today().isoformat()
    targets_filename = make_dated_filename(
        'leakage_free_targets', FLOCK_VERSION, '.csv', today,
    )
    pairs_filename = make_dated_filename(
        'leakage_free_pairs', FLOCK_VERSION, '.csv', today,
    )
    temp_dir = tempfile.mkdtemp()
    targets_out = os.path.join(temp_dir, targets_filename)
    pairs_out = os.path.join(temp_dir, pairs_filename)
    write_csv_with_provenance(target_view, targets_out, sources=sources)
    write_csv_with_provenance(pair_subset, pairs_out, sources=sources)
    logger.info('Wrote %s (%d targets)', targets_out, len(target_view))
    logger.info('Wrote %s (%d pairs)', pairs_out, len(pair_subset))

    if not args.no_upload:
        logger.info('Uploading to %s', COFOLDING_BENCHMARK_S3)
        upload_file_to_s3(targets_out, COFOLDING_BENCHMARK_S3)
        upload_file_to_s3(pairs_out, COFOLDING_BENCHMARK_S3)
    logger.info('Done.')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Derive and persist the leakage-free target set (target view + '
            'pair subset) from the annotated co-folding benchmark.'
        ),
    )
    parser.add_argument(
        '--no-upload',
        action='store_true',
        help='Write the artifacts locally only; do not upload to S3.',
    )
    parser.add_argument(
        '--flock-date',
        default=None,
        help=(
            'Pin the assembled Flock build to compile against (YYYY-MM-DD). '
            'Defaults to the latest in S3. Needed to reproduce a previous '
            'leakage-free set, since the default follows the newest build.'
        ),
    )
    parser.add_argument(
        '--no-ppi3d',
        action='store_true',
        help=(
            'Ignore the PPI3D positive source, reproducing the PINDER-only '
            'leakage-free set. Use the matching assemble_flock --no-ppi3d build.'
        ),
    )
    parser.add_argument(
        '--negative-leakage',
        default=None,
        help=(
            'Per-pair negative leakage table from '
            'flock.cofolding_benchmark.sequence_homology. Without it every '
            'literature-mined negative stays NA on the date flag and no '
            'negative carries a homolog flag.'
        ),
    )
    return parser


if __name__ == '__main__':
    main()
