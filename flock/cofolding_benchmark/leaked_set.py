from __future__ import annotations

import argparse
import io
import logging
import os
import tempfile
import time
from datetime import date as _date

import numpy as np
import pandas as pd
import requests
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp

from flock import FLOCK_VERSION
from flock.aws import upload_file_to_s3
from flock.cofolding_benchmark.date_flags import undirected_pair
from flock.cofolding_benchmark.leakage_free_set import _download
from flock.cofolding_benchmark.leakage_free_set import annotate_benchmark
from flock.cofolding_benchmark.leakage_free_set import build_gene_lookup
from flock.cofolding_benchmark.leakage_free_set import build_pair_subset
from flock.cofolding_benchmark.leakage_free_set import build_positive_pair_metadata
from flock.cofolding_benchmark.leakage_free_set import build_target_view
from flock.cofolding_benchmark.leakage_free_set import clean_pair_masks
from flock.cofolding_benchmark.leakage_free_set import clean_partner_graphs
from flock.cofolding_benchmark.leakage_free_set import compute_leakage_free_targets
from flock.cofolding_benchmark.leakage_free_set import load_benchmark_inputs
from flock.cofolding_benchmark.leakage_free_set import parse_args as parse_leakage_free_args
from flock.logging_utils import setup_logging
from flock.paths import COFOLDING_BENCHMARK_S3
from flock.paths import get_cofolding_benchmark_path
from flock.paths import get_flock_negatome_path
from flock.paths import get_protein_annotation_path
from flock.paths import make_dated_filename
from flock.provenance import write_csv_with_provenance
from flock.uniprot import fetch_uniprot_tsv

# The residue budget for one co-folding job, counted over the whole complex
# (target + partner). Shared by both arms and recorded in the provenance header,
# because it is what makes the two comparable rather than a knob inside one
# function: a pair the cap excludes from one arm must be excluded from the other.
MAX_COMBINED_RESIDUES = 1420

# UniProt return fields for the length top-up, and the accessions-per-request
# chunk. The endpoint takes accessions as one comma-separated query parameter,
# so the chunk bounds URL length rather than server work.
LENGTH_FIELDS = 'accession,length'
LENGTH_CHUNK_SIZE = 300


def leaked_pair_masks(benchmark_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return row masks flagging leaked positive and leaked negative pairs.

    The mirror of clean_pair_masks, and deliberately not its complement. A pair
    is leaked only on positive evidence: either flag explicitly True. Pairs whose
    flags are NA on both sides were never checkable - a positive with no datable
    complex, a negative no MMseqs2 hit reaches - and belong to neither arm, since
    "the model may have seen this" is not the claim a leaked arm is making.

    Both labels use the same rule here. The asymmetry in clean_pair_masks exists
    because NA has to be read as not-clean for one label and as no-evidence for
    the other; asking instead for an explicit True removes the ambiguity, so no
    asymmetry is needed.

    Args:
        benchmark_df: Benchmark with Type, in_training_set and
            homolog_in_training_set columns.

    Returns:
        (leaked_positive_mask, leaked_negative_mask), each aligned to
        benchmark_df and disjoint.
    """
    date_leaked = benchmark_df['in_training_set'].eq(True)
    homolog_leaked = benchmark_df['homolog_in_training_set'].eq(True)
    leaked = date_leaked | homolog_leaked
    return (
        benchmark_df['Type'].eq('Positive') & leaked,
        benchmark_df['Type'].eq('Negative') & leaked,
    )


def _parse_length_tsv(text: str) -> dict[str, int]:
    """Parse the UniProt accessions endpoint's TSV into accession -> length.

    Args:
        text: TSV as returned by fetch_uniprot_tsv for LENGTH_FIELDS.

    Returns:
        Dict of accession to sequence length, empty if the response held no rows.
    """
    frame = pd.read_csv(io.StringIO(text), sep='\t')
    if frame.empty:
        return {}
    # The endpoint labels the columns 'Entry' and 'Length' rather than echoing
    # the requested field names, so take them positionally.
    accessions = frame.iloc[:, 0].astype(str)
    values = pd.to_numeric(frame.iloc[:, 1], errors='coerce')
    return {
        accession: int(length)
        for accession, length in zip(accessions, values)
        if pd.notna(length)
    }


def _fetch_lengths(
    chunk: list[str],
    attempts: int = 3,
    backoff_s: float = 2.0,
) -> dict[str, int]:
    """Fetch one chunk's lengths, retrying transient transport failures.

    The whole build sits behind this: by the time the top-up runs, every S3
    input has been read and both leakage flags annotated, so letting one reset
    connection propagate throws all of that away. Retried on any
    RequestException rather than on a status allowlist, because the failure
    seen in practice was a peer reset with no status at all.

    Args:
        chunk: Accessions for one request.
        attempts: Total tries before giving up.
        backoff_s: Seconds to wait after the first failure, doubling after each.

    Returns:
        Accession to length for the chunk.

    Raises:
        requests.RequestException: If every attempt fails.
    """
    logger = logging.getLogger(__name__)
    for attempt in range(1, attempts + 1):
        try:
            return _parse_length_tsv(fetch_uniprot_tsv(chunk, LENGTH_FIELDS))
        except requests.RequestException:
            if attempt == attempts:
                raise
            wait = backoff_s * 2 ** (attempt - 1)
            logger.warning(
                'UniProt length request failed (attempt %d/%d), retrying in '
                '%.0fs', attempt, attempts, wait,
            )
            time.sleep(wait)
    return {}


def resolve_lengths(
    accessions: set[str],
    cache_path: str,
    chunk_size: int = LENGTH_CHUNK_SIZE,
) -> dict[str, int]:
    """Resolve the UniProt sequence length of every accession, with caching.

    Three sources in order of cost: the local cache from a previous run, the
    published per-protein diversity annotation (one S3 read covering ~21k
    accessions), then the UniProt accessions endpoint for whatever is still
    missing. The annotation table was built over an older, smaller Flock, so the
    top-up is the normal path rather than an edge case.

    Accessions the endpoint does not return - obsoleted or demerged entries -
    are cached with an empty length rather than left out, so they are not asked
    for again on every subsequent run. There are around 760 of them, they will
    never resolve, and re-requesting them each time is both wasted work and a
    recurring chance for a transport error to take down a build that had
    already read every one of its S3 inputs.

    Callers must treat a missing length as a pair they cannot size, not as zero.

    Args:
        accessions: Accessions to resolve.
        cache_path: CSV path to read a previous run's result from and write this
            run's to. Carries accession,length with length blank for the
            asked-but-unavailable ones.
        chunk_size: Accessions per UniProt request.

    Returns:
        Dict of accession to length, covering as many of accessions as could be
        resolved.
    """
    logger = logging.getLogger(__name__)
    lengths: dict[str, int] = {}
    asked: set[str] = set()

    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        asked = set(cached['accession'].astype(str))
        resolved = cached.dropna(subset=['length'])
        lengths.update(
            dict(
                zip(
                    resolved['accession'].astype(str),
                    resolved['length'].astype(int),
                ),
            ),
        )
        logger.info(
            'Loaded %d cached lengths from %s (%d known unavailable)',
            len(lengths), cache_path, len(asked) - len(lengths),
        )

    unseeded = accessions - asked
    if unseeded:
        annotation_path = get_protein_annotation_path()
        logger.info(
            'Seeding lengths from %s...', annotation_path.rsplit('/', 1)[-1],
        )
        annotation = pd.read_csv(
            _download(annotation_path), comment='#',
            usecols=['accession', 'length'],
        )
        annotation = annotation.dropna(subset=['length'])
        seeded = {
            accession: int(length)
            for accession, length in zip(
                annotation['accession'].astype(str), annotation['length'],
            )
            if accession in unseeded
        }
        lengths.update(seeded)
        asked.update(seeded)
        logger.info('Seeded %d lengths from the annotation table', len(seeded))

    missing = sorted(accessions - asked)
    if missing:
        logger.info(
            'Fetching %d lengths from UniProt in chunks of %d...',
            len(missing), chunk_size,
        )
        for start in range(0, len(missing), chunk_size):
            chunk = missing[start:start + chunk_size]
            lengths.update(_fetch_lengths(chunk))
            # Marked asked whether or not the endpoint returned them, which is
            # what stops the permanently-unavailable ones coming back next run.
            asked.update(chunk)

    unresolved = accessions - set(lengths)
    if unresolved:
        logger.warning(
            '%d of %d accessions have no resolvable length; their pairs are '
            'excluded from both arms', len(unresolved), len(accessions),
        )
    cache = pd.DataFrame({'accession': sorted(asked)})
    cache['length'] = cache['accession'].map(lengths).astype('Int64')
    cache.to_csv(cache_path, index=False)
    logger.info(
        'Cached %d lengths and %d known-unavailable accessions to %s',
        len(lengths), len(asked) - len(lengths), cache_path,
    )
    return lengths


def combined_pair_lengths(
    benchmark_df: pd.DataFrame,
    lengths: dict[str, int],
) -> pd.Series:
    """Return each row's complex size: target length plus partner length.

    Args:
        benchmark_df: Benchmark with Target and Partner columns.
        lengths: Accession to length, from resolve_lengths.

    Returns:
        Float Series aligned to benchmark_df, NaN where either side's length is
        unknown.
    """
    target = benchmark_df['Target'].map(lengths)
    partner = benchmark_df['Partner'].map(lengths)
    return target.astype('Float64').add(partner.astype('Float64')).astype(float)


def within_length_cap(
    combined_length: pd.Series,
    cap: int = MAX_COMBINED_RESIDUES,
) -> pd.Series:
    """Return the row mask of pairs a co-folding job can afford.

    An unknown combined length fails the cap: a pair that cannot be sized cannot
    be shown to fit, and admitting it would put unmatched pairs in one arm.

    Args:
        combined_length: Output of combined_pair_lengths.
        cap: Maximum residues over the whole complex.

    Returns:
        Boolean Series aligned to combined_length.
    """
    return combined_length.notna() & combined_length.le(cap)


def select_matched_partners(
    reference_lengths: list[float],
    candidate_lengths: dict[str, float],
) -> list[str]:
    """Pick the candidate partners whose complex sizes match a reference set.

    One partner per reference pair, chosen to minimise the total absolute
    difference in complex size over the whole assignment rather than greedily
    per pair, so the selected set reproduces the reference length distribution
    instead of clustering on its easiest members. When there are fewer
    candidates than reference pairs, every candidate is taken and the shortfall
    is the caller's to report.

    Args:
        reference_lengths: Combined lengths of the reference target's pairs.
        candidate_lengths: Candidate partner accession to combined length.

    Returns:
        The selected partner accessions, at most len(reference_lengths) of them.
    """
    if not reference_lengths or not candidate_lengths:
        return []
    names = sorted(candidate_lengths)
    values = np.array([candidate_lengths[name] for name in names], dtype=float)
    cost = np.abs(
        np.asarray(reference_lengths, dtype=float)[:, None] - values[None, :],
    )
    _, columns = linear_sum_assignment(cost)
    return [names[column] for column in sorted(columns)]


def assign_leaked_targets(
    reference_targets: dict[str, dict[str, int]],
    leaked_candidates: dict[str, dict[str, int]],
    lengths: dict[str, int],
    shortfall_penalty: float = 10_000.0,
) -> tuple[dict[str, str], dict[str, str]]:
    """Map every reference target to the leaked target that will stand in for it.

    A reference target stands in for itself whenever it clears the per-target
    floor on leaked pairs, which is the comparison worth having: same protein,
    same partner-count profile, leaked partners instead of clean ones, so
    leakage is the only variable. That is the minority case in practice - a
    protein reaches the leakage-free set precisely because its structures are
    recent, which is the same reason it has few leaked partners - so most
    reference targets are stood in for by a substitute.

    Substitutes are chosen by one optimal assignment rather than greedily, over
    a cost that puts capacity ahead of similarity: a candidate that cannot
    supply the reference target's pair counts pays shortfall_penalty per missing
    pair, which dominates any length difference, so short candidates are used
    only when nothing better is free. Ties and the remaining slack are then
    settled on protein length, which is what keeps the two arms' complex sizes
    comparable. Greedy nearest-length picking gets both of these wrong: it
    spends the closest candidates on whichever target it happens to visit first
    and never looks at capacity at all.

    Substitutes are drawn only from accessions absent from the reference set, so
    a substituted row never duplicates a target the paired rows already hold.

    Args:
        reference_targets: Capped leakage-free targets, from
            compute_leakage_free_targets.
        leaked_candidates: Capped leaked targets clearing the same floor.
        lengths: Accession to length, used to match substitutes.
        shortfall_penalty: Cost per pair a candidate cannot supply. Must exceed
            the largest plausible length difference for capacity to outrank
            similarity.

    Returns:
        (assignment, match_mode), both keyed by reference target. A reference
        target with no substitute available is absent from both.
    """
    assignment: dict[str, str] = {}
    match_mode: dict[str, str] = {}
    needs_substitute: list[str] = []
    for target in sorted(reference_targets):
        if target in leaked_candidates:
            assignment[target] = target
            match_mode[target] = 'paired'
        else:
            needs_substitute.append(target)

    available = sorted(set(leaked_candidates) - set(reference_targets))
    if not needs_substitute or not available:
        return assignment, match_mode

    def column(source: dict[str, dict[str, int]], keys: list[str], field: str) -> np.ndarray:
        return np.array([float(source[key][field]) for key in keys])

    wanted_length = np.array(
        [float(lengths.get(target, np.nan)) for target in needs_substitute],
    )
    candidate_length = np.array(
        [float(lengths.get(accession, np.nan)) for accession in available],
    )
    gaps = []
    for field in ('n_clean_pos', 'n_clean_neg'):
        wanted = column(reference_targets, needs_substitute, field)
        supplied = column(leaked_candidates, available, field)
        gaps.append(np.clip(wanted[:, None] - supplied[None, :], 0, None))
    shortfall = gaps[0] + gaps[1]
    length_gap = np.abs(wanted_length[:, None] - candidate_length[None, :])
    cost = length_gap + shortfall_penalty * shortfall
    # An unresolvable length cannot happen here - both sides came through the
    # cap - but a huge finite cost keeps the assignment solvable if it ever does.
    cost = np.nan_to_num(cost, nan=1e12, posinf=1e12)

    rows, columns = linear_sum_assignment(cost)
    for row, column_index in zip(rows, columns):
        target = needs_substitute[row]
        assignment[target] = available[column_index]
        match_mode[target] = 'substituted'
    return assignment, match_mode


def sample_matched_pairs(
    assignment: dict[str, str],
    reference_partners: dict[str, set[str]],
    leaked_partners: dict[str, set[str]],
    lengths: dict[str, int],
) -> set[tuple[str, str]]:
    """Choose the leaked pairs that match the reference arm, one label at a time.

    Per assigned target, the reference target's pair complex sizes are the
    distribution to hit and the leaked target's partners are the pool to hit it
    from. Called once for positives and once for negatives, since the two labels
    have separate per-target counts.

    Args:
        assignment: Reference target to leaked target, from
            assign_leaked_targets.
        reference_partners: Capped clean partner sets for one label.
        leaked_partners: Capped leaked partner sets for the same label.
        lengths: Accession to length. Every accession reachable here has one, as
            the cap mask already excluded pairs that could not be sized.

    Returns:
        The selected undirected pairs.
    """
    selected: set[tuple[str, str]] = set()
    for reference_target, leaked_target in sorted(assignment.items()):
        reference_lengths = sorted(
            float(lengths[reference_target] + lengths[partner])
            for partner in reference_partners.get(reference_target, set())
        )
        candidates = {
            partner: float(lengths[leaked_target] + lengths[partner])
            for partner in leaked_partners.get(leaked_target, set())
        }
        for partner in select_matched_partners(reference_lengths, candidates):
            selected.add(undirected_pair(leaked_target, partner))
    return selected


def pair_row_mask(
    benchmark_df: pd.DataFrame,
    base_mask: pd.Series,
    pairs: set[tuple[str, str]],
) -> pd.Series:
    """Return the row mask for a set of undirected pairs, within a base mask.

    Both directed rows of a selected pair are set, so the emitted subset keeps
    the benchmark's directed convention and a pair whose two proteins are both
    targets appears under each of them - the same shape as the leakage-free
    pairs file.

    Args:
        benchmark_df: Benchmark with Target and Partner columns.
        base_mask: Rows eligible for selection, e.g. leaked and within the cap.
        pairs: Selected undirected pairs.

    Returns:
        Boolean Series aligned to benchmark_df.
    """
    mask = pd.Series(False, index=benchmark_df.index)
    if not pairs:
        return mask
    subset = benchmark_df.loc[base_mask, ['Target', 'Partner']]
    mask.loc[subset.index] = [
        undirected_pair(target, partner) in pairs
        for target, partner in zip(subset['Target'], subset['Partner'])
    ]
    return mask


def restrict_to_targets(
    benchmark_df: pd.DataFrame,
    mask: pd.Series,
    targets: set[str],
) -> pd.Series:
    """Narrow a row mask to the rows an arm actually publishes.

    build_pair_subset keeps a selected row only when its Target side is one of
    the arm's targets, so a mask that has not been through this is wider than
    the file it produces - for the reference arm, the whole capped clean pool of
    Flock rather than its 300-odd targets' pairs. Every statistic and assertion
    about an arm has to see the narrowed mask, or it describes a different
    dataset than the one on disk.

    Args:
        benchmark_df: Benchmark with a Target column.
        mask: Selected rows.
        targets: The arm's target accessions.

    Returns:
        Boolean Series aligned to benchmark_df.
    """
    return mask & benchmark_df['Target'].isin(targets)


def count_partners(
    targets: set[str],
    positive_partners: dict[str, set[str]],
    negative_partners: dict[str, set[str]],
) -> dict[str, dict[str, int]]:
    """Count each target's selected partners, for the target-view row.

    Counted over the pairs actually selected rather than over the quota that was
    asked for. The positive partner graph is undirected, so a pair selected for
    one target also counts toward the other when both are targets; the floors are
    guaranteed but the counts are what they are, exactly as in the leakage-free
    build.

    Args:
        targets: The arm's target accessions.
        positive_partners: Selected positive partner sets.
        negative_partners: Selected negative partner sets.

    Returns:
        Dict target -> {'n_clean_pos', 'n_clean_neg'}, in the key names
        build_target_view expects.
    """
    return {
        target: {
            'n_clean_pos': len(positive_partners.get(target, set())),
            'n_clean_neg': len(negative_partners.get(target, set())),
        }
        for target in sorted(targets)
    }


def build_matching_report(
    assignment: dict[str, str],
    match_mode: dict[str, str],
    reference_counts: dict[str, dict[str, int]],
    leaked_counts: dict[str, dict[str, int]],
    reference_lengths: dict[str, list[float]],
    leaked_lengths: dict[str, list[float]],
    gene_lookup: dict[str, str],
) -> pd.DataFrame:
    """Build the per-target record of how the two arms were matched.

    This is the file that says whether the comparison is fair, so it carries the
    requested and achieved counts side by side rather than only the achieved
    ones: a target whose leaked pool ran short is a target the arms differ on,
    and that has to be visible without re-running the build.

    Args:
        assignment: Reference target to leaked target.
        match_mode: Reference target to 'paired' or 'substituted'.
        reference_counts: Achieved counts in the capped leakage-free arm.
        leaked_counts: Achieved counts in the leaked arm.
        reference_lengths: Reference target to its pairs' complex sizes.
        leaked_lengths: Leaked target to its pairs' complex sizes.
        gene_lookup: Accession to gene name.

    Returns:
        One row per reference target, sorted by reference target.
    """
    rows = []
    for reference_target in sorted(reference_counts):
        leaked_target = assignment.get(reference_target, '')
        reference_pairs = reference_lengths.get(reference_target, [])
        leaked_pairs = leaked_lengths.get(leaked_target, [])
        rows.append({
            'reference_target': reference_target,
            'reference_gene': gene_lookup.get(reference_target, ''),
            'leaked_target': leaked_target,
            'leaked_gene': gene_lookup.get(leaked_target, ''),
            'match_mode': match_mode.get(reference_target, 'dropped'),
            'n_pos_reference': reference_counts[reference_target]['n_clean_pos'],
            'n_pos_leaked': leaked_counts.get(
                leaked_target, {},
            ).get('n_clean_pos', 0),
            'n_neg_reference': reference_counts[reference_target]['n_clean_neg'],
            'n_neg_leaked': leaked_counts.get(
                leaked_target, {},
            ).get('n_clean_neg', 0),
            'median_residues_reference': (
                float(
                    np.median(reference_pairs),
                ) if reference_pairs else np.nan
            ),
            'median_residues_leaked': (
                float(np.median(leaked_pairs)) if leaked_pairs else np.nan
            ),
        })
    return pd.DataFrame(rows)


def load_negatome_sources(path: str) -> dict[tuple[str, str], str]:
    """Load the Pdb / Lit source label of every Negatome pair.

    Reported per arm rather than matched on. The literature source is two orders
    of magnitude smaller than the PDB one, so forcing its share to agree between
    the arms would over-constrain a sampler already matching length and counts;
    what the comparison needs is for the drift to be visible.

    Args:
        path: S3 path to the compiled Negatome pairs CSV.

    Returns:
        Dict undirected pair -> comma-joined sorted source labels.
    """
    negatome = pd.read_csv(_download(path), comment='#')
    sources: dict[tuple[str, str], set[str]] = {}
    for target, partner, source in zip(
        negatome['Target'], negatome['Negative'], negatome['source'],
    ):
        sources.setdefault(
            undirected_pair(
                target, partner,
            ), set(),
        ).add(str(source))
    return {pair: ','.join(sorted(labels)) for pair, labels in sources.items()}


def summarise_arms(
    benchmark_df: pd.DataFrame,
    combined_length: pd.Series,
    arms: dict[str, tuple[pd.Series, pd.Series]],
    negatome_sources: dict[tuple[str, str], str],
) -> None:
    """Log the side-by-side comparison the paper's fairness claim rests on.

    Reports pair and target counts, the complex-size distribution with a
    two-sample KS statistic between the arms, and the source composition of each
    label. Nothing here gates the build; it is the evidence for reading the two
    files as comparable.

    Every count and distribution here is over UNDIRECTED pairs. The arms carry
    a target on both ends of a pair at very different rates, so per-row figures
    would compare the two CSVs' row multisets rather than the complexes they
    hold.

    Args:
        benchmark_df: The annotated benchmark.
        combined_length: Per-row complex size.
        arms: Arm name -> (positive mask, negative mask), reference arm first.
        negatome_sources: Undirected pair -> Negatome source label.
    """
    logger = logging.getLogger(__name__)

    def undirected(mask: pd.Series) -> set[tuple[str, str]]:
        return {
            undirected_pair(target, partner)
            for target, partner in zip(
                benchmark_df.loc[mask, 'Target'],
                benchmark_df.loc[mask, 'Partner'],
            )
        }

    def pair_lengths(mask: pd.Series) -> np.ndarray:
        # One value per undirected pair, not per row. A pair carrying a target
        # on both ends appears twice in the file, and the arms double-count at
        # very different rates, so a per-row distribution describes each CSV's
        # row multiset rather than the set of distinct complexes - which is what
        # the GPU cost and the per-pair scores are actually over. Both directed
        # rows of a pair carry the same complex size, so overwriting is safe.
        by_pair: dict[tuple[str, str], float] = {}
        for target, partner, length in zip(
            benchmark_df.loc[
                mask,
                'Target',
            ], benchmark_df.loc[mask, 'Partner'],
            combined_length[mask],
        ):
            if pd.notna(length):
                by_pair[undirected_pair(target, partner)] = float(length)
        return np.array(sorted(by_pair.values()))

    distributions: dict[str, np.ndarray] = {}
    for name, (positive_mask, negative_mask) in arms.items():
        both = positive_mask | negative_mask
        lengths = pair_lengths(both)
        distributions[name] = lengths
        positive_sources = benchmark_df.loc[
            positive_mask, 'Positive_source',
        ].value_counts().to_dict()
        negative_pairs = undirected(negative_mask)
        negative_sources: dict[str, int] = {}
        for pair in negative_pairs:
            label = negatome_sources.get(pair, 'unknown')
            negative_sources[label] = negative_sources.get(label, 0) + 1
        # Undirected counts, because that is how the published set is quoted.
        # The row count is the directed file length, and the two differ by
        # however many pairs carry a target on both ends.
        logger.info(
            '%s: %d targets, %d undirected pairs (%d positive, %d negative) '
            'over %d directed rows',
            name, benchmark_df.loc[both, 'Target'].nunique(),
            len(undirected(both)), len(undirected(positive_mask)),
            len(negative_pairs), int(both.sum()),
        )
        logger.info(
            '%s: complex size over %d distinct pairs, mean %.1f median %.1f '
            'min %.0f max %.0f',
            name, len(lengths), lengths.mean(), np.median(lengths),
            lengths.min(), lengths.max(),
        )
        logger.info('%s: positive sources %s', name, positive_sources)
        logger.info('%s: negative sources %s', name, negative_sources)

    names = list(distributions)
    if len(names) == 2:
        statistic, p_value = ks_2samp(
            distributions[names[0]], distributions[names[1]],
        )
        logger.info(
            'Complex-size KS between %s and %s: D=%.4f p=%.3g',
            names[0], names[1], statistic, p_value,
        )


def _selected_pair_lengths(
    benchmark_df: pd.DataFrame,
    combined_length: pd.Series,
    mask: pd.Series,
) -> dict[str, list[float]]:
    """Group the selected rows' complex sizes by Target, for the report.

    Args:
        benchmark_df: The annotated benchmark.
        combined_length: Per-row complex size.
        mask: Selected rows.

    Returns:
        Dict target -> its selected pairs' complex sizes.
    """
    grouped: dict[str, list[float]] = {}
    for target, length in zip(
        benchmark_df.loc[mask, 'Target'], combined_length[mask],
    ):
        grouped.setdefault(target, []).append(float(length))
    return grouped


def _write_arm(
    name: str,
    benchmark_df: pd.DataFrame,
    targets: dict[str, dict[str, int]],
    positive_partners: dict[str, set[str]],
    positive_mask: pd.Series,
    negative_mask: pd.Series,
    pair_metadata: dict,
    gene_lookup: dict[str, str],
    leakage_status: str,
    extra_target_columns: pd.DataFrame | None,
    sources: dict[str, str],
    out_dir: str,
    today: str,
) -> list[str]:
    """Write one arm's target view and pair subset, mirroring the published schema.

    The column layout is the leakage-free build's, so the downstream chain -
    the AFDB annotation, the pairing JSONs, the ranking notebook - reads either
    arm unchanged. Additions are appended: leakage_status on the pairs, and for
    the leaked arm the reference target it stands in for.

    Args:
        name: File stem prefix, e.g. 'leaked'.
        benchmark_df: The annotated benchmark.
        targets: Achieved per-target counts.
        positive_partners: Selected positive partner sets.
        positive_mask: Selected positive rows.
        negative_mask: Selected negative rows.
        pair_metadata: Source-assembly metadata per selected positive pair.
        gene_lookup: Accession to gene name.
        leakage_status: Value for the pairs file's leakage_status column.
        extra_target_columns: Optional frame to merge onto the target view on
            'target'.
        sources: Provenance entries.
        out_dir: Directory to write into.
        today: Date stamp for the filenames.

    Returns:
        The written file paths.
    """
    logger = logging.getLogger(__name__)
    target_view = build_target_view(
        targets, positive_partners, pair_metadata, gene_lookup,
    )
    if extra_target_columns is not None:
        target_view = target_view.merge(
            extra_target_columns, on='target', how='left',
        )
    pair_subset = build_pair_subset(
        benchmark_df, set(targets), gene_lookup, positive_mask, negative_mask,
    )
    pair_subset['leakage_status'] = leakage_status

    written = []
    for stem, frame in (
        (f'{name}_targets', target_view), (f'{name}_pairs', pair_subset),
    ):
        path = os.path.join(
            out_dir, make_dated_filename(stem, FLOCK_VERSION, '.csv', today),
        )
        write_csv_with_provenance(frame, path, sources=sources)
        logger.info('Wrote %s (%d rows)', path, len(frame))
        written.append(path)
    return written


def main() -> None:
    """Build the leaked comparison arm of the co-folding benchmark, and its match.

    The leakage-free set measures generalisation; on its own it gives an
    absolute number with nothing to read it against. This build adds the other
    arm: pairs a model trained to CUTOFF_DATE demonstrably did see.

    Read the two cohorts the match_mode column separates differently. For a
    PAIRED target, leakage really is the only variable: the same protein,
    the same per-target pair counts, leaked partners instead of clean ones. For
    a SUBSTITUTED one the protein changes, and the match holds only complex
    size and pair capacity - MSA depth, family and fold, taxonomy, interface
    topology and monomer difficulty all move with it, and every one of those
    shifts co-folding scores on its own. A score delta over the substituted
    cohort is therefore not attributable to leakage without stratifying on
    those covariates. Pairing is the contrast to quote and, on this data,
    the minority of the set.

    Matching is done in three steps. Both arms are capped at
    --max-combined-residues over the complex, so a pair too large for one is
    absent from both and neither arm is cheaper to run. The capped leakage-free
    arm is then re-derived from scratch under the same 2-clean-positive /
    20-clean-negative floor - capping only removes pairs, so its target set is a
    subset of the published one - and becomes the profile to hit. Finally each
    reference target is matched to a leaked target, itself where it clears the
    floor on leaked pairs and the nearest unused leaked candidate by protein
    length otherwise, and its per-target pair counts and complex-size
    distribution are reproduced from that target's leaked partners.

    Writes five versioned artifacts under the cofolding_benchmark S3 prefix,
    uploaded unless --no-upload:
      - leaked_targets_<version>_<date>.csv / leaked_pairs_<version>_<date>.csv
      - leakage_free_capped_targets_<version>_<date>.csv and its pairs file, the
        arm the leaked one was matched to. Not the same file as the published
        leakage_free_* pair, which carries no residue cap.
      - leaked_matching_report_<version>_<date>.csv, the per-target record of
        requested versus achieved counts.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = parse_args()
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    inputs = load_benchmark_inputs(args, parser)
    logger.info('Annotating leakage flags (date + homology)...')
    benchmark_df = annotate_benchmark(
        inputs.flock_df, inputs.pinder_metadata_df, inputs.pinder_index_df,
        inputs.negatome_df, inputs.date_lookup, inputs.ppi3d_interfaces_df,
        inputs.ppi3d_cluster_dates_df,
        negative_leakage_df=inputs.negative_leakage_df,
    )

    accessions = set(benchmark_df['Target']) | set(benchmark_df['Partner'])
    lengths = resolve_lengths(
        accessions, os.path.join(args.work_dir, 'pair_lengths.csv'),
    )
    combined_length = combined_pair_lengths(benchmark_df, lengths)
    capped = within_length_cap(combined_length, args.max_combined_residues)
    logger.info(
        '%d of %d benchmark rows are within %d combined residues',
        int(capped.sum()), len(benchmark_df), args.max_combined_residues,
    )

    min_positives = 2
    min_negatives = 20

    logger.info('Deriving the capped leakage-free reference arm...')
    clean_positive_mask, clean_negative_mask = clean_pair_masks(benchmark_df)
    clean_positive_mask &= capped
    clean_negative_mask &= capped
    reference_positive_partners, reference_negative_partners = clean_partner_graphs(
        benchmark_df, clean_positive_mask, clean_negative_mask,
    )
    reference_targets = compute_leakage_free_targets(
        reference_positive_partners, reference_negative_partners,
        min_positives, min_negatives,
    )
    logger.info('Capped leakage-free targets: %d', len(reference_targets))
    if args.flock_date is None and not args.no_ppi3d:
        check_against_published(set(reference_targets))

    logger.info('Deriving the leaked candidate pool...')
    leaked_positive_mask, leaked_negative_mask = leaked_pair_masks(
        benchmark_df,
    )
    leaked_positive_mask &= capped
    leaked_negative_mask &= capped
    leaked_positive_partners, leaked_negative_partners = clean_partner_graphs(
        benchmark_df, leaked_positive_mask, leaked_negative_mask,
    )
    leaked_candidates = compute_leakage_free_targets(
        leaked_positive_partners, leaked_negative_partners,
        min_positives, min_negatives,
    )
    logger.info('Capped leaked candidate targets: %d', len(leaked_candidates))

    assignment, match_mode = assign_leaked_targets(
        reference_targets, leaked_candidates, lengths,
    )
    mode_counts = {
        mode: sum(1 for value in match_mode.values() if value == mode)
        for mode in sorted(set(match_mode.values()))
    }
    unassigned = sorted(set(reference_targets) - set(assignment))
    logger.info(
        'Target assignment: %s, %d reference targets with no stand-in',
        mode_counts, len(unassigned),
    )
    if unassigned:
        # The reference arm is built from its own targets, so leaving these in
        # publishes two arms of different sizes - and nothing downstream
        # compares the counts unless they are dropped here.
        logger.warning(
            'Dropping %d reference targets with no leaked stand-in, so the '
            'arms stay the same size: %s',
            len(unassigned), unassigned[:10],
        )
        reference_targets = {
            target: counts for target, counts in reference_targets.items()
            if target in assignment
        }
    enough_positives = sum(
        1 for target in reference_targets
        if len(leaked_positive_partners.get(target, set())) >= min_positives
    )
    enough_negatives = sum(
        1 for target in reference_targets
        if len(leaked_negative_partners.get(target, set())) >= min_negatives
    )
    logger.info(
        'Of %d reference targets, %d carry >=%d leaked positives and %d carry '
        '>=%d leaked negatives, so pairing is limited by the %s side',
        len(reference_targets), enough_positives, min_positives,
        enough_negatives, min_negatives,
        'positive' if enough_positives < enough_negatives else 'negative',
    )

    selected_positive_pairs = sample_matched_pairs(
        assignment, reference_positive_partners, leaked_positive_partners, lengths,
    )
    selected_negative_pairs = sample_matched_pairs(
        assignment, reference_negative_partners, leaked_negative_partners, lengths,
    )
    selected_positive_mask = pair_row_mask(
        benchmark_df, leaked_positive_mask, selected_positive_pairs,
    )
    selected_negative_mask = pair_row_mask(
        benchmark_df, leaked_negative_mask, selected_negative_pairs,
    )
    leaked_targets = set(assignment.values())
    selected_positive_partners, selected_negative_partners = clean_partner_graphs(
        benchmark_df, selected_positive_mask, selected_negative_mask,
    )
    leaked_counts = count_partners(
        leaked_targets, selected_positive_partners, selected_negative_partners,
    )
    reference_counts = count_partners(
        set(reference_targets),
        reference_positive_partners, reference_negative_partners,
    )

    # Everything from here on describes the two files rather than the pools they
    # were drawn from, so it reads the narrowed masks.
    reference_target_set = set(reference_targets)
    published_masks = {
        'leakage_free_capped': (
            restrict_to_targets(
                benchmark_df, clean_positive_mask, reference_target_set,
            ),
            restrict_to_targets(
                benchmark_df, clean_negative_mask, reference_target_set,
            ),
        ),
        'leaked': (
            restrict_to_targets(
                benchmark_df, selected_positive_mask, leaked_targets,
            ),
            restrict_to_targets(
                benchmark_df, selected_negative_mask, leaked_targets,
            ),
        ),
    }
    assert_arms_comparable(
        benchmark_df, combined_length, args.max_combined_residues,
        published_masks['leakage_free_capped'], published_masks['leaked'],
        reference_counts, leaked_counts, min_positives, min_negatives,
    )

    negatome_sources = load_negatome_sources(get_flock_negatome_path())
    summarise_arms(
        benchmark_df, combined_length, published_masks, negatome_sources,
    )

    gene_lookup = build_gene_lookup(
        benchmark_df, set(reference_targets) | leaked_targets,
    )
    reference_metadata = build_positive_pair_metadata(
        inputs.pinder_metadata_df, inputs.ppi3d_interfaces_df,
        {
            undirected_pair(target, partner)
            for target, partner in zip(
                benchmark_df.loc[clean_positive_mask, 'Target'],
                benchmark_df.loc[clean_positive_mask, 'Partner'],
            )
        },
    )
    leaked_metadata = build_positive_pair_metadata(
        inputs.pinder_metadata_df, inputs.ppi3d_interfaces_df,
        selected_positive_pairs,
    )

    arm_lengths = {
        name: _selected_pair_lengths(
            benchmark_df, combined_length, positive_mask | negative_mask,
        )
        for name, (positive_mask, negative_mask) in published_masks.items()
    }
    report = build_matching_report(
        assignment, match_mode, reference_counts, leaked_counts,
        arm_lengths['leakage_free_capped'], arm_lengths['leaked'], gene_lookup,
    )
    # A paired target that clears the floor but cannot fill its reference counts
    # keeps its pairing rather than being substituted away, which is the right
    # trade but silently shrinks the leaked arm. Say by how much.
    for label, reference_column, leaked_column in (
        ('positives', 'n_pos_reference', 'n_pos_leaked'),
        ('negatives', 'n_neg_reference', 'n_neg_leaked'),
    ):
        requested = int(report[reference_column].sum())
        achieved = int(report[leaked_column].sum())
        short = int(
            (report[leaked_column] < report[reference_column]).sum(),
        )
        logger.info(
            'Matched %s: %d requested, %d achieved (%+d), %d targets short',
            label, requested, achieved, achieved - requested, short,
        )

    sources = {
        'flock_source': inputs.flock_filename,
        'pinder_source': inputs.pinder_filename,
        'pinder_index_source': inputs.pinder_index_filename,
        'pdb_negatome_source': inputs.negatome_filename,
        'ppi3d_interfaces_source': inputs.ppi3d_interfaces_filename,
        'min_clean_positives': str(min_positives),
        'min_clean_negatives': str(min_negatives),
        'leakage_method': (
            'positives: source-union complex date + per-source '
            'interface-cluster homolog (strictest evidence wins); '
            'negatives: co-presence date + sequence-identity homolog'
        ),
        'negative_leakage_source': (
            os.path.basename(args.negative_leakage)
            if args.negative_leakage is not None else 'none (negatives date-only)'
        ),
        'max_combined_residues': str(args.max_combined_residues),
        'length_source': 'UniProt sequence length (annotation table + REST top-up)',
        'match_mode_counts': str(mode_counts),
        'matching_metric': 'total |delta| in complex size, optimal assignment',
    }
    clean_sources = {
        **sources, 'pair_selection': 'clean (both leakage flags explicitly False)',
    }
    leaked_sources = {
        **sources,
        'pair_selection': (
            'leaked (in_training_set is True or homolog_in_training_set is True)'
        ),
        'matched_against': 'leakage_free_capped_' + FLOCK_VERSION,
    }

    today = _date.today().isoformat()
    out_dir = tempfile.mkdtemp()
    written = _write_arm(
        'leakage_free_capped', benchmark_df, reference_counts,
        reference_positive_partners, clean_positive_mask, clean_negative_mask,
        reference_metadata, gene_lookup, 'leakage_free', None,
        clean_sources, out_dir, today,
    )
    assignment_columns = pd.DataFrame(
        {
            'target': list(assignment.values()),
            'matched_reference_target': list(assignment),
            'match_mode': [match_mode[key] for key in assignment],
        },
    )
    written += _write_arm(
        'leaked', benchmark_df, leaked_counts,
        selected_positive_partners, selected_positive_mask, selected_negative_mask,
        leaked_metadata, gene_lookup, 'leaked', assignment_columns,
        leaked_sources, out_dir, today,
    )
    report_path = os.path.join(
        out_dir,
        make_dated_filename(
            'leaked_matching_report', FLOCK_VERSION, '.csv', today,
        ),
    )
    write_csv_with_provenance(report, report_path, sources=leaked_sources)
    logger.info('Wrote %s (%d targets)', report_path, len(report))
    written.append(report_path)

    if not args.no_upload:
        logger.info('Uploading to %s', COFOLDING_BENCHMARK_S3)
        for path in written:
            upload_file_to_s3(path, COFOLDING_BENCHMARK_S3)
    logger.info('Done.')


def check_against_published(capped_targets: set[str]) -> None:
    """Warn about the attrition the residue cap costs the leakage-free arm.

    Capping only removes pairs, so the capped target set must be a subset of the
    published one. A target present here but not there means the cap somehow
    admitted a pair the uncapped build rejected, which is a bug rather than an
    attrition figure, so it raises.

    Args:
        capped_targets: The capped leakage-free target accessions.

    Raises:
        AssertionError: If the capped set is not a subset of the published one.
    """
    logger = logging.getLogger(__name__)
    published = pd.read_csv(
        _download(get_cofolding_benchmark_path('leakage_free_targets')),
        comment='#',
    )
    published_targets = set(published['target'])
    unexpected = capped_targets - published_targets
    assert not unexpected, (
        f'{len(unexpected)} capped leakage-free targets are absent from the '
        f'published set, e.g. {sorted(unexpected)[:5]}; the cap can only remove '
        f'pairs, so this is a bug and not attrition'
    )
    logger.info(
        'Residue cap costs the leakage-free arm %d of %d published targets',
        len(published_targets) - len(capped_targets), len(published_targets),
    )


def assert_arms_comparable(
    benchmark_df: pd.DataFrame,
    combined_length: pd.Series,
    cap: int,
    reference_masks: tuple[pd.Series, pd.Series],
    leaked_masks: tuple[pd.Series, pd.Series],
    reference_counts: dict[str, dict[str, int]],
    leaked_counts: dict[str, dict[str, int]],
    min_positives: int,
    min_negatives: int,
) -> None:
    """Check the four properties a comparison between the arms requires.

    Necessary, not sufficient - holding protein identity fixed is the job of
    match_mode, and these say nothing about it. Each of the four fails
    silently: an over-cap pair makes one arm dearer to run, arms of unequal
    size cannot be compared per target at all, a shared pair puts the same
    complex on both sides of the contrast, and a target below the floor cannot
    be ranked.

    Args:
        benchmark_df: The annotated benchmark.
        combined_length: Per-row complex size.
        cap: The residue cap both arms were built under.
        reference_masks: (positive, negative) masks of the reference arm.
        leaked_masks: (positive, negative) masks of the leaked arm.
        reference_counts: Achieved reference per-target counts.
        leaked_counts: Achieved leaked per-target counts.
        min_positives: Per-target positive floor.
        min_negatives: Per-target negative floor.

    Raises:
        AssertionError: If any of the three properties does not hold.
    """
    def pair_set(masks: tuple[pd.Series, pd.Series]) -> set[tuple[str, str]]:
        both = masks[0] | masks[1]
        return {
            undirected_pair(target, partner)
            for target, partner in zip(
                benchmark_df.loc[
                    both,
                    'Target',
                ], benchmark_df.loc[both, 'Partner'],
            )
        }

    for label, masks in (('reference', reference_masks), ('leaked', leaked_masks)):
        both = masks[0] | masks[1]
        over_cap = combined_length[both].isna() | combined_length[both].gt(cap)
        assert not over_cap.any(), (
            f'{int(over_cap.sum())} {label} rows exceed the {cap}-residue cap '
            f'or have no resolvable length'
        )

    assert len(reference_counts) == len(leaked_counts), (
        f'{len(reference_counts)} reference targets against '
        f'{len(leaked_counts)} leaked ones; the arms have to be the same size '
        f'for a per-target comparison, and a reference target with no stand-in '
        f'should have been dropped before this point'
    )

    shared = pair_set(reference_masks) & pair_set(leaked_masks)
    assert not shared, (
        f'{len(shared)} undirected pairs appear in both arms, e.g. '
        f'{sorted(shared)[:5]}; the leaked arm must be disjoint from the '
        f'leakage-free one'
    )

    for label, counts in (('reference', reference_counts), ('leaked', leaked_counts)):
        below = {
            target: values for target, values in counts.items()
            if min(
                values['n_clean_pos'] - min_positives,
                values['n_clean_neg'] - min_negatives,
            ) < 0
        }
        assert not below, (
            f'{len(below)} {label} targets fall below the {min_positives}-positive '
            f'/ {min_negatives}-negative floor, e.g. {sorted(below)[:5]}'
        )


def parse_args() -> argparse.ArgumentParser:
    parser = parse_leakage_free_args()
    parser.description = (
        'Build the leaked comparison arm of the co-folding benchmark: pairs a '
        'model trained to the cutoff did see, matched to a residue-capped '
        're-derivation of the leakage-free set on targets, per-target pair '
        'counts and complex size.'
    )
    parser.add_argument(
        '--work-dir',
        default='leaked_set_work',
        help=(
            'Directory for the resolved UniProt length cache, so a rerun does '
            'not repeat the ~35k-accession lookup (default: %(default)s).'
        ),
    )
    parser.add_argument(
        '--max-combined-residues',
        type=int,
        default=MAX_COMBINED_RESIDUES,
        help=(
            'Residue budget over the whole complex (target + partner). Applied '
            'to both arms, so a pair too large for one is absent from both '
            '(default: %(default)s).'
        ),
    )
    return parser


if __name__ == '__main__':
    main()
