# The screen's pair calls, unioned across runs and resolved to UniProt accessions.
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from flock.aws import upload_file_to_s3
from flock.negatome_v3.literature import PAIRS_ROOT
from flock.negatome_v3.literature import PAPER_DEFINITIONS_PATH
from flock.negatome_v3.literature import PAPER_ORGANISMS_PATH
from flock.negatome_v3.literature.grounding import normalise as normalise_text
from flock.negatome_v3.literature.names import get_name_index
from flock.negatome_v3.literature.names import normalise_name
from flock.negatome_v3.literature.names import UniprotNameIndex
from flock.negatome_v3.literature.organisms import build_organism_terms
from flock.negatome_v3.literature.organisms import build_prefix_taxa
from flock.negatome_v3.literature.organisms import long_form_variants
from flock.negatome_v3.literature.organisms import scan_organisms
from flock.negatome_v3.literature.organisms import single_letter_taxa
from flock.negatome_v3.literature.organisms import species_prefix
from flock.negatome_v3.literature.organisms import split_organism
from flock.negatome_v3.literature.runs import iter_requests
from flock.negatome_v3.literature.runs import latest_records
from flock.negatome_v3.literature.runs import REQUESTS_SUBDIR
from flock.negatome_v3.literature.runs import RUNS_ROOT
from flock.negatome_v3.literature.vocabulary import CONSTRUCT_PATTERNS
from flock.negatome_v3.literature.vocabulary import DROP_CONSTRUCT
from flock.negatome_v3.literature.vocabulary import DROP_REAGENT
from flock.negatome_v3.literature.vocabulary import DROP_UNGROUNDED
from flock.negatome_v3.literature.vocabulary import DROP_VIRUS_VIRUS
from flock.negatome_v3.literature.vocabulary import ION_RE
from flock.negatome_v3.literature.vocabulary import NO_INTERACTION
from flock.negatome_v3.literature.vocabulary import OUTCOME_PROVENANCE
from flock.negatome_v3.literature.vocabulary import PAIRS_NAME
from flock.negatome_v3.literature.vocabulary import PAIRS_STAGE
from flock.negatome_v3.literature.vocabulary import PICK_OUTCOMES
from flock.negatome_v3.literature.vocabulary import PRIMARY_RUN_ID
from flock.negatome_v3.literature.vocabulary import PROVENANCE_NONE
from flock.negatome_v3.literature.vocabulary import REAGENT_NAMES
from flock.negatome_v3.literature.vocabulary import REPLICATE_RUN_ID
from flock.negatome_v3.literature.vocabulary import RUN_LABELS
from flock.negatome_v3.literature.vocabulary import SIDE_DETERMINED
from flock.negatome_v3.literature.vocabulary import SIDE_DETERMINED_BY_TEXT
from flock.negatome_v3.literature.vocabulary import SIDE_IDENTITY_PICK
from flock.negatome_v3.literature.vocabulary import SIDE_SPECIES_PICK
from flock.negatome_v3.literature.vocabulary import SIDE_STATUS
from flock.negatome_v3.literature.vocabulary import SIDE_TOO_WIDE
from flock.negatome_v3.literature.vocabulary import SIDE_UNRESOLVED
from flock.negatome_v3.literature.vocabulary import SINGLE_LETTER_ORGANISMS
from flock.negatome_v3.literature.vocabulary import SOURCE_DEFINITION
from flock.negatome_v3.literature.vocabulary import SOURCE_NONE
from flock.negatome_v3.literature.vocabulary import SOURCE_PAPER
from flock.negatome_v3.literature.vocabulary import SOURCE_PREFIX
from flock.negatome_v3.literature.vocabulary import STANDALONE_MUTATION_RE
from flock.negatome_v3.literature.vocabulary import STATUS_NEEDS_MODEL_PICK
from flock.negatome_v3.literature.vocabulary import STATUS_ORDER
from flock.negatome_v3.literature.vocabulary import TOKEN_SPLIT_RE
from flock.paths import get_literature_stage_prefix
from flock.paths import LITERATURE_STAGE_VERSIONS
from flock.paths import make_dated_filename
from flock.provenance import build_output_provenance
from flock.provenance import write_output_provenance

logger = logging.getLogger(__name__)

# Most candidates a side can carry and still be handed to a model as a list. A
# routing threshold rather than a filter: nothing is deleted for being wide, a
# side above it goes to deeper curation instead of to the model pick.
#
# Measured after the text scan: the resolved set is 10,598 pairs at every cap
# from 25 to no cap at all, so this value cannot affect what this stage delivers
# deterministically. It only moves the model-pick/curation boundary - 32,568 /
# 34,162 / 35,218 / 36,193 pairs needing a pick at 25 / 50 / 100 / no cap. So the
# value belongs to whoever sizes the model pick's prompt, since uncapped it would
# hand a model a list of 1,791 accessions for CYTB.
DEFAULT_MAX_CANDIDATES = 50


class SideResolution(NamedTuple):
    """What one written name resolved to.

    candidates is the set that survived every narrowing step, which is not what
    the written name alone returns: the paper's own text may have cut it down,
    or supplied it in the first place for a name that resolved to nothing. It is
    deliberately not written to the pair table - it would be a list column of no
    use to a consumer - but the model pick has to render exactly this list, or it
    would ask about a different question than the one that gave the side its
    status.
    """

    outcome: str
    n_candidates: int
    n_groups: int
    accession: str
    is_viral: bool
    candidates: tuple[str, ...]


def iter_final_records(
        run_dir: Path,
        relationship: str = NO_INTERACTION,
) -> list[dict]:
    """Return one record per paper, keeping only the wanted pair calls.

    Last record per custom_id wins, which runs.latest_records is what decides;
    the projection passed to it is where the relationship filter happens, so a
    record is reduced to the pairs this call wants before it is retained rather
    than after. Superseding replaces the paper's pair list outright rather than
    merging, or a retry returning fewer pairs would leave the original attempt's
    extras behind.

    Args:
        run_dir: A run directory holding a records subdirectory.
        relationship: The pair relationship to keep.

    Returns:
        One dict per paper, carrying its identifiers and its surviving pairs.

    Raises:
        FileNotFoundError: If the run has no records directory.
    """
    papers = latest_records(
        run_dir,
        lambda record: {
            'custom_id': record['custom_id'],
            'pmid': record.get('pmid') or record['custom_id'],
            'pmcid': record.get('pmcid') or '',
            'paper_id': record.get('paper_id') or '',
            'text_source': record.get('text_source') or '',
            'input_level': record.get('input_level') or '',
            'pairs': [
                pair for pair in (record.get('pairs') or [])
                if pair.get('relationship') == relationship
            ],
        },
    )
    logger.info(
        '%s: %d papers, %d carrying a %s pair', run_dir.name, len(papers),
        sum(1 for item in papers if item['pairs']), relationship,
    )
    return papers


def load_run_pairs(
        run_dir: Path,
        relationship: str = NO_INTERACTION,
) -> dict[tuple[str, str, str], dict]:
    """Collect one run's pair calls, keyed on paper and normalised name pair.

    The key normalises both names and sorts them, so a pair reported as A||B in
    one run and B||A in the other is recognised as the same pair. It is a sorted
    tuple rather than a set because a homodimer's two sides normalise
    identically, and a set would silently collapse it to a one-sided key.

    The paper part of the key is the custom_id the screen was submitted under,
    not the PMID. They are the same for every paper in this corpus, but the
    submitter falls back to paper_id when two papers share a PMID - the same
    article deposited twice - and keying on the PMID would then merge two
    different articles' pairs into one row and pool their excerpts. It is also
    the only key the recorded requests can be looked up by, which is what the
    grounding gate needs.

    A pair whose side normalises to nothing is dropped: it cannot be looked up in
    the name index either, so carrying it would only inflate the denominator.

    Args:
        run_dir: A run directory.
        relationship: The pair relationship to keep.

    Returns:
        Key to a dict carrying the paper's identifiers, the names as written and
        the pair's confidence and excerpts.
    """
    found: dict[tuple[str, str, str], dict] = {}
    n_unnamed = 0
    for record in iter_final_records(run_dir, relationship):
        for pair in record['pairs']:
            sides = []
            for side in ('protein_a', 'protein_b'):
                written = str(pair.get(side) or '')
                sides.append((normalise_name(written), written))
            if not all(normalised for normalised, _ in sides):
                n_unnamed += 1
                continue
            sides.sort()
            key = (record['custom_id'], sides[0][0], sides[1][0])
            existing = found.get(key)
            excerpts = list(pair.get('excerpts') or [])
            if existing is None:
                found[key] = {
                    'custom_id': record['custom_id'],
                    'pmid': record['pmid'],
                    'pmcid': record['pmcid'],
                    'paper_id': record['paper_id'],
                    'text_source': record['text_source'],
                    'input_level': record['input_level'],
                    'norm_a': sides[0][0], 'norm_b': sides[1][0],
                    'name_a': sides[0][1], 'name_b': sides[1][1],
                    'confidence': pair.get('confidence') or '',
                    'excerpts': excerpts,
                }
                continue
            # The same paper can name one pair twice, usually with different
            # casing. Keep the first spelling and pool the evidence rather than
            # emitting two rows that resolve to the same accessions.
            existing['excerpts'] = _merge_excerpts(
                existing['excerpts'], excerpts,
            )
    if n_unnamed:
        logger.warning(
            '%s: dropped %d pairs with a side that normalises to nothing',
            run_dir.name, n_unnamed,
        )
    return found


def _merge_excerpts(left: list[str], right: list[str]) -> list[str]:
    """Concatenate two excerpt lists, dropping duplicates and keeping order."""
    return list(dict.fromkeys([*left, *right]))


def union_runs(
        run_ids: list[str],
        relationship: str = NO_INTERACTION,
        runs_root: Path = RUNS_ROOT,
) -> pd.DataFrame:
    """Union several screen runs' pair calls into one table.

    The runs are independent screens of the same corpus under the same config,
    so a pair either of them reports is evidence. Unioning rather than
    intersecting is the measured decision: the two full-corpus runs overlap at
    Jaccard 0.561 on no-interaction pairs, and most of the non-overlap is which
    pairs get extracted from a paper both runs screened, not which papers got
    screened.

    Args:
        run_ids: Run directory names under runs_root.
        relationship: The pair relationship to keep.
        runs_root: Directory holding the run directories.

    Returns:
        One row per (paper, normalised name pair), carrying which runs saw it,
        each run's confidence, and the pooled excerpts.

    Raises:
        ValueError: If no run ids were given.
    """
    if not run_ids:
        raise ValueError('No run ids to union.')
    per_run = {
        run_id: load_run_pairs(runs_root / run_id, relationship)
        for run_id in run_ids
    }
    for run_id, found in per_run.items():
        logger.info('%s: %d %s pairs', run_id, len(found), relationship)

    rows: dict[tuple[str, str, str], dict] = {}
    for run_id, found in per_run.items():
        label = RUN_LABELS.get(run_id, run_id)
        for key, pair in found.items():
            row = rows.get(key)
            if row is None:
                # The first run to report a pair supplies the spelling kept in
                # name_a and name_b. The runs often write the same protein with
                # different casing, and the normalised names are what anything
                # downstream matches on, so the written pair is provenance rather
                # than a value a later run should be able to overwrite.
                row = {
                    key_name: value for key_name, value in pair.items()
                    if key_name != 'confidence'
                }
                row['seen_in'] = []
                rows[key] = row
            else:
                row['excerpts'] = _merge_excerpts(
                    row['excerpts'], pair['excerpts'],
                )
            row['seen_in'].append(label)
            row[f'confidence_{label}'] = pair['confidence']

    frame = pd.DataFrame(list(rows.values()))
    frame['seen_in'] = frame['seen_in'].map(
        lambda labels: '+'.join(sorted(labels)),
    )
    for run_id in run_ids:
        column = f'confidence_{RUN_LABELS.get(run_id, run_id)}'
        if column not in frame.columns:
            frame[column] = ''
        frame[column] = frame[column].fillna('')
    return frame.sort_values(['pmid', 'norm_a', 'norm_b']).reset_index(drop=True)


def is_construct_name(name: str) -> bool:
    """Return whether a written name marks a construct rather than a protein.

    Unsafe on its own: several rules here fire on real proteins, and no regex
    separates them. Call it behind the resolution gate that apply_filters
    applies, never directly.

    Args:
        name: A protein name as the screen emitted it.

    Returns:
        True if any light-touch construct pattern matches.
    """
    if any(pattern.search(name) for pattern in CONSTRUCT_PATTERNS):
        return True
    return any(
        STANDALONE_MUTATION_RE.match(token)
        for token in TOKEN_SPLIT_RE.split(name)
    )


def ground_pairs(frame: pd.DataFrame, run_dir: Path) -> pd.Series:
    """Return, per row, whether every excerpt is a verbatim span of the text sent.

    Grounded against one run's request copies, not both. The two runs screened the
    same corpus under the same packaging, so the text they sent for a paper is
    byte-identical; that was checked on 4,000 shared custom_ids before relying on
    it. The requests files hold the whole screened corpus as text, so they are
    streamed and each paper's excerpts are checked as its text goes past, rather
    than building a lookup that would hold a gigabyte.

    The paper's text is normalised once and its excerpts tested against that,
    rather than through ungrounded_excerpts, which normalises the source per
    call. That function is written for the screen, where a paper is grounded
    once; here a paper carries 2.8 pair rows on average and re-normalising a
    3 kB source for each of them is most of this stage's runtime.

    A paper whose request carries no text cannot ground anything and comes back
    False: an excerpt that cannot be checked is not an excerpt that passed.

    Args:
        frame: The unioned pair table.
        run_dir: The run whose request copies to ground against.

    Returns:
        A boolean Series aligned to frame's index.

    Raises:
        FileNotFoundError: If the run has no requests directory.
    """
    requests_dir = run_dir / REQUESTS_SUBDIR
    if not requests_dir.is_dir():
        raise FileNotFoundError(f'No requests directory at {requests_dir}')

    by_paper: dict[str, list[int]] = defaultdict(list)
    for position, custom_id in enumerate(frame['custom_id']):
        by_paper[custom_id].append(position)
    excerpts = frame['excerpts'].tolist()

    grounded = [False] * len(frame)
    seen: set[str] = set()
    n_no_text = 0
    # Yielded in chunk order, so a custom_id a retry re-sent appears more than
    # once; the first copy is the one to ground against, as in iter_final_records
    # the last record wins but every attempt was sent the same text.
    for row in iter_requests(run_dir, set(by_paper)):
        custom_id = row['custom_id']
        if custom_id in seen:
            continue
        seen.add(custom_id)
        text = row.get('text') or ''
        if not text:
            n_no_text += 1
            continue
        source = normalise_text(text)
        for position in by_paper[custom_id]:
            items = [str(item) for item in excerpts[position]]
            grounded[position] = bool(items) and all(
                item.strip() and normalise_text(item) in source
                for item in items
            )

    n_absent = sum(1 for key in by_paper if key not in seen)
    if n_absent or n_no_text:
        logger.warning(
            '%d papers absent from %s requests, %d carrying no text',
            n_absent, run_dir.name, n_no_text,
        )
    return pd.Series(grounded, index=frame.index)


def apply_filters(
        frame: pd.DataFrame,
        run_dir: Path,
        index: UniprotNameIndex | None = None,
) -> pd.DataFrame:
    """Mark every pair the deterministic filters remove.

    Both filters run over the whole table and the first one to fire supplies the
    reason, so the reported rates are per-pair rather than cumulative. Nothing is
    deleted: a drop_reason column keeps the removal rates measurable and lets a
    filter be reconsidered without re-running the union.

    The construct filter is gated per side on whether UniProt records that exact
    name. If it does, the name is a protein and no heuristic may remove it. The
    gate is what makes the filter safe rather than any single rule being careful:
    measured over all 294,154 sides, it protects 655 gene names the earlier
    ungated suffix rule dropped (CD40L, VPS33B, BCL11A, DDX39B), a further 29
    sides of systematic gene-family naming no regex can separate (PPP1R12A,
    TBC1D22A, E1B55K, and the ASFV names pA104R and pS273R), and 57 sides the
    standalone-mutation rule would eat (S100B, S100P, E75A and the poxvirus ORF
    names A46R, B18R, D10R). In exchange the standalone rule can run at all,
    which drops 4,715 genuine construct sides the suffix rule cannot see.

    The non-protein, drug and monoclonal filter is deliberately not here yet. It
    needs the same kind of gate on names resolving to zero accessions: run
    ungated it fires on 6.22% of names that do resolve, including Tau, Fas and
    FLOWERING LOCUS T, and whole-name anchoring does not prevent that.

    Args:
        frame: The unioned pair table.
        run_dir: The run whose request copies to ground against.
        index: The reviewed-UniProt name index. Defaults to the shared one.

    Returns:
        The table with grounded and drop_reason columns added.
    """
    index = index or get_name_index()
    frame = frame.copy()
    frame['grounded'] = ground_pairs(frame, run_dir)
    construct = [
        any(
            is_construct_name(str(name)) and not index.records_name(str(name))
            for name in (name_a, name_b)
        )
        for name_a, name_b in zip(frame['name_a'], frame['name_b'])
    ]
    frame['drop_reason'] = [
        DROP_UNGROUNDED if not is_grounded
        else DROP_CONSTRUCT if is_construct
        else ''
        for is_grounded, is_construct in zip(frame['grounded'], construct)
    ]
    return frame


def is_reagent_name(name: str) -> bool:
    """Return whether a written name is an assay reagent rather than a protein.

    Args:
        name: A protein name as the screen emitted it.

    Returns:
        True if the whole normalised name is on the reagent blocklist, or the
        written name is a bare ion.
    """
    if normalise_name(name) in REAGENT_NAMES:
        return True
    return bool(ION_RE.match(name.strip()))


def apply_final_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark the pairs the two post-resolution filters remove.

    Both run after mapping, for opposite reasons. The reagent blocklist fires on
    names that do resolve, which is precisely why mapping cannot substitute for
    it. The virus-virus drop needs the per-side viral flag, which only exists
    once the candidates are known.

    Only virus-virus pairs are dropped, never host-pathogen ones: a paper
    reporting that a viral effector does not bind a host protein is a negative
    the benchmark wants. The viral flag is per side and is only set where every
    candidate still standing for that side is viral, so a name shared between a
    viral and a cellular protein - 'U1' resolves to both a Tibrogargan virus
    protein and cellular U1 - does not make the pair viral while both readings
    are live. Once the paper's own text has picked one, the flag follows the
    pick: a side determined to a viral accession is viral.

    Args:
        frame: The species-resolved pair table.

    Returns:
        The table with drop_reason written for the newly dropped pairs.
    """
    frame = frame.copy()
    reagent = [
        is_reagent_name(str(name_a)) or is_reagent_name(str(name_b))
        for name_a, name_b in zip(frame['name_a'], frame['name_b'])
    ]
    frame['drop_reason'] = [
        reason if reason
        else DROP_REAGENT if is_reagent
        else DROP_VIRUS_VIRUS if viral_a and viral_b
        else ''
        for reason, is_reagent, viral_a, viral_b in zip(
            frame['drop_reason'], reagent, frame['viral_a'], frame['viral_b'],
        )
    ]
    return frame


def _write_side_columns(
        frame: pd.DataFrame,
        suffix: str,
        sides: list[SideResolution],
        name_sources: list[str] | None = None,
        species_sources: list[str] | None = None,
) -> None:
    """Write one side's resolution columns onto the table, in place.

    Shared by both resolvers so the published table's column set is defined
    once. The two source columns are written only by the with-text resolver,
    which is the only stage that has them.

    Args:
        frame: The table to write onto.
        suffix: 'a' or 'b'.
        sides: One resolution per row, in row order.
        name_sources: Which source supplied the candidates, per row.
        species_sources: Which source supplied the organism, per row.
    """
    frame[f'outcome_{suffix}'] = [side.outcome for side in sides]
    frame[f'n_candidates_{suffix}'] = [side.n_candidates for side in sides]
    frame[f'n_groups_{suffix}'] = [side.n_groups for side in sides]
    frame[f'accession_{suffix}'] = [side.accession for side in sides]
    frame[f'viral_{suffix}'] = [side.is_viral for side in sides]
    frame[f'provenance_{suffix}'] = [
        OUTCOME_PROVENANCE.get(side.outcome, PROVENANCE_NONE)
        for side in sides
    ]
    if name_sources is not None:
        frame[f'name_source_{suffix}'] = name_sources
    if species_sources is not None:
        frame[f'species_source_{suffix}'] = species_sources


def assign_status(frame: pd.DataFrame) -> None:
    """Give each pair whichever of its two sides needs more, in place."""
    frame['pair_status'] = [
        max(
            SIDE_STATUS[outcome_a], SIDE_STATUS[outcome_b],
            key=STATUS_ORDER.index,
        )
        for outcome_a, outcome_b in zip(frame['outcome_a'], frame['outcome_b'])
    ]


def classify_side(
        candidates: tuple[str, ...],
        index: UniprotNameIndex,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        determined_outcome: str = SIDE_DETERMINED,
) -> SideResolution:
    """Turn a side's surviving candidates into its outcome.

    Shared by the name-only and the with-text resolvers so the two stages cannot
    give the same candidate set a different status.

    The viral flag describes the candidates that survive, not every candidate the
    written name could ever mean. A name spanning a viral and a cellular protein
    is not viral on its own, but once the paper's own text has narrowed it to the
    viral one it is, and the virus-virus drop exists to remove exactly that pair.

    Args:
        candidates: The side's candidate accessions. Must not be empty.
        index: The reviewed-UniProt name index.
        max_candidates: Most candidates that can still be given to a model.
        determined_outcome: The outcome to record when one candidate survives.

    Returns:
        The side's outcome, its candidate and group counts, the accession where
        one is determined, and whether every surviving candidate is viral.
    """
    viral = all(index.cards[item].is_viral for item in candidates)
    if len(candidates) == 1:
        return SideResolution(
            determined_outcome, 1, 1, candidates[0], viral, candidates,
        )
    n_groups = len(index.group_candidates(candidates))
    if len(candidates) > max_candidates:
        outcome = SIDE_TOO_WIDE
    elif n_groups == 1:
        outcome = SIDE_SPECIES_PICK
    else:
        outcome = SIDE_IDENTITY_PICK
    return SideResolution(
        outcome, len(candidates), n_groups, '', viral, candidates,
    )


def resolve_name(
        name: str,
        index: UniprotNameIndex,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> SideResolution:
    """Resolve one written name against the reviewed index.

    Orthologue groups are counted even for a name that routes to too_wide, so the
    group counts stay comparable across the whole pool rather than only over the
    names small enough to send to a model.

    The candidate cap is a routing threshold, not a filter: a side carrying more
    than max_candidates cannot be handed to a model as a list, so it goes to
    deeper curation. Nothing is discarded for being wide. The names above the cap
    are pan-species core genes and non-proteins - CYTB at 1,791 candidates, atpE
    at 1,265, NADP at 1,128 - and the last of those is the blocklist's job, not
    this cap's.

    Args:
        name: A protein name as the screen emitted it.
        index: The reviewed-UniProt name index.
        max_candidates: Most candidates that can still be given to a model.

    Returns:
        The side's outcome, its candidate and group counts, the accession where
        the name alone determines one, and whether every candidate is viral.
    """
    candidates = index.candidates(name)
    if not candidates:
        return SideResolution(SIDE_UNRESOLVED, 0, 0, '', False, ())
    return classify_side(candidates, index, max_candidates)


def resolve_pairs(
        frame: pd.DataFrame,
        index: UniprotNameIndex | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> pd.DataFrame:
    """Resolve both sides of every pair and give each pair a status.

    Resolution is cached per distinct written name, since the corpus reuses names
    heavily and grouping a wide candidate set is the expensive part.

    Every side is resolved independently and species is never intersected across
    a pair. Intersecting mislabels host-pathogen negatives - 'MOSPD2 || GRA12'
    came out human when GRA12 is a Toxoplasma protein - and makes any per-pair
    viral call unreliable, since 'U1' resolves to both a Tibrogargan virus protein
    and cellular U1.

    Args:
        frame: The filtered pair table.
        index: The reviewed-UniProt name index. Defaults to the shared one.
        max_candidates: Most candidates that can still be given to a model.

    Returns:
        The table with per-side resolution columns, a mapping provenance per side
        and a status per pair.
    """
    index = index or get_name_index()
    distinct = {
        str(name) for column in ('name_a', 'name_b') for name in frame[column]
    }
    resolved = {
        name: resolve_name(name, index, max_candidates)
        for name in distinct
    }
    logger.info('resolved %d distinct names', len(resolved))

    frame = frame.copy()
    for suffix in ('a', 'b'):
        _write_side_columns(
            frame, suffix,
            [resolved[str(name)] for name in frame[f'name_{suffix}']],
        )
    assign_status(frame)
    # Carried on the table rather than left to the run's logs. Entries are added,
    # renamed and demerged between releases, so the release is what a rebuild has
    # to match, and it is what the published table's provenance records. Parquet
    # dictionary-encodes a column of one repeated value to almost nothing.
    frame['uniprot_release'] = index.release
    return frame


class SpeciesEvidence(NamedTuple):
    """The paper's own statements about organisms, keyed for one side lookup."""

    paper_organisms: dict[str, frozenset[int]]
    definitions: dict[tuple[str, str], list[str]]
    organism_terms: dict[str, frozenset[int]]
    prefix_codes: dict[str, frozenset[int]]
    single_letter: dict[str, frozenset[int]]
    taxon_scientific: dict[int, str]

    def scientific_name(self, accession: str, index: UniprotNameIndex) -> str:
        """Return the scientific name of the organism an accession belongs to.

        Read through the taxid rather than held as its own accession-keyed map:
        UniProt writes one organism string per taxid, so a second table would be
        575k entries restating 15k.

        Args:
            accession: A reviewed accession present in the index.
            index: The reviewed-UniProt name index.

        Returns:
            The scientific name, empty if the taxid is not in the scan's reach.
        """
        card = index.cards.get(accession)
        if card is None:
            return ''
        return self.taxon_scientific.get(card.organism_id, '')


def load_species_evidence(
        index: UniprotNameIndex,
        organisms_path: Path,
        definitions_path: Path,
) -> SpeciesEvidence:
    """Load the corpus scan's caches and the lookups the species check needs.

    Species is compared on scientific name rather than taxid. UniProt carries a
    separate taxid per strain, so 'Saccharomyces cerevisiae' alone spans many of
    them, and comparing taxids would make a side whose candidates sit in two
    strains of one species look ambiguous when it is not.

    Args:
        index: The reviewed-UniProt name index.
        organisms_path: The per-paper organism cache.
        definitions_path: The per-paper abbreviation cache.

    Returns:
        Everything the per-side species check reads.
    """
    organisms = pd.read_parquet(organisms_path)
    paper_organisms = {
        str(paper_id): frozenset(int(item) for item in ids)
        for paper_id, ids in zip(organisms['paper_id'], organisms['organism_ids'])
    }
    definition_frame = pd.read_parquet(definitions_path)
    definitions: dict[tuple[str, str], list[str]] = defaultdict(list)
    for paper_id, abbrev, long_form in zip(
        definition_frame['paper_id'], definition_frame['abbrev'], definition_frame['long_form'],
    ):
        definitions[(str(paper_id), str(abbrev))].append(str(long_form))

    codes, taxon_name = build_prefix_taxa(index)
    single = {
        letter: single_letter_taxa(letter, taxon_name)
        for letter in SINGLE_LETTER_ORGANISMS
    }
    taxon_scientific = {
        taxid: split_organism(organism)[0]
        for taxid, organism in taxon_name.items()
    }
    logger.info(
        'species evidence: %d papers with organisms, %d defined abbreviations',
        len(paper_organisms), len(definitions),
    )
    return SpeciesEvidence(
        paper_organisms=paper_organisms,
        definitions=dict(definitions),
        organism_terms=build_organism_terms(index),
        prefix_codes=codes,
        single_letter=single,
        taxon_scientific=taxon_scientific,
    )


def species_sources(
        name: str,
        paper_id: str,
        evidence: SpeciesEvidence,
        allow_prefix: bool = True,
) -> list[tuple[str, frozenset[str]]]:
    """Return the organism assertions available for one side, strongest first.

    A definition in the paper's own text ties an organism to this exact written
    name, which is stronger than a prefix convention, which is stronger than the
    organisms the paper names anywhere. The caller takes the first that the
    candidates actually support rather than merging them.

    Args:
        name: The written name.
        paper_id: The paper the name was written in.
        evidence: The loaded species evidence.
        allow_prefix: Whether the species-prefix rule may fire. Pass False for a
            name UniProt carries whole, where the leading letter is part of the
            name rather than a prefix.

    Returns:
        (source, scientific names) in priority order, sources that assert
        nothing omitted.
    """
    found: list[tuple[str, frozenset[str]]] = []

    definition_taxa: set[int] = set()
    for long_form in evidence.definitions.get(
        (paper_id, normalise_name(name)), (),
    ):
        definition_taxa |= scan_organisms(long_form, evidence.organism_terms)
    if definition_taxa:
        found.append(
            (SOURCE_DEFINITION, _names_for(definition_taxa, evidence)),
        )

    paper_taxa = evidence.paper_organisms.get(paper_id)
    paper_names = _names_for(
        paper_taxa, evidence,
    ) if paper_taxa else frozenset()

    # The prefix must be an organism the paper itself names, whenever the paper
    # names any. A leading letter is a weak convention and collides with names
    # that simply start with one: 'mTOR' is mechanistic target of rapamycin, not
    # a mouse TOR; 'bFGF' and 'bZIP' are basic, not bovine; 'zDHHC17' is a zinc
    # finger, not zebrafish. Unchecked, the single-letter rule fired on 437 mTOR
    # sides and assigned an organism the paper never mentions on 915 of 2,165
    # sides where the paper said what it studied. Intersecting fixes all of those
    # at once and leaves 'mNTE' in a mouse paper alone, which is what the rule is
    # for. A paper naming no organism at all gets the prefix unchecked, since
    # there is nothing to check it against.
    prefix = species_prefix(
        name, evidence.prefix_codes, evidence.single_letter,
    ) if allow_prefix else None
    if prefix is not None:
        supported = prefix_organisms(prefix.taxids, paper_id, evidence)
        if supported:
            found.append((SOURCE_PREFIX, supported))

    if paper_names:
        found.append((SOURCE_PAPER, paper_names))
    return found


def _names_for(
        taxids: frozenset[int] | set[int],
        evidence: SpeciesEvidence,
) -> frozenset[str]:
    """Map taxids to the scientific names they belong to."""
    return frozenset(
        evidence.taxon_scientific[taxid] for taxid in taxids
        if taxid in evidence.taxon_scientific
    )


def prefix_organisms(
        taxids: frozenset[int],
        paper_id: str,
        evidence: SpeciesEvidence,
) -> frozenset[str]:
    """Return the organisms a species prefix may claim in a given paper.

    The prefix must name an organism the paper itself names, whenever the paper
    names any. A paper naming none gets the prefix unchecked, since there is
    nothing to check it against. See species_sources for what this rule costs
    and why it is worth it.

    Args:
        taxids: The organisms the prefix denotes.
        paper_id: The paper the name was written in.
        evidence: The loaded species evidence.

    Returns:
        The scientific names the prefix may claim here, empty if the paper
        contradicts it.
    """
    wanted = _names_for(taxids, evidence)
    paper_taxa = evidence.paper_organisms.get(paper_id)
    if not paper_taxa:
        return wanted
    return wanted & _names_for(paper_taxa, evidence)


def rescue_candidates(
        name: str,
        paper_id: str,
        index: UniprotNameIndex,
        evidence: SpeciesEvidence,
) -> tuple[str, tuple[str, ...]]:
    """Find candidates for a side whose written name resolves to nothing.

    Two of the species sources double as name rescues, because both strip
    something off the written name and leave something the index does carry.
    A species prefix leaves a bare symbol - 'AtCGL160' resolves to nothing but
    'CGL160' resolves. A definition supplies a whole phrase, which resolves once
    reduced from a phrase to a name.

    The prefix is tried first because it is the more reliable of the two; the
    definition route was measured to clear only a few hundred pairs on its own and
    is here because the definitions are already loaded, not because it carries
    the stage.

    A rescue keeps only the candidates its own evidence supports, because the
    source that stripped the name is the only reason those candidates are on the
    table at all. Without that check the prefix is just a string-trimming device:
    'GmMYB93' would resolve on 'MYB93' to an Arabidopsis accession and, being the
    only one, be recorded as determined - 845 of 1,080 single-candidate rescues
    landed in an organism their own prefix contradicted, 'OsPYL12' on Arabidopsis
    and 'mGlu5' on rice among them. A rescue nothing supports is a coincidence of
    spelling, so the side stays unresolved.

    Args:
        name: The written name.
        paper_id: The paper the name was written in.
        index: The reviewed-UniProt name index.
        evidence: The loaded species evidence.

    Returns:
        The source that supplied the candidates and the candidates, or an empty
        source and no candidates.
    """
    prefix = species_prefix(
        name, evidence.prefix_codes,
        evidence.single_letter,
    )
    if prefix is not None:
        # Filtered on what the prefix itself asserts, deliberately without the
        # paper-corroboration gate that species_sources applies. That gate
        # guards using a prefix to narrow candidates the name found on its own,
        # where a wrong reading of a leading letter costs a real accession. Here
        # the prefix is the only reason there are candidates at all, so the
        # question is just whether they are the organism it named. Requiring the
        # paper to name it too was tried and is too strong: the organism scan
        # only knows the common names UniProt records, so a yeast paper writing
        # 'budding yeast' corroborates nothing and 574 resolved pairs went with
        # it, 'Abp1p' and 'Sec1p' among them.
        candidates = _in_organisms(
            index.candidates(prefix.stem),
            _names_for(prefix.taxids, evidence), index, evidence,
        )
        if candidates:
            return SOURCE_PREFIX, candidates
    for long_form in evidence.definitions.get(
        (paper_id, normalise_name(name)), (),
    ):
        # A definition is already a statement the paper made about this exact
        # name, so it is not intersected with the whole-paper scan the way a
        # prefix is: the scan reads a capped title, abstract and methods, and a
        # definition stated in the results would fail a check against it.
        stated = _names_for(
            scan_organisms(long_form, evidence.organism_terms), evidence,
        )
        for variant in long_form_variants(long_form):
            candidates = _in_organisms(
                index.candidates(variant), stated, index, evidence,
            )
            if candidates:
                return SOURCE_DEFINITION, candidates
    return SOURCE_NONE, ()


def _in_organisms(
        candidates: tuple[str, ...],
        organisms: frozenset[str],
        index: UniprotNameIndex,
        evidence: SpeciesEvidence,
) -> tuple[str, ...]:
    """Keep the candidates belonging to one of these organisms.

    Args:
        candidates: Candidate accessions.
        organisms: Scientific names the source asserts. An empty set asserts
            nothing, so every candidate survives.
        index: The reviewed-UniProt name index.
        evidence: The loaded species evidence.

    Returns:
        The surviving candidates, in input order.
    """
    if not organisms:
        return candidates
    return tuple(
        item for item in candidates
        if evidence.scientific_name(item, index) in organisms
    )


def resolve_side_with_text(
        name: str,
        paper_id: str,
        index: UniprotNameIndex,
        evidence: SpeciesEvidence,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[SideResolution, str, str]:
    """Resolve one side, using the paper's own text where the name alone fails.

    The candidate cap is applied only after the text has had its say. A name
    like 'atpE' carries 1,265 candidates and would once have been discarded for
    being wide, but a paper that names its organism collapses it to one.

    Args:
        name: The written name.
        paper_id: The paper the name was written in.
        index: The reviewed-UniProt name index.
        evidence: The loaded species evidence.
        max_candidates: Most candidates that can still go to a model as a list.

    Returns:
        The resolution, the source that supplied the candidates, and the source
        that supplied the organism.
    """
    candidates = index.candidates(name)
    # A name that UniProt carries whole has no species prefix to read: the
    # leading letter is part of it. 'mTOR' is mechanistic target of rapamycin,
    # 'zDHHC17' a zinc finger, 'bZIP' a basic leucine zipper, and all three
    # resolve outright, where a genuine prefix like 'hMLH1' or 'mNTE' does not.
    # Without this the single-letter rule read 437 mTOR sides as mouse.
    resolves_alone = bool(candidates)
    name_source = SOURCE_NONE
    if not candidates:
        name_source, candidates = rescue_candidates(
            name, paper_id, index, evidence,
        )
    if not candidates:
        return SideResolution(SIDE_UNRESOLVED, 0, 0, '', False, ()), name_source, SOURCE_NONE

    species_source = SOURCE_NONE
    if len(candidates) > 1:
        for source, wanted in species_sources(
            name, paper_id, evidence, allow_prefix=not resolves_alone,
        ):
            survivors = tuple(
                item for item in candidates
                if evidence.scientific_name(item, index) in wanted
            )
            if survivors:
                candidates, species_source = survivors, source
                break

    determined = (
        SIDE_DETERMINED if species_source == SOURCE_NONE and not name_source
        else SIDE_DETERMINED_BY_TEXT
    )
    return (
        classify_side(
            candidates, index, max_candidates, determined,
        ),
        name_source, species_source,
    )


def resolve_pairs_with_text(
        frame: pd.DataFrame,
        evidence: SpeciesEvidence,
        index: UniprotNameIndex | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> pd.DataFrame:
    """Re-resolve every side using the paper's own text, and restate each status.

    Cached on (name, paper), not on name alone: the whole point is that the same
    written name means different things in different papers.

    Args:
        frame: The resolved pair table.
        evidence: The loaded species evidence, from load_species_evidence.
        index: The reviewed-UniProt name index. Defaults to the shared one.
        max_candidates: Most candidates that can still go to a model as a list.

    Returns:
        The table with its resolution columns rewritten and a species_source and
        name_source recorded per side.
    """
    index = index or get_name_index()
    keys = {
        (str(name), str(paper_id))
        for suffix in ('a', 'b')
        for name, paper_id in zip(frame[f'name_{suffix}'], frame['paper_id'])
    }
    cache = {
        key: resolve_side_with_text(
            key[0], key[1], index, evidence, max_candidates,
        )
        for key in keys
    }

    frame = frame.copy()
    for suffix in ('a', 'b'):
        rows = [
            cache[(str(name), str(paper_id))]
            for name, paper_id in zip(frame[f'name_{suffix}'], frame['paper_id'])
        ]
        _write_side_columns(
            frame, suffix,
            [side for side, _, _ in rows],
            name_sources=[source for _, source, _ in rows],
            species_sources=[source for _, _, source in rows],
        )
    assign_status(frame)
    logger.info('resolved %d distinct (name, paper) sides', len(cache))
    return frame


def read_kept_pairs(source: str) -> pd.DataFrame:
    """Read a pair table and keep only the rows this stage kept.

    Unguarded on the column: every table a downstream stage accepts carries
    drop_reason, and one that did not would be the wrong table rather than a
    table to keep whole.

    Args:
        source: A published pair table.

    Returns:
        The kept rows, renumbered.
    """
    frame = pd.read_parquet(source)
    return frame[frame['drop_reason'] == ''].reset_index(drop=True)


def write_table(frame: pd.DataFrame, path_text: str) -> Path:
    """Write a stage's table, creating its directory.

    Args:
        frame: The table to write.
        path_text: Where to write it.

    Returns:
        The path written, for the caller to report.
    """
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def print_counts(
        values: pd.Series,
        width: int = 14,
        total: int | None = None,
        skip_blank: bool = False,
        labels: dict | None = None,
        order: tuple[str, ...] | None = None,
) -> None:
    """Print a value-count breakdown, optionally as a share of a total.

    Args:
        values: The column to count.
        width: Column width for the label.
        total: Denominator for a percentage column, omitted if None.
        skip_blank: Drop the empty-string label, which several of these columns
            use as their 'nothing fired' value.
        labels: Display names for the values that have one.
        order: Print these values in this order, for a column whose values have
            an order of their own. Ordered by frequency if omitted.
    """
    counts = values.value_counts()
    if order is not None:
        counts = counts.reindex([
            value for value in order if value in counts.index
        ])
    for value, count in counts.items():
        if skip_blank and not value:
            continue
        label = (labels or {}).get(value, str(value) or '(none)')
        share = f'  {100.0 * count / total:>5.1f}%' if total else ''
        print(f'  {label:<{width}} {count:>8,}{share}')


def _report_resolution(frame: pd.DataFrame) -> None:
    """Print the per-side outcome split and the per-pair status split."""
    outcomes = pd.concat([frame['outcome_a'], frame['outcome_b']])
    print(f'\n{len(outcomes):,} sides:')
    print_counts(outcomes, width=20, total=len(outcomes))

    print(f'\n{len(frame):,} pairs:')
    for status in STATUS_ORDER:
        selected = frame['pair_status'] == status
        count = int(selected.sum())
        print(
            f'  {status:<20} {count:>8,}  '
            f'{100.0 * count / len(frame):>5.1f}%   '
            f'{frame.loc[selected, "pmid"].nunique():>7,} papers',
        )

    both_viral = (frame['viral_a'] & frame['viral_b']).sum()
    any_viral = (frame['viral_a'] | frame['viral_b']).sum()
    print(f'\nviral: {both_viral:,} virus-virus pairs (droppable), '
          f'{any_viral - both_viral:,} host-pathogen (kept)')

    picks = frame[frame['pair_status'] == STATUS_NEEDS_MODEL_PICK]
    n_picks = sum(
        1 for outcome in pd.concat([picks['outcome_a'], picks['outcome_b']])
        if outcome in PICK_OUTCOMES
    )
    print(f'side-picks the model stage would face: {n_picks:,}')


def _report_final(frame: pd.DataFrame) -> None:
    """Print what the reagent blocklist and the virus-virus drop removed."""
    for reason in (DROP_REAGENT, DROP_VIRUS_VIRUS):
        dropped = frame[frame['drop_reason'] == reason]
        print(f'\n{reason}: {len(dropped):,} pairs '
              f'({100.0 * len(dropped) / len(frame):.2f}%)')
        for status in STATUS_ORDER:
            selected = frame['pair_status'] == status
            hit = int((selected & (frame['drop_reason'] == reason)).sum())
            print(
                f'  {status:<20} {hit:>6,} of {int(selected.sum()):>7,}',
            )

    reagent = frame[frame['drop_reason'] == DROP_REAGENT]
    names = pd.concat([reagent['name_a'], reagent['name_b']])
    hits = pd.Series(
        [name for name in names if is_reagent_name(str(name))], dtype=object,
    ).value_counts()
    print(f'\n{len(hits)} reagent names fired, top 15 by side count:')
    for name, count in hits.head(15).items():
        print(f'  {str(name):<24} {count:>5,}')


def _report(frame: pd.DataFrame) -> None:
    """Print the headline counts for a pair table."""
    print(f'{len(frame):,} pairs over {frame["pmid"].nunique():,} papers')
    print(
        f'  unique name pairs {frame.groupby(["norm_a", "norm_b"]).ngroups:,}',
    )
    print('\nseen in:')
    print_counts(frame['seen_in'], total=len(frame))


def _run_union(args: argparse.Namespace) -> None:
    """Union the runs' pair calls and write the unioned table."""
    frame = union_runs(args.runs, args.relationship)
    out_path = write_table(frame, args.out)
    _report(frame)
    print('\ntext source:')
    print_counts(frame['text_source'])
    print(f'\nwrote {out_path}')


def _run_filter(args: argparse.Namespace) -> None:
    """Apply the pre-mapping filters and write the filtered table."""
    frame = pd.read_parquet(args.source)
    frame = apply_filters(frame, RUNS_ROOT / args.ground_against)
    out_path = write_table(frame, args.out)

    kept = frame[frame['drop_reason'] == '']
    print(f'{len(frame):,} pairs in, {len(kept):,} kept')
    print('\ndropped:')
    print_counts(frame['drop_reason'], total=len(frame), skip_blank=True)
    print()
    _report(kept)
    print(f'\nwrote {out_path}')


def _run_resolve(args: argparse.Namespace) -> None:
    """Resolve both sides against UniProt and write the resolved table."""
    frame = pd.read_parquet(args.source)
    frame = frame[frame['drop_reason'] == ''].reset_index(drop=True)
    frame = resolve_pairs(
        frame, max_candidates=args.max_candidates,
    )
    out_path = write_table(frame, args.out)
    print(f'candidate cap {args.max_candidates}')
    _report_resolution(frame)
    print(f'\nwrote {out_path}')


def _run_species(args: argparse.Namespace) -> None:
    """Re-resolve each side against its own paper's text and write the table."""
    frame = pd.read_parquet(args.source)
    index = get_name_index()
    evidence = load_species_evidence(
        index, Path(args.organisms), Path(args.definitions),
    )
    frame = resolve_pairs_with_text(
        frame, evidence, index,
        max_candidates=args.max_candidates,
    )
    out_path = write_table(frame, args.out)
    _report_resolution(frame)
    print('\norganism source, where one fired:')
    print_counts(
        pd.concat([frame['species_source_a'], frame['species_source_b']]),
        skip_blank=True,
    )
    print('\ncandidates rescued for a name that resolved to nothing:')
    print_counts(
        pd.concat([frame['name_source_a'], frame['name_source_b']]),
        skip_blank=True,
    )
    print(f'\nwrote {out_path}')


def _run_finalise(args: argparse.Namespace) -> None:
    """Apply the post-mapping filters, write the dated table and its provenance.

    Two tables come out, and only one of them is the deliverable. The dated file
    holds the kept pairs and nothing else, because it is what gets published and
    a consumer reading it should not have to know that a drop_reason column
    exists. The annotated table, every row with its reason, stays beside the
    other stage outputs so removal rates stay measurable and a filter can be
    reconsidered without re-running the union.

    Raises:
        KeyError: If the source table predates the uniprot_release column, which
            is what a rebuild has to match and so what the provenance records.
    """
    frame = pd.read_parquet(args.source)
    if 'uniprot_release' not in frame.columns or frame.empty:
        raise KeyError(
            f'{args.source} is empty or predates the uniprot_release column. '
            'Re-run resolve and species: the release is what a rebuild has to '
            'match, so the published table has to carry it.',
        )
    frame = apply_final_filters(frame)
    kept = frame[frame['drop_reason'] == ''].reset_index(drop=True)

    audit_path = write_table(frame, args.audit_out)
    # Resolved here rather than inside the filename, so the provenance records
    # the stamp the file actually carries rather than a null for the default.
    stamp = args.date or date.today().isoformat()
    out_path = write_table(
        kept,
        str(
            Path(args.out_dir) / make_dated_filename(
                PAIRS_NAME, LITERATURE_STAGE_VERSIONS[PAIRS_STAGE], '.parquet',
                stamp,
            ),
        ),
    )

    provenance_path = out_path.with_suffix('.provenance.json')
    write_output_provenance(
        provenance_path,
        build_output_provenance(
            workflow='flock.negatome_v3.literature.pairs.finalise',
            parameters={'source': args.source, 'date': stamp},
            input_paths={'species_pairs': args.source},
            extra={
                'uniprot_release': frame['uniprot_release'].iloc[0],
                'screen_runs': [PRIMARY_RUN_ID, REPLICATE_RUN_ID],
                'n_pairs': len(frame),
                'n_kept': len(kept),
                'n_kept_by_status': {
                    status: int((kept['pair_status'] == status).sum())
                    for status in STATUS_ORDER
                },
            },
            source_name='flock',
            repo_root=Path(__file__).resolve().parents[3],
        ),
    )
    _report_final(frame)
    print(f'\n{len(kept):,} pairs kept of {len(frame):,}')
    _report_resolution(kept)
    print(f'\nwrote {out_path} ({len(kept):,} kept pairs)')
    print(f'wrote {provenance_path}')
    print(f'wrote {audit_path} (all {len(frame):,} rows, with drop_reason)')
    if args.upload:
        prefix = get_literature_stage_prefix(PAIRS_STAGE)
        upload_file_to_s3(str(out_path), prefix)
        upload_file_to_s3(str(provenance_path), prefix)
        print(f'uploaded both to {prefix}')


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments, carrying the handler for the chosen
        subcommand as `run`.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Turn the screen runs\' no-interaction pair calls into the '
            'deterministic pair table, one step at a time.'
        ),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    union = subparsers.add_parser(
        'union', help='Union the runs\' pair calls into one table.',
    )
    union.set_defaults(run=_run_union)
    union.add_argument(
        '--runs', nargs='+', default=[PRIMARY_RUN_ID, REPLICATE_RUN_ID],
        help='Run ids to union, as directory names under the runs root.',
    )
    union.add_argument(
        '--relationship', default=NO_INTERACTION,
        help=(
            'Which pair calls to keep. The default is the only one the '
            'benchmark treats as a negative; "both" is deliberately dropped, '
            'having cost 10.8 points of gold recall for 12% of volume.'
        ),
    )
    union.add_argument(
        '--out', default=str(PAIRS_ROOT / 'screen_pairs.parquet'),
        help='Where to write the unioned table.',
    )

    filters = subparsers.add_parser(
        'filter', help='Apply the deterministic filters that precede mapping.',
    )
    filters.set_defaults(run=_run_filter)
    filters.add_argument(
        '--source', default=str(PAIRS_ROOT / 'screen_pairs.parquet'),
        help='The unioned table to filter.',
    )
    filters.add_argument(
        '--ground-against', default=PRIMARY_RUN_ID,
        help=(
            'Run id whose request copies supply the text excerpts are grounded '
            'against. Either run does: both sent byte-identical text.'
        ),
    )
    filters.add_argument(
        '--out', default=str(PAIRS_ROOT / 'screen_pairs_filtered.parquet'),
        help='Where to write the filtered table.',
    )

    resolve = subparsers.add_parser(
        'resolve',
        help='Resolve both sides against UniProt and give each pair a status.',
    )
    resolve.set_defaults(run=_run_resolve)
    resolve.add_argument(
        '--source', default=str(PAIRS_ROOT / 'screen_pairs_filtered.parquet'),
        help='The filtered table to resolve.',
    )
    resolve.add_argument(
        '--max-candidates', type=int, default=DEFAULT_MAX_CANDIDATES,
        help=(
            'Most candidates a side can carry and still be given to a model as a '
            'list. A routing threshold, not a filter: wider sides go to deeper '
            'curation rather than being dropped.'
        ),
    )
    resolve.add_argument(
        '--out', default=str(PAIRS_ROOT / 'screen_pairs_resolved.parquet'),
        help='Where to write the resolved table.',
    )

    species = subparsers.add_parser(
        'species',
        help='Re-resolve each side using the organism its own paper names.',
    )
    species.set_defaults(run=_run_species)
    species.add_argument(
        '--source', default=str(PAIRS_ROOT / 'screen_pairs_resolved.parquet'),
        help='The resolved table to re-resolve.',
    )
    species.add_argument(
        '--organisms', default=str(PAPER_ORGANISMS_PATH),
        help='Per-paper organism cache, written by the organisms module.',
    )
    species.add_argument(
        '--definitions', default=str(PAPER_DEFINITIONS_PATH),
        help='Per-paper abbreviation cache, written by the organisms module.',
    )
    species.add_argument(
        '--max-candidates', type=int, default=DEFAULT_MAX_CANDIDATES,
        help='Most candidates a side can carry and still go to a model.',
    )
    species.add_argument(
        '--out', default=str(PAIRS_ROOT / 'screen_pairs_species.parquet'),
        help='Where to write the table.',
    )

    finalise = subparsers.add_parser(
        'finalise',
        help='Drop reagent and virus-virus pairs, and write the pair table.',
    )
    finalise.set_defaults(run=_run_finalise)
    finalise.add_argument(
        '--source', default=str(PAIRS_ROOT / 'screen_pairs_species.parquet'),
        help='The species-resolved table to finalise.',
    )
    finalise.add_argument(
        '--out-dir', default=str(PAIRS_ROOT),
        help='Directory for the dated pair table and its provenance file.',
    )
    finalise.add_argument(
        '--audit-out', default=str(PAIRS_ROOT / 'screen_pairs_final.parquet'),
        help=(
            'Where to write every row with its drop_reason. Local only: the '
            'dated table is what publishes.'
        ),
    )
    finalise.add_argument(
        '--date', help='Date stamp for the output filename (YYYY-MM-DD).',
    )
    finalise.add_argument(
        '--upload', action='store_true',
        help=(
            'Publish the table and its provenance to the pairs stage prefix in '
            'S3. Off by default: this writes to the shared bucket.'
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build the deterministic pair table from the screen runs' records.

    Five steps, run in order and each writing its own table.

    'union' reads each run's parsed records, keeps the last record per paper so a
    retry supersedes the attempt it replaced, filters to the wanted relationship,
    and keys every pair on its paper plus its two normalised names. Pairs seen by
    more than one run merge into one row that pools their excerpts and keeps each
    run's confidence separately. Nothing is dropped except a pair with a side that
    normalises to nothing.

    'filter' adds the two deterministic filters that are safe before mapping: the
    grounding gate, and the light-touch construct filter. Both record a reason
    rather than deleting the row.

    'resolve' maps both sides against the reviewed UniProt name index and gives
    each pair a status, and 'species' re-resolves each side using the organism
    its own paper names, which is what fills the resolved set.

    'finalise' applies the two filters that can only run once the sides are
    mapped - the reagent and tag blocklist, and the virus-virus drop - and writes
    the dated pair table with a provenance file beside it. Pass --upload to
    publish both to the pairs stage prefix in S3.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    args = parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
