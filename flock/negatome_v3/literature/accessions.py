# The model pick that turns ambiguous names into single UniProt accessions.
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pandas as pd

from flock import REPO_ROOT
from flock.aws import upload_file_to_s3
from flock.logging_utils import setup_logging
from flock.negatome_v3.literature import batch
from flock.negatome_v3.literature import CORPUS_BLOCKS_PATH
from flock.negatome_v3.literature import packaging
from flock.negatome_v3.literature import PAIRS_ROOT
from flock.negatome_v3.literature import PAPER_DEFINITIONS_PATH
from flock.negatome_v3.literature import PAPER_ORGANISMS_PATH
from flock.negatome_v3.literature import runs
from flock.negatome_v3.literature.config import PICK_MODEL
from flock.negatome_v3.literature.config import PICK_PROMPT_VERSION
from flock.negatome_v3.literature.config import PICK_RESULT_MODEL
from flock.negatome_v3.literature.config import PickConfig
from flock.negatome_v3.literature.grounding import normalise as normalise_text
from flock.negatome_v3.literature.grounding import ungrounded_excerpts
from flock.negatome_v3.literature.names import get_name_index
from flock.negatome_v3.literature.names import UniprotNameIndex
from flock.negatome_v3.literature.pairs import assign_status
from flock.negatome_v3.literature.pairs import DEFAULT_MAX_CANDIDATES
from flock.negatome_v3.literature.pairs import load_species_evidence
from flock.negatome_v3.literature.pairs import print_counts
from flock.negatome_v3.literature.pairs import read_kept_pairs
from flock.negatome_v3.literature.pairs import resolve_side_with_text
from flock.negatome_v3.literature.pairs import write_table
from flock.negatome_v3.literature.vocabulary import ACCESSIONS_NAME
from flock.negatome_v3.literature.vocabulary import ACCESSIONS_STAGE
from flock.negatome_v3.literature.vocabulary import DROP_VIRUS_VIRUS
from flock.negatome_v3.literature.vocabulary import OUTCOME_PROVENANCE
from flock.negatome_v3.literature.vocabulary import PAIRS_NAME
from flock.negatome_v3.literature.vocabulary import PAIRS_STAGE
from flock.negatome_v3.literature.vocabulary import PARAGRAPH_FROM_BLOCK
from flock.negatome_v3.literature.vocabulary import PARAGRAPH_FROM_EXCERPT
from flock.negatome_v3.literature.vocabulary import PICK_ABSTAINED
from flock.negatome_v3.literature.vocabulary import PICK_ANSWERED
from flock.negatome_v3.literature.vocabulary import PICK_LABELS
from flock.negatome_v3.literature.vocabulary import PICK_NO_ANSWER
from flock.negatome_v3.literature.vocabulary import PICK_NONE
from flock.negatome_v3.literature.vocabulary import PICK_ORDER
from flock.negatome_v3.literature.vocabulary import PICK_OUTCOMES
from flock.negatome_v3.literature.vocabulary import PICK_UNGROUNDED
from flock.negatome_v3.literature.vocabulary import SIDE_MODEL_PICKED
from flock.negatome_v3.literature.vocabulary import STATUS_NEEDS_MODEL_PICK
from flock.negatome_v3.literature.vocabulary import STATUS_ORDER
from flock.paths import get_literature_stage_prefix
from flock.paths import LITERATURE_STAGE_VERSIONS
from flock.paths import make_dated_filename
from flock.provenance import build_output_provenance
from flock.provenance import write_output_provenance

logger = logging.getLogger(__name__)

# Who answers a request Sonnet 5 declined. Sonnet 4.6 rather than the screen's
# full cascade down to Haiku 4.5: see _run_retry. This is the model the delivered
# screen's own retry chunk ran under, and it accepts output_config.effort, so it
# must stay out of batch.MODELS_WITHOUT_EFFORT.
FALLBACK_MODEL = 'claude-sonnet-4-6'

# Where the stage's own tables live, beside the pair tables they are built from.
PICK_INPUTS_PATH = PAIRS_ROOT / 'pick_inputs.parquet'
PICK_ANSWERS_PATH = PAIRS_ROOT / 'pick_answers.parquet'

# The pair table both subcommands read by default: the dated file the pairs stage
# published, not finalise's audit output, whose filename carries no date and so
# is overwritten in place by any later run on any branch. Repoint this when the
# pairs stage publishes a new date.
PAIRS_TABLE_DATE = '2026-08-11'
DEFAULT_PAIRS_TABLE = PAIRS_ROOT / make_dated_filename(
    PAIRS_NAME, LITERATURE_STAGE_VERSIONS[PAIRS_STAGE], '.parquet',
    PAIRS_TABLE_DATE,
)


def pick_custom_id(paper_id: str, name: str) -> str:
    """The Batch API id for one side, stable across rebuilds of the inputs.

    The paper is kept readable for debugging and the name is hashed, since a
    protein name can hold anything - slashes, Greek letters, spaces - and the
    API constrains the id's characters and length.

    Args:
        paper_id: The paper the name was written in.
        name: The written protein name.

    Returns:
        An id of the form '10021361-3f9a1c2b04'.
    """
    digest = hashlib.sha1(
        f'{paper_id}\x00{name}'.encode(),
    ).hexdigest()[:10]
    return f'{paper_id}-{digest}'


def collect_pick_sides(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce the pair table to one row per side needing a pick.

    The unit is the (paper, name) side, not the pair: the same name in the same
    paper means the same protein however many pairs it appears in, which takes
    50,366 pick sides down to 38,766 requests. Not deduplicated across papers,
    though - that would assume two papers mean the same entry. Excerpts are
    pooled across the side's pairs, since any of them may locate a paragraph.

    Args:
        frame: The pair table, already filtered to the rows to keep.

    Returns:
        One row per side, with paper_id, pmid, the written name, its recorded
        outcome, how many pairs it came from and the pooled excerpts.
    """
    sides = []
    for suffix in ('a', 'b'):
        wanted = frame[frame[f'outcome_{suffix}'].isin(PICK_OUTCOMES)]
        sides.append(
            pd.DataFrame({
                'paper_id': wanted['paper_id'].astype(str),
                'pmid': wanted['pmid'].astype(str),
                'name': wanted[f'name_{suffix}'].astype(str),
                'recorded_outcome': wanted[f'outcome_{suffix}'],
                'excerpts': wanted['excerpts'],
            }),
        )
    long = pd.concat(sides, ignore_index=True)
    logger.info(
        '%d pick sides over %d pairs', len(long), len(frame),
    )

    grouped = []
    for (paper_id, name), group in long.groupby(['paper_id', 'name']):
        grouped.append({
            'paper_id': paper_id,
            'pmid': group['pmid'].iloc[0],
            'name': name,
            'recorded_outcome': group['recorded_outcome'].iloc[0],
            # How many pairs one answer settles, which is what makes the
            # per-side cost readable against the pair counts.
            'n_pairs': len(group),
            'excerpts': list(
                dict.fromkeys(
                    str(item) for items in group['excerpts'] for item in items
                ),
            ),
        })
    out = pd.DataFrame(grouped)
    logger.info('%d distinct (paper, name) sides to ask about', len(out))
    return out


def add_candidates(
        sides: pd.DataFrame,
        index: UniprotNameIndex,
        source_release: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        organisms_path: Path = PAPER_ORGANISMS_PATH,
        definitions_path: Path = PAPER_DEFINITIONS_PATH,
) -> pd.DataFrame:
    """Re-derive the candidate list each side's status was decided on.

    The pair table records how many candidates a side had, not which, so this
    re-runs the same resolver. Re-running rather than looking up matters: the
    surviving set is what the paper's own text left, which a fresh lookup of the
    written name would not reproduce. A side whose outcome no longer matches the
    table is dropped rather than asked about, and the count logged.

    That drift check compares outcomes, which is weaker than it looks: a UniProt
    release can add or retire an entry a name matches, changing the candidate set
    while leaving the outcome 'species_pick' either side of the change. The side
    would then be asked about a list the published table never described, and
    apply would stamp that table's older release into the provenance. So the
    index has to be the release the table was resolved against, and a mismatch
    stops the build rather than being logged.

    Args:
        sides: One row per side, from collect_pick_sides.
        index: The reviewed-UniProt name index.
        source_release: The uniprot_release the source pair table carries.
        max_candidates: Most candidates a side can carry and still be asked about.
        organisms_path: Per-paper organism cache, written by the organisms module.
        definitions_path: Per-paper abbreviation cache, from the same module.

    Returns:
        The sides still resolving to a pick, with their candidate accessions and
        each candidate's viral flag.

    Raises:
        ValueError: If the name index is not the release the pair table records.
    """
    if index.release != source_release:
        raise ValueError(
            f'The pair table was resolved against UniProt {source_release} and '
            f'the name index is {index.release}. Candidate sets can differ '
            f'between releases without the outcome changing, so the picks would '
            f'be made on lists the published table never described. Re-run the '
            f'pairs stage against {index.release}, or build the index from '
            f'{source_release}.',
        )
    evidence = load_species_evidence(index, organisms_path, definitions_path)
    resolutions = [
        resolve_side_with_text(
            str(name), str(paper_id), index, evidence, max_candidates,
        )[0]
        for name, paper_id in zip(sides['name'], sides['paper_id'])
    ]
    sides = sides.copy()
    sides['outcome'] = [item.outcome for item in resolutions]
    sides['candidates'] = [list(item.candidates) for item in resolutions]
    sides['candidates_viral'] = [
        [
            bool(index.cards[accession].is_viral)
            for accession in item.candidates
        ]
        for item in resolutions
    ]
    sides['n_candidates'] = [item.n_candidates for item in resolutions]

    drifted = sides['outcome'] != sides['recorded_outcome']
    if drifted.any():
        logger.warning(
            '%d of %d sides no longer resolve to the outcome the pair table '
            'records and are dropped: %s', int(drifted.sum()), len(sides),
            dict(Counter(sides.loc[drifted, 'outcome'])),
        )
    return sides[~drifted].reset_index(drop=True)


def add_paragraphs(
        sides: pd.DataFrame,
        blocks_path: str | Path,
) -> pd.DataFrame:
    """Attach the paragraph each side's excerpt was copied from.

    One streaming pass over the block table, which does not fit in memory; each
    paper's blocks are normalised once and every side of that paper tested
    against them. A side whose excerpts match no block falls back to the excerpt
    itself, the weaker arm at 64.1% accuracy against the paragraph's 75.6%, so
    the share that happens to is reported rather than buried.

    Args:
        sides: One row per side, carrying paper_id and pooled excerpts.
        blocks_path: Local parquet of parsed blocks from the corpus stage.

    Returns:
        The sides with a paragraph and the source it came from.
    """
    wanted: dict[str, list[int]] = {}
    for position, paper_id in enumerate(sides['paper_id']):
        wanted.setdefault(str(paper_id), []).append(position)

    paragraphs: list[str] = [''] * len(sides)
    excerpts = sides['excerpts'].tolist()
    n_papers = 0

    for paper_id, paper in packaging.iter_papers(blocks_path):
        positions = wanted.get(paper_id)
        if not positions:
            continue
        n_papers += 1
        blocks = [str(text) for text in paper.sort_values('block_index').text]
        normalised = [normalise_text(text) for text in blocks]
        for position in positions:
            for excerpt in excerpts[position]:
                target = normalise_text(str(excerpt))
                if not target:
                    continue
                match = next(
                    (
                        blocks[number] for number, block in enumerate(normalised)
                        if target in block
                    ),
                    '',
                )
                if match:
                    paragraphs[position] = match
                    break

    sides = sides.copy()
    sides['paragraph'] = [
        text or ' '.join(str(item) for item in excerpts[position])
        for position, text in enumerate(paragraphs)
    ]
    # Read off the paragraphs rather than tracked in step with them: a located
    # paragraph is the only thing that separates the two sources.
    sides['paragraph_source'] = [
        PARAGRAPH_FROM_BLOCK if text else PARAGRAPH_FROM_EXCERPT
        for text in paragraphs
    ]
    found = sides['paragraph_source'].eq(PARAGRAPH_FROM_BLOCK).sum()
    logger.info(
        'located a paragraph for %d of %d sides (%.1f%%) over %d papers; the '
        'rest fall back to the excerpt',
        found, len(sides), 100 * found / max(len(sides), 1), n_papers,
    )
    if (sides['paragraph'].str.strip() == '').any():
        raise ValueError(
            'Some sides have neither a matched paragraph nor any excerpt text. '
            'There is nothing to ground an answer against, so they must not be '
            'sent.',
        )
    return sides


def render_candidates(
        accessions: list[str],
        index: UniprotNameIndex,
) -> str:
    """Number the candidate entries for the request.

    Protein name, genes and organism: the organism decides an orthologue
    question and the protein name an identity one.

    Args:
        accessions: Candidate accessions, in the order the resolver left them.
        index: The reviewed-UniProt name index.

    Returns:
        One numbered line per candidate.
    """
    lines = []
    for position, accession in enumerate(accessions, start=1):
        card = index.cards[accession]
        lines.append(
            f'{position}. {accession}  {card.protein_name}  '
            f'[genes: {card.gene_names}]  [organism: {card.organism}]',
        )
    return '\n'.join(lines)


def build_request_text(
        template: str,
        name: str,
        paragraph: str,
        candidates: str,
) -> str:
    """Render the whole request for one side into a single user message.

    Args:
        template: The prompt template, carrying name, text and candidates
            placeholders.
        name: The protein name as the paper writes it.
        paragraph: The paper text the pick is made from.
        candidates: The rendered candidate list.

    Returns:
        The user message, instructions included.
    """
    return template.format(name=name, text=paragraph, candidates=candidates)


def build_inputs(
        frame: pd.DataFrame,
        blocks_path: str | Path,
        index: UniprotNameIndex,
        source_release: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        organisms_path: Path = PAPER_ORGANISMS_PATH,
        definitions_path: Path = PAPER_DEFINITIONS_PATH,
) -> pd.DataFrame:
    """Build the table of sides to ask about, one row per request.

    Args:
        frame: The pair table, already filtered to the rows to keep.
        blocks_path: Local parquet of parsed blocks from the corpus stage.
        index: The reviewed-UniProt name index.
        source_release: The uniprot_release the source pair table carries.
        max_candidates: Most candidates a side can carry and still be asked about.
        organisms_path: Per-paper organism cache.
        definitions_path: Per-paper abbreviation cache.

    Returns:
        The input table, carrying the exact text each request will send.

    Raises:
        ValueError: If two sides collide on one custom_id, which the Batch API
            rejects for the whole batch rather than for the offending request,
            or if the name index is not the pair table's UniProt release.
    """
    sides = collect_pick_sides(frame)
    sides = add_candidates(
        sides, index, source_release, max_candidates, organisms_path,
        definitions_path,
    )
    sides = add_paragraphs(sides, blocks_path)

    sides['custom_id'] = [
        pick_custom_id(str(paper_id), str(name))
        for paper_id, name in zip(sides['paper_id'], sides['name'])
    ]
    duplicated = sides['custom_id'].duplicated()
    if duplicated.any():
        raise ValueError(
            f'{int(duplicated.sum())} sides collide on a custom_id, e.g. '
            f'{sides.loc[duplicated, "custom_id"].iloc[0]}.',
        )
    # The candidate list but not the whole request: the prompt template is
    # applied at submit time, so one input table serves any prompt version
    # instead of needing a rebuild, which is another pass over the block table.
    sides['candidates_text'] = [
        render_candidates(list(candidates), index)
        for candidates in sides['candidates']
    ]
    # recorded_outcome equals outcome on every surviving row, add_candidates
    # having dropped the disagreements, and is kept for that reason: it is the
    # only evidence in this table that re-running the resolver reproduces what
    # the pair table was published on. Only the excerpts go, being superseded by
    # the paragraph each one located.
    return sides.drop(columns=['excerpts'])


def iter_items(
        sides: pd.DataFrame,
        done: set[str],
        template: str,
) -> Iterator[dict]:
    """Yield the requests still to send, in table order.

    Resuming filters on what the run already recorded rather than skipping a row
    count: the work is a table, not a stream, so the ids sent are known exactly
    and a rebuilt table cannot shift the boundary.

    Args:
        sides: The input table.
        done: custom_ids the run has already sent.
        template: The prompt template to render each request with.

    Yields:
        One item per request, carrying everything a record is parsed against.
    """
    for row in sides.itertuples():
        if row.custom_id in done:
            continue
        yield {
            'custom_id': str(row.custom_id),
            'paper_id': str(row.paper_id),
            'pmid': str(row.pmid),
            'name': str(row.name),
            'outcome': str(row.outcome),
            'candidates': [str(item) for item in row.candidates],
            'candidates_viral': [bool(item) for item in row.candidates_viral],
            'paragraph': str(row.paragraph),
            'paragraph_source': str(row.paragraph_source),
            'text': build_request_text(
                template, str(row.name), str(row.paragraph),
                str(row.candidates_text),
            ),
        }


def parse_pick(entry: dict, sent: dict) -> dict:
    """Turn one raw batch result into a record.

    The evidence is grounded against the paragraph alone, not the whole request.
    Grounding against the request would let a model quote the candidate list it
    was shown back as its evidence and pass, which is the answer the gate exists
    to catch: a pick resting on the list rather than on the paper.

    Args:
        entry: One deserialized line of a raw results jsonl.
        sent: The recorded request for this custom_id, empty if there is none.
    """
    candidates = list(sent.get('candidates') or [])
    record: dict = {
        'custom_id': entry.get('custom_id'),
        'paper_id': sent.get('paper_id'),
        'pmid': sent.get('pmid'),
        'name': sent.get('name'),
        'outcome': sent.get('outcome'),
        'n_candidates': len(candidates),
        'paragraph_source': sent.get('paragraph_source'),
        'result_type': None,
        'pick_outcome': PICK_NO_ANSWER,
        'choice': None,
        'accession': '',
        'is_viral': False,
        'evidence': '',
        'reason': '',
        'error': None,
    }
    parsed, _ = batch.unwrap_result(entry, record, PICK_RESULT_MODEL)
    if parsed is None:
        return record

    record['choice'] = parsed.choice
    record['reason'] = parsed.reason
    record['evidence'] = parsed.evidence
    if not candidates:
        record['error'] = 'no recorded request for this custom_id'
        return record
    if parsed.choice == 0:
        record['pick_outcome'] = PICK_ABSTAINED
        return record
    if not 1 <= parsed.choice <= len(candidates):
        # Not an abstention: the model meant to pick and named a candidate that
        # was never offered, so the answer is unusable rather than declined.
        record['error'] = (
            f'choice {parsed.choice} is outside the {len(candidates)} '
            f'candidates offered'
        )
        return record

    paragraph = str(sent.get('paragraph') or '')
    if not paragraph:
        record['error'] = 'no recorded paragraph; evidence not checked'
        return record
    if ungrounded_excerpts([parsed.evidence], paragraph):
        record['pick_outcome'] = PICK_UNGROUNDED
        return record

    # Checked after grounding, so an abstention or an ungrounded pick is still
    # classified as what it was. A pick cannot be recorded without this flag:
    # apply reads it to re-run the virus-virus drop, and a missing flag would
    # read as not viral and let the pair through - the permissive direction on a
    # filter that already missed 49 pairs once. So it is recorded as a failure,
    # which leaves the side needing a pick and puts it in retry's list.
    viral = list(sent.get('candidates_viral') or [])
    if len(viral) != len(candidates):
        record['error'] = (
            f'recorded request carries {len(viral)} viral flags for '
            f'{len(candidates)} candidates, so the virus-virus drop cannot be '
            f're-run over this pick'
        )
        return record

    record['pick_outcome'] = PICK_ANSWERED
    record['accession'] = candidates[parsed.choice - 1]
    record['is_viral'] = bool(viral[parsed.choice - 1])
    return record


def tally_pick(record: dict, counts: Counter) -> None:
    """Add one record to a run's outcome counters."""
    batch.tally_errors(record, counts)
    counts[f"pick {record['pick_outcome']}"] += 1
    if record['pick_outcome'] == PICK_ANSWERED:
        counts['picks'] += 1
        counts[f"from {record.get('outcome')}"] += 1
        if record.get('paragraph_source') == PARAGRAPH_FROM_EXCERPT:
            counts['picks made on the excerpt fallback'] += 1


def load_answers(run_dir: Path) -> pd.DataFrame:
    """Read a run's answers, one row per side.

    Args:
        run_dir: A run directory holding a records subdirectory.

    Returns:
        One row per side answered.
    """
    frame = pd.DataFrame(runs.latest_records(run_dir))
    logger.info('%s: %d answered sides', run_dir.name, len(frame))
    return frame


def apply_picks(frame: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """Fold the picks back into the pair table and restate each pair's status.

    A side the model answered becomes model_picked: it counts as resolved and
    carries model_picked provenance, so the weaker basis travels with the row.
    Everything else keeps the outcome it had, and its pair keeps needing a pick.

    Args:
        frame: The pair table.
        answers: One row per side answered, from load_answers.

    Returns:
        The table with its picked sides rewritten and its statuses restated.
    """
    picked = {
        (str(row.paper_id), str(row.name)): row
        for row in answers.itertuples()
    }
    frame = frame.copy()
    for suffix in ('a', 'b'):
        outcomes = list(frame[f'outcome_{suffix}'])
        accessions = list(frame[f'accession_{suffix}'])
        virals = list(frame[f'viral_{suffix}'])
        provenances = list(frame[f'provenance_{suffix}'])
        pick_outcomes = [PICK_NONE] * len(frame)
        for position, (outcome, paper_id, name) in enumerate(
            zip(outcomes, frame['paper_id'], frame[f'name_{suffix}']),
        ):
            if outcome not in PICK_OUTCOMES:
                continue
            answer = picked.get((str(paper_id), str(name)))
            if answer is None:
                pick_outcomes[position] = PICK_NO_ANSWER
                continue
            pick_outcomes[position] = answer.pick_outcome
            if answer.pick_outcome != PICK_ANSWERED:
                continue
            outcomes[position] = SIDE_MODEL_PICKED
            accessions[position] = answer.accession
            # Recomputed from the accession the pick landed on: a flag over a
            # candidate set says nothing once one has been chosen, and the
            # virus-virus drop below reads this.
            virals[position] = bool(answer.is_viral)
            provenances[position] = OUTCOME_PROVENANCE[SIDE_MODEL_PICKED]
        frame[f'outcome_{suffix}'] = outcomes
        frame[f'accession_{suffix}'] = accessions
        frame[f'viral_{suffix}'] = virals
        frame[f'provenance_{suffix}'] = provenances
        frame[f'pick_outcome_{suffix}'] = pick_outcomes
    assign_status(frame)
    return frame


def drop_virus_virus(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop pairs the picks have just made virus-virus, in place of nothing.

    The pair stage ran this filter against the viral flags a candidate *set*
    carried. A pick can turn a side viral that was not, so it runs again over the
    sides this stage narrowed - and only those, since a row whose flags did not
    move would reach the same answer.

    Args:
        frame: The pair table, after apply_picks.

    Returns:
        The table with newly virus-virus pairs marked in drop_reason.
    """
    touched = frame['pick_outcome_a'].eq(PICK_ANSWERED)
    touched |= frame['pick_outcome_b'].eq(PICK_ANSWERED)
    newly = touched & frame['viral_a'] & frame['viral_b']
    newly &= frame['drop_reason'].eq('')
    frame = frame.copy()
    frame.loc[newly, 'drop_reason'] = DROP_VIRUS_VIRUS
    logger.info(
        '%d pairs became virus-virus once a side was picked and are dropped',
        int(newly.sum()),
    )
    return frame


def models_answered(answers: pd.DataFrame) -> dict[str, int]:
    """Count the picks each model actually made.

    The configured primary model identifies the run, but it is not necessarily
    who answered: a retry runs under the fallback and its picks enter the table
    under the same model_picked provenance as the rest. Counted over answered
    picks only, since those are the sides that carry an accession.

    Args:
        answers: One row per side answered, from load_answers.

    Returns:
        Model id to how many picks it made.
    """
    if 'model' not in answers.columns:
        return {}
    picked = answers.loc[answers['pick_outcome'] == PICK_ANSWERED, 'model']
    return {
        str(model): int(count) for model, count in picked.value_counts().items()
    }


def source_provenance_path(inputs_path: str | Path) -> Path:
    """Where the input table's record of its own source pair table lives."""
    return Path(inputs_path).with_suffix('.source.json')


def write_source_provenance(
        inputs_path: str | Path,
        source: str,
        release: str,
) -> Path:
    """Record which pair table an input table was built from, beside it.

    Args:
        inputs_path: The input table just written.
        source: The pair table it was built from.
        release: That table's uniprot_release.

    Returns:
        The path written.
    """
    path = source_provenance_path(inputs_path)
    payload = {
        'source': str(source),
        'uniprot_release': release,
        'identity': runs.corpus_identity(source),
    }
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_source_identity(inputs_path: str | Path) -> dict | None:
    """The recorded identity of an input table's source, or None if unrecorded."""
    path = source_provenance_path(inputs_path)
    if not path.exists():
        return None
    with open(path) as handle:
        payload: dict = json.load(handle)
    return payload


def require_matching_source(
        run_dir: Path,
        source: str,
        allow_unverified: bool = False,
) -> bool:
    """Refuse a source pair table the run's answers were not built against.

    apply joins each answer onto the pair table on a key that is not unique
    across tables, so a stale answer would be applied to a row it was never made
    about. Recorded at submit, checked here.

    An unbound run is refused rather than warned about: a run predating the
    binding cannot prove what it was built from, so the operator has to say so
    with allow_unverified, or backfill the binding.

    Args:
        run_dir: The run's directory.
        source: The pair table apply was pointed at.
        allow_unverified: Proceed when the run recorded no binding at all. Does
            not weaken the check when one exists - a recorded mismatch is always
            refused.

    Returns:
        Whether the source was checked against a recorded binding. False means it
        was taken on the operator's word, which belongs in the provenance of
        anything published from it.

    Raises:
        RuntimeError: If the run recorded a different source table, or recorded
            none and allow_unverified is False.
    """
    recorded = runs.read_manifest(run_dir).get('pairs_source')
    if not recorded:
        if not allow_unverified:
            raise RuntimeError(
                f'{run_dir.name} recorded no source pair table, so --source '
                f'cannot be checked. Backfill the binding, or pass '
                f'--allow-unverified-source to proceed anyway.',
            )
        logger.warning(
            'This run recorded no source pair table and --allow-unverified-'
            'source was passed, so --source is not checked. Confirm %s is the '
            'table the inputs were built from.', source,
        )
        return False
    current = runs.corpus_identity(source)
    if recorded['identity'] != current:
        raise RuntimeError(
            f'This run was built from {recorded["source"]} '
            f'({recorded["identity"]}) and --source is {source} ({current}).',
        )
    return True


def _run_id(args: argparse.Namespace) -> str:
    """The run a subcommand addresses: named directly, or derived from the config."""
    return args.run_id or PickConfig(
        prompt_version=args.prompt_version, label=args.label,
    ).run_id


def _run_inputs(args: argparse.Namespace) -> None:
    """Build the table of sides to ask about."""
    frame = read_kept_pairs(args.source)
    frame = frame[
        frame['pair_status'] == STATUS_NEEDS_MODEL_PICK
    ].reset_index(drop=True)
    print(
        f'{len(frame):,} pairs needing a pick over '
        f'{frame.paper_id.nunique():,} papers',
    )
    index = get_name_index()
    sides = build_inputs(
        frame, args.blocks, index, str(frame['uniprot_release'].iloc[0]),
        max_candidates=args.max_candidates,
        organisms_path=Path(args.organisms),
        definitions_path=Path(args.definitions),
    )
    out_path = write_table(sides, args.out)

    print(f'\n{len(sides):,} requests')
    print('\npick type:')
    print_counts(
        sides['outcome'], width=34,
        total=len(sides), order=PICK_OUTCOMES,
    )
    print('\npaper text shown:')
    print_counts(
        sides['paragraph_source'], width=34, total=len(sides),
        order=(PARAGRAPH_FROM_BLOCK, PARAGRAPH_FROM_EXCERPT),
    )
    # Sized under the default prompt, since the template is applied at submit
    # time and a different version would shift every request by its own length.
    template = PickConfig().prompt_text
    chars = pd.Series([
        len(build_request_text(template, name, paragraph, candidates))
        for name, paragraph, candidates in zip(
            sides['name'], sides['paragraph'], sides['candidates_text'],
        )
    ])
    print(
        f'\nrequest text under {PICK_PROMPT_VERSION}: mean '
        f'{chars.mean():,.0f} chars, median {chars.median():,.0f}, max '
        f'{chars.max():,.0f}',
    )
    print(f'candidates per request: median {sides.n_candidates.median():.0f}, '
          f'max {sides.n_candidates.max()}')

    # The input table cannot carry which pair table it was built from - its
    # columns are the request payload and are frozen by any run already resuming
    # against them - so the identity goes in a file beside it. submit copies it
    # into the manifest and apply refuses a source that disagrees, which is what
    # stops a stale answer being joined onto a newer table's rows.
    provenance_path = write_source_provenance(
        out_path, args.source, str(frame['uniprot_release'].iloc[0]),
    )
    print(f'\nwrote {out_path}')
    print(f'wrote {provenance_path}')


def _run_submit(args: argparse.Namespace) -> None:
    """Submit the input table as a series of batches."""
    config = PickConfig(prompt_version=args.prompt_version, label=args.label)
    sides = pd.read_parquet(args.inputs)
    if sides.empty:
        raise ValueError(f'No requests in {args.inputs}.')
    run_dir = runs.make_run_dir(config.run_id)
    done = runs.sent_custom_ids(run_dir)
    if done:
        logger.info('%d requests already sent, resuming past them', len(done))
    # Both bindings go to submit rather than being written here, so they are
    # validated and persisted in the same manifest write. Recording pairs_source
    # first and letting submit reject the resume afterwards would leave the run
    # bound to a pair table its own answers never came from, and apply would
    # then accept that table for the old candidates.
    bindings = {'corpus': runs.corpus_identity(args.inputs)}
    recorded_source = read_source_identity(args.inputs)
    if recorded_source:
        bindings['pairs_source'] = recorded_source
    else:
        logger.warning(
            'No %s beside the input table, so apply cannot check its --source. '
            'Rebuild the inputs to record it.', source_provenance_path(
                args.inputs,
            ).name,
        )
    batch.submit(
        config,
        iter_items(sides, done, config.prompt_text),
        first_chunk_index=runs.resume_offset(run_dir)[0],
        bindings=bindings,
        max_requests=args.max_requests,
        max_bytes=args.max_bytes,
        max_batches=args.max_batches,
    )


def _run_collect(args: argparse.Namespace) -> None:
    """Archive and parse every ended batch."""
    batch.collect(
        _run_id(args),
        upload=not args.no_upload,
        stage=ACCESSIONS_STAGE,
        parse=parse_pick,
        count=tally_pick,
    )


def _run_retry(args: argparse.Namespace) -> None:
    """Re-send the sides that came back with no answer, under another model.

    Requests are rebuilt from the run's own recorded text, so a retry cannot
    drift from what was first sent. The fallback is Sonnet 4.6, not the screen's
    cascade down to Haiku 4.5: the screen was a reading task where any answer
    beat none, this is a pick whose precision is the point, and Haiku 4.5 was
    measured on it at 41.2% accuracy and 47.1% abstention on the paragraph arm. A
    Haiku retry warns rather than being blocked, since every record carries the
    model that answered it.
    """
    config = PickConfig(prompt_version=args.prompt_version, label=args.label)
    if 'haiku' in args.model:
        logger.warning(
            'Haiku 4.5 was measured on this task at 41.2%% accuracy and 47.1%% '
            'abstention on the paragraph arm. Its picks will be worse than the '
            '85-87%% this stage is published against, and they enter the table '
            'under the same model_picked provenance. Split them out by the '
            'model field on each record if you go ahead.',
        )
    batch.retry(
        config,
        _run_id(args),
        max_requests=args.max_requests,
        max_bytes=args.max_bytes,
        max_batches=args.max_batches,
        model=args.model,
    )


def _run_poll(args: argparse.Namespace) -> None:
    """Report each batch's processing status."""
    batch.poll(_run_id(args), wait=args.wait, interval=args.interval)


def _run_apply(args: argparse.Namespace) -> None:
    """Fold the picks into the pair table and write the dated output."""
    run_id = _run_id(args)
    run_dir = runs.require_run_dir(run_id)
    answers = load_answers(run_dir)
    source_verified = require_matching_source(
        run_dir, args.source, allow_unverified=args.allow_unverified_source,
    )
    write_table(answers, args.answers_out)
    frame = read_kept_pairs(args.source)
    before = frame['pair_status'].copy()
    frame = apply_picks(frame, answers)
    frame = drop_virus_virus(frame)
    kept = frame[frame['drop_reason'] == ''].reset_index(drop=True)

    stamp = args.date or date.today().isoformat()
    out_path = write_table(
        kept,
        str(
            Path(args.out_dir) / make_dated_filename(
                ACCESSIONS_NAME, LITERATURE_STAGE_VERSIONS[ACCESSIONS_STAGE],
                '.parquet', stamp,
            ),
        ),
    )
    provenance_path = out_path.with_suffix('.provenance.json')
    write_output_provenance(
        provenance_path,
        build_output_provenance(
            workflow='flock.negatome_v3.literature.accessions.apply',
            parameters={
                'source': args.source,
                'allow_unverified_source': bool(args.allow_unverified_source),
                'run_id': run_id, 'date': stamp,
            },
            input_paths={'pairs': args.source},
            extra={
                'uniprot_release': frame['uniprot_release'].iloc[0],
                'pick_run_id': run_id,
                # False means the source table was taken on the operator's word
                # rather than checked against what the run was built from, which
                # anything reading this table downstream has to be able to see.
                'source_verified': source_verified,
                'model': PICK_MODEL,
                'models_answered': models_answered(answers),
                'n_sides_answered': int(len(answers)),
                'n_picks': int(
                    (answers['pick_outcome'] == PICK_ANSWERED).sum(),
                ),
                'n_kept': len(kept),
                'n_kept_by_status': {
                    status: int((kept['pair_status'] == status).sum())
                    for status in STATUS_ORDER
                },
            },
            source_name='flock',
            repo_root=REPO_ROOT,
        ),
    )

    print('\npick outcome, over the sides that needed one:')
    print_counts(
        answers['pick_outcome'], width=34, total=len(answers),
        labels=PICK_LABELS, order=PICK_ORDER,
    )
    print('\npair status before:')
    print_counts(before, width=34, total=len(before), order=STATUS_ORDER)
    print('\npair status after:')
    print_counts(
        kept['pair_status'], width=34, total=len(kept), order=STATUS_ORDER,
    )
    print(f'\nwrote {out_path} ({len(kept):,} kept pairs)')
    print(f'wrote {provenance_path}')
    if args.upload:
        prefix = get_literature_stage_prefix(ACCESSIONS_STAGE)
        upload_file_to_s3(str(out_path), prefix)
        upload_file_to_s3(str(provenance_path), prefix)
        print(f'uploaded both to {prefix}')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Ask a model which UniProt entry each ambiguous protein name '
            'means, and fold the answers into the pair table.'
        ),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_run_id_arg(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            '--run-id', default='',
            help=(
                'Address an existing run directly, for when an edit to the '
                'prompt or the schema has moved the run_id.'
            ),
        )

    def add_config_args(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            '--prompt-version', default=PICK_PROMPT_VERSION,
            help='Prompt stem.',
        )
        target.add_argument(
            '--label', default='',
            help='Extra slug in the run_id, e.g. pilot.',
        )

    inputs = subparsers.add_parser(
        'inputs', help='Build the table of sides to ask about.',
    )
    inputs.set_defaults(run=_run_inputs)
    inputs.add_argument(
        '--source', default=str(DEFAULT_PAIRS_TABLE),
        help='The pair table to take the pairs needing a pick from.',
    )
    inputs.add_argument(
        '--blocks', default=str(CORPUS_BLOCKS_PATH),
        help='Local corpus blocks parquet, for the containing paragraphs.',
    )
    inputs.add_argument(
        '--organisms', default=str(PAPER_ORGANISMS_PATH),
        help='Per-paper organism cache, written by the organisms module.',
    )
    inputs.add_argument(
        '--definitions', default=str(PAPER_DEFINITIONS_PATH),
        help='Per-paper abbreviation cache, written by the organisms module.',
    )
    inputs.add_argument(
        '--max-candidates', type=int, default=DEFAULT_MAX_CANDIDATES,
        help='Most candidates a side can carry and still be asked about.',
    )
    inputs.add_argument(
        '--out', default=str(PICK_INPUTS_PATH),
        help='Where to write the input table.',
    )

    submit = subparsers.add_parser(
        'submit', help='Submit the input table to the Batch API.',
    )
    submit.set_defaults(run=_run_submit)
    add_config_args(submit)
    batch.add_batch_args(submit)
    submit.add_argument(
        '--inputs', default=str(PICK_INPUTS_PATH),
        help='The input table to submit.',
    )

    retry = subparsers.add_parser(
        'retry',
        help='Re-send the sides that came back with no answer, e.g. refusals.',
    )
    retry.set_defaults(run=_run_retry)
    add_config_args(retry)
    add_run_id_arg(retry)
    batch.add_batch_args(retry)
    retry.add_argument(
        '--model', default=FALLBACK_MODEL,
        help=(
            'Model to answer the failed requests. Defaults to the peer-class '
            'fallback; Haiku 4.5 was measured unfit for this task below full '
            'text and warns rather than refusing.'
        ),
    )

    poll = subparsers.add_parser(
        'poll', help='Report each batch\'s processing status.',
    )
    poll.set_defaults(run=_run_poll)
    add_config_args(poll)
    add_run_id_arg(poll)
    poll.add_argument(
        '--wait', action='store_true',
        help='Keep polling until every batch has ended.',
    )
    poll.add_argument(
        '--interval', type=float, default=300.0,
        help='Seconds between polls when waiting.',
    )

    collect = subparsers.add_parser(
        'collect', help='Archive and parse every ended batch.',
    )
    collect.set_defaults(run=_run_collect)
    add_config_args(collect)
    add_run_id_arg(collect)
    collect.add_argument(
        '--no-upload', action='store_true',
        help='Skip the S3 archive of raw results.',
    )

    apply_picks_parser = subparsers.add_parser(
        'apply', help='Fold the picks into the pair table and write it.',
    )
    apply_picks_parser.set_defaults(run=_run_apply)
    add_config_args(apply_picks_parser)
    add_run_id_arg(apply_picks_parser)
    apply_picks_parser.add_argument(
        '--source', default=str(DEFAULT_PAIRS_TABLE),
        help=(
            'The pair table to fold the picks into. Must be the one the inputs '
            'were built from.'
        ),
    )
    apply_picks_parser.add_argument(
        '--allow-unverified-source', action='store_true',
        help=(
            'Apply even though the run recorded no source pair table. Only for '
            'a run predating that record, and only once you have confirmed '
            '--source is what its inputs were built from.'
        ),
    )
    apply_picks_parser.add_argument(
        '--answers-out', default=str(PICK_ANSWERS_PATH),
        help='Where to write one row per answered side.',
    )
    apply_picks_parser.add_argument(
        '--out-dir', default=str(PAIRS_ROOT),
        help='Directory for the dated table and its provenance file.',
    )
    apply_picks_parser.add_argument(
        '--date', help='Date stamp for the output filename (YYYY-MM-DD).',
    )
    apply_picks_parser.add_argument(
        '--upload', action='store_true',
        help=(
            'Publish the table and its provenance to the accessions stage '
            'prefix in S3. Off by default: this writes to the shared bucket.'
        ),
    )
    return parser


def main() -> None:
    """Settle each ambiguous protein name with one cheap model pick.

    The pairs needing a pick are those whose sides resolve to several UniProt
    entries: an orthologue question, where the candidates are one protein across
    organisms, or an identity question, where they are different proteins. Both
    are settled by what the paper itself says, so each side is sent once with the
    paragraph its excerpt was copied from and a numbered list of its candidates,
    and the model returns the entry the paper means or declines.

    Six subcommands. 'inputs' reduces those pairs to one row per (paper, name)
    side, re-derives each side's candidates by re-running the resolver, and
    locates the containing paragraph in one pass over the block table. 'submit',
    'poll', 'collect' and 'retry' drive the Batch API through the screen's own
    runner. 'apply' folds the answers back in: an answered side becomes
    model_picked and counts as resolved, while an abstention, an ungrounded pick
    or a missing answer leaves its pair still needing one.

    Precision over the picks made was measured at 85-87%, against 100% by
    construction for the deterministic sides, so the provenance column is what a
    consumer controls that difference with and curation downstream has to verify
    a model-picked accession rather than trust it. The pick is deliberately not
    replicated and asks for no confidence: both were measured and rejected, and
    errors repeat rather than varying.

    Reads FLOCK_ANTHROPIC_API_KEY through the shared runner and fails hard if it
    is unset. This script is executed by the user, not by an agent.
    """
    setup_logging()
    parser = parse_args()
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
