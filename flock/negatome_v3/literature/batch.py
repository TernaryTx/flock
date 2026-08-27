# Submit, poll and collect one screening run against the Anthropic Batch API.
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Callable
from collections.abc import Iterator
from datetime import datetime
from datetime import timezone
from typing import TypeVar

import anthropic
import pandas as pd
from pydantic import BaseModel
from pydantic import ValidationError

from flock.aws import upload_file_to_s3
from flock.logging_utils import setup_logging
from flock.negatome_v3.literature import packaging
from flock.negatome_v3.literature import runs
from flock.negatome_v3.literature.config import DEFAULT_ABSTRACT_LEVEL
from flock.negatome_v3.literature.config import DEFAULT_INPUT_LEVEL
from flock.negatome_v3.literature.config import DEFAULT_PROMPT_VERSION
from flock.negatome_v3.literature.config import MODEL
from flock.negatome_v3.literature.config import RequestConfig
from flock.negatome_v3.literature.config import RESULT_MODEL
from flock.negatome_v3.literature.config import STAGE
from flock.negatome_v3.literature.config import StageConfig
from flock.negatome_v3.literature.grounding import ungrounded_excerpts
from flock.paths import get_literature_run_raw_prefix

logger = logging.getLogger(__name__)

# The answer type a stage validates against, so unwrap_result hands each
# caller back its own model rather than something it has to re-narrow.
ResultT = TypeVar('ResultT', bound=BaseModel)

# Runs authenticate with this variable and nothing else. The agent that wrote this
# file must never read it, and the SDK must never find a credential on its own.
API_KEY_VAR = 'FLOCK_ANTHROPIC_API_KEY'

# Requests is the documented ceiling; the byte budget sits below the documented
# 256 MB so JSON framing cannot push a chunk over. Size is what binds in practice:
# at cue_window packaging a 261k-paper corpus is 2-3 GB of text, a dozen-odd batches
# by size and well inside one batch by count.
MAX_REQUESTS_PER_BATCH = 100_000
MAX_BATCH_BYTES = 200 * 1024 * 1024

# Fallback models that reject output_config.effort outright (400
# invalid_request_error: "This model does not support the effort parameter."),
# confirmed against claude-haiku-4-5 in the TERN-2720 retry. PARAMS carries
# effort for the primary model, so build_request must drop it for these.
MODELS_WITHOUT_EFFORT = frozenset({'claude-haiku-4-5', 'claude-sonnet-4-5'})


def get_client() -> anthropic.Anthropic:
    """Build a client bound explicitly to FLOCK_ANTHROPIC_API_KEY.

    Never construct a bare Anthropic(): the SDK would fall back to
    ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile, and the
    run would bill an account nobody chose. Failing hard is the point.

    Raises:
        RuntimeError: If the variable is unset or empty.
    """
    api_key = os.environ.get(API_KEY_VAR)
    if not api_key:
        raise RuntimeError(
            f'{API_KEY_VAR} is not set. Export it in the shell you run this from. '
            f'Do not fall back to ANTHROPIC_API_KEY or an ant auth profile.',
        )
    return anthropic.Anthropic(api_key=api_key)


def build_request(
        config: RequestConfig,
        custom_id: str,
        paper_text: str,
        model: str = '',
) -> dict:
    """Build one Batch request.

    A plain dict rather than the SDK's Request TypedDict, which is what it is at
    runtime anyway; building it here keeps the exact bytes that get serialized
    visible to the size accounting in chunk_requests.

    Args:
        config: The run configuration.
        custom_id: The only join key back to the paper; results arrive unordered.
        paper_text: The exact text to send.
        model: Model to send this request to, defaulting to the config's own. A
            plain call-scoped parameter, like retry below, rather than a
            StageConfig field: it changes who answers a request, not which run the
            request belongs to, so it must never affect config.run_id.
    """
    model = model or config.model
    params = dict(config.params)
    output_config = {
        'format': {
            'type': 'json_schema',
            'schema': config.output_json_schema,
        },
    }
    if model not in MODELS_WITHOUT_EFFORT:
        output_config.update(config.params['output_config'])
    params['output_config'] = output_config
    params.update({
        'model': model,
        'max_tokens': config.max_tokens,
        'messages': [{'role': 'user', 'content': paper_text}],
    })
    # Omitted rather than sent empty when a stage puts its instructions in
    # the user message instead, which is how the accession pick was measured.
    if config.system_prompt:
        params['system'] = config.system_prompt
    return {'custom_id': custom_id, 'params': params}


def iter_items(
        config: StageConfig,
        papers: pd.DataFrame,
        blocks_path: str,
        skip_items: int = 0,
) -> Iterator[dict]:
    """Stream the papers the screen covers, packaged and ready to send.

    One run covers the whole corpus. Each paper is packaged by what text the corpus
    stage got for it - the full-text level where the full text is usable, the abstract
    level otherwise - so a page scan, a 404 or a failed parse is screened on its
    abstract rather than dropped, at about half the tokens of a full-text paper. The
    corpus stage guarantees every paper at least one block, which is what makes the
    block table a complete enumeration of what this run covers.

    Papers whose packaging selects no text are yielded with an empty 'text' rather
    than dropped, so they stay counted without spending a request. The two reasons
    that happens are recorded separately, since the share of papers with no
    negation cue anywhere is what the cue-filtered packaging is judged on.

    Args:
        config: The run configuration, carrying both packagings.
        papers: The corpus papers table, carrying paper_id, text_source, pmid, pmcid
            and usable_fulltext.
        blocks_path: Local parquet of parsed blocks from the corpus stage.
        skip_items: Advance past this many papers without packaging them, for
            resuming a partly submitted run. Packaging is the expensive step, so
            skipping before it is what keeps a resume cheap.

    Yields:
        Dicts with custom_id, paper_id, text_source, input_level, pmid, pmcid, text
        and skipped, in block-table order.
    """
    usable_fulltext = papers.usable_fulltext.fillna(False).astype(bool)
    covered = dict(
        zip(
            papers.paper_id.astype(str),
            zip(
                papers.pmid.fillna('').astype(str),
                papers.pmcid.fillna('').astype(str),
                papers.text_source.fillna('').astype(str),
                usable_fulltext,
            ),
        ),
    )
    logger.info(
        '%d of %d papers have no usable full text and are packaged at %s',
        int((~usable_fulltext).sum()), len(papers), config.abstract_level,
    )
    # The Batch API refuses a batch that repeats a custom_id, and the id here is the
    # pmid with the paper_id as fallback. Two papers carrying one PMID - the same
    # article deposited twice - collide there, so both fall back to the paper_id,
    # which cannot repeat because the block table is one group per paper_id. Only the
    # repeats are kept: the counts of the other ~260k pmids would be held for the
    # whole submit, since this is a generator.
    pmids = papers.pmid.dropna().astype(str)
    ambiguous = set(pmids[pmids.duplicated(keep=False) & (pmids != '')])
    position = 0
    for paper_id, paper in packaging.iter_papers(blocks_path):
        # A block table can hold papers this run's corpus table does not, e.g. a
        # scoped rebuild. Filtering here keeps skip_items counting only the papers
        # this run would have packaged.
        if paper_id not in covered:
            continue
        position += 1
        if position <= skip_items:
            continue
        pmid, pmcid, text_source, usable_fulltext = covered[paper_id]
        level = config.level_for(usable_fulltext)
        skipped = None
        try:
            text = packaging.build_paper_input(paper, level, paper_id=paper_id)
        except ValueError:
            # No text at all at this level, before any cue filtering: a parse that
            # produced nothing but excluded sections. A data problem rather than a
            # paper with nothing to say.
            text = ''
            skipped = f'no {level.name} text parsed'
        if not text:
            skipped = skipped or f'no {level.name} block selected'
        yield {
            'custom_id': paper_id if pmid in ambiguous else (pmid or paper_id),
            'paper_id': paper_id,
            'text_source': text_source,
            'input_level': level.name,
            'pmid': pmid,
            'pmcid': pmcid,
            'text': text,
            'skipped': skipped,
        }


def chunk_requests(
        config: RequestConfig,
        items: Iterator[dict],
        max_requests: int = MAX_REQUESTS_PER_BATCH,
        max_bytes: int = MAX_BATCH_BYTES,
        model: str = '',
) -> Iterator[tuple[list[dict], list[dict], int]]:
    """Group packaged items into batches respecting both Batch API limits.

    Chunking is deterministic given the same items, which is what lets a resumed
    submit pick up at the next chunk index knowing the boundaries fall in the same
    places. Items are consumed lazily, so only one chunk's text is ever in memory.

    Args:
        config: The run configuration.
        items: Packaged items to chunk, as iter_items yields them.
        max_requests: Requests per batch.
        max_bytes: Serialized bytes per batch.
        model: Model every request in this call builds against; see build_request.

    Yields:
        (items, requests, serialized bytes) per chunk. Skipped items carry no
        request and are yielded with their chunk for recording only. The byte total
        is the one the chunking decision was made on.

    Raises:
        ValueError: If two items in one chunk share a custom_id.
    """
    chunk_items: list[dict] = []
    chunk_out: list[dict] = []
    chunk_ids: set[str] = set()
    chunk_bytes = 0

    for item in items:
        if not item['text']:
            chunk_items.append(item)
            continue
        custom_id = item['custom_id']
        request = build_request(config, custom_id, item['text'], model)
        request_bytes = len(json.dumps(request))
        if request_bytes > max_bytes:
            logger.warning(
                '%s alone serializes to %.1f MB, over the %.0f MB budget; '
                'submitting it on its own',
                custom_id, request_bytes / 1024 ** 2, max_bytes / 1024 ** 2,
            )
        over_count = len(chunk_out) >= max_requests
        over_bytes = bool(chunk_out) and chunk_bytes + \
            request_bytes > max_bytes
        if over_count or over_bytes:
            yield chunk_items, chunk_out, chunk_bytes
            chunk_items, chunk_out, chunk_bytes = [], [], 0
            chunk_ids = set()
        # The API refuses the whole submission on a repeated custom_id, so a
        # collision has to fail here rather than after the create call is billed.
        if custom_id in chunk_ids:
            raise ValueError(
                f'Duplicate custom_id {custom_id!r} within one chunk. The Batch '
                f'API rejects a batch that repeats a custom_id.',
            )
        chunk_ids.add(custom_id)
        chunk_items.append(item)
        chunk_out.append(request)
        chunk_bytes += request_bytes

    if chunk_items:
        yield chunk_items, chunk_out, chunk_bytes


def submit(
        config: RequestConfig,
        items: Iterator[dict],
        first_chunk_index: int = 0,
        bindings: dict[str, dict] | None = None,
        retry: bool = False,
        max_requests: int = MAX_REQUESTS_PER_BATCH,
        max_bytes: int = MAX_BATCH_BYTES,
        max_batches: int | None = None,
        model: str = '',
) -> None:
    """Chunk a run's items and submit them as a series of batches.

    Safe to re-invoke, provided the items are already positioned past the recorded
    chunks - which is what runs.resume_offset is for. Both halves of that offset are
    passed in, because reading it here as well would mean two reads of one manifest
    that can raise on an interrupted run.

    Args:
        config: The run configuration.
        items: The work units, already advanced past any recorded chunks.
        first_chunk_index: Chunk index to number the first new chunk from.
        bindings: What this run's requests were built from, keyed by manifest
            key. Each is recorded on the first submit and refused if a later one
            passes something different, in the same manifest write, so a
            rejected resume cannot leave a new binding behind. None for a retry,
            which re-sends recorded text and reads no input table at all.
        retry: Whether these chunks re-send papers already counted, which keeps
            them out of the item offset a later resume advances by.
        max_requests: Requests per batch.
        max_bytes: Serialized bytes per batch.
        max_batches: Stop after this many new batches, for the pilot submission
            that checks the pass rate before committing the rest of the corpus.
        model: Model every request in this call builds against, defaulting to the
            config's own. A parameter, not part of config, so a retry can answer
            under a different model without moving config.run_id - see
            build_request.

    Raises:
        RuntimeError: If the run is already bound to a different input table.
        ValueError: If model differs from the config's primary model while retry
            is False, or if model is not a valid Anthropic model id.
    """
    model = model or config.model
    if model != config.model and not retry:
        raise ValueError(
            f'submit() was called with model={model!r} and retry=False. Only a '
            f'retry may answer under a different model than the primary '
            f'{config.model!r} - config.json always records the primary model '
            f'regardless of what any individual chunk actually used, so a '
            f'non-retry chunk sent to a different model would be undocumented '
            f'there.',
        )

    run_dir = runs.make_run_dir(config.run_id)
    runs.write_run_config(run_dir, config)
    manifest = runs.read_manifest(run_dir)
    manifest['run_id'] = config.run_id
    # Bound before a single request is sent, so a resume against a different
    # input table is refused rather than half-answered.
    runs.bind_manifest(manifest, bindings or {})

    client = get_client()
    try:
        client.models.retrieve(model)
    except anthropic.NotFoundError as error:
        raise ValueError(
            f'{model!r} is not a valid Anthropic model id ({error}). Fix the '
            f'--model value before submitting the rest of this batch.',
        ) from error
    new_batches = 0

    for chunk_index, (chunk, requests, chunk_bytes) in enumerate(
        chunk_requests(config, items, max_requests, max_bytes, model),
        start=first_chunk_index,
    ):
        runs.write_requests(run_dir, chunk_index, chunk)
        # Every chunk is recorded, including one that selected no text, so the item
        # counts stay complete enough for resume_offset to rely on. The intent is
        # recorded before the create call, so an interruption between the two leaves
        # a pending entry that stops the next submit instead of letting it pay again.
        entry = {
            'chunk_index': chunk_index,
            'batch_id': None,
            'n_requests': len(requests),
            'n_skipped': len(chunk) - len(requests),
            'bytes': chunk_bytes,
            'retry': retry,
            'model': model,
            'pending_at': datetime.now(timezone.utc).isoformat(),
            'submitted_at': None,
            'collected_at': None,
        }
        manifest['batches'].append(entry)
        runs.write_manifest(run_dir, manifest)
        if not requests:
            logger.info(
                'Chunk %d selected no text at all, nothing to submit', chunk_index,
            )
            continue

        batch = client.messages.batches.create(requests=requests)
        entry['batch_id'] = batch.id
        entry['submitted_at'] = datetime.now(timezone.utc).isoformat()
        runs.write_manifest(run_dir, manifest)
        logger.info(
            'Chunk %d: submitted %d requests as %s (%d skipped)',
            chunk_index, len(requests), batch.id, len(chunk) - len(requests),
        )
        new_batches += 1
        # Tested here rather than at the top of the loop: the generator has already
        # packaged the whole next chunk by the time a top-of-loop test could run, so
        # breaking there doubled the cost of the --max-batches 1 pilot to buy
        # nothing. One item of lookahead is left and is inherent.
        if max_batches is not None and new_batches >= max_batches:
            logger.info('Reached max_batches=%d, stopping', max_batches)
            break


def poll(run_id: str, wait: bool = False, interval: float = 300.0) -> dict:
    """Report each batch's processing status, optionally until all have ended.

    Returns:
        Mapping of batch_id to processing_status.
    """
    manifest = runs.read_manifest(runs.require_run_dir(run_id))
    client = get_client()

    while True:
        statuses = {}
        for entry in runs.submitted_batches(manifest):
            batch = client.messages.batches.retrieve(entry['batch_id'])
            statuses[entry['batch_id']] = batch.processing_status
            counts = batch.request_counts
            logger.info(
                'chunk %d %s %s: %d succeeded, %d errored, %d canceled, '
                '%d expired, %d processing',
                entry['chunk_index'], entry['batch_id'], batch.processing_status,
                counts.succeeded, counts.errored, counts.canceled,
                counts.expired, counts.processing,
            )
        if not wait or all(status == 'ended' for status in statuses.values()):
            return statuses
        time.sleep(interval)


def unwrap_result(
        entry: dict,
        record: dict,
        result_model: type[ResultT],
) -> tuple[ResultT | None, str]:
    """Validate one raw batch result against a schema, or record why it failed.

    The half of parsing that is the same for any stage: what the Batch API can
    return instead of an answer, and how each is recorded. Every outcome is
    written onto the record rather than raised, so one refusal or schema failure
    cannot cost a run. The API rejects the 'fallbacks' parameter, so there is no
    server-side refusal recovery to lean on; this is the whole of it.

    Args:
        entry: One deserialized line of a raw results jsonl.
        record: The record being built, mutated in place with result_type,
            stop_reason, usage, error and raw_text as each applies.
        result_model: The Pydantic model the answer must validate against.

    Returns:
        The validated answer and the text it was validated from, or (None, '')
        if the API returned no answer, in which case the record carries the
        reason. The text is returned rather than written onto the record, which
        would give a successful screen record a second copy of every pair.
    """
    result = entry.get('result') or {}
    record['result_type'] = result.get('type')

    if result.get('type') != 'succeeded':
        # errored / canceled / expired all mean no answer, and each needs a
        # different response from the operator: retry, resubmit, or resubmit sooner.
        record['error'] = f"{result.get('type')}: {json.dumps(result.get('error'))}"
        return None, ''

    message = result.get('message') or {}
    record['stop_reason'] = message.get('stop_reason')
    record['usage'] = message.get('usage')

    # Sonnet 5 carries elevated cyber safeguards and can decline on an otherwise
    # successful response, returning empty or partial content.
    if message.get('stop_reason') == 'refusal':
        details = message.get('stop_details') or {}
        record['error'] = f"refusal: {details.get('category')}"
        return None, ''
    if message.get('stop_reason') == 'max_tokens':
        record['error'] = 'max_tokens: output truncated, schema will not parse'
        return None, ''

    text = ''.join(
        block.get('text', '') for block in message.get('content') or []
        if block.get('type') == 'text'
    )
    try:
        return result_model.model_validate_json(text), text
    except ValidationError as error:
        record['error'] = f'ValidationError: {error.error_count()} error(s)'
        record['raw_text'] = text
        return None, text


def parse_result(entry: dict, sent: dict) -> dict:
    """Turn one raw batch result into a record.

    Every outcome is recorded rather than raised, so one refusal or schema failure
    cannot cost a run, and a record always carries enough to tell why it holds no
    pairs. The Batch API rejects the 'fallbacks' parameter, so there is no
    server-side refusal recovery to lean on; this is the whole of it.

    paper_id, pmid and pmcid are copied from the request rather than left to be
    re-derived: a record carrying only its custom_id cannot be joined to the corpus
    tables without re-reading the requests files and reconstructing which identifier
    the id was built from. text_source and input_level ride along too, so every number
    a run reports can be split by how the paper was retrieved and how it was packaged
    without a second join.

    Args:
        entry: One deserialized line of a raw results jsonl.
        sent: The recorded request for this custom_id, empty if there is none.
    """
    source_text = sent.get('text', '')
    record: dict = {
        'custom_id': entry.get('custom_id'),
        'paper_id': sent.get('paper_id'),
        'text_source': sent.get('text_source'),
        'input_level': sent.get('input_level'),
        'pmid': sent.get('pmid'),
        'pmcid': sent.get('pmcid'),
        'result_type': None,
        'pairs': [],
        'error': None,
    }
    parsed, raw_text = unwrap_result(entry, record, RESULT_MODEL)
    if parsed is None:
        return record

    if not source_text:
        # A result whose custom_id is not in the chunk's requests file. The gate
        # cannot run without the text that was sent, and pairs that skipped the gate
        # must not be recorded as if they had passed it.
        record['error'] = 'no sent text for this custom_id; grounding not checked'
        record['raw_text'] = raw_text
        return record

    record['pairs'] = [
        {
            **pair.model_dump(),
            'ungrounded_excerpts': ungrounded_excerpts(pair.excerpts, source_text),
        }
        for pair in parsed.pairs
    ]
    return record


def tally_errors(record: dict, counts: Counter) -> None:
    """Count one record's envelope: that it arrived, and how it failed if it did.

    The counting counterpart of unwrap_result. The error key keeps only the
    prefix before the first colon - refusal, ValidationError, expired - without
    the per-record detail that would give every record its own counter.
    """
    counts['records'] += 1
    if record.get('error'):
        counts['failed'] += 1
        counts[f"error {record['error'].split(':')[0]}"] += 1


def tally(record: dict, counts: Counter) -> None:
    """Add one record to a run's outcome counters.

    The taxonomy is what the operator acts on and each branch needs a different
    response: a refusal is resubmittable, a validation failure is a prompt or schema
    problem, an expiry means submitting sooner, an ungrounded excerpt means the model
    is not copying from the text it was sent. Counting them here is what makes the
    --max-batches 1 pilot readable, since every one of these is otherwise computed
    per record and discarded.
    """
    tally_errors(record, counts)
    if record.get('stop_reason') == 'max_tokens':
        counts['truncated'] += 1
    if not record['pairs']:
        counts['papers with no pairs'] += 1
    for pair in record['pairs']:
        counts['pairs'] += 1
        counts[f"call {pair['relationship']}"] += 1
        counts[f"confidence {pair['confidence']}"] += 1
        counts['excerpts'] += len(pair['excerpts'])
        counts['ungrounded excerpts'] += len(pair['ungrounded_excerpts'])
        if pair['ungrounded_excerpts']:
            counts['pairs with an ungrounded excerpt'] += 1


def log_tally(label: str, counts: Counter) -> None:
    """Log one set of outcome counters, with the grounding pass rate derived.

    Only 'records' is named here, every stage having one. A stage's own counter
    - the screen's pairs, the pick's picks - is left to the sorted dump below,
    since naming one in the header printed a hardcoded zero for the other stage.
    """
    logger.info('%s: %d records', label, counts['records'])
    for key in sorted(counts):
        if key != 'records':
            logger.info('  %-34s %d', key, counts[key])
    if counts['excerpts']:
        grounded = counts['excerpts'] - counts['ungrounded excerpts']
        logger.info(
            '  %-34s %.1f%% (%d/%d)', 'excerpt grounding pass rate',
            100 * grounded / counts['excerpts'], grounded, counts['excerpts'],
        )


def collect(
        run_id: str,
        upload: bool = True,
        stage: str = STAGE,
        parse: Callable[[dict, dict], dict] = parse_result,
        count: Callable[[dict, Counter], None] = tally,
) -> None:
    """Download, archive and parse the results of every ended batch.

    Raw jsonl is written and pushed to S3 before anything is parsed out of it,
    because the API deletes results 29 days after a batch is created - not after it
    ends - and a parse can be redone from the archive while a fetch cannot.

    Everything above the answer itself is the same for any stage - which batches
    have ended, archiving before parsing, superseding on retry - so the three
    things that are not are parameters, defaulting to the screen's.

    Args:
        run_id: The run to collect.
        upload: Whether to archive each batch's raw jsonl to S3.
        stage: Pipeline stage, which is what picks the S3 archive prefix.
        parse: Turns one raw result and its recorded request into a record.
        count: Adds one record to the run's outcome counters.
    """
    run_dir = runs.require_run_dir(run_id)
    manifest = runs.read_manifest(run_dir)
    client = get_client()
    raw_s3 = get_literature_run_raw_prefix(stage, run_id)
    run_counts: Counter = Counter()

    for entry in runs.submitted_batches(manifest):
        if entry.get('collected_at'):
            continue
        batch = client.messages.batches.retrieve(entry['batch_id'])
        if batch.processing_status != 'ended':
            logger.info(
                'chunk %d %s is %s, not collecting yet',
                entry['chunk_index'], entry['batch_id'], batch.processing_status,
            )
            continue

        raw_path = runs.raw_path(run_dir, entry['batch_id'])
        with open(raw_path, 'w') as handle:
            for result in client.messages.batches.results(entry['batch_id']):
                handle.write(result.model_dump_json() + '\n')
        logger.info('chunk %d: wrote %s', entry['chunk_index'], raw_path)
        if upload:
            upload_file_to_s3(str(raw_path), raw_s3)
            logger.info(
                'chunk %d: archived to %s',
                entry['chunk_index'], raw_s3,
            )

        sent = runs.read_requests(run_dir, entry['chunk_index'])
        counts: Counter = Counter()
        records_path = runs.records_path(run_dir, entry['chunk_index'])
        # Falls back to the primary constant for manifests written before this
        # field existed, so already-collected runs stay readable. One lookup per
        # chunk, not per record: entry is the same for every record below.
        chunk_model = entry.get('model', MODEL)
        with open(raw_path) as raw, open(records_path, 'w') as out:
            for line in raw:
                result_entry = json.loads(line)
                custom_id = result_entry.get('custom_id')
                record = parse(result_entry, sent.get(custom_id, {}))
                record['batch_id'] = entry['batch_id']
                record['model'] = chunk_model
                out.write(json.dumps(record) + '\n')
                count(record, counts)
        log_tally(f"chunk {entry['chunk_index']}", counts)
        run_counts.update(counts)

        entry['collected_at'] = datetime.now(timezone.utc).isoformat()
        runs.write_manifest(run_dir, manifest)

    if run_counts['records']:
        log_tally('run total', run_counts)
        if run_counts['failed']:
            logger.warning(
                '%d records carry no answer. Re-send them with the retry mode, '
                'which rebuilds their requests from the recorded text.',
                run_counts['failed'],
            )


def retry(
        config: RequestConfig,
        run_id: str,
        max_requests: int = MAX_REQUESTS_PER_BATCH,
        max_bytes: int = MAX_BATCH_BYTES,
        max_batches: int | None = None,
        model: str = '',
) -> None:
    """Re-submit the requests of a run whose records carry no answer.

    Errored, expired, refused and schema-failed requests are otherwise lost for
    good: collect stamps the whole chunk collected and submit only resumes past
    everything recorded, so nothing ever revisits them.

    The text is taken from the run's own requests files rather than re-packaged from
    the corpus, so a retry cannot drift from what the first attempt sent even if the
    block table has since been rebuilt, and needs no --blocks at all.

    Args:
        config: The run configuration. Its run_id must match the run being retried.
        run_id: The run to retry.
        max_requests: Requests per batch.
        max_bytes: Serialized bytes per batch.
        max_batches: Stop after this many new batches.
        model: Model to send this retry's requests to, e.g. a fallback for a model
            that refused them the first time. Does not affect config.run_id, so
            retrying the same run under several models in turn - collecting between
            each - is just calling this again: failed_custom_ids only returns
            whatever the previous model still could not answer.

    Raises:
        RuntimeError: If the run was submitted under a different configuration than
            the one now resolved, which would re-send those papers under a prompt or
            schema the rest of the run never saw.
    """
    run_dir = runs.require_run_dir(run_id)
    recorded = runs.read_run_config(run_dir).get('run_id')
    if recorded and recorded != config.run_id:
        raise RuntimeError(
            f'Run {run_id} was submitted as {recorded} and the configuration now '
            f'resolves to {config.run_id}. Retrying would send these papers under a '
            f'different prompt, packaging or schema than the rest of the run. Check '
            f'out the code that produced {recorded} to retry it.',
        )

    failed = set(runs.failed_custom_ids(run_dir))
    if not failed:
        logger.info('No failed records in %s, nothing to retry', run_id)
        return
    # Later chunks win, so a custom_id already re-sent once resolves to its most
    # recent recorded text rather than the first attempt's.
    latest: dict[str, dict] = {}
    for row in runs.iter_requests(run_dir, failed):
        latest[row['custom_id']] = row
    missing = failed - set(latest)
    if missing:
        logger.warning(
            '%d failed records have no recorded request text and cannot be '
            'retried: %s', len(missing), sorted(missing)[:10],
        )
    logger.info('Retrying %d of %d failed records', len(latest), len(failed))

    submit(
        config,
        iter(latest.values()),
        first_chunk_index=runs.resume_offset(run_dir)[0],
        retry=True,
        max_requests=max_requests,
        max_bytes=max_bytes,
        max_batches=max_batches,
        model=model,
    )


def main() -> None:
    """Run the literature Negatome v3 screen through the Anthropic Batch API.

    Submitting, polling and collecting are separate invocations because a
    corpus-scale batch takes hours: the run directory and its manifest carry
    everything needed to resume, and re-invoking any mode is safe.

    submit covers the whole corpus in one run, packaging each paper at the full-text
    level or the abstract level according to what the corpus stage retrieved for it. It
    chunks on serialized size as well as request count, writes what it sent, and
    records each batch in the manifest. poll reports processing status. collect
    archives each ended batch's raw jsonl to S3 before parsing it into records,
    validating against the schema and checking every excerpt against the text that was
    actually sent, then reports the outcome taxonomy and the grounding pass rate. retry
    re-submits the requests whose records came back with no answer, rebuilding them
    from the recorded text. Pass --model on a retry to send those requests to a
    different model than the run's primary one; model is a plain parameter to retry
    and submit, not part of StageConfig, so run_id does not move. Cascading through
    more than one fallback model is just re-invoking retry with each --model in
    turn, collecting between them: a later retry's failed set is read fresh from
    the latest record per custom_id, so it only picks up whatever the previous
    model still could not answer.

    Every record carries text_source, input_level and model, so any number the run
    reports can be split by how the paper was retrieved, how it was packaged and
    which model actually answered it. Records collected before this field existed
    have no 'model' key at all, not a stale value - read it as
    record.get('model', MODEL) rather than record['model'] when a script might
    walk records/*.jsonl written both before and after this change.

    Reads FLOCK_ANTHROPIC_API_KEY and fails hard if it is unset, so a run cannot
    silently bill the wrong account. This script is executed by the user, not by an
    agent.

    A run is addressed by its content-hashed run_id, which moves if the prompt, either
    packaging or the schema is edited. Pass --run-id to poll or collect a run whose
    inputs have since changed, rather than letting them look in the wrong place.
    """
    setup_logging()
    parser = parse_args()
    args = parser.parse_args()

    if args.mode == 'submit':
        if not args.papers or not args.blocks:
            parser.error(
                'submit needs --papers and --blocks from the corpus stage',
            )
        if args.run_id:
            # Submitting addresses the run by the config's own hash. Honouring a
            # --run-id here would report one run directory while reading resume state
            # from another, so resume_offset would see an empty manifest and the
            # corpus would be re-packaged and re-submitted at full cost. Checked
            # before the run_id is printed, so the operator is never shown one.
            parser.error(
                '--run-id addresses an existing run for poll and collect only; '
                'submit derives the run_id from the prompt, packaging and schema',
            )
    if args.mode != 'retry' and args.model:
        parser.error(
            '--model only applies to retry; submit, poll and collect always use '
            'the primary MODEL constant.',
        )

    config = StageConfig(
        prompt_version=args.prompt_version,
        input_level=args.input_level,
        abstract_level=args.abstract_level,
        label=args.label,
    )
    run_id = args.run_id or config.run_id
    print(f'run_id {run_id}')

    if args.mode == 'submit':
        papers = pd.read_parquet(
            args.papers,
            columns=[
                'paper_id', 'text_source', 'pmid', 'pmcid', 'usable_fulltext',
            ],
        )
        logger.info('Screening %d corpus papers', len(papers))
        if papers.empty:
            parser.error(f'No papers in {args.papers}.')
        # Resuming skips packaging the papers already submitted, rather than
        # re-deriving them to find where the chunk boundaries fell.
        run_dir = runs.make_run_dir(config.run_id)
        n_chunks, n_items = runs.resume_offset(run_dir)
        submit(
            config,
            iter_items(config, papers, args.blocks, skip_items=n_items),
            first_chunk_index=n_chunks,
            bindings={'corpus': runs.corpus_identity(args.blocks)},
            max_requests=args.max_requests,
            max_bytes=args.max_bytes,
            max_batches=args.max_batches,
        )
    elif args.mode == 'retry':
        retry(
            config, run_id,
            max_requests=args.max_requests,
            max_bytes=args.max_bytes,
            max_batches=args.max_batches,
            model=args.model,
        )
    elif args.mode == 'poll':
        poll(run_id, wait=args.wait, interval=args.interval)
    else:
        collect(run_id, upload=not args.no_upload)


def add_batch_args(target: argparse.ArgumentParser) -> None:
    """Add the flags that configure this runner's Batch API limits.

    Declared beside the limits so every stage driving the runner offers the same
    three flags with the same defaults.
    """
    target.add_argument(
        '--max-requests', type=int, default=MAX_REQUESTS_PER_BATCH,
        help='Requests per batch.',
    )
    target.add_argument(
        '--max-bytes', type=int, default=MAX_BATCH_BYTES,
        help='Serialized bytes per batch.',
    )
    target.add_argument(
        '--max-batches', type=int,
        help='Submit at most this many new batches, for a pilot before the rest.',
    )


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Submit, poll, collect or retry a literature Negatome v3 screening run.',
    )
    parser.add_argument('mode', choices=['submit', 'retry', 'poll', 'collect'])
    parser.add_argument(
        '--prompt-version', default=DEFAULT_PROMPT_VERSION,
        help='Prompt stem.',
    )
    parser.add_argument(
        '--input-level', default=DEFAULT_INPUT_LEVEL,
        help='Packaging for papers with usable full text.',
    )
    parser.add_argument(
        '--abstract-level', default=DEFAULT_ABSTRACT_LEVEL,
        help='Packaging for papers without usable full text.',
    )
    parser.add_argument(
        '--label', default='',
        help='Extra slug in the run_id, e.g. pilot.',
    )
    parser.add_argument(
        '--model', default='',
        help='Model to send this retry\'s requests to (retry only); defaults to '
             'the primary MODEL constant.',
    )
    parser.add_argument(
        '--run-id', default='',
        help='Address an existing run directly (retry, poll and collect only), for '
             'when an edit to the prompt, packaging or schema has moved the run_id.',
    )
    parser.add_argument(
        '--papers', help='Local corpus papers parquet (submit only).',
    )
    parser.add_argument(
        '--blocks', help='Local corpus blocks parquet (submit only).',
    )
    add_batch_args(parser)
    parser.add_argument(
        '--wait', action='store_true',
        help='Keep polling until every batch has ended (poll only).',
    )
    parser.add_argument(
        '--interval', type=float, default=300.0,
        help='Seconds between polls when waiting.',
    )
    parser.add_argument(
        '--no-upload', action='store_true',
        help='Skip the S3 archive of raw results (collect only).',
    )
    return parser


if __name__ == '__main__':
    main()
