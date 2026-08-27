# Everything under one run's directory: its config, manifest, requests and results.
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from collections.abc import Iterator
from datetime import datetime
from datetime import timezone
from pathlib import Path

from flock.negatome_v3.literature import RUN_ROOT
from flock.negatome_v3.literature.config import RunConfig

logger = logging.getLogger(__name__)

RUNS_ROOT = RUN_ROOT / 'runs'

CONFIG_FILE = 'config.json'
FROZEN_PROMPT_FILE = 'prompt.md'
MANIFEST_FILE = 'manifest.json'

# One file per chunk, so a rewrite replaces that chunk's file rather than appending
# to a shared one. That is what makes re-invoking submit or collect idempotent: a
# run interrupted mid-chunk cannot leave half a chunk's rows behind the ones that
# replace them.
REQUESTS_SUBDIR = 'requests'
RECORDS_SUBDIR = 'records'
RAW_SUBDIR = 'raw'


def make_run_dir(run_id: str) -> Path:
    """Create and return a run's directory and its subdirectories."""
    run_dir = RUNS_ROOT / run_id
    for subdir in (REQUESTS_SUBDIR, RECORDS_SUBDIR, RAW_SUBDIR):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def require_run_dir(run_id: str) -> Path:
    """Return an existing run's directory, refusing to invent an empty one.

    A run_id hashes the prompt text, the packaging and the schema, so editing any
    of them between submitting and collecting moves it. Creating the directory on
    demand would leave poll and collect reading an empty manifest and reporting
    nothing to do, on batches that have already been paid for.

    Raises:
        FileNotFoundError: If no such run directory exists.
    """
    run_dir = RUNS_ROOT / run_id
    if not (run_dir / MANIFEST_FILE).exists():
        existing = sorted(
            path.name for path in RUNS_ROOT.glob('*/')
        ) if RUNS_ROOT.exists() else []
        raise FileNotFoundError(
            f'No run manifest at {run_dir / MANIFEST_FILE}. The run_id is derived '
            f'from the prompt text, packaging and schema, so an edit to any of '
            f'them since submitting will have moved it. Pass --run-id to name the '
            f'run directly. Existing runs: {existing}',
        )
    return run_dir


def raw_path(run_dir: Path, batch_id: str) -> Path:
    return run_dir / RAW_SUBDIR / f'{batch_id}.jsonl'


def records_path(run_dir: Path, chunk_index: int) -> Path:
    return run_dir / RECORDS_SUBDIR / f'{chunk_index:05d}.jsonl'


def requests_path(run_dir: Path, chunk_index: int) -> Path:
    return run_dir / REQUESTS_SUBDIR / f'{chunk_index:05d}.jsonl'


def corpus_identity(blocks_path: str | Path) -> dict:
    """Identify the block table a run screened, by name, size and modification time.

    The run_id hashes the prompt, packaging and schema but not the corpus, and
    --blocks is a free-form path, so without this a resume against a rebuilt table
    would keep the same run_id, advance its item iterator over the first N papers of
    the *new* table, and screen two different corpora under one identity.
    """
    path = Path(blocks_path)
    stat = path.stat()
    return {
        'blocks': path.name,
        'bytes': stat.st_size,
        'mtime': datetime.fromtimestamp(
            stat.st_mtime, timezone.utc,
        ).isoformat(),
    }


def write_run_config(run_dir: Path, config: RunConfig) -> None:
    """Write the run's self-description and a frozen copy of its prompt.

    Stores resolved values only, so nothing here is a pointer to a file that can
    change underneath it and the run stays interpretable after the prompt moves on.
    """
    payload = {
        'run_id': config.run_id, 'label': config.label, **config.fingerprint(),
    }
    with open(run_dir / CONFIG_FILE, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    (run_dir / FROZEN_PROMPT_FILE).write_text(config.prompt_text)


def read_run_config(run_dir: Path) -> dict:
    """Read back the run's self-description, or an empty dict if it has none."""
    path = run_dir / CONFIG_FILE
    if not path.exists():
        return {}
    with open(path) as handle:
        config: dict = json.load(handle)
    return config


def read_manifest(run_dir: Path) -> dict:
    """Read the run's batch manifest, or an empty one if the run is new."""
    path = run_dir / MANIFEST_FILE
    if not path.exists():
        return {'batches': []}
    with open(path) as handle:
        manifest: dict = json.load(handle)
    return manifest


def write_manifest(run_dir: Path, manifest: dict) -> None:
    """Write the manifest mapping each chunk to the batch that carries it.

    This is what makes an interrupted run resumable by polling rather than by
    resubmitting, so a second submit skips work already paid for. Written to a
    temporary file and moved into place, because it is the only record of those batch
    ids: truncating it to rewrite it would let an interruption mid-write - including
    the write immediately after a batch is created - leave invalid JSON behind and
    lose the ids the recovery path reads.
    """
    path = run_dir / MANIFEST_FILE
    part_path = path.with_name(path.name + '.part')
    with open(part_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part_path, path)


def bind_manifest(manifest: dict, bindings: dict[str, dict]) -> None:
    """Record what a run is bound to, refusing a resume that disagrees.

    Validated and written together, into the manifest the caller is about to
    write. A caller that recorded a binding itself and then checked afterwards
    would persist it even when the check rejects the resume, leaving the run
    bound to an input table its own answers never came from.

    Args:
        manifest: The run's manifest, modified in place.
        bindings: Manifest key to the identity this invocation was built from.

    Raises:
        RuntimeError: If the run is already bound to a different value of any of
            them.
    """
    for key, value in bindings.items():
        recorded = manifest.get(key)
        if recorded and recorded != value:
            raise RuntimeError(
                f'This run is bound to {key}={recorded}, and {value} was passed. '
                f'Pass what the run was built from, or start a new run with '
                f'--label.',
            )
        manifest[key] = value


def resume_offset(run_dir: Path) -> tuple[int, int]:
    """Return where a resume starts: the next chunk index, and papers already sent.

    The two are counted differently, which matters once a run has retried anything.
    Chunk indices name the requests and records files, so the next one must clear
    every index used, retries included. The item count drives how far the paper
    iterator is advanced, so it counts only the chunks that consumed new papers: a
    retry chunk re-sends text already recorded, and counting it would skip that many
    unscreened papers.

    Raises:
        RuntimeError: If a chunk holding requests was recorded without a batch id,
            meaning a previous submit was interrupted mid-create. The create call
            may or may not have gone through, and resubmitting blind would pay for
            the same papers twice, so this stops rather than guesses.
    """
    entries = read_manifest(run_dir)['batches']
    pending = [
        entry['chunk_index'] for entry in entries
        if entry['n_requests'] and not entry.get('batch_id')
    ]
    if pending:
        raise RuntimeError(
            f'Chunks {pending} are recorded as pending with no batch id, so a '
            f'previous submit was interrupted mid-create. List the account\'s '
            f'batches, and either write the matching batch id into '
            f'{run_dir / MANIFEST_FILE} or delete the pending entries if no '
            f'batch was created.',
        )
    next_chunk_index = max(
        (entry['chunk_index'] for entry in entries), default=-1,
    ) + 1
    n_items = sum(
        entry['n_requests'] + entry['n_skipped'] for entry in entries
        if not entry.get('retry')
    )
    return next_chunk_index, n_items


def submitted_batches(manifest: dict) -> list[dict]:
    """Return the manifest entries that actually carry a batch.

    A chunk that selected no text has no batch and nothing to report; one left
    pending by an interrupted submit has one that cannot be addressed. Both are
    filtered here so poll and collect do not each carry the distinction.
    """
    ready = []
    for entry in manifest['batches']:
        if entry.get('batch_id'):
            ready.append(entry)
        elif entry['n_requests']:
            logger.warning(
                'chunk %d is pending with no batch id; submit was interrupted '
                'mid-create and needs reconciling', entry['chunk_index'],
            )
    return ready


def write_requests(run_dir: Path, chunk_index: int, chunk: list[dict]) -> None:
    """Record what a chunk sent, before its batch is created.

    This copy is what the grounding gate is checked against at collection.

    Whatever an item carries is recorded, not a fixed key set: a stage sending
    something other than a whole paper has its own fields to check an answer
    against. Items must therefore hold only JSON-serializable values.
    """
    with open(requests_path(run_dir, chunk_index), 'w') as out:
        for item in chunk:
            out.write(json.dumps(item) + '\n')


def read_requests(run_dir: Path, chunk_index: int) -> dict[str, dict]:
    """Read back what a chunk sent, keyed by custom_id.

    Carries the text for the grounding gate and the identifiers a record is joined
    on. Read back rather than re-derived from the block table: re-deriving would
    silently pass if the corpus had moved underneath the run, which is the one
    thing the gate exists to catch.
    """
    sent = {}
    with open(requests_path(run_dir, chunk_index)) as handle:
        for line in handle:
            row = json.loads(line)
            sent[row['custom_id']] = row
    return sent


def iter_requests(run_dir: Path, custom_ids: set[str]) -> Iterator[dict]:
    """Yield the recorded requests for these custom_ids, across every chunk.

    Scanned line by line rather than loaded, because at corpus scale the requests
    files hold the whole screened corpus as text. Yielded in chunk order, so a
    custom_id that a retry has already re-sent appears more than once.
    """
    for path in sorted((run_dir / REQUESTS_SUBDIR).glob('*.jsonl')):
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                if row['custom_id'] in custom_ids:
                    yield row


def sent_custom_ids(run_dir: Path) -> set[str]:
    """Every custom_id this run has already recorded a request for.

    The resume state for a stage whose work is a table, where resume_offset's row
    count serves one whose work is a stream over the corpus.

    Only chunks the manifest records are read, never every file in the directory.
    submit writes a chunk's requests file before it appends that chunk to the
    manifest, so an interruption between the two leaves a requests file no
    manifest entry covers - and reading it here would mark those ids sent while
    resume_offset, which counts manifest entries, hands the same chunk index to
    the next submit. The chunk file would be overwritten with different items and
    the ids in it never sent at all.

    Raises:
        FileNotFoundError: If the manifest records a chunk whose requests file is
            missing, which means the run directory has lost data rather than
            merely been interrupted.
    """
    ids = set()
    for entry in read_manifest(run_dir)['batches']:
        path = requests_path(run_dir, entry['chunk_index'])
        if not path.exists():
            raise FileNotFoundError(
                f'The manifest records chunk {entry["chunk_index"]} but there is '
                f'no {path}. Resuming would re-send that chunk and pay for it '
                f'twice. Restore the file, or drop the entry if the chunk was '
                f'never really sent.',
            )
        with open(path) as handle:
            for line in handle:
                ids.add(json.loads(line)['custom_id'])
    return ids


def latest_records(
        run_dir: Path,
        project: Callable[[dict], dict] | None = None,
) -> list[dict]:
    """The run's records, keeping only the last one per custom_id.

    A custom_id appears in more than one chunk once a run has retried anything -
    the delivered screen re-sent Sonnet 5's refusals under Sonnet 4.6 and those
    errors under Haiku 4.5. Chunk files sort in the order they were written, so
    the last record for a custom_id is the one that answered it and supersedes
    every earlier attempt.

    Args:
        run_dir: A run directory holding a records subdirectory.
        project: Reduces one record to what the caller keeps, applied before it
            is stored rather than after - the screen's 150,217 whole records cost
            920 MB retained against 30 MB for the two fields the retry list
            reads. A caller wanting the whole record passes nothing.

    Returns:
        One record per custom_id, projected if a projection was given.

    Raises:
        FileNotFoundError: If the run has no records directory.
    """
    records_dir = run_dir / RECORDS_SUBDIR
    if not records_dir.is_dir():
        raise FileNotFoundError(f'No records directory at {records_dir}')
    latest: dict[str, dict] = {}
    n_lines = 0
    for path in sorted(records_dir.glob('*.jsonl')):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                n_lines += 1
                latest[record['custom_id']] = (
                    record if project is None else project(record)
                )
    logger.info(
        '%s: %d record lines over %d custom_ids', run_dir.name, n_lines,
        len(latest),
    )
    return list(latest.values())


def failed_custom_ids(run_dir: Path) -> list[str]:
    """The custom_ids whose most recent record carries no usable answer.

    A custom_id can appear in several chunks once retries exist, so the last record
    wins: a retry that succeeded must not leave the original failure in the list.
    Skipped papers are not in here at all, having never been sent.
    """
    return [
        record['custom_id'] for record in latest_records(
            run_dir, lambda item: {
                'custom_id': item['custom_id'], 'error': item.get('error'),
            },
        )
        if record['error']
    ]
