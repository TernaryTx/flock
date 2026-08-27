from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from flock.negatome_v3.literature import RUN_ROOT

logger = logging.getLogger(__name__)

EUROPEPMC_BASE = 'https://www.ebi.ac.uk/europepmc/webservices/rest'

# Every fetch writes here first and is read back on rerun, so reruns cost nothing
# and an S3 mirror is a copy rather than the working set. The environment variable
# exists so a long sweep can be pointed at a larger volume without editing code.
DEFAULT_CACHE_DIR = os.environ.get(
    'NEGATOME_LIT_CACHE_DIR', str(RUN_ROOT / 'cache'),
)

FULLTEXT_CACHE_SUBDIR = 'europepmc_fulltext'
SEARCH_CACHE_SUBDIR = 'europepmc_search'

MAX_PAGE_SIZE = 1000

# Rate limiting and transient server errors. A 404 is a real answer (no full text
# for this article) and must not be retried.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# isOpenAccess, inEPMC and pmcid are three different things and only the first
# predicts whether fullTextXML resolves, so all three are recorded and none is used
# as a proxy.
METADATA_FIELDS = (
    'id', 'source', 'pmid', 'pmcid', 'doi', 'title', 'pubYear',
    'firstPublicationDate', 'isOpenAccess', 'inEPMC', 'inPMC', 'license',
    'hasPDF', 'hasSuppl', 'hasTextMinedTerms', 'hasBook', 'language',
    'publicationStatus', 'citedByCount',
)


def cache_path(*parts: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Build a path inside the local cache, creating its parent directories."""
    path = os.path.join(cache_dir, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _http_get(
        url: str,
        timeout: float = 180.0,
        max_attempts: int = 4,
        backoff: float = 2.0,
) -> tuple[int, bytes]:
    """GET a URL, retrying only on rate limiting and transient server errors.

    Returns:
        (HTTP status, body). The body is empty for non-200 responses.

    Raises:
        OSError: If the request keeps failing at the network level.
        http.client.HTTPException: If the response keeps arriving malformed.
    """
    last_status = 0
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            last_status = error.code
            if error.code not in RETRY_STATUSES:
                return error.code, b''
        except (OSError, http.client.HTTPException):
            # Broader than URLError, which urllib only raises for a socket error
            # while opening the connection. A timeout, reset or IncompleteRead during
            # response.read() surfaces as a bare TimeoutError, ConnectionResetError or
            # IncompleteRead - none of them a URLError - and would otherwise abort a
            # multi-hour corpus fetch on one dropped connection. URLError is itself an
            # OSError, and HTTPError is caught above it.
            if attempt == max_attempts - 1:
                raise
        sleep_for = backoff ** attempt
        attempt_number = attempt + 1
        endpoint = url.split('?')[0]
        logger.warning(
            'Retrying %s after status %s (attempt %d/%d, sleeping %.1fs)',
            endpoint, last_status, attempt_number, max_attempts, sleep_for,
        )
        time.sleep(sleep_for)
    return last_status, b''


def search_pages(
        query: str,
        cache_dir: str = DEFAULT_CACHE_DIR,
        result_type: str = 'core',
        page_size: int = MAX_PAGE_SIZE,
        sleep_between: float = 0.2,
        max_pages: int | None = None,
) -> Iterator[dict]:
    """Page through a free-text Europe PMC search, caching each page.

    Pagination is by cursorMark, the only form Europe PMC supports past the first
    1,000 hits. A cursor mark is opaque and valid only for the query and sort it
    came from, so pages are cached by (query, page index) instead: a resumed sweep
    replays the cached pages and continues from the last cursor. That makes a cached
    sweep a frozen snapshot rather than a live view - delete the cache subdirectory
    to re-sweep against a moved index.

    Args:
        query: Europe PMC query string.
        cache_dir: Local cache root.
        result_type: Europe PMC resultType; 'core' carries abstracts and the
            coverage flags, 'idlist' is identifiers only.
        page_size: Hits per page, capped at MAX_PAGE_SIZE by the API.
        sleep_between: Courtesy pause after an uncached request, in seconds.
        max_pages: Stop after this many pages; None sweeps to exhaustion.

    Raises:
        RuntimeError: If a page request does not return HTTP 200.
    """
    # resultType is part of the key, not just the query. A hit-count probe asks for
    # 'idlist' over the same query a full sweep later runs as 'core', and a key that
    # ignored it would serve the sweep a cached page with no abstract and no
    # pubType - papers that would then be dropped as unusable, silently.
    cache_key = f'{result_type}\n{query}'.encode()
    query_hash = hashlib.sha1(cache_key).hexdigest()[:16]
    cursor = '*'
    page_index = 0

    while True:
        if max_pages is not None and page_index >= max_pages:
            return
        local = cache_path(
            SEARCH_CACHE_SUBDIR, query_hash, f'{page_index:05d}.json',
            cache_dir=cache_dir,
        )
        if os.path.exists(local):
            with open(local) as cached:
                payload = json.load(cached)
        else:
            url = f'{EUROPEPMC_BASE}/search?' + urllib.parse.urlencode({
                'query': query,
                'resultType': result_type,
                'format': 'json',
                'pageSize': str(min(page_size, MAX_PAGE_SIZE)),
                'cursorMark': cursor,
            })
            status, body = _http_get(url)
            if status != 200:
                raise RuntimeError(
                    f'Europe PMC search returned HTTP {status} for '
                    f'{query_hash} page {page_index}',
                )
            payload = json.loads(body)
            # Written via .part so a kill mid-write cannot leave a truncated page
            # that a resumed sweep fails to parse.
            part = f'{local}.part'
            with open(part, 'wb') as out:
                out.write(body)
            os.replace(part, local)
            if sleep_between:
                time.sleep(sleep_between)

        results = payload.get('resultList', {}).get('result', [])
        yield payload
        if page_index == 0:
            logger.info(
                'Query %s: %s hits', query_hash, payload.get('hitCount'),
            )

        next_cursor = payload.get('nextCursorMark')
        # Europe PMC stops advancing the cursor on the last page and returns a short
        # page there. Checking both means a truncated final page cannot loop forever.
        if not results or not next_cursor or next_cursor == cursor:
            return
        cursor = next_cursor
        page_index += 1


def flatten_metadata_record(identifier: str, record: dict) -> dict:
    """Reduce a Europe PMC search result to the coverage row used downstream."""
    abstract = record.get('abstractText') or ''
    pub_types = record.get('pubTypeList', {}).get('pubType') or []
    if isinstance(pub_types, str):
        pub_types = [pub_types]
    flat = {
        'identifier': identifier,
        'has_abstract': bool(abstract.strip()),
        'abstract_chars': len(abstract),
        'abstract_text': abstract or None,
        'pub_types': ';'.join(pub_types),
        'journal': (record.get('journalInfo') or {}).get('journal', {}).get('title'),
    }
    for field in METADATA_FIELDS:
        flat[field] = record.get(field)
    return flat


def fetch_full_text(
        pmcid: str,
        cache_dir: str = DEFAULT_CACHE_DIR,
        sleep_between: float = 0.2,
) -> tuple[int, bytes | None]:
    """Attempt to fetch JATS full-text XML for a PMCID, caching the outcome.

    Called for every record with a PMCID regardless of its isOpenAccess or inEPMC
    flags, because those do not reliably predict whether the endpoint resolves:
    records exist with inEPMC=Y and a PMCID that still 404 here. Measuring the fetch
    is the only honest way to report coverage, and an answered status is cached so a
    failure is not silently retried into a different answer later.

    Returns:
        (HTTP status, XML bytes or None if the fetch did not return 200).
    """
    xml_path = cache_path(
        FULLTEXT_CACHE_SUBDIR, f'{pmcid}.xml', cache_dir=cache_dir,
    )
    status_path = cache_path(
        FULLTEXT_CACHE_SUBDIR, f'{pmcid}.status', cache_dir=cache_dir,
    )
    if os.path.exists(status_path):
        with open(status_path) as cached:
            status = int(cached.read().strip())
        if status == 200 and os.path.exists(xml_path):
            with open(xml_path, 'rb') as cached_xml:
                return status, cached_xml.read()
        return status, None

    status, body = _http_get(f'{EUROPEPMC_BASE}/{pmcid}/fullTextXML')
    # The status file is the cache's commit record, so it is written last and only
    # after the XML is on disk under its final name. Writing it first would let an
    # interruption cache a 200 with no readable XML behind it, and since the read
    # path above trusts the status, that paper would be reported as retrieved, never
    # screened and never retried.
    if status == 200 and body:
        part_path = f'{xml_path}.part'
        with open(part_path, 'wb') as out:
            out.write(body)
        os.replace(part_path, xml_path)
    # An exhausted retryable status is the fetcher giving up, not Europe PMC's answer
    # about the article, so it is not committed to the cache. Caching it would
    # short-circuit every later run on that paper and make it indistinguishable from a
    # real 404 - the opposite of the honest coverage figure this module exists for.
    if status in RETRY_STATUSES:
        logger.warning(
            '%s: giving up at HTTP %s after retries; not caching the status',
            pmcid, status,
        )
    elif status == 200 and not body:
        # A 200 with an empty body is not an answer about the article either: nothing
        # was written, so committing the status would cache the paper as retrieved with
        # no XML behind it and the read path above would never fetch it again.
        logger.warning(
            '%s: HTTP 200 with an empty body; not caching the status', pmcid,
        )
    else:
        with open(status_path, 'w') as out:
            out.write(str(status))
    if sleep_between:
        time.sleep(sleep_between)
    return status, (body if status == 200 and body else None)


# A fetch is almost entirely waiting, so a corpus sweep's tens of thousands of them run
# on a small pool. The ceiling is Europe PMC's patience rather than this machine's: each
# worker still takes fetch_full_text's courtesy pause, so the request rate is roughly the
# worker count over one fetch's round trip. Raise it and the 429 backoff in _http_get is
# what absorbs the consequences.
DEFAULT_FETCH_WORKERS = 8

# How far ahead of its consumer the pool may run. Every queued result is a whole
# article's XML held in memory, so this is what stops a pool handed a 100,000-PMCID
# corpus from reading that corpus into RAM. Two per worker keeps them all busy across
# one slow article without the queue growing past it.
QUEUE_DEPTH_PER_WORKER = 2


def fetch_full_text_ordered(
        pmcids: Iterable[str],
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_workers: int = DEFAULT_FETCH_WORKERS,
        queue_depth_per_worker: int = QUEUE_DEPTH_PER_WORKER,
) -> Iterator[tuple[str, int, bytes | None]]:
    """Fetch many PMCIDs concurrently, yielding the results in the order given.

    The order is the point. The caller parses each result and writes that paper's
    blocks straight into a parquet file whose readers require one paper's rows to be
    contiguous and the papers themselves to arrive sorted, so the concurrency has to
    stay invisible downstream: the tables a sweep writes are identical whatever
    max_workers is, and only its wall clock moves.

    ThreadPoolExecutor.map preserves order too, but it submits every task up front,
    which over a corpus sweep means every article's XML resident at once. Here at most
    max_workers * queue_depth_per_worker fetches are ever in flight.

    The workers share nothing: fetch_full_text keys its cache files on the PMCID, and
    the corpus stage has already reduced its fetch list to one row per paper, so no two
    workers write the same path.

    Args:
        pmcids: PMCIDs to fetch, in the order the results are wanted.
        cache_dir: Local cache root.
        max_workers: Concurrent fetches.
        queue_depth_per_worker: In-flight fetches allowed per worker.

    Yields:
        (pmcid, HTTP status, XML bytes or None), one tuple per PMCID given.

    Raises:
        OSError: If a fetch keeps failing at the network level. Raised from the worker
            when the consumer reaches that PMCID, so a sweep still stops on it rather
            than recording a network failure as the article's own answer.
        http.client.HTTPException: Likewise for a persistently malformed response.
    """
    remaining = iter(pmcids)
    in_flight: deque[tuple[str, Future[tuple[int, bytes | None]]]] = deque()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:

        def submit(count: int) -> None:
            for pmcid in islice(remaining, count):
                in_flight.append(
                    (
                        pmcid, pool.submit(
                            fetch_full_text, pmcid, cache_dir=cache_dir,
                        ),
                    ),
                )

        submit(max_workers * queue_depth_per_worker)
        while in_flight:
            pmcid, pending = in_flight.popleft()
            status, xml = pending.result()
            # Refilled after that result is in hand rather than before, so the queue
            # depth is a true ceiling on how much XML is alive at once.
            submit(1)
            yield pmcid, status, xml
