# PPI3D bulk-download utilities. PPI3D has no REST API; its "Download data" page
# is a form POST that enqueues a server-side job, which is then polled for a
# gzipped CSV. These helpers wrap that flow so it can be driven from a script.
from __future__ import annotations

import gzip
import logging
import os
import re
import time
from datetime import date
from datetime import timedelta

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PPI3D_BASE_URL = 'https://bioinformatics.lt/ppi3d/'
SUBMIT_URL = PPI3D_BASE_URL + 'clusters/submit_interfaces_request'
JOB_URL_TEMPLATE = PPI3D_BASE_URL + 'clusters/data_request/{job_id}'

# The submit form posts every field on every request, so the full set is kept
# here and overridden selectively by build_criteria(). Values mirror the form
# defaults except where noted in build_criteria's signature.
BASE_CRITERIA: dict[str, str] = {
    'interaction_types[protein_protein_interactions]': '1',
    'interaction_types[protein_peptide_interactions]': '0',
    'interaction_types[protein_nucleic_interactions]': '0',
    'interaction_types[domain_nucleic_interactions]': '0',
    'interaction_types[intra_chain_domain_interactions]': '0',
    'complex[min_number_of_residues]': '1',
    'complex[max_number_of_residues]': '1000000000',
    'complex[min_number_of_residues_visible]': '1',
    'complex[max_number_of_residues_visible]': '1000000000',
    'subunits[min_number_of_residues]': '1',
    'subunits[max_number_of_residues]': '100000',
    'subunits[min_number_of_residues_visible]': '1',
    'subunits[max_number_of_residues_visible]': '100000',
    'interface[max_area]': '100000',
    'interface[max_number_of_contacts]': '10000',
    'interface[allow_ligands]': '1',
    'include_clustering_data_in_table': '1',
    'submit': 'Submit',
}

# Artifacts the job page exposes once the request completes. The log records how
# many complexes and interfaces the server retrieved and filtered, and the json
# echoes the submitted criteria; both are kept for provenance alongside the data.
RESULT_EXTENSIONS = ('.csv.gz', '.json', '.log')


def make_session(retries: int = 5, backoff_factor: float = 2.0) -> requests.Session:
    """Build a session that survives the transient drops of a long pull.

    A full-history pull is hundreds of requests over several hours, during which
    the server will occasionally close a connection without responding. Without
    transport-level retries a single such drop aborts the whole run, so retries
    are mounted here rather than left to callers. POST is included: a dropped
    submit almost certainly never reached the server, and a duplicate job is
    merely wasteful rather than harmful.

    Args:
        retries: Maximum retry attempts per request.
        backoff_factor: Exponential backoff multiplier between attempts.

    Returns:
        Configured requests session.
    """
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({'GET', 'POST'}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def get_database_date(session: requests.Session | None = None) -> str:
    """Return the PDB release date PPI3D's current snapshot goes up to.

    Requesting a release_date_to beyond this is silently rejected: the server
    re-renders the form instead of enqueueing a job, which surfaces as a missing
    job id. The download form's own default upper bound is the server telling us
    the maximum valid date, so it is read from there rather than hard-coded.

    Args:
        session: Requests session to reuse; a fresh one is created if omitted.

    Returns:
        Date string (YYYY-MM-DD).

    Raises:
        RuntimeError: If the date cannot be parsed from the form.
    """
    owns_session = session is None
    session = session or make_session()
    try:
        response = session.get(PPI3D_BASE_URL + 'clusters', timeout=120)
        response.raise_for_status()
        match = re.search(
            r'name="PDB_data\[release_date_to\]"[^>]*value="(\d{4}-\d{2}-\d{2})"',
            response.text,
        )
        if match is None:
            raise RuntimeError(
                'Could not read the PPI3D snapshot date from the download form',
            )
        return match.group(1)
    finally:
        if owns_session:
            session.close()


def build_criteria(
        release_date_from: str,
        release_date_to: str,
        resolution: float = 4.0,
        min_area: float = 100.0,
        min_contacts: int = 1,
        min_subunits: int = 2,
        max_subunits: int = 10000,
        min_distinct_subunits: int = 2,
        max_distinct_subunits: int = 10000,
        include_homo: bool = False,
        include_hetero: bool = True,
        clustering: str = 'sequence40_structure50',
) -> dict[str, str]:
    """Build the POST payload for a PPI3D interface-cluster data request.

    Defaults request hetero protein-protein interfaces from multi-subunit
    assemblies, which is the subset Flock treats as candidate positive pairs.
    Thresholds are left permissive here and applied properly downstream, so that
    one cached download can serve several filter settings.

    Args:
        release_date_from: Inclusive lower bound on PDB release date (YYYY-MM-DD).
        release_date_to: Inclusive upper bound on PDB release date (YYYY-MM-DD).
        resolution: Keep structures with resolution better than this, in angstroms.
            PPI3D itself only ingests structures better than 4 A, so values above
            4.0 do not widen the result.
        min_area: Minimum interface area in A^2. PPI3D only stores interfaces
            above 100 A^2, so values below that do not widen the result.
        min_contacts: Minimum number of interface contacts.
        min_subunits: Minimum number of subunits in the assembly.
        max_subunits: Maximum number of subunits in the assembly.
        min_distinct_subunits: Minimum number of distinct subunits in the assembly.
        max_distinct_subunits: Maximum number of distinct subunits in the assembly.
        include_homo: Whether to include homo-interactions.
        include_hetero: Whether to include hetero-interactions.
        clustering: Clustering level, one of 'none', 'sequence95_structure50',
            'sequence70_structure50', 'sequence40_structure50',
            'sequence40_structure50_area'.

    Returns:
        Form payload mapping field names to string values.
    """
    criteria = dict(BASE_CRITERIA)
    criteria.update({
        'PDB_data[resolution]': str(resolution),
        'PDB_data[release_date_from]': release_date_from,
        'PDB_data[release_date_to]': release_date_to,
        'complex[min_number_of_subunits]': str(min_subunits),
        'complex[max_number_of_subunits]': str(max_subunits),
        'complex[min_number_of_protein_subunits]': str(min_subunits),
        'complex[max_number_of_protein_subunits]': str(max_subunits),
        'complex[min_number_of_different_subunits]': str(min_distinct_subunits),
        'complex[max_number_of_different_subunits]': str(max_distinct_subunits),
        'interface[min_area]': str(min_area),
        'interface[min_number_of_contacts]': str(min_contacts),
        'interface[homo]': '1' if include_homo else '0',
        'interface[hetero]': '1' if include_hetero else '0',
        'clustering': clustering,
    })
    return criteria


def submit_request(criteria: dict[str, str], session: requests.Session, timeout: int = 300) -> str:
    """Submit a data request and return the job id assigned by the server.

    Args:
        criteria: Form payload from build_criteria().
        session: Requests session to issue the POST through.
        timeout: Socket timeout in seconds for the submit call.

    Returns:
        The job id, e.g. 'jzR38D2B'.

    Raises:
        RuntimeError: If the response URL does not carry a job id.
    """
    response = session.post(SUBMIT_URL, data=criteria, timeout=timeout)
    response.raise_for_status()
    match = re.search(r'/clusters/data_request/([A-Za-z0-9]+)', response.url)
    if match is None:
        # The server re-renders the form rather than reporting an error when
        # criteria are out of range. Overshooting release_date_to past the
        # current snapshot is the usual cause; see get_database_date.
        raise RuntimeError(
            f'PPI3D did not return a job id; landed on {response.url}. '
            f'Check that release_date_to '
            f'({criteria.get("PDB_data[release_date_to]")}) is within the '
            f'current snapshot (get_database_date()).',
        )
    return match.group(1)


def _find_result_urls(page_html: str) -> dict[str, str]:
    """Extract result artifact URLs from a completed job page.

    Args:
        page_html: HTML of the job page.

    Returns:
        Mapping of extension (e.g. '.csv.gz') to URL, empty while the job is
        still queued or running.
    """
    urls = {}
    for href in re.findall(r'href="([^"]+)"', page_html):
        if '/downloads/data_requests/' not in href:
            continue
        for extension in RESULT_EXTENSIONS:
            if href.endswith(extension):
                urls[extension] = href
    return urls


def poll_request(
        job_id: str,
        session: requests.Session,
        poll_interval: int = 20,
        timeout: int = 7200,
) -> dict[str, str]:
    """Poll a PPI3D job until its result artifacts appear.

    Args:
        job_id: Job id from submit_request().
        session: Requests session to poll through.
        poll_interval: Seconds to wait between polls.
        timeout: Maximum seconds to wait before giving up.

    Returns:
        Mapping of extension to result URL, always containing '.csv.gz'.

    Raises:
        TimeoutError: If the job does not complete within timeout.
    """
    logger = logging.getLogger(__name__)
    job_url = JOB_URL_TEMPLATE.format(job_id=job_id)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # A failed poll says nothing about the job, which is running server-side
        # and unaffected by our connection dropping. Keep polling until the
        # deadline rather than abandoning work that is probably still in flight.
        try:
            response = session.get(job_url, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning('Poll of job %s failed (%s); retrying', job_id, exc)
        else:
            urls = _find_result_urls(response.text)
            if '.csv.gz' in urls:
                return urls
            logger.debug('Job %s still running', job_id)
        time.sleep(poll_interval)
    raise TimeoutError(f'PPI3D job {job_id} did not finish within {timeout}s')


def download_results(urls: dict[str, str], dest_dir: str, stem: str, session: requests.Session) -> dict[str, str]:
    """Download a completed job's artifacts into dest_dir.

    Args:
        urls: Mapping of extension to URL from poll_request().
        dest_dir: Directory to write into; created if absent.
        stem: Filename stem to use, e.g. 'ppi3d_2024Q1'.
        session: Requests session to download through.

    Returns:
        Mapping of extension to local file path.
    """
    os.makedirs(dest_dir, exist_ok=True)
    paths = {}
    for extension, url in sorted(urls.items()):
        local_path = os.path.join(dest_dir, stem + extension)
        with session.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            # iter_content, not copyfileobj(response.raw): raw skips
            # Content-Encoding handling, so a transfer-gzipped response would be
            # written still compressed on top of its own .gz.
            with open(local_path, 'wb') as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
        paths[extension] = local_path
    return paths


def fetch_window(
        release_date_from: str,
        release_date_to: str,
        dest_dir: str,
        stem: str | None = None,
        session: requests.Session | None = None,
        poll_interval: int = 20,
        **criteria_kwargs,
) -> dict[str, str]:
    """Fetch one release-date window of PPI3D interfaces, caching to dest_dir.

    Skips the request entirely if the target CSV already exists, so an
    interrupted multi-window pull can be resumed without re-querying the server.

    Args:
        release_date_from: Inclusive lower bound on PDB release date (YYYY-MM-DD).
        release_date_to: Inclusive upper bound on PDB release date (YYYY-MM-DD).
        dest_dir: Directory to cache results in.
        stem: Filename stem; defaults to 'ppi3d_<from>_<to>'.
        session: Requests session to reuse; a fresh one is created if omitted.
        poll_interval: Seconds between job polls.
        **criteria_kwargs: Passed through to build_criteria().

    Returns:
        Mapping of extension to local file path.
    """
    logger = logging.getLogger(__name__)
    if stem is None:
        stem = f'ppi3d_{release_date_from}_{release_date_to}'
    cached = os.path.join(dest_dir, stem + '.csv.gz')
    if os.path.exists(cached):
        logger.info('Using cached %s', cached)
        return {
            extension: os.path.join(dest_dir, stem + extension)
            for extension in RESULT_EXTENSIONS
            if os.path.exists(os.path.join(dest_dir, stem + extension))
        }

    owns_session = session is None
    session = session or make_session()
    try:
        criteria = build_criteria(
            release_date_from, release_date_to, **criteria_kwargs,
        )
        job_id = submit_request(criteria, session)
        logger.info(
            'Submitted %s to %s as job %s', release_date_from,
            release_date_to, job_id,
        )
        urls = poll_request(job_id, session, poll_interval=poll_interval)
        paths = download_results(urls, dest_dir, stem, session)
        logger.info('Downloaded %s', paths['.csv.gz'])
        return paths
    finally:
        if owns_session:
            session.close()


def date_windows(start: str, end: str, months: int = 3) -> list[tuple[str, str]]:
    """Split a date range into inclusive windows aligned to calendar boundaries.

    Chunking keeps any single PPI3D request bounded, since the server materialises
    the whole result set before returning it. Windows align to calendar boundaries
    (quarters for months=3, years for months=12) so repeated pulls reuse the same
    window names and therefore the same cache entries.

    Args:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).
        months: Window length in months. Should divide evenly into 12 for the
            alignment to be meaningful.

    Returns:
        List of (from, to) date-string pairs covering [start, end] with no gaps
        or overlaps.
    """
    end_date = date.fromisoformat(end)
    windows = []
    cursor = date.fromisoformat(start)
    while cursor <= end_date:
        boundary_month = ((cursor.month - 1) // months + 1) * months
        if boundary_month >= 12:
            period_end = date(cursor.year, 12, 31)
        else:
            period_end = date(
                cursor.year, boundary_month + 1, 1,
            ) - timedelta(days=1)
        window_end = min(period_end, end_date)
        windows.append((cursor.isoformat(), window_end.isoformat()))
        cursor = window_end + timedelta(days=1)
    return windows


def quarter_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Split a date range into inclusive calendar-quarter windows.

    Args:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).

    Returns:
        List of (from, to) date-string pairs.
    """
    return date_windows(start, end, months=3)


def scaled_windows(
        start: str,
        end: str,
        schedule: tuple[tuple[str, int], ...] = (
            ('2000-01-01', 12),
            ('2015-01-01', 6),
            ('9999-12-31', 3),
        ),
) -> list[tuple[str, str]]:
    """Build windows whose length shrinks as PDB deposition volume grows.

    A single quarter of 2023 holds more structures than several years of the
    1980s, so fixed-width chunking spends hundreds of requests on near-empty early
    years while still straining on recent ones. Each schedule entry gives an
    exclusive upper date bound and the window length in months to use below it.

    Args:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).
        schedule: Ordered (upper_bound, months) pairs. The final entry should carry
            a bound beyond any plausible end date.

    Returns:
        List of (from, to) date-string pairs covering [start, end] with no gaps
        or overlaps.
    """
    end_date = date.fromisoformat(end)
    windows: list[tuple[str, str]] = []
    cursor = date.fromisoformat(start)
    for bound, months in schedule:
        bound_date = date.fromisoformat(bound) - timedelta(days=1)
        segment_end = min(bound_date, end_date)
        if segment_end < cursor:
            continue
        windows.extend(
            date_windows(
                cursor.isoformat(), segment_end.isoformat(), months=months,
            ),
        )
        cursor = segment_end + timedelta(days=1)
        if cursor > end_date:
            break
    return windows


def fetch_range(
        start: str,
        end: str,
        dest_dir: str,
        windows: list[tuple[str, str]] | None = None,
        pause_seconds: float = 2.0,
        **criteria_kwargs,
) -> list[str]:
    """Fetch a date range as a series of cached windows.

    Windows are fetched serially with a short pause between them; PPI3D is a
    single academic server and should not be hit in parallel. A window whose CSV
    is already cached is skipped, so an interrupted pull resumes where it stopped.

    Args:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).
        dest_dir: Directory to cache results in.
        windows: Explicit (from, to) windows to fetch. Defaults to calendar
            quarters; pass scaled_windows(...) for a full-history pull.
        pause_seconds: Delay between consecutive requests.
        **criteria_kwargs: Passed through to build_criteria().

    Returns:
        Local paths to each window's CSV, in chronological order.
    """
    logger = logging.getLogger(__name__)
    if windows is None:
        windows = quarter_windows(start, end)
    logger.info(
        'Fetching %d windows from %s to %s', len(windows), start, end,
    )
    csv_paths = []
    failed: list[tuple[str, str]] = []
    with make_session() as session:
        for index, (window_from, window_to) in enumerate(windows, start=1):
            # One bad window must not discard the hours already spent on the
            # others. Record it and carry on; the cache makes a targeted re-run
            # of just the failures cheap.
            try:
                paths = fetch_window(
                    window_from, window_to, dest_dir,
                    session=session, **criteria_kwargs,
                )
            # Deliberately broad: after hours of fetching, no single window's
            # failure mode is worth losing the run over, and every one of them
            # is recoverable by re-running against the cache. Failures are
            # re-reported in full below so nothing goes silently missing.
            except Exception:
                logger.exception(
                    'Window %d/%d (%s..%s) failed',
                    index, len(windows), window_from, window_to,
                )
                failed.append((window_from, window_to))
            else:
                csv_paths.append(paths['.csv.gz'])
                logger.info('Window %d/%d done', index, len(windows))
            time.sleep(pause_seconds)
    if failed:
        logger.error(
            '%d of %d windows failed and are absent from the result: %s',
            len(failed), len(windows),
            ', '.join(f'{start}..{end}' for start, end in failed),
        )
    return csv_paths


def read_interfaces(csv_paths: list[str]) -> pd.DataFrame:
    """Concatenate downloaded PPI3D window CSVs into one DataFrame.

    Empty windows (early PDB years frequently return no rows) are skipped.

    Args:
        csv_paths: Local paths to gzipped PPI3D CSVs.

    Returns:
        Concatenated interface table with a reset index.
    """
    logger = logging.getLogger(__name__)
    frames = []
    for path in csv_paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            logger.debug('Empty window %s', path)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_log_counts(log_path: str) -> dict[str, int]:
    """Parse the retrieved/returned counts out of a PPI3D job log.

    The log is the only place the server reports how many interfaces it filtered
    out, which is worth carrying into provenance.

    Args:
        log_path: Local path to a job's .log file.

    Returns:
        Dict with any of 'complexes', 'interfaces_retrieved', 'interfaces_returned'
        that could be parsed.
    """
    with gzip.open(log_path, 'rt') if log_path.endswith('.gz') else open(log_path) as handle:
        text = handle.read()
    counts = {}
    patterns = {
        'complexes': r'Retrieved (\d+) protein complexes',
        'interfaces_retrieved': r'Retrieved (\d+) interfaces from DB',
        'interfaces_returned': r'Returning (\d+) filtered interfaces',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is not None:
            counts[key] = int(match.group(1))
    return counts
