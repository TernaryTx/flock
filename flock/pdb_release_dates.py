from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from flock.pdb_metadata import PdbMetadata

logger = logging.getLogger(__name__)

_GRAPHQL_URL = 'https://data.rcsb.org/graphql'
_HOLDINGS_ENTRY_IDS_URL = 'https://data.rcsb.org/rest/v1/holdings/current/entry_ids'

# RCSB GraphQL rejects entries() queries with more than this many entry_ids.
GRAPHQL_MAX_IDS = 1000


def get_current_entry_ids() -> list[str]:
    """Return every PDB entry ID currently held by the RCSB.

    Returns:
        List of uppercase 4-character PDB entry IDs.

    Raises:
        requests.HTTPError: If the holdings request fails.
    """
    response = requests.get(_HOLDINGS_ENTRY_IDS_URL)
    response.raise_for_status()
    return response.json()


def _fetch_release_dates_batch(entry_ids: list[str], max_retries: int = 3) -> dict[str, str]:
    """Fetch initial release dates for up to GRAPHQL_MAX_IDS entries via GraphQL.

    Args:
        entry_ids: Entry IDs to query (must not exceed GRAPHQL_MAX_IDS).
        max_retries: Number of attempts before giving up on transient errors.

    Returns:
        Mapping of uppercase entry ID to release date (YYYY-MM-DD). Entries
        whose release date is missing are omitted.

    Raises:
        requests.HTTPError: If the request still fails after max_retries.
    """
    query = (
        '{ entries(entry_ids: %s) '
        '{ rcsb_id rcsb_accession_info { initial_release_date } } }'
    ) % json.dumps(entry_ids)
    for attempt in range(max_retries):
        try:
            response = requests.post(
                _GRAPHQL_URL, json={'query': query}, timeout=120,
            )
            response.raise_for_status()
            entries = (response.json().get('data') or {}).get('entries') or []
            dates: dict[str, str] = {}
            for entry in entries:
                if entry is None:
                    continue
                accession_info = entry.get('rcsb_accession_info') or {}
                released = accession_info.get('initial_release_date')
                if released:
                    dates[entry['rcsb_id'].upper()] = released[:10]
            return dates
        except (requests.RequestException, ValueError) as error:
            if attempt == max_retries - 1:
                raise
            logger.warning(
                'RCSB batch failed (attempt %d/%d), retrying: %s',
                attempt + 1, max_retries, error,
            )
            time.sleep(2 ** attempt)
    raise RuntimeError('max_retries must be >= 1')


def get_release_dates(
    entry_ids: list[str],
    batch_size: int = GRAPHQL_MAX_IDS,
    max_workers: int = 8,
) -> dict[str, str]:
    """Return initial release dates for many PDB entries via batched GraphQL.

    Args:
        entry_ids: Entry IDs to look up.
        batch_size: IDs per GraphQL request (RCSB hard limit is GRAPHQL_MAX_IDS).
        max_workers: Number of concurrent requests.

    Returns:
        Mapping of uppercase entry ID to release date (YYYY-MM-DD). Entries with
        no release date are omitted.
    """
    batches = [
        entry_ids[i:i + batch_size]
        for i in range(0, len(entry_ids), batch_size)
    ]
    dates: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_dates in executor.map(_fetch_release_dates_batch, batches):
            dates.update(batch_dates)
    return dates


class PdbReleaseDateData(PdbMetadata):
    """Whole-PDB initial release dates, cached and versioned in S3.

    Subclasses PdbMetadata to inherit its load-or-fetch caching. The base
    class assumes a single downloadable flatfile read via urlretrieve; release
    dates instead come from the RCSB holdings endpoint plus batched GraphQL,
    so retrieve_new_data is overridden. The cached table has two columns:
    entry_id (uppercase) and release_date (YYYY-MM-DD).
    """

    def __init__(self) -> None:
        super().__init__()
        self._date_lookup: dict[str, str] | None = None

    def retrieve_new_data(self) -> None:
        """Fetch every current PDB entry's release date from RCSB and cache it."""
        entry_ids = get_current_entry_ids()
        release_dates = get_release_dates(entry_ids)
        data_frame = pd.DataFrame(
            sorted(release_dates.items()),
            columns=['entry_id', 'release_date'],
        )
        self.data = data_frame
        self._date_lookup = None
        self._save_and_upload_df(data_frame)

    def get_release_date_dict(self) -> dict[str, str]:
        """Return a cached uppercase entry_id -> release_date (YYYY-MM-DD) map."""
        if self._date_lookup is None:
            self._date_lookup = dict(
                zip(
                    self.data['entry_id'].str.upper(),
                    self.data['release_date'],
                ),
            )
        return self._date_lookup

    def is_pre_cutoff(self, entry_id: str, cutoff: str) -> bool | None:
        """Return whether a PDB entry was released on or before the cutoff.

        Args:
            entry_id: PDB entry ID (case-insensitive).
            cutoff: Release-date cutoff (YYYY-MM-DD). Releases on or before this
                date count as pre-cutoff; the cutoff is treated as inclusive.

        Returns:
            True if released on/before the cutoff, False if strictly after it,
            or None if the entry is absent from the table.
        """
        release_date = self.get_release_date_dict().get(entry_id.upper())
        if release_date is None:
            return None
        return release_date <= cutoff


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build or refresh the whole-PDB release-date table cached in S3.',
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Force a fresh fetch from RCSB even if a cached table already exists.',
    )
    return parser


def main() -> None:
    """Build or refresh the whole-PDB release-date table.

    Instantiating PdbReleaseDateData loads the latest cached table from S3, or
    fetches it from RCSB (holdings endpoint + batched GraphQL) and uploads a
    dated CSV when no cache exists. Pass --refresh to force a fresh fetch even
    when a cached table is present (e.g. to pick up newly released entries).
    """
    parser = parse_args()
    args = parser.parse_args()
    data = PdbReleaseDateData()
    if args.refresh:
        data.retrieve_new_data()
    print(f'{len(data.data):,} PDB entries with release dates')
    print(
        f'release_date range: {data.data.release_date.min()} .. {data.data.release_date.max()}',
    )


if __name__ == '__main__':
    main()
