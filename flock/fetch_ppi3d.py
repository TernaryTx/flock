from __future__ import annotations

import argparse
import logging
import os
from datetime import date as _date

from flock import PPI3D_VERSION
from flock.aws import list_files_from_s3
from flock.aws import upload_folder_to_s3
from flock.logging_utils import setup_logging
from flock.paths import PPI3D_S3
from flock.ppi3d import fetch_range
from flock.ppi3d import get_database_date
from flock.ppi3d import read_interfaces
from flock.ppi3d import scaled_windows

# First PDB release dates. PPI3D indexes the whole archive, so a full pull starts
# here rather than at any Flock-specific cutoff: the interface clusters used for
# the homolog leakage flag are only meaningful if every member is present,
# including pre-cutoff ones.
PPI3D_EPOCH = '1973-01-01'


def fetch_to_s3(
        start: str,
        end: str,
        cache_dir: str,
        version: str = PPI3D_VERSION,
        pull_date: str | None = None,
        upload: bool = True,
        **criteria_kwargs,
) -> str:
    """Fetch a PPI3D date range into a local cache and mirror it to S3.

    Args:
        start: Inclusive start date (YYYY-MM-DD).
        end: Inclusive end date (YYYY-MM-DD).
        cache_dir: Local directory to fetch into. Existing windows are reused.
        version: Version string for the S3 prefix.
        pull_date: Date stamp for the S3 prefix. Defaults to today.
        upload: Whether to mirror the cache to S3 once fetching completes.
        **criteria_kwargs: Passed through to build_criteria().

    Returns:
        The S3 prefix the pull was written to (or would be written to).
    """
    logger = logging.getLogger(__name__)
    pull_date = pull_date or _date.today().isoformat()
    s3_prefix = PPI3D_S3 + f'filteredforflock_{version}_{pull_date}/'

    windows = scaled_windows(start, end)
    logger.info(
        'Pull %s..%s in %d windows -> %s', start, end, len(windows), s3_prefix,
    )
    # Whatever was fetched is worth persisting even if the run then fell over —
    # a late failure must not discard hours of successful windows.
    try:
        csv_paths = fetch_range(
            start, end, cache_dir, windows=windows, **criteria_kwargs,
        )
        interfaces = read_interfaces(csv_paths)
        n_interfaces = len(interfaces)
        n_entries = interfaces['pdb_id'].nunique() if n_interfaces else 0
        logger.info(
            'Fetched %d interfaces across %d PDB entries',
            n_interfaces, n_entries,
        )
        # An incomplete pull is still worth keeping, but it must not be mistaken
        # for a whole one: re-running the same command fills only the gaps.
        if len(csv_paths) < len(windows):
            logger.error(
                'INCOMPLETE PULL: %d of %d windows retrieved. Re-run with the '
                'same arguments to fetch the remainder.',
                len(csv_paths), len(windows),
            )
    finally:
        if upload:
            logger.info('Uploading %s to %s', cache_dir, s3_prefix)
            upload_folder_to_s3(cache_dir, s3_prefix)
            verify_upload(cache_dir, s3_prefix)
    return s3_prefix


def verify_upload(cache_dir: str, s3_prefix: str) -> None:
    """Check that every cached file reached S3.

    upload_folder_to_s3 logs per-file failures without raising, so an expired
    credential or a permissions problem otherwise ends with the script cheerfully
    reporting success over an empty prefix. Comparing names is the only way to
    know the mirror is real.

    Args:
        cache_dir: Local directory that was uploaded.
        s3_prefix: S3 prefix it was uploaded to.

    Raises:
        RuntimeError: If any local file is absent from S3.
    """
    logger = logging.getLogger(__name__)
    local = {
        name for name in os.listdir(cache_dir)
        if os.path.isfile(os.path.join(cache_dir, name))
    }
    remote = {
        path.rsplit('/', 1)[-1]
        for path in list_files_from_s3(s3_prefix, recursive=True)
    }
    absent = sorted(local - remote)
    if absent:
        raise RuntimeError(
            f'{len(absent)} of {len(local)} files did not reach {s3_prefix}; '
            f'first missing: {absent[:3]}. The local cache is intact — re-run '
            f'the same command once the cause is fixed.',
        )
    logger.info('Verified %d files present at %s', len(local), s3_prefix)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fetch PPI3D interface data by release-date window and mirror it to S3.',
    )
    parser.add_argument(
        '--start', default=PPI3D_EPOCH,
        help='Inclusive start release date (YYYY-MM-DD).',
    )
    parser.add_argument(
        '--end', default=None,
        help='Inclusive end release date (YYYY-MM-DD). Defaults to the PDB '
             'release date of PPI3D\'s current snapshot; requesting beyond it '
             'is rejected by the server.',
    )
    parser.add_argument(
        '--cache-dir', default='ppi3d_cache',
        help='Local directory to fetch into; existing windows are reused.',
    )
    parser.add_argument(
        '--version', default=PPI3D_VERSION,
        help='Version string for the S3 prefix.',
    )
    parser.add_argument(
        '--pull-date', default=None,
        help='Date stamp for the S3 prefix (YYYY-MM-DD). Defaults to today.',
    )
    parser.add_argument(
        '--no-upload', action='store_true',
        help='Fetch into the local cache without mirroring to S3.',
    )
    return parser


def main() -> None:
    """Pull PPI3D interface data and mirror it to S3.

    PPI3D has no REST API and serves bulk data one release-date window at a time,
    so a pull is a series of form-POST jobs polled to completion. Windows are
    sized by scaled_windows: a quarter of recent PDB holds more structures than
    several years of the 1980s, so early years use wider windows and recent years
    narrower ones. Requests are serial with a pause between them, since PPI3D is
    one small academic server.

    Each window contributes three files to the cache: the gzipped CSV of
    interfaces, the JSON criteria that produced it, and the server log recording
    how many interfaces were retrieved and filtered. All three are mirrored to
    s3://.../PPI3D/filteredforflock_<version>_<pull_date>/ so downstream builds
    do not depend
    on the server being reachable.

    The cache is resumable: a window whose CSV is already present is skipped, so
    an interrupted pull can be restarted with the same arguments. A full pull from
    1973 takes several hours.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = parse_args()
    args = parser.parse_args()

    end = args.end or get_database_date()
    logger.info('PPI3D snapshot covers PDB releases up to %s', end)

    os.makedirs(args.cache_dir, exist_ok=True)
    s3_prefix = fetch_to_s3(
        start=args.start,
        end=end,
        cache_dir=args.cache_dir,
        version=args.version,
        pull_date=args.pull_date,
        upload=not args.no_upload,
    )
    logger.info('Done. Pull at %s', s3_prefix)


if __name__ == '__main__':
    main()
