from __future__ import annotations

import logging
import os
import re
import tempfile
import urllib.request
from datetime import date as _date

import pandas as pd

from flock import INTACT_VERSION
from flock.aws import upload_file_to_s3
from flock.logging_utils import setup_logging
from flock.paths import FLOCK_S3
from flock.paths import INTACT_S3
from flock.paths import make_dated_filename
from flock.provenance import write_csv_with_provenance

# Points to the latest IntAct release.
INTACT_FTP_URL = (
    'https://ftp.ebi.ac.uk/pub/databases/intact/current/'
    'psimitab/intact-micluster.txt'
)

# Regex to extract a valid UniProt accession from a PSI-MITAB identifier.
# UniProt accessions are 6 or 10 characters with a digit in position 2.
# Handles isoforms ("uniprotkb:P12345-2") and feature chains
# ("uniprotkb:P12345-PRO_...") by stopping at the hyphen.
# Rejects gene names that also appear under the uniprotkb: prefix in micluster.
UNIPROT_RE = re.compile(
    r'^uniprotkb:([A-Z][0-9][A-Z0-9]{3}[0-9](?:[A-Z0-9]{4})?)\b',
)

# PSI-MITAB 2.7 column indices used in the micluster file
ALT_ID_A_COL = 2
ALT_ID_B_COL = 3
NEGATIVE_FLAG_COL = 35
MIN_COLUMNS = NEGATIVE_FLAG_COL + 1


def extract_uniprot_accession(field: str) -> str | None:
    """Extract the base UniProt accession from a PSI-MITAB interactor field.

    Args:
        field: Raw interactor identifier, e.g. "uniprotkb:P12345-2".

    Returns:
        Base accession string (e.g. "P12345"), or None if the field is not a
        UniProt identifier.
    """
    match = UNIPROT_RE.match(field)
    return match.group(1) if match else None


def parse_intact_pairs(file_path: str) -> set[tuple[str, str]]:
    """Parse the IntAct micluster file and extract positive UniProt pairs.

    Reads the file line-by-line to keep memory usage manageable (~6GB). UniProt
    accessions are extracted from the alternative ID columns (2 and 3, 0-indexed)
    since the micluster format uses IntAct IDs as primary identifiers. Only rows
    where both interactors map to UniProt and the negative flag (column 35) is
    not "true" are included. Pair order is normalised (a <= b).

    Args:
        file_path: Path to the intact-micluster.txt file.

    Returns:
        Set of (uniprot_a, uniprot_b) tuples with a <= b.
    """
    logger = logging.getLogger(__name__)
    pairs: set[tuple[str, str]] = set()
    skipped_non_uniprot = 0
    skipped_negative = 0
    total = 0

    with open(file_path, encoding='utf-8') as intact_file:
        for line in intact_file:
            if line.startswith('#'):
                continue
            total += 1
            cols = line.rstrip('\n').split('\t', MIN_COLUMNS)
            if len(cols) < MIN_COLUMNS:
                continue

            # Skip rows flagged as negative interactions in IntAct
            if cols[NEGATIVE_FLAG_COL].lower() == 'true':
                skipped_negative += 1
                continue

            acc_a = extract_uniprot_accession(cols[ALT_ID_A_COL])
            acc_b = extract_uniprot_accession(cols[ALT_ID_B_COL])
            if acc_a is None or acc_b is None:
                skipped_non_uniprot += 1
                continue

            pair = (acc_a, acc_b) if acc_a <= acc_b else (acc_b, acc_a)
            pairs.add(pair)

            if total % 500_000 == 0:
                logger.info(
                    'Processed %d rows, %d unique pairs so far',
                    total, len(pairs),
                )

    logger.info(
        'Finished parsing: %d total rows, %d unique positive UniProt pairs',
        total, len(pairs),
    )
    logger.info(
        'Skipped: %d non-UniProt, %d negative',
        skipped_non_uniprot, skipped_negative,
    )
    return pairs


def main(version: str = INTACT_VERSION, date: str | None = None) -> None:
    """Download IntAct clustered interactions, archive raw to S3, extract UniProt pairs, upload to S3.

    Args:
        version: Version string for the IntAct raw archive, e.g. 'v1'.
        date: Date the data was downloaded (YYYY-MM-DD). Defaults to today.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    upload_date = date or _date.today().isoformat()
    raw_filename = f'intact-micluster_{version}_{upload_date}.txt'
    raw_s3_path = INTACT_S3 + raw_filename

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = os.path.join(tmpdir, 'intact-micluster.txt')

        logger.info('Downloading IntAct micluster from %s ...', INTACT_FTP_URL)
        urllib.request.urlretrieve(INTACT_FTP_URL, local_file)
        size_mb = os.path.getsize(local_file) / (1024 * 1024)
        logger.info('Downloaded %.1f MB', size_mb)

        # upload_file_to_s3 uses the local filename as the S3 key, so rename before uploading
        local_versioned = os.path.join(tmpdir, raw_filename)
        os.rename(local_file, local_versioned)
        logger.info('Archiving raw download to %s', raw_s3_path)
        upload_file_to_s3(local_versioned, INTACT_S3)

        logger.info('Parsing positive UniProt pairs...')
        pairs = parse_intact_pairs(local_versioned)

        df = pd.DataFrame(sorted(pairs), columns=['uniprot_a', 'uniprot_b'])
        logger.info('Lookup table: %d rows', len(df))

        output_filename = make_dated_filename(
            'intact_positive_pairs', INTACT_VERSION, '.csv', upload_date,
        )
        local_out = os.path.join(tmpdir, output_filename)
        write_csv_with_provenance(
            df, local_out, sources={
                'intact_raw_source': raw_filename,
            },
        )

        logger.info('Uploading to %s', FLOCK_S3 + output_filename)
        upload_file_to_s3(local_out, FLOCK_S3)

    logger.info('Done.')


if __name__ == '__main__':
    main()
