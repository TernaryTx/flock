from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3Bucket = Any


def get_bucket_name_key_from_path(bucket_path: str) -> tuple[str, str]:
    """Parse an S3 path into its bucket name and object key/prefix components.

    Args:
        bucket_path: Full S3 path (e.g. 's3://my-bucket/path/to/objects').

    Returns:
        Tuple of (bucket name, object key).
    """
    bucket_name_key_str = bucket_path.replace('s3://', '')
    if '/' in bucket_name_key_str:
        bucket_name, key = bucket_name_key_str.split('/', 1)
        return bucket_name, key
    return bucket_name_key_str, ''


def get_authenticated_bucket(bucket_path: str) -> tuple[S3Bucket, str]:
    """Set up an authenticated bucket resource based on an S3 path.

    Args:
        bucket_path: S3 bucket path (e.g. 's3://my-bucket/folder/test.txt').

    Returns:
        Tuple of (authenticated Bucket resource, object key/prefix).
    """
    s3 = boto3.resource('s3')
    bucket_name, key = get_bucket_name_key_from_path(bucket_path)
    return s3.Bucket(bucket_name), key


def download_file_from_s3(s3_path: str, filename: str, local_path: str) -> None:
    """Download a file from an AWS S3 bucket.

    Args:
        s3_path: Path to the bucket/folder (e.g. 's3://my-bucket/my-folder').
        filename: Name of the file in the bucket (appended to s3_path).
        local_path: Local path to save the downloaded file to.
    """
    if os.path.dirname(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    bucket, prefix = get_authenticated_bucket(s3_path)
    full_key = os.path.join(prefix, filename)
    if full_key.startswith('/'):
        full_key = full_key[1:]
    bucket.download_file(full_key, local_path)


def upload_file_to_s3(local_file: str, s3_folder: str) -> None:
    """Upload a local file to an AWS S3 bucket.

    Args:
        local_file: Path of the local file to upload.
        s3_folder: Path to the bucket folder to upload the file to
            (e.g. 's3://my-bucket/my-folder').
    """
    bucket, prefix = get_authenticated_bucket(s3_folder)
    file_name = os.path.basename(local_file)
    key = os.path.join(prefix, file_name)
    if key.startswith('/'):
        key = key[1:]
    bucket.upload_file(Filename=local_file, Key=key)


def upload_folder_to_s3(local_folder: str, s3_destination: str) -> None:
    """Upload the contents of a local folder recursively to an S3 prefix.

    Per-file failures are logged rather than raised, so one bad object does not
    abort the rest of the mirror; callers that need certainty the mirror is
    complete should verify the prefix contents afterwards.

    Args:
        local_folder: Path to the local folder to upload.
        s3_destination: S3 prefix to upload the folder contents to
            (e.g. 's3://my-bucket/my-folder').
    """
    bucket, prefix = get_authenticated_bucket(s3_destination)
    for root, _, file_names in os.walk(local_folder):
        for file_name in file_names:
            local_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_path, local_folder)
            key = os.path.join(prefix, relative_path)
            if key.startswith('/'):
                key = key[1:]
            try:
                bucket.upload_file(Filename=local_path, Key=key)
            except (BotoCoreError, ClientError) as err:
                logger.error(
                    'Failed to upload %s to s3://%s/%s: %s',
                    local_path, bucket.name, key, err,
                )


def download_folder_contents(
        bucket_input_folder: str,
        local_destination_folder: str,
) -> None:
    """Download all contents of an S3 folder to a local folder.

    Args:
        bucket_input_folder: S3 folder path (e.g. 's3://my-bucket/my-folder').
        local_destination_folder: Local folder to download files into. Created
            (along with any subfolders) if it does not exist.
    """
    bucket, prefix = get_authenticated_bucket(bucket_input_folder)
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith('/'):
            continue
        relative_path = obj.key[len(prefix):]
        local_path = os.path.join(local_destination_folder, relative_path)
        if os.path.dirname(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        bucket.download_file(obj.key, local_path)


def load_dataframe_from_s3(s3_path: str, filename: str) -> pd.DataFrame:
    """Load a CSV file from an AWS S3 bucket, without overwriting local files.

    Args:
        s3_path: Path to the bucket (e.g. 's3://my-bucket/my-folder').
        filename: Name of the CSV file in the bucket.

    Returns:
        DataFrame containing the data from the CSV file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, filename)
        download_file_from_s3(s3_path, filename, local_path)
        return pd.read_csv(local_path)


def list_files_from_s3(s3_path: str, recursive: bool = False) -> list[str]:
    """List all files and folders from a given S3 bucket path.

    Args:
        s3_path: Path to the bucket (e.g. 's3://my-bucket/my-folder').
        recursive: Whether to descend into all subdirectories and return every
            file within them.

    Returns:
        List of S3 paths (str) to files and folders.
    """
    bucket_name, prefix = get_bucket_name_key_from_path(s3_path)
    client = boto3.client('s3')

    if prefix and not prefix.endswith('/'):
        prefix += '/'
    results = []
    if recursive:
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in pages:
            for obj in page.get('Contents', []):
                results.append(f"s3://{bucket_name}/{obj['Key']}")
    else:
        response = client.list_objects_v2(
            Bucket=bucket_name, Prefix=prefix, Delimiter='/',
        )
        results.extend(
            f"s3://{bucket_name}/{obj['Key']}"
            for obj in response.get('Contents', [])
        )
        results.extend(
            f"s3://{bucket_name}/{item['Prefix']}"
            for item in response.get('CommonPrefixes', [])
        )

    return results
