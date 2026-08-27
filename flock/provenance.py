from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import date
from datetime import datetime
from datetime import timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import pandas as pd

from flock import FLOCK_VERSION

OUTPUT_PROVENANCE_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024
# Distributions whose version can change an output, so worth recording beside it.
_VERSION_DISTRIBUTIONS = (
    'biopython',
    'boto3',
    'flock',
    'numpy',
    'pandas',
    'scipy',
)


def _run_git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ['git', *arguments],
            cwd=Path(__file__).parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _get_git_commit(package: str = 'flock') -> str:
    """Return the short HEAD sha, marked when the package does not match it.

    A bare sha is a promise that checking it out reproduces the file. Writing
    one from a worktree that does not match HEAD breaks that promise silently,
    and the artifact outlives the worktree that made it - the first build of the
    leaked comparison arm recorded a HEAD that did not yet contain the module
    that wrote it.

    That incident is why the check reads `git status --porcelain` and not
    `git diff HEAD`: the offending module was brand new and therefore
    untracked, which a diff against HEAD does not see at all. The status output
    is hashed together with the diff so the identifier still changes when a
    tracked file is edited without any file being added.

    Scoped to the package directory rather than the whole repo, because the
    claim being made is that this code produced this file. An untracked
    notebook or editor config says nothing about that, and letting it mark
    every build dirty would make the marker mean nothing.

    Args:
        package: Repo-relative directory whose state the sha is claiming to
            describe.

    Returns:
        The short sha, 'unknown' if git is unavailable, or
        '<sha>-dirty-<hash>' when the package differs from HEAD in any way,
        additions included.
    """
    commit = _run_git('rev-parse', '--short', 'HEAD')
    if commit is None:
        return 'unknown'
    status = _run_git('status', '--porcelain', '--', package)
    diff = _run_git('diff', 'HEAD', '--', package)
    if status is None or diff is None:
        return commit
    if not status and not diff:
        return commit
    fingerprint = hashlib.sha256(f'{status}\n{diff}'.encode()).hexdigest()
    return f'{commit}-dirty-{fingerprint[:8]}'


def write_csv_with_provenance(
    df: pd.DataFrame,
    path: str,
    sources: dict[str, str],
    sep: str = ',',
    header: bool = True,
    index: bool = False,
) -> None:
    """Write a DataFrame to CSV/TSV with provenance metadata in # comment lines.

    Writes flock_version, date, commit, and all entries from sources as
    # key=value lines before the CSV data. Readers skip these lines with
    pandas.read_csv(..., comment='#').

    Args:
        df: DataFrame to write.
        path: Local file path to write to.
        sources: Mapping of label -> filename (not full S3 path) for each input
            that produced this file.
        sep: Field separator (default ',').
        header: Whether to write column headers (default True).
        index: Whether to write the row index (default False).
    """
    meta = {
        'flock_version': FLOCK_VERSION,
        'date': date.today().isoformat(),
        'commit': _get_git_commit(),
        **sources,
    }
    with open(path, 'w') as file_out:
        for key, value in meta.items():
            file_out.write(f'# {key}={value}\n')
        df.to_csv(file_out, sep=sep, header=header, index=index)


def _provenance_from_lines(lines: Iterable[str]) -> dict[str, str]:
    """Collect '# key=value' lines into a record, stopping at the first data line.

    Args:
        lines: The file's lines, in order. Consumed lazily, so a file object reads only
            as far as the header.

    Returns:
        Mapping of provenance key to value, empty if there is no header.
    """
    record: dict[str, str] = {}
    for line in lines:
        if not line.startswith('#'):
            break
        key, _, value = line[1:].strip().partition('=')
        if key:
            record[key.strip()] = value.strip()
    return record


def read_csv_provenance(path: str) -> dict[str, str]:
    """Read the provenance header from a CSV written by write_csv_with_provenance.

    Values are returned as strings exactly as written; nested values were JSON-encoded on
    the way out and are left encoded here.

    Args:
        path: Local path to a CSV carrying leading '# key=value' comment lines.

    Returns:
        Mapping of provenance key to value, empty if the file has no header.
    """
    with open(path) as file_in:
        return _provenance_from_lines(file_in)


def csv_provenance_lines(path: str) -> int:
    """Count the leading '# key=value' provenance lines a CSV carries.

    Counted rather than passing comment='#' to the reader, which would also truncate any
    field containing a hash and so could silently corrupt a value.

    Args:
        path: Local path to the CSV.

    Returns:
        Number of leading comment lines, zero for a gzipped file.
    """
    if path.endswith('.gz'):
        return 0
    count = 0
    with open(path) as file_in:
        for line in file_in:
            if not line.startswith('#'):
                break
            count += 1
    return count


def _utc_timestamp(timestamp: float | None = None) -> str:
    """Return an ISO-8601 UTC timestamp.

    Args:
        timestamp: Optional Unix timestamp. Defaults to the current time.

    Returns:
        ISO-8601 timestamp with a UTC offset.
    """
    if timestamp is None:
        value = datetime.now(timezone.utc)
    else:
        value = datetime.fromtimestamp(timestamp, timezone.utc)
    return value.isoformat()


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file.

    Args:
        path: File to hash.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        while chunk := input_file.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_input_file(
    role: str,
    input_path: str | Path,
    index: int,
) -> dict[str, Any]:
    """Describe and checksum one input without modifying it.

    An S3 or other remote path is recorded by its path alone, with a marker
    that it was not hashed locally, so building provenance never needs AWS.

    Args:
        role: Stable label describing the input's purpose.
        input_path: Local path or external URI.
        index: Position when a role contains multiple inputs.

    Returns:
        JSON-serializable input description.
    """
    path_text = str(input_path)
    record: dict[str, Any] = {'role': role, 'path': path_text}
    if index > 0:
        record['index'] = index
    if path_text.startswith('s3://'):
        record.update({'external': True, 'hashed_locally': False})
        return record
    if '://' in path_text:
        record.update({'external': True, 'exists': False})
        return record

    path = Path(input_path).expanduser()
    record['resolved_path'] = str(path.resolve())
    if not path.is_file():
        record['exists'] = False
        return record
    file_stat = path.stat()
    record.update({
        'exists': True,
        'size_bytes': file_stat.st_size,
        'modified_at_utc': _utc_timestamp(file_stat.st_mtime),
        'sha256': _sha256_file(path),
    })
    return record


def _config_paths(argv: Sequence[str]) -> list[str]:
    """Extract config-file arguments from a command line.

    Args:
        argv: Command-line arguments to inspect.

    Returns:
        Config paths supplied with ``-c`` or ``--config``.
    """
    paths = []
    for argument_index, argument in enumerate(argv):
        if argument in ('-c', '--config') and argument_index + 1 < len(argv):
            paths.append(argv[argument_index + 1])
        elif argument.startswith('--config='):
            paths.append(argument.split('=', 1)[1])
    return paths


def _input_records(
    input_paths: Mapping[str, str | Path | Sequence[str | Path] | None],
    argv: Sequence[str],
) -> list[dict[str, Any]]:
    """Return ordered provenance records for user and config inputs.

    Args:
        input_paths: Input roles mapped to paths.
        argv: Command arguments used to discover config inputs.

    Returns:
        Ordered input provenance records.
    """
    records = []
    configured_paths = dict(input_paths)
    config_paths = _config_paths(argv)
    if config_paths and 'config' not in configured_paths:
        configured_paths['config'] = config_paths
    for role, paths in configured_paths.items():
        if paths is None:
            continue
        if isinstance(paths, (str, Path)):
            role_paths: Sequence[str | Path] = [paths]
        else:
            role_paths = paths
        for index, input_path in enumerate(role_paths):
            records.append(_describe_input_file(role, input_path, index))
    return records


def _sanitise_git_origin(origin: str | None) -> str | None:
    """Remove embedded credentials from a Git origin URL.

    Args:
        origin: Git remote URL, if available.

    Returns:
        Credential-free remote URL.
    """
    if origin is None:
        return None
    parsed_origin = urlsplit(origin)
    if not parsed_origin.scheme or parsed_origin.hostname is None:
        return origin
    host = parsed_origin.hostname
    if parsed_origin.port is not None:
        host = f'{host}:{parsed_origin.port}'
    return urlunsplit((
        parsed_origin.scheme,
        host,
        parsed_origin.path,
        '',
        '',
    ))


def _package_version(source_name: str, source_version: str | None) -> str | None:
    """Resolve the version for the source package.

    Args:
        source_name: Distribution name.
        source_version: Explicit version override.

    Returns:
        Package version when installed or explicitly supplied.
    """
    if source_version is not None:
        return source_version
    try:
        return metadata.version(source_name)
    except metadata.PackageNotFoundError:
        return None


def _source_identity(
    source_name: str,
    source_version: str | None,
    repo_root: Path,
) -> dict[str, Any]:
    """Return package and Git identity for the code creating an output.

    Args:
        source_name: Package or application name.
        source_version: Explicit package version, if any.
        repo_root: Git checkout containing the source.

    Returns:
        JSON-serializable source identity.
    """
    git_commit = _run_git('-C', str(repo_root), 'rev-parse', 'HEAD')
    return {
        'name': source_name,
        'version': _package_version(source_name, source_version),
        'git_origin': _sanitise_git_origin(
            _run_git('-C', str(repo_root), 'remote', 'get-url', 'origin'),
        ),
        'git_commit': git_commit,
        'git_branch': _run_git(
            '-C', str(repo_root), 'rev-parse', '--abbrev-ref', 'HEAD',
        ),
        'git_commit_date': _run_git(
            '-C', str(repo_root), 'show', '-s', '--format=%cI', 'HEAD',
        ),
        'identity_source': 'git' if git_commit else None,
    }


def _distribution_versions(
    distribution_names: Sequence[str],
) -> dict[str, str | None]:
    """Return installed versions for selected distributions.

    Args:
        distribution_names: Distributions relevant to the workflow.

    Returns:
        Distribution names mapped to installed versions or ``None``.
    """
    versions: dict[str, str | None] = {}
    for distribution_name in distribution_names:
        try:
            versions[distribution_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            versions[distribution_name] = None
    return versions


def _json_safe(value: Any) -> Any:
    """Convert common scientific and path values into JSON-safe structures.

    Args:
        value: Value to convert.

    Returns:
        JSON-safe equivalent.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'tolist'):
        return _json_safe(value.tolist())
    return str(value)


def build_output_provenance(
    workflow: str,
    parameters: Mapping[str, Any],
    input_paths: Mapping[str, str | Path | Sequence[str | Path] | None],
    extra: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
    repo_root: Path | None = None,
    source_name: str = 'flock',
    source_version: str | None = None,
    distribution_names: Sequence[str] = _VERSION_DISTRIBUTIONS,
) -> dict[str, Any]:
    """Build reproducible, versioned provenance for one output.

    Args:
        workflow: Stable workflow identifier.
        parameters: Fully resolved workflow parameters.
        input_paths: Input-file roles mapped to one path or a sequence of paths.
        extra: Additional workflow metadata retained at top level.
        argv: Command arguments to record. Defaults to the current process arguments.
        repo_root: Checkout used for Git identity. Defaults to the flock repo root.
        source_name: Package or application producing the output.
        source_version: Optional explicit source package version.
        distribution_names: Scientific distributions whose versions should be recorded.

    Returns:
        JSON-serializable provenance dictionary.
    """
    command_argv = list(sys.argv if argv is None else argv)
    source_root = (
        Path(repo_root) if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    provenance = dict(_json_safe(extra or {}))
    provenance.update({
        'schema_version': OUTPUT_PROVENANCE_SCHEMA_VERSION,
        'created_at_utc': _utc_timestamp(),
        'workflow': workflow,
        'source': _source_identity(
            source_name=source_name,
            source_version=source_version,
            repo_root=source_root,
        ),
        'command': {
            'argv': command_argv,
            'rendered': shlex.join(command_argv),
            'working_directory': str(Path.cwd()),
        },
        'parameters': _json_safe(parameters),
        'inputs': _input_records(input_paths, command_argv),
        'software': {
            'python': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'platform': platform.platform(),
            'distributions': _distribution_versions(distribution_names),
        },
    })
    return provenance


def write_output_provenance(
    output_path: str | Path,
    provenance: Mapping[str, Any],
) -> None:
    """Atomically write provenance as formatted JSON.

    Args:
        output_path: Destination JSON file.
        provenance: Provenance record to write.
    """
    provenance_path = Path(output_path)
    temporary_path = provenance_path.with_name(f'{provenance_path.name}.tmp')
    with temporary_path.open('w') as provenance_file:
        json.dump(provenance, provenance_file, indent=2, sort_keys=True)
        provenance_file.write('\n')
        provenance_file.flush()
        os.fsync(provenance_file.fileno())
    os.replace(temporary_path, provenance_path)
