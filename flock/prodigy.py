# Client for PRODIGY-cryst, the BIO/XTAL interface classifier PINDER's `label`
# column comes from. The tool pins scikit-learn==0.22 and numpy<1.21, which cannot
# coexist with the `flock` env's numpy 2.x, so it is installed in a separate conda
# env and driven as a subprocess. See the PPI3D section of the README for the
# env setup.
#
# Named prodigy.py, not prodigy_cryst.py: the runner it launches lives in this
# same directory, which lands on sys.path[0] in the subprocess, so a module here
# sharing the installed package's name would shadow it.
from __future__ import annotations

import logging
import os
import subprocess
import time

import pandas as pd
import requests

from flock.ppi3d import make_session

PRODIGY_ENV = 'prodigy'
RUNNER_PATH = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__),
    ), '_prodigy_runner.py',
)

# Columns the runner emits, in order.
RUNNER_COLUMNS = (
    'pdb_path', 'prodigy_class', 'prodigy_prob_bio',
    'prodigy_prob_xtal', 'prodigy_contacts', 'prodigy_link_density',
    'prodigy_error',
)


def interface_filename(download_url: str) -> str:
    """Return the local filename for a PPI3D interface coordinate URL.

    Args:
        download_url: Value of a PPI3D row's download_url column.

    Returns:
        Basename, e.g. 'protein_protein-9mbz-1-9mbz_A-1-9mbz_J-1.pdb'.
    """
    return download_url.rsplit('/', 1)[-1]


def download_interfaces(
        download_urls: list[str],
        dest_dir: str,
        pause_seconds: float = 0.3,
        session: requests.Session | None = None,
) -> list[str]:
    """Download PPI3D interface coordinate files, skipping any already cached.

    Serial with a pause: this is the same small academic server the bulk download
    comes from, and an interface-per-file fetch is far more requests than the
    windowed pull. Callers should classify only the interfaces they actually
    need rather than the whole database.

    Args:
        download_urls: PPI3D download_url values.
        dest_dir: Directory to cache the PDB files in.
        pause_seconds: Delay between consecutive downloads.
        session: Requests session to reuse; a retrying one is created if omitted.

    Returns:
        Local paths of successfully cached files, in input order.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(dest_dir, exist_ok=True)
    owns_session = session is None
    session = session or make_session()
    paths = []
    try:
        for index, url in enumerate(download_urls, start=1):
            local_path = os.path.join(dest_dir, interface_filename(url))
            if os.path.exists(local_path):
                paths.append(local_path)
                continue
            try:
                response = session.get(url, timeout=120)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning('Download failed for %s: %s', url, exc)
                continue
            with open(local_path, 'wb') as handle:
                handle.write(response.content)
            paths.append(local_path)
            if index % 100 == 0:
                logger.info(
                    'Downloaded %d/%d interfaces',
                    index, len(download_urls),
                )
            time.sleep(pause_seconds)
    finally:
        if owns_session:
            session.close()
    return paths


def classify_interfaces(
        pdb_paths: list[str],
        env: str = PRODIGY_ENV,
        list_path: str | None = None,
) -> pd.DataFrame:
    """Run PRODIGY-cryst over interface PDB files via the legacy conda env.

    The whole batch runs in a single subprocess: the classifier costs a couple of
    seconds per structure but roughly as long again to import, so per-file
    invocation would roughly double the cost.

    Args:
        pdb_paths: Local paths to interface coordinate files.
        env: Name of the conda env holding prodigy-cryst.
        list_path: Where to write the batch's file list. Defaults to a file
            alongside the first input.

    Returns:
        DataFrame with RUNNER_COLUMNS. Rows that failed carry
        prodigy_class == 'ERROR'.

    Raises:
        RuntimeError: If the subprocess itself fails.
    """
    logger = logging.getLogger(__name__)
    if not pdb_paths:
        return pd.DataFrame(columns=list(RUNNER_COLUMNS))

    if list_path is None:
        list_path = os.path.join(
            os.path.dirname(pdb_paths[0]) or '.', '_prodigy_batch.txt',
        )
    with open(list_path, 'w') as handle:
        handle.write('\n'.join(pdb_paths) + '\n')

    logger.info(
        'Classifying %d interfaces in conda env %r',
        len(pdb_paths), env,
    )
    completed = subprocess.run(
        [
            'conda', 'run', '--no-capture-output', '-n',
            env, 'python', RUNNER_PATH, list_path,
        ],
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'PRODIGY-cryst runner failed (exit {completed.returncode}): '
            f'{completed.stderr[-2000:]}',
        )

    rows = []
    for line in completed.stdout.splitlines():
        fields = line.split('\t')
        if len(fields) == len(RUNNER_COLUMNS):
            rows.append(fields)
    frame = pd.DataFrame(rows, columns=list(RUNNER_COLUMNS))
    for column in ('prodigy_prob_bio', 'prodigy_prob_xtal', 'prodigy_link_density'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['prodigy_contacts'] = pd.to_numeric(
        frame['prodigy_contacts'], errors='coerce',
    ).astype('Int64')

    n_error = (frame['prodigy_class'] == 'ERROR').sum()
    if n_error:
        logger.warning(
            '%d of %d interfaces failed to classify',
            n_error, len(frame),
        )
    return frame
