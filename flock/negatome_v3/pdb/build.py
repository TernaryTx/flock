from __future__ import annotations

import datetime
import logging
import os
from argparse import ArgumentParser
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from tempfile import mkdtemp

import pandas as pd
from Bio.PDB.PDBExceptions import PDBConstructionException

from flock import NEGATOME_PDB_VERSION
from flock.aws import upload_file_to_s3
from flock.logging_utils import setup_logging
from flock.negatome_v3.pdb.metadata import _fetch_pdb_metadata
from flock.negatome_v3.pdb.structure_filtering import get_non_interacting_protein_pairs
from flock.negatome_v3.pdb.structure_filtering import measure_interchain_dist
from flock.paths import NEGATOME_S3
from flock.structure import download_all_bio_assemblies
from flock.structure import PdbFile
from flock.structure import read_pdb


_MAX_PDB_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


def _get_next_chunk(relevant_pdbs: list[str], pdb_files_dir: str, chunk_size: int) -> tuple[list[str], int]:
    already_downloaded = {
        os.path.splitext(
            f,
        )[0].lower()[:4] for f in os.listdir(pdb_files_dir)
    }
    remaining = [
        pdb for pdb in relevant_pdbs if pdb.lower()
        not in already_downloaded
    ]
    return remaining[:chunk_size], len(remaining)


def _process_pdb_file(
    pdb_file: PdbFile,
    pdb_chain_to_uniprot: dict[str, dict[str, set[str]]],
    max_file_size_bytes: int = _MAX_PDB_FILE_SIZE_BYTES,
) -> list[tuple[str, str, str]]:
    if os.path.getsize(pdb_file.path) > max_file_size_bytes:
        return []
    try:
        pdb_structure = read_pdb(pdb_file)
    except PDBConstructionException as exc:
        # Skip malformed cifs (e.g. duplicate atom records) rather than
        # letting the exception bubble through future.result() and kill
        # the whole job.
        logging.warning(
            'Skipping %s: BioPython parse failed: %s', pdb_file.path, exc,
        )
        return []
    interchain_dist = measure_interchain_dist(pdb_structure)
    pdb_id = pdb_file.return_pdb_id().lower()
    chain_to_uniprot = pdb_chain_to_uniprot.get(pdb_id, {})
    protein_pairs = get_non_interacting_protein_pairs(
        interchain_dist, chain_to_uniprot,
    )
    return [(u_a, u_b, pdb_id) for (u_a, u_b) in protein_pairs]


def _extract_negatome_pairs(
    pdb_files: list,
    pdb_chain_to_uniprot: dict[str, dict[str, set[str]]],
    max_file_size_bytes: int = _MAX_PDB_FILE_SIZE_BYTES,
) -> list[tuple[str, str, str]]:
    negatome_pairs: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(
                _process_pdb_file, f, pdb_chain_to_uniprot,
                max_file_size_bytes,
            ) for f in pdb_files
        ]
        for future in as_completed(futures):
            negatome_pairs.extend(future.result())
    return negatome_pairs


def _merge_pdb_codes(codes: pd.Series) -> str:
    all_codes: set[str] = set()
    for code_str in codes:
        all_codes.update(code_str.split(','))
    return ','.join(sorted(all_codes))


def _append_to_tsv(df_new: pd.DataFrame, output_tsv: str) -> None:
    try:
        df_combined = pd.concat(
            [pd.read_csv(output_tsv, sep='\t', index_col=False), df_new],
        )
    except FileNotFoundError:
        df_combined = df_new
    df_combined = df_combined[
        df_combined['ProteinA'] != df_combined['ProteinB']
    ]
    df_combined = df_combined.groupby(['ProteinA', 'ProteinB'], as_index=False)[
        'PDB_Code'
    ].agg(_merge_pdb_codes)
    df_combined.to_csv(output_tsv, sep='\t', index=False)


def main(
    output_dir: str = mkdtemp(),
    chunk_size: int = 500,
    pdb_ids: list[str] | None = None,
    max_file_size_mb: int = 500,
    output_suffix: str = '',
) -> str:
    setup_logging()
    pdb_files_dir = os.path.join(output_dir, 'pdbs/')
    os.makedirs(pdb_files_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    output_tsv = os.path.join(
        output_dir,
        f'pdb_{NEGATOME_PDB_VERSION}_{today}{output_suffix}.txt',
    )
    relevant_pdbs, pdb_chain_to_uniprot = _fetch_pdb_metadata()
    if pdb_ids is not None:
        relevant_pdbs = [pdb.lower() for pdb in pdb_ids]
    chunk, n_remaining = _get_next_chunk(
        relevant_pdbs, pdb_files_dir, chunk_size,
    )
    if not chunk:
        logging.info(f"All {len(relevant_pdbs)} PDBs already processed.")
        return output_tsv
    logging.info(
        f"Processing chunk of {len(chunk)} PDBs ({len(relevant_pdbs) - n_remaining} done, {n_remaining} remaining).",
    )
    pdb_files = download_all_bio_assemblies(
        pdb_ids=chunk, output_directory=pdb_files_dir,
    )
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    negatome_pairs = _extract_negatome_pairs(
        pdb_files, pdb_chain_to_uniprot, max_file_size_bytes,
    )
    if negatome_pairs:
        df_new = pd.DataFrame(
            negatome_pairs, columns=[
                'ProteinA', 'ProteinB', 'PDB_Code',
            ],
        )
        df_new = df_new.groupby(['ProteinA', 'ProteinB'], as_index=False)[
            'PDB_Code'
        ].agg(_merge_pdb_codes)
        _append_to_tsv(df_new, output_tsv)
    elif not os.path.exists(output_tsv):
        # Empty result with no existing output — write a header-only TSV so
        # downstream merge tools see a valid file rather than failing on
        # FileNotFoundError.
        pd.DataFrame(
            columns=['ProteinA', 'ProteinB', 'PDB_Code'],
        ).to_csv(output_tsv, sep='\t', index=False)
    logging.info('Uploading %s to %s', output_tsv, NEGATOME_S3)
    upload_file_to_s3(output_tsv, NEGATOME_S3)
    return output_tsv


def parse_args():
    parser = ArgumentParser(
        description='Build a PDB-derived negatome by scanning heterocomplexes for non-interacting chain pairs.',
    )
    parser.add_argument(
        '-o', '--output_dir', type=str,
        help='Directory to write output files (pdbs/ subfolder and negatome CSV). Created if it does not exist.',
        required=True,
    )
    parser.add_argument(
        '-c', '--chunk_size', type=int,
        help='Number of PDB structures to process per run.',
        default=500,
    )
    parser.add_argument(
        '--pdb-ids', type=str, nargs='+', default=None,
        help=(
            'Optional explicit list of PDB IDs to process, overriding the '
            'full heterocomplex set from SIFTS. Use to target large '
            'assemblies in a separate job.'
        ),
    )
    parser.add_argument(
        '--max-file-size-mb', type=int, default=500,
        help=(
            'Skip PDB files larger than this size in MB (default 500). '
            'Increase when targeting large assemblies on a memory-rich worker.'
        ),
    )
    parser.add_argument(
        '--output-suffix', type=str, default='',
        help=(
            'Optional suffix appended to the output filename stem (before '
            '.txt), e.g. "_large_8wq9" to avoid collisions with the main run.'
        ),
    )
    return parser


if __name__ == '__main__':
    args = parse_args().parse_args()
    main(
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        pdb_ids=args.pdb_ids,
        max_file_size_mb=args.max_file_size_mb,
        output_suffix=args.output_suffix,
    )
