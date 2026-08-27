from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.PDB import PDBList
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure


@dataclass(frozen=True)
class PdbFile:
    path: str

    def return_pdb_id(self) -> str:
        return Path(self.path).stem[:4]


def read_pdb(pdb: PdbFile) -> Structure:
    """Read a PDB/MMCIF file and return a Biopython Structure object.

    Args:
        pdb: PdbFile whose path points to the input file.

    Returns:
        Biopython Structure object.
    """
    fname = pdb.path
    parser = PDBParser() if fname.endswith('.pdb') else MMCIFParser()
    return parser.get_structure(Path(fname).stem, fname)


def download_all_bio_assemblies(
        pdb_ids: list[str],
        output_directory: str,
        num_threads: int = 12,
) -> list[PdbFile]:
    """Download all biological assembly mmCIF files from RCSB for the given PDB IDs.

    The asymmetric unit file is downloaded first (required before assembly retrieval),
    then all biological assemblies for each entry are fetched. Already-present files
    are skipped without re-downloading.

    Args:
        pdb_ids: PDB IDs to download.
        output_directory: Directory to write .cif files into.
        num_threads: Number of parallel download threads.

    Returns:
        List of PdbFile objects for all successfully downloaded assemblies.
    """
    os.makedirs(output_directory, exist_ok=True)
    downloader = PDBList(verbose=False)
    downloader.download_pdb_files(
        pdb_ids, pdir=output_directory, file_format='mmCif', max_num_threads=num_threads,
    )
    pdb_ids_lower = {pdb_id.lower() for pdb_id in pdb_ids}
    relevant_assemblies = [
        (code, num) for code, num in downloader.get_all_assemblies()
        if code in pdb_ids_lower
    ]
    pdb_files = []
    for pdb_code, assembly_num in relevant_assemblies:
        filename = downloader.retrieve_assembly_file(
            pdb_code, assembly_num, pdir=output_directory, file_format='mmcif',
        )
        if filename and os.path.exists(filename):
            pdb_files.append(PdbFile(filename))
    return pdb_files
