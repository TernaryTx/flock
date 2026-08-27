from __future__ import annotations

import gzip
import os
import re
import xml.etree.ElementTree as ET
from tempfile import mkdtemp
from urllib.request import urlretrieve

# EBI's SIFTS split-XML mirror, one gzipped file per PDB entry, bucketed into
# subdirectories by the middle two characters of the (lowercase) PDB ID —
# the same "divided" layout EBI/RCSB use for their other per-entry mirrors.
_SIFTS_SPLIT_XML_URL = 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/split_xml/{mid}/{pdb_id}.xml.gz'
# Format of XML namespace is {namespaceURI}tagname - so need {} - not a typo
BASE_EXT = '{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}'


def extract_pdb_uniprot_mapping(sifts_xml_gz_path: str) -> dict[tuple[str, int, str], tuple[str, int]]:
    """Extract a mapping from PDB residues to UniProt residues from a SIFTS XML file.

    Args:
        sifts_xml_gz_path: Input SIFTS file gz path.

    Returns:
        Mapping of (chain, PDB residue number, insertion code) to (UniProt
        accession, UniProt residue number). The insertion code is an empty
        string if there is no insertion.
    """
    with gzip.open(sifts_xml_gz_path, 'rt') as f:
        root = ET.parse(f).getroot()
    res_mapping = {}

    for residue in root.findall(f".//{BASE_EXT}residue"):
        pdb_chain, pdb_resnum = None, None
        uniprot_acc, uniprot_resnum = None, None
        for xref in residue.findall(f"{BASE_EXT}crossRefDb"):
            source = xref.get('dbSource')
            if source == 'PDB':
                pdb_chain = xref.get('dbChainId')
                pdb_resnum = xref.get('dbResNum')
            elif source == 'UniProt':
                uniprot_acc = xref.get('dbAccessionId')
                uniprot_resnum = xref.get('dbResNum')
        if pdb_chain and pdb_resnum and uniprot_acc and uniprot_resnum and pdb_resnum != 'null':
            match = re.match(r'(\d+)([A-Za-z]?)', pdb_resnum)
            if match:
                resnum_int = int(match.group(1))
                insertion_code = match.group(2)
                res_mapping[(pdb_chain, resnum_int, insertion_code)] = (
                    uniprot_acc, int(uniprot_resnum),
                )
    return res_mapping


def get_pdb_to_uniprot_dict(
        pdb_id: str,
        keep_sifts: bool = False,
        local_folder: str | None = None,
) -> dict[tuple[str, int, str], tuple[str, int]]:
    """Download a SIFTS file from EBI and extract the PDB-to-UniProt residue mapping.

    Args:
        pdb_id: PDB ID of the protein structure.
        keep_sifts: Whether to keep the downloaded SIFTS XML file afterwards.
        local_folder: Local folder to download the SIFTS XML file to. Defaults
            to a fresh temp directory.

    Returns:
        Mapping of (chain, PDB residue number, insertion code) to (UniProt
        accession, UniProt residue number). The insertion code is an empty
        string if there is no insertion.
    """
    pdb_id = pdb_id.lower()
    local_folder = local_folder or mkdtemp()
    local_sifts_file = os.path.join(local_folder, f'{pdb_id}.xml.gz')
    url = _SIFTS_SPLIT_XML_URL.format(mid=pdb_id[1:3], pdb_id=pdb_id)
    urlretrieve(url, local_sifts_file)
    mapping_dict = extract_pdb_uniprot_mapping(local_sifts_file)
    if not keep_sifts:
        os.remove(local_sifts_file)
    return mapping_dict
