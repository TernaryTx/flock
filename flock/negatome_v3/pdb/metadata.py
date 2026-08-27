from __future__ import annotations

from collections import defaultdict

import pandas as pd

from flock.pdb_metadata import PdbUniprotData


def _fetch_pdb_metadata():
    metadata = PdbUniprotData().data
    relevant_pdbs = get_heterocomplexes(metadata)
    pdb_chain_to_uniprot = get_pdb_chain_uniprot_map(relevant_pdbs, metadata)
    return relevant_pdbs, pdb_chain_to_uniprot


def get_heterocomplexes(metadata: pd.DataFrame) -> list[str]:
    """Return PDB IDs whose structures contain at least three distinct UniProt accessions.

    Structures with fewer than three unique accessions cannot yield a non-interacting
    chain pair that is also not a nearest neighbour, so they are excluded from the
    PDB-based negatome pipeline.

    Args:
        metadata: PDB-UniProt mapping table, e.g. PdbUniprotData().data.
    Returns:
        List of PDB IDs (strings) with three or more unique mapped UniProt accessions.
    """
    groupby_pdb = metadata.groupby('PDB')['Uniprot_Acc'].nunique()
    return groupby_pdb[groupby_pdb > 2].index.to_list()


def get_pdb_chain_uniprot_map(
    pdb_ids: list[str], metadata: pd.DataFrame,
) -> dict[str, dict[str, set[str]]]:
    """Build a nested mapping from PDB ID to {chain ID: UniProt accessions}.

    Args:
        pdb_ids: PDB IDs to include, typically from get_heterocomplexes().
        metadata: PDB-UniProt mapping table, e.g. PdbUniprotData().data.
    Returns:
        Dict mapping pdb_id to a dict mapping chain_id to a set of UniProt accessions.
    """
    metadata = metadata[metadata['PDB'].isin(pdb_ids)]
    pdb_chain_uniprot_map: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set),
    )
    for pdb, chain, acc in zip(metadata['PDB'], metadata['Chain'], metadata['Uniprot_Acc']):
        pdb_chain_uniprot_map[pdb][chain].add(acc)
    return {pdb: dict(chains) for pdb, chains in pdb_chain_uniprot_map.items()}
