from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from itertools import product

import numpy as np
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
from scipy.spatial import cKDTree


INTERACTION_DIST_THRES_ANG = 8.0
GLYCINE_RESNAME = 'GLY'
CB_ATM_NAME = 'CB'
CA_ATM_NAME = 'CA'


def _get_cb_ca_coords(chain: Chain) -> np.ndarray:
    """Return Cb/Ca coordinates for each residue as an (N, 3) array.
    Uses Cb for all residues except glycine, which falls back to Ca.
    """
    coords = []
    for residue in chain.get_residues():
        if residue.id[0] == ' ':
            central_atom_name = CA_ATM_NAME if residue.resname == GLYCINE_RESNAME else CB_ATM_NAME
            if central_atom_name not in residue:
                central_atom_name = CA_ATM_NAME
            if central_atom_name in residue:
                coords.append(residue[central_atom_name].coord)
    return np.array(coords, dtype=np.float64)


def measure_interchain_dist(pdb_structure: Structure) -> dict[tuple[str, str], float]:
    """Measure the minimum inter-chain Cb/Ca distance for every chain pair.
    Uses Cb atoms for all residues except glycine, which uses Ca. A chain pair
    is considered non-interacting when its minimum distance exceeds
    INTERACTION_DIST_THRES_ANG (8 Å).
    Args:
        pdb_structure: Biopython Structure (first model is used).
    Returns:
        Dict mapping (chain_i, chain_j) to minimum inter-chain distance in Å,
        sorted ascending so the nearest-neighbour pair comes first.
    """
    model = pdb_structure[0]
    chain_coords: dict[str, np.ndarray] = {}
    for chain in model.get_chains():
        coords = _get_cb_ca_coords(chain)
        if len(coords):
            chain_coords[chain.id] = coords
    min_distances: dict[tuple[str, str], float] = {}
    for chain_i, chain_j in combinations(chain_coords, 2):
        tree_j = cKDTree(chain_coords[chain_j])
        dists, _ = tree_j.query(chain_coords[chain_i], workers=-1)
        min_distances[(chain_i, chain_j)] = float(dists.min())
    return dict(sorted(min_distances.items(), key=lambda item: item[1]))


def _compute_nearest_neighbour(min_distances: dict[tuple[str, str], float]) -> dict[str, str]:
    """Return each chain's nearest neighbour by minimum Cb/Ca distance.

    Args:
        min_distances: Output of measure_interchain_dist — (chain_i, chain_j) → distance in Å.
    Returns:
        Dict mapping each chain id to the chain id of its nearest neighbour.
    """
    nearest: dict[str, tuple[str, float]] = {}
    for (chain_i, chain_j), dist in min_distances.items():
        for chain, other in [(chain_i, chain_j), (chain_j, chain_i)]:
            if chain not in nearest or dist < nearest[chain][1]:
                nearest[chain] = (other, dist)
    return {chain: nn for chain, (nn, _) in nearest.items()}


def get_non_interacting_chain_pairs(min_distances: dict[tuple[str, str], float]) -> list[tuple[str, str]]:
    """Return chain pairs that are both distant and not nearest neighbours.
    A pair (i, j) is returned when:
      - their minimum Cb/Ca distance exceeds INTERACTION_DIST_THRES_ANG, and
      - neither chain considers the other its nearest neighbour.
    Args:
        min_distances: Output of measure_interchain_dist — (chain_i, chain_j) → distance in Å.
    Returns:
        List of (chain_i, chain_j) tuples satisfying both criteria.
    """
    nearest_neighbour = _compute_nearest_neighbour(min_distances)
    return [
        pair for pair, dist in min_distances.items()
        if all((
            dist > INTERACTION_DIST_THRES_ANG,
            nearest_neighbour.get(pair[0]) != pair[1],
            nearest_neighbour.get(pair[1]) != pair[0],
        ))
    ]


def get_non_interacting_protein_pairs(
    min_distances: dict[tuple[str, str], float],
    chain_to_uniprot: dict[str, set[str]],
) -> set[tuple[str, str]]:
    """Aggregate chain-level non-interactions up to the protein-pair level.

    A protein pair (u_a, u_b) with u_a != u_b is non-interacting in this PDB
    iff every chain pair (c_i, c_j) backing it — i.e. where u_a is mapped to
    c_i and u_b to c_j — is classified non-interacting at the chain level
    (>8 Å Cb/Ca minimum AND not mutual nearest-neighbour).

    Aggregating at the protein-pair level prevents emitting a spurious
    negative when the same two proteins also have at least one interacting
    chain pair in the same PDB (the chain-pair-vs-protein-pair bug fixed by
    TERN-2291).

    Args:
        min_distances: Output of measure_interchain_dist — (chain_i, chain_j) → distance in Å.
        chain_to_uniprot: Mapping from chain id to the set of UniProt
            accessions associated with that chain in this PDB.
    Returns:
        Set of (u_a, u_b) tuples (sorted alphabetically) that are non-interacting.
    """
    non_interacting_chain_pairs = {
        frozenset(pair) for pair in get_non_interacting_chain_pairs(min_distances)
    }
    protein_pair_chains: dict[
        tuple[str, str],
        list[frozenset[str]],
    ] = defaultdict(list)
    for chain_i, chain_j in min_distances:
        for u_i, u_j in product(
            chain_to_uniprot.get(chain_i, set()),
            chain_to_uniprot.get(chain_j, set()),
        ):
            if u_i == u_j:
                continue
            key = (min(u_i, u_j), max(u_i, u_j))
            protein_pair_chains[key].append(frozenset({chain_i, chain_j}))
    return {
        pair for pair, chain_pairs in protein_pair_chains.items()
        if all(cp in non_interacting_chain_pairs for cp in chain_pairs)
    }
