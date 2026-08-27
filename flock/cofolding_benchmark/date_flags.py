from __future__ import annotations

import pandas as pd

from flock.cofolding_benchmark import CUTOFF_DATE


def undirected_pair(protein_a: str, protein_b: str) -> tuple[str, str]:
    """Return a UniProt pair as a sorted tuple so (A, B) and (B, A) collapse."""
    return (protein_a, protein_b) if protein_a <= protein_b else (protein_b, protein_a)


def parse_pinder_uniprot_pair(ppi_id: str) -> tuple[str, str] | None:
    """Parse the undirected UniProt pair from a PINDER system id.

    The id format is {pdb}__{chain}_{uniprot}--{pdb}__{chain}_{uniprot}.

    Args:
        ppi_id: PINDER system identifier.

    Returns:
        Sorted (uniprot_a, uniprot_b) tuple, or None if either side is
        UNDEFINED or the id does not parse.
    """
    try:
        uniprot_a, uniprot_b = (
            part.split('__')[1].split('_')[1] for part in ppi_id.split('--')
        )
    except (IndexError, ValueError):
        return None
    if uniprot_a == 'UNDEFINED' or uniprot_b == 'UNDEFINED':
        return None
    return undirected_pair(uniprot_a, uniprot_b)


def build_pinder_pair_index(pinder_df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Map each undirected PINDER UniProt pair to its source PDB entry IDs.

    Every PINDER system whose pair parses (any label or resolution) contributes
    its entry_id, so the index captures every PDB entry in which the exact A-B
    complex appears. This is the per-pair source set for the positive
    in_training_set check.

    Args:
        pinder_df: PINDER metadata with at least the id and entry_id columns.

    Returns:
        Dict mapping (uniprot_a, uniprot_b) to a set of source PDB entry IDs.
    """
    index: dict[tuple[str, str], set[str]] = {}
    for ppi_id, entry_id in zip(pinder_df['id'], pinder_df['entry_id']):
        pair = parse_pinder_uniprot_pair(ppi_id)
        if pair is None:
            continue
        index.setdefault(pair, set()).add(entry_id)
    return index


def build_ppi3d_pair_index(ppi3d_df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Map each undirected PPI3D UniProt pair to its source PDB entry IDs.

    The PPI3D counterpart of build_pinder_pair_index. Every interface
    contributes its pdb_id regardless of BIO/XTAL call: the date flag asks
    whether a model could have SEEN the two chains together, and a crystal
    contact in a pre-cutoff entry is just as memorisable as a biological one.

    Args:
        ppi3d_df: Classified PPI3D interfaces with uniprot_1, uniprot_2 and
            pdb_id columns.

    Returns:
        Dict mapping (uniprot_a, uniprot_b) to a set of source PDB entry IDs,
        uppercased to match the release-date lookup.
    """
    index: dict[tuple[str, str], set[str]] = {}
    for uniprot_1, uniprot_2, pdb_id in zip(
        ppi3d_df['uniprot_1'], ppi3d_df['uniprot_2'], ppi3d_df['pdb_id'],
    ):
        pair = undirected_pair(uniprot_1, uniprot_2)
        index.setdefault(pair, set()).add(str(pdb_id).upper())
    return index


def merge_pair_indexes(
    *indexes: dict[tuple[str, str], set[str]],
) -> dict[tuple[str, str], set[str]]:
    """Union several pair -> source-PDB indexes.

    Unlike the homolog check, which must stay per-source because interface
    clusters only have meaning inside one resource's clustering, the date check
    asks a source-agnostic question: which PDB entries contain this pair. More
    sources strictly sharpen that, so the source sets are unioned before the
    date test rather than evaluated separately and combined.

    Args:
        *indexes: Pair -> source PDB ID sets, e.g. from build_pinder_pair_index
            and build_ppi3d_pair_index.

    Returns:
        Merged index whose value for each pair is the union across inputs.
    """
    merged: dict[tuple[str, str], set[str]] = {}
    for index in indexes:
        for pair, sources in index.items():
            merged.setdefault(pair, set()).update(sources)
    return merged


def build_negatome_pdb_index(pdb_negatome_df: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    """Map each undirected PDB-negatome UniProt pair to its source PDB IDs.

    Args:
        pdb_negatome_df: Raw PDB negatome with ProteinA, ProteinB and a
            comma-joined PDB_Code column.

    Returns:
        Dict mapping (uniprot_a, uniprot_b) to a list of source PDB IDs.
    """
    index: dict[tuple[str, str], list[str]] = {}
    for protein_a, protein_b, codes in zip(
        pdb_negatome_df['ProteinA'],
        pdb_negatome_df['ProteinB'],
        pdb_negatome_df['PDB_Code'],
    ):
        index[undirected_pair(protein_a, protein_b)] = str(codes).split(',')
    return index


def pair_in_training_set(
    source_pdb_ids: set[str] | list[str] | None,
    date_lookup: dict[str, str],
    cutoff: str = CUTOFF_DATE,
) -> bool | None:
    """Decide whether a pair's source structures predate the training cutoff.

    A pair counts as in the training set if any of its source PDB entries was
    released on or before the cutoff. Uncertain cases default to leaked so the
    clean (post-cutoff) set stays a true upper bound: a source ID absent from
    the current-holdings date table is an obsoleted/superseded entry, i.e. old,
    and is treated as pre-cutoff.

    Args:
        source_pdb_ids: Source PDB entry IDs for the pair, or None/empty when no
            source structure is known (e.g. literature-only negatives).
        date_lookup: Uppercase entry_id -> release_date (YYYY-MM-DD) map.
        cutoff: Release-date cutoff (YYYY-MM-DD). Releases on/before it count as
            pre-cutoff (in the training set).

    Returns:
        True if any source structure is pre-cutoff (or absent/obsolete), False
        if every source structure is post-cutoff, or None if no source PDB is
        known for the pair.
    """
    if not source_pdb_ids:
        return None
    for pdb_id in source_pdb_ids:
        release_date = date_lookup.get(pdb_id.upper())
        if release_date is None or release_date <= cutoff:
            return True
    return False


def annotate_date_flags(
    flock_df: pd.DataFrame,
    pinder_index: dict[tuple[str, str], set[str]],
    negatome_index: dict[tuple[str, str], list[str]],
    date_lookup: dict[str, str],
    cutoff: str = CUTOFF_DATE,
) -> pd.DataFrame:
    """Add the date-only leakage flags to every Flock row.

    Adds two nullable-boolean columns:
      - in_training_set: from release dates of the pair's source structures
        (PINDER complexes for positives, co-presence PDBs for negatives).
      - homolog_in_training_set: left as NA here. Both homolog flags are added
        downstream - positives' by pinder_clusters and ppi3d_clusters,
        negatives' by the sequence-identity rule - and writing either one
        before combine_homolog_flags has run would destroy it, since the
        cluster steps rebuild this column wholesale.

    Args:
        flock_df: Flock benchmark with Target, Partner and Type columns.
        pinder_index: Positives' pair -> source PDB IDs. Pass
            merge_pair_indexes(pinder_index, ppi3d_index) when Flock's positives
            are the union of both sources; a pair missing from this index gets
            NA and is excluded downstream, so omitting a source silently drops
            every pair only that source contributes.
        negatome_index: Output of build_negatome_pdb_index (negatives' sources).
        date_lookup: Uppercase entry_id -> release_date map.
        cutoff: Release-date cutoff (YYYY-MM-DD).

    Returns:
        Copy of flock_df with the two flag columns appended.
    """
    annotated = flock_df.copy()
    flags: list[bool | None] = []
    for target, partner, row_type in zip(
        annotated['Target'], annotated['Partner'], annotated['Type'],
    ):
        pair = undirected_pair(target, partner)
        if row_type == 'Positive':
            sources: set[str] | list[str] | None = pinder_index.get(pair)
        else:
            sources = negatome_index.get(pair)
        flags.append(pair_in_training_set(sources, date_lookup, cutoff))
    annotated['in_training_set'] = pd.array(flags, dtype='boolean')
    annotated['homolog_in_training_set'] = pd.array(
        [pd.NA] * len(annotated), dtype='boolean',
    )
    return annotated
