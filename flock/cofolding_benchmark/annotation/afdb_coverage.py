from __future__ import annotations

import logging
import math
import re
import tempfile
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from flock.sifts import get_pdb_to_uniprot_dict
from flock.structure import PdbFile
from flock.structure import read_pdb

logger = logging.getLogger(__name__)

_AFDB_API = 'https://alphafold.ebi.ac.uk/api/prediction/{acc}'
_AFDB_TIMEOUT = 30

# Type alias: SIFTS mapping for one PDB entry.
_SiftsMap = dict[tuple[str, int, str], tuple[str, int]]


def _fetch_sifts(pdb_id: str) -> tuple[str, _SiftsMap | None]:
    """Download and parse the SIFTS mapping for one PDB entry.

    Args:
        pdb_id: PDB entry ID (case-insensitive).

    Returns:
        (pdb_id_lower, mapping) or (pdb_id_lower, None) on failure.
    """
    pdb_id = pdb_id.strip().lower()
    try:
        return pdb_id, get_pdb_to_uniprot_dict(pdb_id)
    except Exception:
        logger.warning('SIFTS fetch failed for %s', pdb_id)
        return pdb_id, None


def _build_sifts_cache(
    all_pdb_ids: set[str],
    n_workers: int,
) -> dict[str, _SiftsMap]:
    """Pre-download SIFTS mappings for all unique PDB IDs in parallel.

    Each PDB is downloaded exactly once regardless of how many accessions
    reference it, dramatically reducing S3 round-trips versus per-accession
    lookups.

    Args:
        all_pdb_ids: All unique PDB entry IDs needed across all accessions.
        n_workers: Number of parallel download threads.

    Returns:
        Dict mapping lowercase PDB ID to its SIFTS residue mapping. Failed
        downloads are omitted.
    """
    cache: dict[str, _SiftsMap] = {}
    logger.info(
        'Pre-fetching SIFTS for %d unique PDBs (%d workers)...',
        len(all_pdb_ids), n_workers,
    )
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_fetch_sifts, pdb_id): pdb_id
            for pdb_id in all_pdb_ids
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            pdb_id, mapping = future.result()
            if mapping is not None:
                cache[pdb_id] = mapping
            if done % 100 == 0 or done == len(all_pdb_ids):
                logger.info(
                    '  SIFTS %d/%d done (%d cached)',
                    done, len(all_pdb_ids), len(cache),
                )
    return cache


def _observed_resnums_from_cache(
    uniprot_acc: str,
    source_pdb_ids: set[str],
    sifts_cache: dict[str, _SiftsMap],
) -> set[int]:
    """Return the union of observed UniProt residue numbers across source PDBs.

    Uses the pre-built SIFTS cache so no S3 downloads happen here.

    Args:
        uniprot_acc: UniProt accession to filter by.
        source_pdb_ids: Source PDB entry IDs for this accession.
        sifts_cache: Pre-downloaded SIFTS mappings keyed by lowercase PDB ID.

    Returns:
        Union of observed UniProt residue numbers across all source PDBs.
    """
    observed: set[int] = set()
    for pdb_id in source_pdb_ids:
        mapping = sifts_cache.get(pdb_id.strip().lower())
        if mapping is None:
            continue
        observed.update(
            resnum for _, (acc, resnum)
            in mapping.items() if acc == uniprot_acc
        )
    return observed


def download_afdb_fragments(uniprot_acc: str, dest_dir: str) -> list[Path]:
    """Download all AlphaFold DB fragment PDB files for a UniProt accession.

    Fetches the EBI AlphaFold API to discover fragment URLs, then downloads
    each PDB file. Handles multi-fragment proteins (sequences > 1400 aa).

    Args:
        uniprot_acc: UniProt accession.
        dest_dir: Directory to write downloaded PDB files into.

    Returns:
        List of local paths to downloaded PDB files, in fragment order.
        Returns an empty list if the API call fails or no AFDB entry exists.
    """
    url = _AFDB_API.format(acc=uniprot_acc)
    try:
        response = requests.get(url, timeout=_AFDB_TIMEOUT)
        response.raise_for_status()
        entries = response.json()
    except Exception:
        logger.warning('AFDB API request failed for %s', uniprot_acc)
        return []

    if not entries:
        logger.warning('No AFDB entries found for %s', uniprot_acc)
        return []

    # The AFDB API can return alternative models (e.g. AF-Q96N11-2-F1) alongside
    # the canonical fragments (AF-Q96N11-F1, AF-Q96N11-F2, ...). Alternative
    # models have a numeric suffix before the fragment index and use a different
    # numbering scheme that can overwrite canonical pLDDT values. Keep only
    # entries whose entryId matches the canonical AF-{acc}-F{N} pattern.
    canonical_re = re.compile(
        r'^AF-' + re.escape(uniprot_acc) + r'-F\d+$',
        re.IGNORECASE,
    )
    paths = []
    for entry in entries:
        entry_id = entry.get('entryId', '')
        if not canonical_re.match(entry_id):
            continue
        pdb_url = entry.get('pdbUrl')
        if not pdb_url:
            continue
        local_path = Path(dest_dir) / f'{entry_id}.pdb'
        try:
            pdb_response = requests.get(pdb_url, timeout=_AFDB_TIMEOUT)
            pdb_response.raise_for_status()
            local_path.write_bytes(pdb_response.content)
            paths.append(local_path)
        except Exception:
            logger.warning(
                'Failed to download AFDB PDB for %s (%s)', uniprot_acc, pdb_url,
            )
    return paths


def build_plddt_map(pdb_paths: list[Path]) -> dict[int, float]:
    """Build a {UniProt_resnum: pLDDT} map from AlphaFold DB PDB files.

    In AFDB PDB format, pLDDT is stored in the CA B-factor field and residue
    numbers correspond to UniProt positions (sequential across fragments).
    Combines all fragments into a single dict.

    Args:
        pdb_paths: Paths to AFDB fragment PDB files.

    Returns:
        Dict mapping UniProt residue number to pLDDT score.
    """
    plddt_map: dict[int, float] = {}
    for pdb_path in pdb_paths:
        try:
            struct = read_pdb(PdbFile(path=str(pdb_path)))
        except Exception:
            logger.warning('Failed to parse AFDB PDB: %s', pdb_path)
            continue
        for residue in struct.get_residues():
            if 'CA' in residue:
                plddt_map[residue.id[1]] = residue['CA'].bfactor
    return plddt_map


def _compute_one(
    uniprot_acc: str,
    source_pdb_ids: set[str],
    sifts_cache: dict[str, _SiftsMap],
) -> tuple[str, float]:
    """Compute AFDB pLDDT>=65 fraction for a single accession.

    Uses the pre-built SIFTS cache for observed-residue lookup; downloads the
    AFDB model fresh (no AFDB cache needed — one call per unique accession).

    Args:
        uniprot_acc: UniProt accession.
        source_pdb_ids: Source PDB entry IDs for SIFTS observed-residue lookup.
        sifts_cache: Pre-downloaded SIFTS mappings (from _build_sifts_cache).

    Returns:
        (uniprot_acc, fraction) where fraction is NaN if AFDB is unavailable or
        no observed residues are found.
    """
    observed = _observed_resnums_from_cache(
        uniprot_acc, source_pdb_ids, sifts_cache,
    )
    if not observed:
        logger.warning(
            'No observed SIFTS residues for %s (pdbs=%s)',
            uniprot_acc, source_pdb_ids,
        )
        return uniprot_acc, float('nan')

    with tempfile.TemporaryDirectory() as dest_dir:
        pdb_paths = download_afdb_fragments(uniprot_acc, dest_dir)
        if not pdb_paths:
            return uniprot_acc, float('nan')
        plddt_map = build_plddt_map(pdb_paths)

    if not plddt_map:
        logger.warning('Empty pLDDT map for %s', uniprot_acc)
        return uniprot_acc, float('nan')

    n_above = sum(
        1 for resnum in observed if plddt_map.get(
            resnum, 0.0,
        ) >= 65.0
    )
    return uniprot_acc, n_above / len(observed)


def compute_coverage(
    acc_to_source_pdbs: dict[str, set[str]],
    n_workers: int = 16,
) -> dict[str, float]:
    """Compute AFDB pLDDT>=65 coverage for all accessions, in parallel.

    Two-phase approach to minimise S3 round-trips:
      Phase 1 — pre-download all unique SIFTS files once (one per PDB ID).
      Phase 2 — compute per-accession coverage using the SIFTS cache plus a
                 fresh AFDB download per accession.

    For each accession, coverage is the fraction of its SIFTS-observed UniProt
    residues (union across all source PDBs) that have AFDB pLDDT >= 65. Returns
    NaN when AFDB is unavailable or no observed residues are found.

    Args:
        acc_to_source_pdbs: Dict mapping UniProt accession to the set of source
            PDB entry IDs from which its observed residues are derived.
        n_workers: Number of parallel threads (used in both phases).

    Returns:
        Dict mapping each accession to its coverage fraction (or NaN).
    """
    all_pdb_ids = {
        pdb.strip().lower()
        for pdbs in acc_to_source_pdbs.values() for pdb in pdbs
    }

    sifts_cache = _build_sifts_cache(all_pdb_ids, n_workers)

    results: dict[str, float] = {}
    total = len(acc_to_source_pdbs)
    logger.info(
        'Computing AFDB coverage for %d accessions (%d workers)...', total, n_workers,
    )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compute_one, acc, pdbs, sifts_cache): acc
            for acc, pdbs in acc_to_source_pdbs.items()
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            acc, fraction = future.result()
            results[acc] = fraction
            if done % 50 == 0 or done == total:
                nan_count = sum(1 for v in results.values() if math.isnan(v))
                logger.info(
                    '  AFDB %d/%d done (%d NaN so far)',
                    done, total, nan_count,
                )

    return results
