# Utilities for querying the RCSB PDB search and data APIs.
# These are used by the structure curation agent but may be more broadly useful too.
#
# Search API:   https://search.rcsb.org/rcsbsearch/v2/query
# Data API:     https://data.rcsb.org/rest/v1/core/
# GraphQL API:  https://data.rcsb.org/graphql
from __future__ import annotations

import logging

import requests


_SEARCH_URL = 'https://search.rcsb.org/rcsbsearch/v2/query'
_DATA_URL = 'https://data.rcsb.org/rest/v1/core'
_GRAPHQL_URL = 'https://data.rcsb.org/graphql'

# Chain counts for many assemblies at once. The REST data API is one request per
# assembly, which is untenable for whole-dataset annotation; GraphQL takes a list.
_ASSEMBLY_INFO_QUERY = '''
query($ids: [String!]!) {
  assemblies(assembly_ids: $ids) {
    rcsb_id
    rcsb_assembly_info {
      polymer_entity_instance_count
      polymer_entity_instance_count_protein
    }
  }
}
'''


def _build_search_payload(
    uniprot_accession: str,
    rows: int,
    start: int,
    sort_by: str,
    single_entity_only: bool,
) -> dict:
    """Build an RCSB search API payload filtering by UniProt accession.

    Args:
        uniprot_accession: UniProt accession to search for (e.g. "P38011").
        rows: Maximum number of results to return.
        start: Offset for pagination.
        sort_by: Sort field — "score" (relevance) or "deposit_date".
        single_entity_only: If True, restrict to entries with exactly one
            polymer entity (monomers/homomers only).

    Returns:
        JSON-serialisable dict ready to POST to the RCSB search endpoint.
    """
    nodes = [
        {
            'type': 'terminal',
            'service': 'text',
            'parameters': {
                'operator': 'exact_match',
                'value': uniprot_accession,
                'attribute': 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession',
            },
        },
        {
            'type': 'terminal',
            'service': 'text',
            'parameters': {
                'operator': 'exact_match',
                'value': 'UniProt',
                'attribute': 'rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name',
            },
        },
    ]
    if single_entity_only:
        nodes.append({
            'type': 'terminal',
            'service': 'text',
            'parameters': {
                'operator': 'equals',
                'value': 1,
                'attribute': 'rcsb_entry_info.polymer_entity_count',
            },
        })
    return {
        'query': {'type': 'group', 'logical_operator': 'and', 'nodes': nodes},
        'return_type': 'entry',
        'request_options': {
            'paginate': {'start': start, 'rows': rows},
            'sort': [{'sort_by': sort_by, 'direction': 'desc'}],
        },
    }


def search_structures(
    uniprot_accession: str,
    rows: int = 25,
    start: int = 0,
    sort_by: str = 'score',
    single_entity_only: bool = False,
) -> list[str]:
    """Return PDB entry IDs for a UniProt accession, relevance-ranked by default.

    Args:
        uniprot_accession: UniProt accession to search for.
        rows: Maximum number of results to return.
        start: Offset for pagination.
        sort_by: Sort field — "score" (relevance, default) or "deposit_date".
        single_entity_only: If True, restrict to entries with exactly one
            polymer entity. Useful for finding monomeric structures when the
            default results are dominated by complexes.

    Returns:
        List of PDB entry IDs (e.g. ["3FRX", "1LDS"]).
    """
    payload = _build_search_payload(
        uniprot_accession, rows, start, sort_by, single_entity_only,
    )
    response = requests.post(_SEARCH_URL, json=payload)
    if response.ok and response.text:
        return [hit['identifier'] for hit in response.json().get('result_set', [])]
    return []


def get_entry_summary(pdb_id: str) -> dict:
    """Return a flat summary of entry-level metadata from the RCSB data API.

    Args:
        pdb_id: 4-character PDB ID (e.g. "3FRX").

    Returns:
        Dict with keys: title, method, resolution, deposit_date, release_date,
        status, polymer_entity_count, polymer_entity_count_protein. Returns
        empty dict if the request fails.
    """
    response = requests.get(f"{_DATA_URL}/entry/{pdb_id}")
    if not response.ok:
        return {}
    data = response.json()
    info = data.get('rcsb_entry_info', {})
    accession = data.get('rcsb_accession_info', {})
    return {
        'title': data.get('struct', {}).get('title', ''),
        'method': info.get('experimental_method', ''),
        'resolution': (info.get('resolution_combined') or [None])[0],
        'deposit_date': accession.get('deposit_date', ''),
        'release_date': accession.get('initial_release_date', ''),
        'status': accession.get('status_code', ''),
        'polymer_entity_count': info.get('polymer_entity_count'),
        'polymer_entity_count_protein': info.get('polymer_entity_count_protein'),
    }


def get_polymer_entity_data(pdb_id: str, entity_id: str) -> dict:
    """Return the full polymer entity data record from the RCSB data API.

    Args:
        pdb_id: 4-character PDB ID (e.g. "3FRX").
        entity_id: Numeric entity ID within the PDB entry (e.g. "1").

    Returns:
        Full entity data dict, or empty dict if the request fails.
    """
    response = requests.get(f"{_DATA_URL}/polymer_entity/{pdb_id}/{entity_id}")
    return response.json() if response.ok else {}


def fetch_assembly_chain_counts(
    assembly_ids: list[str],
    batch_size: int = 100,
    protein_only: bool = False,
    timeout: int = 120,
) -> dict[str, int]:
    """Batch-fetch polymer chain counts for PDB biological assemblies.

    Uses the GraphQL API rather than the REST data API, which would need one
    request per assembly.

    Assemblies RCSB does not recognise are omitted from the response rather than
    returned as null, so a caller must treat a missing key as unknown rather than
    assuming every requested id comes back.

    Args:
        assembly_ids: Assembly identifiers of the form '<PDB_ID>-<N>', e.g.
            '9MBZ-1'. Case-insensitive; RCSB echoes them uppercased.
        batch_size: Assemblies per request.
        protein_only: Count protein chains only, rather than all polymer chains
            (which include nucleic acids).
        timeout: Per-request socket timeout in seconds.

    Returns:
        Dict mapping uppercased assembly id -> chain count, omitting any
        assembly RCSB did not return.
    """
    logger = logging.getLogger(__name__)
    field = (
        'polymer_entity_instance_count_protein' if protein_only
        else 'polymer_entity_instance_count'
    )
    counts: dict[str, int] = {}
    unique_ids = sorted({assembly_id.upper() for assembly_id in assembly_ids})
    for start in range(0, len(unique_ids), batch_size):
        batch = unique_ids[start:start + batch_size]
        response = requests.post(
            _GRAPHQL_URL,
            json={'query': _ASSEMBLY_INFO_QUERY, 'variables': {'ids': batch}},
            timeout=timeout,
        )
        if not response.ok:
            logger.warning(
                'RCSB GraphQL batch %d-%d failed: %s',
                start, start + len(batch), response.status_code,
            )
            continue
        payload = response.json()
        if payload.get('errors'):
            logger.warning('RCSB GraphQL errors: %s', payload['errors'][:1])
        for record in (payload.get('data', {}) or {}).get('assemblies') or []:
            if record is None:
                continue
            info = record.get('rcsb_assembly_info') or {}
            value = info.get(field)
            if value is not None:
                counts[record['rcsb_id']] = int(value)
    logger.info(
        'Resolved chain counts for %d of %d assemblies',
        len(counts), len(unique_ids),
    )
    return counts
