from __future__ import annotations

import logging

import pandas as pd
import requests
from UniProtMapper import ProtMapper

logger = logging.getLogger(__name__)

UNIPROT_ACCESSIONS_URL = 'https://rest.uniprot.org/uniprotkb/accessions'


def fetch_uniprot_tsv(
        accessions: list[str],
        fields: str,
        timeout_s: int = 60,
) -> str:
    """Fetch UniProt return fields for a batch of accessions in one request.

    The direct counterpart to fetch_uniprot_fields, which goes through the
    ID-mapping service and polls a job per call. This endpoint takes the same
    fields in one GET, so it is the cheaper route when the accessions are known
    to be UniProtKB entries and no cross-database mapping is needed.

    Args:
        accessions: Accessions to describe. The endpoint takes them as one
            comma-separated list, so the caller chunks to bound URL length.
        fields: Comma-separated UniProt return field names, e.g.
            'accession,protein_name,organism_name,length'.
        timeout_s: Request timeout.

    Returns:
        The endpoint's TSV, headers included.

    Raises:
        requests.RequestException: On any transport or HTTP failure.
    """
    response = requests.get(
        UNIPROT_ACCESSIONS_URL,
        params={
            'accessions': ','.join(accessions),
            'fields': fields,
            'format': 'tsv',
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.text


def fetch_uniprot_fields(
        uniprots: list[str],
        fields: list[str],
        batch_size: int = 100,
        to_db: str = 'UniProtKB',
) -> pd.DataFrame:
    """Fetch arbitrary UniProt return fields for a list of accessions.

    Submits accessions in batches so that a single unmappable accession only
    costs one batch rather than the whole request. Batches that fail outright
    are logged and skipped, so the returned frame may cover fewer accessions
    than were requested; callers should reindex against their input list rather
    than assume one row per accession.

    Args:
        uniprots: UniProt accession strings. Deduplicated, order preserved.
        fields: UniProt return field names, e.g. ['organism_id', 'xref_pfam'].
            Note that 'xref_gene3d' carries CATH superfamilies and 'xref_supfam'
            carries SCOP superfamilies; 'xref_cath' is not a valid field name.
        batch_size: Number of accessions per request.
        to_db: UniProt dataset to resolve against. Defaults to 'UniProtKB', which
            includes unreviewed (TrEMBL) entries — most accessions in the
            PDB-derived Negatome are unreviewed, and UniProtMapper's own default
            of 'UniProtKB-Swiss-Prot' silently drops them.

    Returns:
        DataFrame with one row per successfully mapped accession, carrying a
        'From' column with the queried accession plus one column per field.
        Empty DataFrame if nothing could be mapped.
    """
    unique = list(dict.fromkeys(uniprots))

    mapper = ProtMapper()
    frames = []
    n_failed = 0
    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        try:
            result, failed = mapper.get(ids=batch, fields=fields, to_db=to_db)
        except Exception:
            logger.exception(
                'UniProt batch %d-%d failed, skipping',
                start, start + len(batch),
            )
            n_failed += len(batch)
            continue
        n_failed += len(failed)
        frames.append(result)

    if n_failed:
        logger.warning(
            'Failed to map %d / %d accessions',
            n_failed, len(unique),
        )
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        'Retrieved %d fields for %d / %d accessions',
        len(fields), len(combined), len(unique),
    )
    return combined


def fetch_gene_names(
    uniprots: list[str],
    chunk_size: int = 5000,
    max_attempts: int = 3,
    strict: bool = True,
) -> dict[str, str]:
    """Fetch the primary gene name for each UniProt accession via UniProtMapper.

    Accessions with no gene name annotation are mapped to an empty string. Where
    an accession maps to multiple names (e.g. CALM1/2/3), the first returned name
    is used.

    Requests are chunked and each chunk is retried independently. UniProtMapper
    submits an asynchronous job and polls it, and that poll returns a transient
    400 often enough to matter on a list this size — a single failure would
    otherwise discard every accession fetched so far, at the end of a long
    pipeline run. A chunk that fails all attempts is logged and skipped; those
    accessions simply have no gene name, which is the same outcome as an
    unannotated one.

    Args:
        uniprots: List of UniProt accession strings.
        chunk_size: Accessions per chunk.
        max_attempts: Attempts per chunk before giving up on it.
        strict: Raise if nothing could be retrieved at all. Leave True where the
            gene names are part of the output; set False where they merely
            enrich an existing column and an outage should not fail the run.

    Returns:
        Dict mapping each resolved accession to its primary gene name.
    """
    unique = list(dict.fromkeys(uniprots))  # deduplicate, preserve order

    mapper = ProtMapper()
    frames = []
    for start in range(0, len(unique), chunk_size):
        chunk = unique[start:start + chunk_size]
        for attempt in range(1, max_attempts + 1):
            try:
                # UniProtMapper defaults to_db to UniProtKB-Swiss-Prot
                # But we want to map against all of UniProtKB.
                result, failed = mapper.get(
                    ids=chunk, fields=['gene_names'], to_db='UniProtKB',
                )
            except Exception as exc:
                logger.warning(
                    'Gene-name chunk %d-%d attempt %d/%d failed: %s',
                    start, start + len(chunk), attempt, max_attempts, exc,
                )
                continue
            if failed:
                logger.warning('Failed to map %d accessions', len(failed))
            frames.append(result)
            break
        else:
            logger.error(
                'Giving up on gene-name chunk %d-%d after %d attempts',
                start, start + len(chunk), max_attempts,
            )

    # Tolerating a failed chunk is one thing; tolerating total failure is
    # another. An outage at UniProt would otherwise hand back an empty map and
    # let the caller write a dataset whose gene columns are silently blank,
    # which is worse than not writing one at all.
    if strict and unique and not frames:
        raise RuntimeError(
            f'No gene names retrieved for any of {len(unique)} accessions; '
            f'UniProt ID mapping is likely unavailable.',
        )
    if not frames:
        logger.warning(
            'No gene names retrieved for %d accessions; continuing without them',
            len(unique),
        )

    gene_name_map: dict[str, str] = {}
    combined = pd.concat(
        frames, ignore_index=True,
    ) if frames else pd.DataFrame()
    for _, row in combined.iterrows():
        accession = row['From']
        gene_names_field = str(row.get('Gene Names', '')) or ''
        # gene_names is space-separated; first token is the primary name
        primary = gene_names_field.split(
        )[0] if gene_names_field.strip() else ''
        if accession not in gene_name_map:
            gene_name_map[accession] = primary

    logger.info(
        'Retrieved gene names for %d / %d accessions',
        len(gene_name_map), len(unique),
    )
    return gene_name_map


def add_gene_name_columns(
    df: pd.DataFrame,
    uniprot_cols: list[str],
    gene_name_cols: list[str],
) -> pd.DataFrame:
    """Add gene name columns to a DataFrame by looking up UniProt accessions.

    Fetches gene names for all unique accessions across uniprot_cols in a
    single batched request, then inserts the corresponding gene name column
    immediately after each UniProt column.

    Args:
        df: DataFrame containing UniProt accession columns.
        uniprot_cols: Column names containing UniProt accessions.
        gene_name_cols: Names for the new gene name columns, one per uniprot_col.

    Returns:
        DataFrame with gene name columns inserted after each UniProt column.
    """
    all_uniprots = pd.concat(
        [df[col] for col in uniprot_cols], ignore_index=True,
    ).dropna().unique().tolist()

    gene_name_map = fetch_gene_names(all_uniprots)

    df = df.copy()
    for uniprot_col, gene_name_col in zip(uniprot_cols, gene_name_cols):
        insert_pos = df.columns.get_loc(uniprot_col) + 1
        gene_names = df[uniprot_col].map(gene_name_map).fillna('')
        df.insert(insert_pos, gene_name_col, gene_names)

    return df
