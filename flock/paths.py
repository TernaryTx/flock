# S3 paths and AWS constants for the flock benchmark dataset.
from __future__ import annotations

import os

from flock import DIVERSITY_VERSION
from flock import FLOCK_VERSION
from flock import INTACT_VERSION
from flock import NEGATOME_LIT_VERSION
from flock import NEGATOME_PDB_VERSION
from flock import PINDER_VERSION
from flock import PPI3D_VERSION
from flock.aws import list_files_from_s3

# Base S3 location for all Flock data. Override with your own bucket via the
# FLOCK_S3_ROOT environment variable before running any of the scripts below.
FLOCK_S3_ROOT = os.environ.get(
    'FLOCK_S3_ROOT', 's3://your-bucket-name/flock/data/',
)

# Negatome - negative PPIs
NEGATOME_S3 = FLOCK_S3_ROOT + 'negatome/'

# Literature-derived Negatome v3: the LLM-mined dataset that supersedes the
# published Negatome 2.0 manual set rather than extending it. The version sits in
# the prefix, so bumping NEGATOME_LIT_VERSION is what creates the new tree.
NEGATOME_LITERATURE_S3 = NEGATOME_S3 + f'literature_{NEGATOME_LIT_VERSION}/'

# Raw Europe PMC responses (search pages, metadata chunks, full-text XML),
# archived for provenance. Shared across stages rather than owned by one, since
# the same fetch feeds the corpus and any later re-screen.
NEGATOME_LITERATURE_RAW_S3 = NEGATOME_LITERATURE_S3 + 'raw/'

# One version per pipeline stage, moving independently: re-screening a fixed
# corpus bumps 'screen' alone. Kept as one dict rather than four *_VERSION
# constants so that adding a stage does not add another module-level name.
LITERATURE_STAGE_VERSIONS = {
    'corpus': 'v1',
    'screen': 'v1',
    'curation': 'v1',
    'pairs': 'v1',
    'accessions': 'v1',
}

# PINDER - positive PPIs from the PDB
PINDER_S3 = FLOCK_S3_ROOT + 'PINDER/'

# PPI3D - positive PPIs from the PDB, tracking current holdings. Unlike the other
# sources this is not a single file: a pull is a directory of per-window
# CSV/JSON/log triples, since PPI3D serves bulk data one release-date window at a
# time. Hence a prefix resolver rather than resolve_latest_raw. Pulls are named
# filteredforflock_* because they are a subset of PPI3D (see get_ppi3d_prefix).
PPI3D_S3 = FLOCK_S3_ROOT + 'PPI3D/'

# IntAct - known positive interactions for stringent filtering
INTACT_S3 = FLOCK_S3_ROOT + 'IntAct/'

# Flock - output benchmark dataset
FLOCK_S3 = FLOCK_S3_ROOT + 'flock/'

# Cache of whole-PDB metadata tables (SIFTS PDB-UniProt map, PDB release
# dates, ...) fetched by flock.pdb_metadata.PdbMetadata subclasses.
PDB_METADATA_CACHE_S3 = FLOCK_S3_ROOT + 'pdb_metadata_cache/'

# Curated monomer structures (output of flock.agent)
MONOMER_STRUCTURES = FLOCK_S3 + f'monomer_structures_{FLOCK_VERSION}.csv'

# Co-folding benchmark - leakage-aware annotation layer over Flock. The
# leakage-free target set and pair subset (leakage_free_*) live under this prefix.
COFOLDING_BENCHMARK_S3 = FLOCK_S3 + 'cofolding_benchmark/'

# Diversity analysis - shared per-protein annotation and structure clustering
# tables consumed by the diversity notebooks. Expensive to rebuild (a ~25k
# accession UniProt pull and an AFDB download + Foldseek clustering run), so the
# results are versioned in S3 rather than cached per notebook.
DIVERSITY_S3 = FLOCK_S3 + 'diversity/'


def make_dated_filename(prefix: str, version: str, ext: str, date: str) -> str:
    """Build a versioned, date-stamped filename, e.g. negatome_pairs_v3_2026-04-16.csv.

    Args:
        prefix: Base name before the version, e.g. 'negatome_pairs'.
        version: Version string, e.g. 'v3'.
        ext: File extension including leading dot, e.g. '.csv'.
        date: Date string in YYYY-MM-DD format.

    Returns:
        Filename string.
    """
    return f'{prefix}_{version}_{date}{ext}'


def resolve_latest_raw(
        s3_prefix: str,
        name_prefix: str,
        version: str,
        name_suffix: str,
        date: str | None = None,
) -> str:
    """Return the S3 path for the latest dated raw file for a given version.

    If date is provided, constructs the path directly. Otherwise lists the
    S3 prefix and returns the lexicographically latest match (YYYY-MM-DD dates
    sort correctly as strings).

    Args:
        s3_prefix: S3 folder prefix, e.g. NEGATOME_S3.
        name_prefix: Filename stem before the version, e.g. 'pdb'.
        version: Version string, e.g. 'v3'.
        name_suffix: File extension including any fixed suffix, e.g. '.txt'.
        date: Optional explicit date override (YYYY-MM-DD).

    Returns:
        Full S3 path to the raw file.

    Raises:
        FileNotFoundError: If no matching file is found in S3.
    """
    if date is not None:
        return s3_prefix + f'{name_prefix}_{version}_{date}{name_suffix}'
    stem_prefix = f'{name_prefix}_{version}_'
    candidates = []
    for path in list_files_from_s3(s3_prefix):
        filename = path.rsplit('/', 1)[-1]
        if not filename.startswith(stem_prefix) or not filename.endswith(name_suffix):
            continue
        # Only match files where the middle portion is a YYYY-MM-DD date (10 chars)
        middle = filename[len(stem_prefix):-len(name_suffix)]
        if len(middle) == 10 and middle[4] == '-' and middle[7] == '-':
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f'No {name_prefix} {version} raw files found in {s3_prefix}',
        )
    return sorted(candidates)[-1]


# Convenience wrappers for the default versions
def get_negatome_pdb_path(version: str = NEGATOME_PDB_VERSION, date: str | None = None) -> str:
    return resolve_latest_raw(NEGATOME_S3, 'pdb', version, '.txt', date)


def get_negatome_lit_path(version: str, date: str | None = None) -> str:
    """Return the published Negatome 2.0 manually curated file for a given version.

    version is required rather than defaulted to NEGATOME_LIT_VERSION, which now
    points at the in-house literature Negatome v3 - a differently shaped dataset
    under NEGATOME_LITERATURE_S3, not a filename-version bump of this file.
    Defaulting would inherit that bump and resolve a file that does not exist.

    Args:
        version: Version string; 'v2' is the published Negatome 2.0 release.
        date: Optional explicit date override (YYYY-MM-DD).

    Returns:
        Full S3 path.
    """
    return resolve_latest_raw(NEGATOME_S3, 'manual_stringent', version, '.txt', date)


def get_negatome_pdb_stringent_path(version: str, date: str | None = None) -> str:
    """Return the IntAct-filtered PDB-derived Negatome file for a given version.

    Resolves pdb_stringent_<version>_<date>.txt. Both the legacy Negatome 2.0
    file (v2) and the by-product written by compile_negatome (v3) share this
    stem, so the same resolver serves the version comparison. version is
    required rather than defaulted, since the point of this resolver is to name
    a specific version explicitly.

    Args:
        version: Version string, 'v2' (published Negatome 2.0) or 'v3' (in-house).
        date: Optional explicit date override (YYYY-MM-DD).

    Returns:
        Full S3 path.
    """
    return resolve_latest_raw(NEGATOME_S3, 'pdb_stringent', version, '.txt', date)


def get_negatome_literature_negatives_path(
        version: str = NEGATOME_LIT_VERSION,
        date: str | None = None,
) -> str:
    """Return the in-house literature-mined Negatome negative source.

    Resolves literature_negatives_<version>_<date>.txt: the flat two-column TSV
    the mining pipeline's final filter publishes for compile_negatome to read.
    It sits beside the PDB source rather than under NEGATOME_LITERATURE_S3
    because it is a Negatome source file, not a stage output, and it is named in
    full so it cannot be confused with the literature_<version>/ stage tree.

    Args:
        version: Version string, e.g. 'v3'.
        date: Optional explicit date override (YYYY-MM-DD).

    Returns:
        Full S3 path.
    """
    return resolve_latest_raw(
        NEGATOME_S3, 'literature_negatives', version, '.txt', date,
    )


def get_pinder_raw_path(version: str = PINDER_VERSION, date: str | None = None) -> str:
    return resolve_latest_raw(PINDER_S3, 'PINDER_raw', version, '.parquet', date)


def get_pinder_index_path(version: str = PINDER_VERSION, date: str | None = None) -> str:
    # PINDER's index.parquet (interface cluster_id + component clusters), paired
    # by date with the PINDER_raw metadata table from the same gs://pinder release.
    return resolve_latest_raw(PINDER_S3, 'PINDER_index', version, '.parquet', date)


def get_intact_raw_path(version: str = INTACT_VERSION, date: str | None = None) -> str:
    return resolve_latest_raw(INTACT_S3, 'intact-micluster', version, '.txt', date)


def get_ppi3d_prefix(version: str = PPI3D_VERSION, date: str | None = None) -> str:
    """Return the S3 prefix holding one PPI3D pull.

    A pull is a directory rather than a file: PPI3D serves bulk data one
    release-date window at a time, so a full snapshot is many per-window
    CSV/JSON/log triples. The directory is named
    filteredforflock_<version>_<date>/ where date is the day it was taken.

    "filteredforflock" rather than "raw" because the pull is a deliberate subset
    of PPI3D, not a mirror of it: hetero protein-protein interfaces only, from
    assemblies with at least two distinct subunits. Protein-peptide,
    protein-nucleic and domain-domain interactions are excluded, as are
    homo-interactions. Each window's criteria JSON records the exact query.

    Args:
        version: Version string, e.g. 'v1'.
        date: Pull date (YYYY-MM-DD). Defaults to the latest pull in S3 for the
            given version.

    Returns:
        Full S3 prefix ending in '/'.

    Raises:
        FileNotFoundError: If no matching pull directory is found in S3.
    """
    if date is not None:
        return PPI3D_S3 + f'filteredforflock_{version}_{date}/'
    stem_prefix = f'filteredforflock_{version}_'
    candidates = set()
    for path in list_files_from_s3(PPI3D_S3, recursive=True):
        relative = path[len(PPI3D_S3):] if path.startswith(PPI3D_S3) else path
        directory = relative.split('/', 1)[0]
        if not directory.startswith(stem_prefix):
            continue
        pull_date = directory[len(stem_prefix):]
        if len(pull_date) == 10 and pull_date[4] == '-' and pull_date[7] == '-':
            candidates.add(directory)
    if not candidates:
        raise FileNotFoundError(
            f'No PPI3D {version} pulls found in {PPI3D_S3}',
        )
    return PPI3D_S3 + sorted(candidates)[-1] + '/'


# Convenience wrappers for processed (output) files
def get_flock_negatome_path(date: str | None = None) -> str:
    return resolve_latest_raw(FLOCK_S3, 'negatome_pairs', NEGATOME_PDB_VERSION, '.csv', date)


def get_flock_pinder_path(date: str | None = None) -> str:
    return resolve_latest_raw(FLOCK_S3, 'pinder_pairs', PINDER_VERSION, '.csv', date)


def get_flock_ppi3d_path(date: str | None = None) -> str:
    return resolve_latest_raw(FLOCK_S3, 'ppi3d_pairs', PPI3D_VERSION, '.csv', date)


def get_flock_ppi3d_interfaces_path(date: str | None = None) -> str:
    # Per-interface table with the PRODIGY-cryst BIO call and the PPI3D interface
    # cluster. The cofolding benchmark's homolog check needs it to scope a pair's
    # query clusters to the interfaces that made it a positive.
    return resolve_latest_raw(FLOCK_S3, 'ppi3d_interfaces', PPI3D_VERSION, '.csv', date)


def get_flock_ppi3d_cluster_dates_path(date: str | None = None) -> str:
    # Earliest release date per PPI3D interface cluster, computed over the whole
    # pull before Flock-specific filtering. The homolog check needs unscoped
    # membership: a model saw every pre-cutoff structure in a cluster.
    return resolve_latest_raw(
        FLOCK_S3, 'ppi3d_cluster_dates', PPI3D_VERSION, '.csv', date,
    )


def get_flock_v1_path(date: str | None = None) -> str:
    return resolve_latest_raw(FLOCK_S3, 'flock', FLOCK_VERSION, '.csv', date)


def get_intact_pairs_path(date: str | None = None) -> str:
    return resolve_latest_raw(FLOCK_S3, 'intact_positive_pairs', INTACT_VERSION, '.csv', date)


def get_cofolding_benchmark_path(name: str, date: str | None = None) -> str:
    """Return the S3 path for a versioned cofolding benchmark CSV.

    All cofolding benchmark files live under COFOLDING_BENCHMARK_S3 and follow
    the same <name>_<FLOCK_VERSION>_<date>.csv naming convention. Valid names:
      leakage_free_targets, leakage_free_pairs,
      leakage_free_targets_annotated, leakage_free_pairs_annotated,
      curated_cofolding_benchmark_targets, curated_cofolding_benchmark_pairs,
      leaked_targets, leaked_pairs, leakage_free_capped_targets,
      leakage_free_capped_pairs, leaked_matching_report,
      leaked_targets_annotated, leaked_pairs_annotated,
      leakage_free_capped_targets_annotated,
      leakage_free_capped_pairs_annotated.

    The leaked_* and leakage_free_capped_* pair are the two arms of the leakage
    comparison, both capped at a combined residue budget; leakage_free_capped is
    therefore not the same file as leakage_free, which carries no cap.

    Args:
        name: File stem (see above).
        date: Optional explicit date override (YYYY-MM-DD).

    Returns:
        Full S3 path.
    """
    return resolve_latest_raw(COFOLDING_BENCHMARK_S3, name, FLOCK_VERSION, '.csv', date)


def get_negative_leakage_path(date: str | None = None) -> str:
    """Return the S3 path for the per-pair negative leakage table.

    Written by flock.cofolding_benchmark.sequence_homology and read by both
    leakage_free_set and annotate. It lives under the cofolding benchmark
    prefix like the artifacts it feeds, and is a parquet rather than a CSV
    because it carries one row per Flock negative pair.

    Args:
        date: Optional explicit date override (YYYY-MM-DD).

    Returns:
        Full S3 path.
    """
    return resolve_latest_raw(
        COFOLDING_BENCHMARK_S3, 'negative_leakage', FLOCK_VERSION,
        '.parquet', date,
    )


def get_literature_stage_prefix(stage: str) -> str:
    """Return the S3 prefix for one stage of the literature Negatome v3 pipeline.

    Args:
        stage: A key of LITERATURE_STAGE_VERSIONS.

    Returns:
        Full S3 prefix, e.g. '<...>/negatome/literature_v3/screen/v1/'.

    Raises:
        ValueError: If the stage is not a known pipeline stage.
    """
    if stage not in LITERATURE_STAGE_VERSIONS:
        raise ValueError(
            f'Unknown literature stage {stage!r}; expected one of '
            f'{tuple(LITERATURE_STAGE_VERSIONS)}.',
        )
    return NEGATOME_LITERATURE_S3 + f'{stage}/{LITERATURE_STAGE_VERSIONS[stage]}/'


def get_literature_run_raw_prefix(stage: str, run_id: str) -> str:
    """Return the S3 prefix archiving one stage run's raw API results.

    Distinct from NEGATOME_LITERATURE_RAW_S3, which archives Europe PMC responses
    shared across stages. This one is per run and holds the batch results, which
    the Anthropic API deletes 29 days after a batch is created, not after it ends.

    Args:
        stage: A key of LITERATURE_STAGE_VERSIONS.
        run_id: The run's content-hashed identifier.

    Returns:
        Full S3 prefix, e.g. '<...>/screen/v1/raw/screen__v1__cue__a3f81c/'.
    """
    return get_literature_stage_prefix(stage) + f'raw/{run_id}/'


def get_negatome_literature_path(
        stage: str,
        name: str,
        ext: str = '.parquet',
        date: str | None = None,
) -> str:
    """Return the S3 path for a versioned literature Negatome v3 output file.

    Every stage output follows the same <name>_<stage version>_<date><ext>
    convention under its own stage prefix, so one parameterised resolver serves all
    of them rather than a wrapper per file. Names in use: corpus/papers,
    corpus/blocks, curation/curated_pairs, pairs/literature_pairs.

    The screen stage has no entry there on purpose. It publishes only its raw batch
    results, under get_literature_run_raw_prefix, and keeps its parsed records local
    until TERN-2301 settles what curation reads from them - a published schema now
    would be one that has to change. This resolver already serves 'hits' when there
    is something to write.

    Args:
        stage: A key of LITERATURE_STAGE_VERSIONS.
        name: File stem before the version, e.g. 'papers'.
        ext: File extension including the leading dot.
        date: Optional explicit date override (YYYY-MM-DD).

    Returns:
        Full S3 path.
    """
    return resolve_latest_raw(
        get_literature_stage_prefix(stage), name,
        LITERATURE_STAGE_VERSIONS[stage], ext, date,
    )


def get_protein_annotation_path(date: str | None = None) -> str:
    return resolve_latest_raw(DIVERSITY_S3, 'protein_annotation', DIVERSITY_VERSION, '.csv', date)


def get_sequence_neighbours_path(date: str | None = None) -> str:
    return resolve_latest_raw(
        DIVERSITY_S3, 'sequence_neighbours', DIVERSITY_VERSION, '.csv', date,
    )


def get_sequence_novelty_path(date: str | None = None) -> str:
    return resolve_latest_raw(
        DIVERSITY_S3, 'sequence_novelty', DIVERSITY_VERSION, '.csv', date,
    )
