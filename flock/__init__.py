from __future__ import annotations

from pathlib import Path

# Local working tree for anything too big or too transient for S3: agent runs, API
# response caches, batch run directories. Gitignored. Derived once here rather than
# by counting parent directories in each module that wants it, since that count is
# wrong the moment a package moves a level deeper.
REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_RUNS_ROOT = REPO_ROOT / 'agent_runs'

# Bump these when reprocessing a source or rebuilding the combined dataset
NEGATOME_PDB_VERSION = 'v3'
NEGATOME_LIT_VERSION = 'v3'
PINDER_VERSION = 'v1'
PPI3D_VERSION = 'v1'
FLOCK_VERSION = 'v1'
INTACT_VERSION = 'v1'
DIVERSITY_VERSION = 'v1'
