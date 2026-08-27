from __future__ import annotations

from flock import AGENT_RUNS_ROOT

# Local working tree for this pipeline: Europe PMC fetch caches, corpus tables and
# batch run directories. Under agent_runs/ rather than /tmp, where the TERN-2296
# full-text cache lived and nearly went missing with it. Every module derives its
# own subdirectory from here; S3 knowledge stays in flock.paths.
RUN_ROOT = AGENT_RUNS_ROOT / 'negatome_lit_v3'

# The pair stage's local working directory and the two per-paper caches the
# corpus scan writes into it. Named here because the scan writes them and the
# pair stage reads them, so neither module owns the contract.
PAIRS_ROOT = RUN_ROOT / 'pairs'
PAPER_ORGANISMS_PATH = PAIRS_ROOT / 'paper_organisms.parquet'
PAPER_DEFINITIONS_PATH = PAIRS_ROOT / 'paper_definitions.parquet'

# The parsed block table every later stage streams over, dated because the corpus
# is append-only and a rebuild becomes the next one. Named here for the same
# reason as the caches above: the corpus stage writes it and three stages read it,
# so a rebuild is one edit rather than a search for every spelling of the date.
CORPUS_BLOCKS_PATH = RUN_ROOT / 'corpus' / 'blocks_v1_2026-08-05.parquet'

# How a paper reached the corpus, and so what text it can be screened on. Named
# here rather than in corpus.py because the screen records them too and has no other
# reason to import the corpus module. Both routes are part of one corpus; the query
# syntax that produces each stays in corpus.py.
SOURCE_FULLTEXT = 'fulltext'
SOURCE_ABSTRACT = 'abstract'
TEXT_SOURCES = (SOURCE_FULLTEXT, SOURCE_ABSTRACT)
