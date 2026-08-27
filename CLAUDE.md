# Flock — project instructions for Claude

## Project overview

The project publishes two datasets, and which one a change belongs to is the first question to ask about it:

- **Flock** — an ungated PPI model **training set**. It admits every pair its sources supply, on both labels: no minimum number of pairs per protein anywhere, no intersection on shared targets. Homodimers are dropped and the negatives are IntAct-filtered.
- **Flock leakage-free** — the co-folding **benchmark**, carved out of Flock by `flock/cofolding_benchmark/`. It alone requires a minimum number of clean pairs per target (`min_clean_positives=2`, `min_clean_negatives=20`) and applies the leakage rules.

`leaked_set.py` adds a third artifact that is neither dataset but a companion to the second: a **leaked** arm of pairs a model trained to the cutoff did see, plus a residue-capped re-derivation of the leakage-free arm matched to it on targets, per-target pair counts and complex size. It exists so a co-folding score on the benchmark has a contrast to be read against, and it is the only place a residue cap appears — that cap is run cost, not a leakage rule, so it applies to both arms.

Both are built from three literature sources:
- **Negatome** — non-interacting pairs ([Blohm et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3965096/))
- **PINDER** — structural PPI dataset ([Kovtun et al. 2024](https://www.biorxiv.org/content/10.1101/2024.07.17.603980v4))
- **PPI3D** — structural interfaces tracking current PDB holdings ([Dapkūnas et al. 2024](https://academic.oup.com/nar/article/52/W1/W264/7645776))

Positives are the union of PINDER and PPI3D; PINDER's snapshot stops at 2024-02-07 and is no longer maintained, so PPI3D supplies everything released since.

The negatome is filtered against **IntAct** positive interactions before building the benchmark, following Blohm et al. 2014.

Negatives come from two in-house v3 sources, labelled `Pdb` and `Lit`. The published Blohm set (`Manual`) is no longer compiled in — the literature-mined source supersedes it — but stays resolvable and reachable through `load_negatome(include=...)` for version comparisons. The literature source is filtered down to its final form by `negatives.py` and published as a flat TSV beside the PDB source, not under the literature stage tree.

## Environment

All code runs in the `flock` conda environment (see [environment.yml](environment.yml)). Use `conda run -n flock ...` or assume it is activated.

## Data

All data lives in S3, not in the repo. S3 paths are defined in [paths.py](paths.py) and rooted at the `FLOCK_S3_ROOT` environment variable. When adding new datasets or reprocessing, bump the version suffix in the relevant processed path (e.g. `v1/` → `v2/`) rather than overwriting in place.

## Repo structure

```
notebooks/           # EDA and analysis notebooks
flock/
  paths.py           # S3 path constants and version strings
  compile_negatome.py
  compile_pinder.py
  fetch_ppi3d.py     # Pull PPI3D by release-date window
  compile_ppi3d.py   # PPI3D interfaces -> positive UniProt pairs
  prodigy.py         # PRODIGY-cryst client (BIO/XTAL), runs in the `prodigy` env
  compile_intact.py  # Download and parse IntAct positive pairs
  intact_filter.py   # Filter pairs against IntAct
  assemble_flock.py
  uniprot.py         # UniProt API utilities
  rcsb.py            # RCSB PDB API utilities
  negatome_v3/       # In-house Negatome v3 sources
    pdb/             # PDB-derived negatives
    literature/      # Europe PMC corpus + LLM screen (corpus.py, batch.py), then
                     # vocabulary.py (patterns, term sets, output values),
                     # names.py (reviewed-UniProt name index + normalisation),
                     # organisms.py (species evidence per paper),
                     # pairs.py (union -> filter -> resolve -> species -> finalise),
                     # accessions.py (model pick over the ambiguous accessions),
                     # curation.py (agentic curation of the resolved pairs) and
                     # negatives.py (final filter -> the Negatome source file)
  cofolding_benchmark/  # The benchmark: the leakage-free subset for co-folding
                     # evaluation (leakage_free_set.py, date_flags.py, the two
                     # interface-cluster modules for positives, and
                     # sequence_homology.py for the negative-side leakage rules),
                     # plus leaked_set.py: the leaked comparison arm and the
                     # residue-capped leakage-free arm it is matched to
```

## Code conventions

- `flock` is installed as a dev package (`pip install -e .`); use `from flock.xxx import ...` for imports
- Type hints on all function signatures; `from __future__ import annotations` at the top of every file
- Google-style docstrings (Args / Returns / Raises)
- No tests directory for now
- Do not add features or refactor beyond what is asked
