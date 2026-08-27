# Flock

Two datasets of protein-protein interaction (PPI) pairs, built by one pipeline:

- **Flock** — an ungated **training set** for PPI models. Every pair its sources supply is admitted, on both labels: there is no minimum number of pairs a protein must have on either label, and no intersection on shared targets. Homodimers are dropped and the negatives are IntAct-filtered.
- **Flock leakage-free** — the co-folding **benchmark**, carved out of Flock. It alone carries the per-target discrimination gate (2 clean positive and 20 clean negative partners) and the leakage rules that make a prediction a genuine test of generalisation.

The distinction is worth holding while reading the rest of this file. The only such minimum left in the project belongs to the benchmark, and it counts clean partners *after* leakage filtering rather than before it; the compile steps below filter nothing on pair counts.

Both draw from three literature sources:

- **Negatome** — manually curated non-interacting protein pairs ([Blohm et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3965096/))
- **PINDER** — structural PPI dataset ([Kovtun et al. 2024](https://www.biorxiv.org/content/10.1101/2024.07.17.603980v4))
- **PPI3D** — structural interfaces tracking current PDB holdings ([Dapkūnas et al. 2024](https://academic.oup.com/nar/article/52/W1/W264/7645776))

Positives are the **union of PINDER and PPI3D**. PINDER's snapshot stops at 2024-02-07 and
is no longer maintained, so it cannot see anything released since; PPI3D is updated weekly.
The evaluation behind that decision is in [notebooks/ppi3d_vs_pinder.ipynb](notebooks/ppi3d_vs_pinder.ipynb)
and [notebooks/prodigy_vs_tabular.ipynb](notebooks/prodigy_vs_tabular.ipynb).

Negatome pairs are filtered against **IntAct** — a curated database of experimentally validated PPIs — to remove known positive interactions that would otherwise be false negatives in the benchmark (following Blohm et al. 2014).

## Environment

```bash
conda env create -f environment.yml
conda activate flock
pip install -r requirements.txt
pip install -e .
```

Once installed, the code is importable as `from flock.xxx import ...` from anywhere, including the notebooks in [notebooks/](notebooks/).

The LLM screen in [flock/negatome_v3/literature/](flock/negatome_v3/literature/) additionally needs the Anthropic SDK and Pydantic, which are not in `requirements.txt` because the model-calling stages are the only part of the repo that uses them and the rest of the pipeline installs without them:

```bash
pip install 'anthropic>=0.69,<1' 'pydantic>=2'
```

Without them, `python -m flock.negatome_v3.literature.batch` fails at import with `ModuleNotFoundError`. Version 0.120.2 of the SDK is what the screen has been run on.

## Data

All raw and processed files live in S3, under a prefix you control. Set the `FLOCK_S3_ROOT` environment variable to your own bucket before running any of the scripts below:

```bash
export FLOCK_S3_ROOT=s3://your-bucket-name/flock/data/
```

Rather than hard-coding paths, use the resolver helpers in [flock/paths.py](flock/paths.py), which pick the latest date-stamped file for the version currently declared in [flock/__init__.py](flock/__init__.py):

| Helper | Resolves to |
|---|---|
| `get_pinder_raw_path()` | PINDER raw metadata parquet (version from `PINDER_VERSION`) |
| `get_negatome_pdb_path()` | Stringent PDB-derived Negatome file (`NEGATOME_PDB_VERSION`) |
| `get_negatome_lit_path(version)` | Published Negatome 2.0 manually curated file; `version` is required and is `'v2'`. Still resolvable, no longer compiled in |
| `get_negatome_literature_path(stage, name)` | Any literature Negatome v3 stage output (`NEGATOME_LIT_VERSION` + `LITERATURE_STAGE_VERSIONS`) |
| `get_negatome_literature_negatives_path()` | Published in-house literature Negatome source file (flat TSV, beside the PDB source) |
| `get_intact_raw_path()` | IntAct micluster file (`INTACT_VERSION`) |
| `get_intact_pairs_path()` | IntAct positive UniProt pairs lookup CSV |
| `get_flock_negatome_path()` | Compiled Negatome pairs (IntAct-filtered) |
| `get_flock_pinder_path()` | Compiled PINDER positives (tag-filtered) |
| `get_flock_v1_path()` | Combined Flock v1 benchmark dataset |
| `get_pinder_index_path()` | PINDER interface index parquet (`cluster_id`; `PINDER_VERSION`) |
| `get_ppi3d_prefix()` | PPI3D pull directory (`PPI3D_VERSION`) — a prefix, not a file |
| `get_flock_ppi3d_path()` | Compiled PPI3D positives (BIO-classified, tag-filtered) |
| `get_flock_ppi3d_interfaces_path()` | Per-interface PPI3D table with BIO call and interface cluster |
| `get_cofolding_benchmark_path(name)` | Any cofolding benchmark CSV by stem name (see below) |

All processed CSVs carry `# key=value` provenance headers — read them with `pd.read_csv(..., comment='#')`.

## Structure

```
notebooks/          # EDA and analysis notebooks
flock/
  paths.py          # S3 path constants and version strings
  rcsb.py           # RCSB PDB search and data API utilities
  compile_negatome.py
  compile_pinder.py
  ppi3d.py          # PPI3D bulk-download client (form POST + job poll)
  fetch_ppi3d.py    # Entry point: pull PPI3D by release-date window -> S3
  compile_ppi3d.py  # Entry point: PPI3D interfaces -> positive UniProt pairs
  prodigy.py        # PRODIGY-cryst client (BIO/XTAL), runs in the `prodigy` env
  compile_intact.py # Download and parse IntAct positive pairs
  intact_filter.py  # Filter negatome pairs against IntAct
  assemble_flock.py
  uniprot.py        # UniProt API utilities
  negatome_v3/               # In-house Negatome v3 sources, one subpackage per source
    pdb/                     # PDB-derived negatives (chain pairs with Cbeta > 8 A)
      build.py               # Entry point: scan heterocomplex bio-assemblies
      metadata.py            # PDB/SIFTS metadata and heterocomplex selection
      structure_filtering.py # Interchain distance measurement and pair filtering
    literature/              # Literature-mined negatives (Europe PMC + LLM screen)
      corpus.py              # Entry point: negation queries -> papers + parsed blocks
      batch.py               # Entry point: submit / poll / collect a screening run
      runs.py                # A run directory: config, manifest, requests, results
      config.py              # The settled screen config and its content-hashed run_id
      europepmc.py           # Europe PMC search, metadata and full-text fetching
      jats.py                # JATS XML -> section-tagged text blocks
      cues.py                # Negation cue patterns and sentence splitting
      packaging.py           # Which text a run sends, per input level
      schemas.py             # The screen's structured-output contract
      grounding.py           # Verbatim-excerpt gate
      vocabulary.py          # Patterns, term sets and output values in one place
      names.py               # Reviewed-UniProt name index and normalisation
      organisms.py           # Species evidence per paper
      pairs.py               # Entry point: union -> filter -> resolve -> species -> finalise
      accessions.py          # Entry point: model pick over the ambiguous accessions
      curation.py            # Entry point: agentic curation of the resolved pairs
      negatives.py           # Entry point: final filter -> the Negatome source file
      screen_v1.md           # The frozen screening prompt
      pick_v2.md             # The frozen accession-pick prompt
      curate_v2.md           # The frozen curation rubric
  cofolding_benchmark/       # Leakage-aware subset for evaluating co-folding models
    __init__.py              # CUTOFF_DATE (training-data cutoff)
    leakage_free_set.py      # Entry point: derive + publish the leakage-free set
    leaked_set.py            # Entry point: the leaked comparison arm + its matched reference
    date_flags.py            # in_training_set flag (PDB release-date lookup)
    pinder_clusters.py       # positives' homolog flag (PINDER interface clusters)
    ppi3d_clusters.py        # positives' homolog flag (PPI3D interface clusters)
    sequence_homology.py     # Entry point: negatives' leakage flags (co-presence + MMseqs2)
    annotation/              # AFDB coverage annotation layer
      afdb_coverage.py       # SIFTS observed-residue + AFDB pLDDT>=65 computation
      annotate.py            # Entry point: annotate one arm (--benchmark-set) + publish to S3
  npmi_score/                # Query pair scoring against a published NPMI table
    __init__.py              # EXTRACTION_FEATURE_SPACE_KEYS (the feature-space fingerprint)
    score_pairs.py           # Entry point: score a pair list against an NPMI table
    score.py                 # The scoring maths (eqs 17-18) + the table as a sparse lookup
    query.py                 # Feature Parquet -> per-protein top-k active features
    normalise.py             # IDF normalisation and the active-feature cut (torch-free)
    schema.py                # Table, pair list, scores and feature column contracts
    io.py                    # Config, pair list and feature file resolution
    FEATURES.md              # The feature contract a query extraction must satisfy
configs/
  npmi.yaml         # Tuned scoring parameters for flock.npmi_score
tests/              # pytest suite for flock.npmi_score
requirements.txt    # extra pip dependencies
```

## Reproducing the dataset

Run scripts from the repo root in order:

### 1. IntAct (positive interaction pairs for filtering)

```bash
python flock/compile_intact.py
```

Downloads the IntAct micluster file (~6 GB) from EBI's `/current/` endpoint, extracts positive UniProt interaction pairs from the PSI-MITAB format, and uploads a lookup CSV to `INTACT_PAIRS`. Must run before compiling the negatome. The `/current/` URL always points to the latest IntAct release — v1 of this dataset was accessed on 2026-04-14.

| Metric | Value |
|---|---|
| Total rows parsed | 1,173,270 |
| Unique positive UniProt pairs | 1,048,301 |
| Unique UniProt accessions | 110,316 |

### 2. Negatome (negative pairs)

```bash
python flock/compile_negatome.py
```

Downloads the two in-house Negatome source files (PDB v3 and literature v3), deduplicates pairs, filters against IntAct positive interactions, and writes a CSV of directed `(Target, Negative, source)` pairs to `FLOCK_NEGATOME`. Every pair is expanded into both directions, so the output is a symmetric edge list and `Target` carries no information beyond which way a row happens to be written.

**There is no minimum number of negatives per protein.** Flock is a training set and admits every pair its sources supply. `min_negatives`, and the contract that it be kept in sync with `min_clean_negatives` in the cofolding benchmark, are both gone — parameter, provenance key and all. The only such minimum left in the project belongs to the leakage-free benchmark, which counts clean partners after leakage filtering rather than before it, so a threshold here could only starve it. `count_negatives_per_protein` survives with no caller, because the EDA notebooks read the distribution of negatives per protein and this is the honest place to compute it.

**The published Negatome 2.0 `Manual` set is no longer compiled in.** `DEFAULT_SOURCES` is `('Pdb', 'Lit')`: the in-house literature source supersedes Blohm et al.'s hand curation rather than extending it, and mixing a superseded dataset into the benchmark is not what the source column is for. Pass `include=ALL_SOURCES` to `load_negatome` to get it back for the version comparison the EDA notebooks make.

`source` names every source supporting a pair, comma-joined and sorted: `Pdb`, `Lit`, or both. `Lit` is the in-house literature-mined set — `Manual` is literature-derived too, being Blohm et al.'s hand curation, so the label names the pipeline rather than the medium. The literature source arrives already filtered against the pairs the PDB source holds, so `Lit` never appears combined with it in practice.

The three source files have different layouts — the published Manual v2 file is a bare TSV, the in-house PDB v3 file written by `flock/negatome_v3/pdb/build.py` carries a `ProteinA / ProteinB / PDB_Code` header row, and the literature v3 file written by `flock/negatome_v3/literature/negatives.py` carries `#` provenance lines above a `uniprot_a / uniprot_b` header row — so `load_negatome` sets the header row per source and skips comment lines for all three.

**The header row index is load-bearing, and nothing catches it any more.** Reading a headered file headerless silently admits a literal `(ProteinA, ProteinB)` pair, which was the case for the PDB source up to and including the `flock_v1_2026-06-02` build. It did no harm then only because the two fake accessions have one partner each and never cleared `min_negatives`. That safety net went with the minimum: a header regression in `load_negatome` would now ship a phantom pair straight into the dataset, and the per-source header index is the only thing between the two.

`load_negatome` resolves each source's path only if that source is included, because the literature stage calls it with `include=('Pdb',)` to find what the Negatome already held before its own output joined it — at which point the file it is about to write does not exist.

| Metric | Value |
|---|---|
| Pairs removed by IntAct filter | 22,403 |
| Unique undirected pairs in output | 378,877 |
| Directed rows in output | 757,754 |

(Figures from the `negatome_pairs_v3_2026-08-18` build, over `pdb_v3_2026-05-26`, `literature_negatives_v3_2026-08-18` and `intact_positive_pairs_v1_2026-08-17`.)

#### Rebuilding the Negatome PDB source

The stringent PDB-derived Negatome (`pdb_v3_<date>.txt`) that `compile_negatome` consumes is produced by `flock/negatome_v3/pdb/build.py`, which scans all heterocomplex PDB bio-assemblies for chain pairs whose Cβ atoms remain >8 Å apart and aggregates them to the protein-pair level. This is a long-running rebuild (~50–70h on a 32-vCPU box for the full ~19k heterocomplex set) and is normally only re-triggered when the aggregation logic, the SIFTS mapping, or the set of relevant heterocomplexes materially changes.

#### Validating changes locally first

Two canonical PDBs exercise the chain → protein aggregation logic and should be re-checked before any full re-run:

- **7ju4** — must NOT emit `(A0A0A1GYE8, P04690)` as non-interacting (FAP172/TUBB1 interact via 8 of their 14 chain pairs; a correct protein-level aggregator drops the pair from the negative set)
- **1u0n** — the canonical Blohm spec example. Must emit exactly one pair, `(P07359, P22029)`, the botrocetin α / GpIb α interaction

Pattern: load the SIFTS PDB-chain → UniProt map via `_fetch_pdb_metadata()`, download the bio-assemblies via `flock.structure.download_all_bio_assemblies`, and run `_process_pdb_file` on each. Assert the per-case predicate. Both cases must pass before launching the full run.

#### Running the full pipeline

The pipeline is a single long Python invocation, not a chunked / resumable workflow — worker temp directories don't persist across restarts, so the whole run has to be one job. Run it on a long-lived, CPU-only compute node with a multi-day timeout (a full run over the ~19k relevant heterocomplexes takes on the order of 50–70h on a 32-vCPU box):

```
python -m flock.negatome_v3.pdb.build -o <output-dir> -c <chunk-size>
```

Set `chunk_size` larger than the total relevant-PDB count for a full run. The pipeline downloads bio-assemblies on the worker, processes them with a `ThreadPoolExecutor`, and uploads `pdb_v3_<date>.txt` to `NEGATOME_S3` itself.

The job is GIL-bound (BioPython parse + dict aggregation), so a 32-vCPU box averages ~1.3 effective cores. GPU is unused; a CPU-only node is the right choice.

#### Handling the large-PDB tail

Bio-assembly cifs >500 MB are skipped by default to bound per-worker memory. The 40-odd oversize PDBs need separate per-PDB runs with more memory available (~240 GB for the largest assemblies, e.g. a 2.2 GB cif):

- `--pdb-ids <id>` to target a single PDB
- `--max-file-size-mb 5000` to disable the skip
- `--output-suffix _large_<id>` to namespace the output file (otherwise it would clobber the main run's `pdb_v3_<date>.txt`)

Each large-PDB job runs in the multi-hour range; budget a multi-day timeout on these too.

The list of oversize PDBs is best captured by probing the running main job mid-flight (`find <pdbs_dir> -size +500M`) once the download phase is complete. Reconstructing it from RCSB metadata after the fact is possible but requires ~19k API calls.

#### Merging the outputs

After the main run and all per-PDB tail jobs land in S3, the per-PDB outputs need to be concatenated and deduplicated into a single canonical `pdb_v3_<date>.txt` for `compile_negatome` to pick up via `get_negatome_pdb_path()`. The merge is straightforward:

1. List all `pdb_v3_<date>_*.txt` outputs in `NEGATOME_S3` (main + per-PDB tail)
2. Concat into a single DataFrame
3. Filter `ProteinA != ProteinB`
4. Group by `(ProteinA, ProteinB)`, aggregating the `PDB_Code` column by comma-joining unique codes (the same `_merge_pdb_codes` helper used by `pdb/build.py:_append_to_tsv`)
5. Sort by `(ProteinA, ProteinB)` for determinism
6. Upload to `NEGATOME_S3` with no suffix (e.g. `pdb_v3_<merge_date>.txt`)

`resolve_latest_raw` picks the freshest date automatically, so once the merged file is uploaded, every downstream consumer (`compile_negatome`, `assemble_flock`, the EDA notebooks) will use it.

#### Rebuilding the Negatome literature source (v3)

The literature-derived Negatome v3 is a new dataset mined from Europe PMC with an LLM screen, and it supersedes the published Negatome 2.0 `manual_stringent` file rather than extending it. `compile_negatome` reads this source and the PDB source; the v2 Manual file is still named by its own `NEGATOME_MANUAL_VERSION` constant and still resolvable, but no longer contributes rows.

The pipeline is corpus → screen → pairs → accessions → curation, and each stage has its own version in `LITERATURE_STAGE_VERSIONS` so that re-screening a fixed corpus does not force a corpus rebuild. Outputs live under `NEGATOME_LITERATURE_S3` (`negatome/literature_v3/`), resolved with `get_negatome_literature_path(stage, name)`.

A sixth step, `negatives.py`, is not a stage in that tree. It takes the curated table and writes the flat two-column source file `compile_negatome` consumes, published as `literature_negatives_v3_<date>.txt` next to `pdb_v3_<date>.txt` under `NEGATOME_S3` and resolved with `get_negatome_literature_negatives_path()`. It publishes there rather than under a stage prefix because what it writes is a Negatome source, not a mining artifact.

```bash
python -m flock.negatome_v3.literature.negatives \
  --source agent_runs/negatome_lit_v3/pairs/literature_pairs_curated_v1_<date>.parquet --upload
```

Four filters run over the distinct pairs at least one paper judged usable as a negative, each recorded in the audit table rather than deleted, and each charged to the first filter that catches it:

| Filter | Why | Pairs dropped |
|---|---|---|
| `contradicted` | Another paper judged the same pair not usable. Dropped rather than resolved in favour of usable — the two readings are of different papers and nothing in the table says which is right. All 436 are cross-paper; within-paper disagreement is already resolved by `dedupe_verdicts`. | 436 (4.3%) |
| `intact_positive` | IntAct records the pair as a positive, following Blohm et al. 2014. | 879 (8.7%) |
| `structural_positive` | PINDER or PPI3D has a solved structure for the pair. Not redundant with the IntAct filter, which misses pairs whose only positive evidence is a deposited complex, and not the same check as `assemble_flock`'s conflict rule, which resolves the conflict inside the assembled dataset while leaving the source file that supplied it untouched. | 18 (0.2%) |
| `known_negative` | The pair is already in the in-house PDB Negatome source. Dropped rather than merged into a combined `source` value. **The published Negatome 2.0 Manual set is deliberately not consulted**: it contributes no rows to the dataset, so matching a pair against it would delete a curated negative rather than deduplicate one. This is a new dataset, not an update to Negatome 2.0, and overlap with it is a figure to report rather than a reason to drop. | 43 (0.4%) |

Of the 10,082 distinct usable pairs, 1,376 are dropped and **8,706 are published** (measured, 2026-08-18). The virus-virus drop the ticket that built this also named needs no filter here: `pairs.apply_final_filters` already applies it upstream, and no pair reaching this step has both sides viral. Host-pathogen pairs are deliberately kept.

Two things worth knowing about what these filters cost. The IntAct filter removes the third-protein-bridged negatives the curation rubric was written to keep — IntAct stores complex-derived binary pairs, so a pair that only associates via a third protein is a positive there, and bridged-negative language is around three times as common in the dropped set as in the survivors. The structural-positive filter is a mix: of the pairs it catches that IntAct does not, roughly half are that same bridged class and the rest are genuine curation errors, where a negative scoped to a mutant, a construct or one cell line was marked `full_length_pair`.

#### Where each stage's data lives

| Stage | Module | Published to S3 | Local working files |
|---|---|---|---|
| Corpus | `corpus.py` | `corpus/v1/papers_v1_<date>.parquet`, `corpus/v1/blocks_v1_<date>.parquet` | `agent_runs/negatome_lit_v3/corpus/`, plus the Europe PMC fetch cache under `cache/` |
| Screen | `batch.py` | `screen/v1/raw/<run_id>/<batch_id>.jsonl` only — no table | `agent_runs/negatome_lit_v3/runs/<run_id>/` (`config.json`, `manifest.json`, `requests/`, `records/`) |
| Pairs | `pairs.py` | `pairs/v1/literature_pairs_v1_<date>.parquet` and its `.provenance.json` | `agent_runs/negatome_lit_v3/pairs/screen_pairs*.parquet`, the per-paper `paper_*.parquet` caches |
| Accessions | `accessions.py` | `accessions/v1/literature_pairs_picked_v1_<date>.parquet` and its `.provenance.json`, plus `accessions/v1/raw/<run_id>/` | `agent_runs/negatome_lit_v3/pairs/pick_inputs.parquet`, `pick_answers.parquet`, and the run directory under `runs/` |
| Curation | `curation.py` | `curation/v1/literature_pairs_curated_v1_<date>.parquet` and its `.provenance.json` | `agent_runs/negatome_lit_v3/pairs/curation_inputs.parquet`, `curation_verdicts.parquet`, `uniprot_cards.parquet`, and the run directory under `runs/` |
| Negatives | `negatives.py` | `literature_negatives_v3_<date>.txt` under `NEGATOME_S3`, not under a stage prefix | `agent_runs/negatome_lit_v3/pairs/literature_negatives_audit.parquet` and the dated file's `.provenance.json` |
| — | — | `raw/` — archived Europe PMC responses, shared across stages | `agent_runs/negatome_lit_v3/cache/` (~21 GB, not backed up) |

Publishing is opt-in on the four table-writing steps: `pairs finalise`, `accessions apply`, `curation apply` and `negatives` all need `--upload`, since it writes to the shared bucket. The raw batch jsonl is the exception and uploads as each batch is collected, because the API deletes results 29 days after a batch is created. Curation archives no raw results, having no batches at all: it serves its own requests and writes each paper's answer straight into the run's `records/`.

**The screen deliberately publishes no table of its own.** Its parsed records stay local under `agent_runs/negatome_lit_v3/runs/<run_id>/records/`, and only the raw batch jsonl goes to S3. Publishing a schema before the downstream shape settled would have fixed one that had to change; the pairs stage reads those local records directly. The records carry `pmid` and `pmcid` alongside `custom_id`, so anything consuming them can join to the corpus tables.

**The pairs and accessions stages both stay live, and a consumer picks one.** `pairs` is the deterministic assignment: every accession comes from matching a written name against reviewed UniProt, narrowed by what the paper's own text says. `accessions` is that same table with a model's picks folded in for the names the deterministic pass could not settle alone. Each pair carries `pair_status` — `resolved`, `needs_model_pick` or `needs_curation`, whichever of its two sides needs more — and each side carries a `provenance_a`/`provenance_b` recording how its accession was arrived at: `deterministic_by_name`, `deterministic_by_text` or `model_picked`. Model-picked sides count as resolved on a weaker basis, measured at 85–87% precision against 100% by construction for the deterministic ones, so anything curating this set has to verify a model-picked accession rather than trust it.

**1. Assemble the corpus.** Check the query hit counts first — one request per query, and it is what verifies the Europe PMC filter syntax before a sweep commits to it:

```bash
python -m flock.negatome_v3.literature.corpus --hit-counts
python -m flock.negatome_v3.literature.corpus
```

The sweep runs each query to exhaustion, dedupes to one row per paper keeping which queries matched, drops retracted and non-evidential records, reduces to one row per PMCID (the sweep can only dedupe on the Europe PMC id, and everything downstream joins on the pmcid), then attempts JATS full-text retrieval for every PMCID and parses what returns into section-tagged blocks. Availability is measured rather than read off the `isOpenAccess` / `inEPMC` flags, which disagree with the endpoint in both directions. Every response is cached under `agent_runs/negatome_lit_v3/cache/`, so a resumed sweep costs nothing; deleting the search cache is what re-sweeps against a moved index.

`--queries` and `--max-pages` scope a trial sweep, but they do not change where its output goes: both tables still upload under `corpus/v1/`, and the resolver picks the newest file there. Pass `--no-upload` with them unless the partial corpus is meant to become the current one.

The query set has not yet been validated against known positives. That pre-step — run the queries against the Negatome v2 positive PMIDs and measure what fraction they catch — gives the corpus-entry recall ceiling, which no later stage can recover.

**2. Screen.** Export `FLOCK_ANTHROPIC_API_KEY` first; the runner reads that variable and nothing else, and never falls back to `ANTHROPIC_API_KEY` or an `ant auth login` profile.

```bash
python -m flock.negatome_v3.literature.batch submit --papers <papers.parquet> --blocks <blocks.parquet> --max-batches 1
python -m flock.negatome_v3.literature.batch poll --wait
python -m flock.negatome_v3.literature.batch collect
python -m flock.negatome_v3.literature.batch retry   # only if collect reports failures
```

Submit, poll and collect are separate invocations because a corpus-scale batch takes hours, and all of them are safe to re-run: the run directory's manifest maps each chunk to its batch, so a resumed submit skips what it already paid for. The manifest also records the block table a run screened, by name, size and modification time, and a resume against a different one is refused rather than silently screening two corpora under one run_id. Papers whose retrieved full text is not usable are skipped before packaging, since what survives section selection on a page-scan stub is usually the abstract alone. Batches are chunked on serialized size (200 MB, under the API's 256 MB) as well as on request count; size is what binds, since at cue-filtered packaging the corpus is a few GB of text. Raw results are archived to S3 as each batch is collected, because the API deletes them 29 days after a batch is created — not after it ends.

`--max-batches 1` is the pilot. Collect reports the outcome taxonomy for each chunk and the run — refusals, validation failures, truncations, expiries, the call and confidence mix, and the excerpt grounding pass rate — so the pilot can be read before the rest of the corpus is committed. Requests that came back with no answer are not lost: `retry` rebuilds them from the text recorded in the run's own requests files, so a retry cannot drift from what was first sent and needs no corpus. Its chunks are marked in the manifest and excluded from the item offset a later resume advances by, and a custom_id can therefore appear in more than one chunk's records, most recent last.

`retry` also takes `--model`, to re-send a chunk's still-unanswered requests under a different model than the run's primary one — a fallback for requests the primary model refuses outright, e.g. an effort-parameter mismatch on `claude-haiku-4-5` or `claude-sonnet-4-5` (TERN-2720). `--model` is a call-scoped parameter, not part of `StageConfig`, so it never moves `run_id`; the model actually used is recorded per chunk in the manifest and per record in `records/*.jsonl`, alongside the primary model recorded in `config.json`.

**The screening configuration is settled and should not be reopened:** Sonnet 5 at low effort, cue-filtered packaging, `screen_v1`, at roughly $2,105 on Batch intro pricing over a 261k-paper open-access corpus. The evidence is in TERN-2297 — four experiments covering the model choice, the effort ladder, the packaging and the prompt. Two results matter for anyone tempted to tune it: recall is noise-limited, since three byte-identical replicates scored 39, 41 and 44 of 52 recall-gold rows, and the config omits `thinking` deliberately (which on Sonnet 5 means adaptive), so adding it would change the run fingerprint and de-link the config from its measurement. That covers the primary `submit` config only — a retry's `--model` fallback answers requests the primary model refused, it does not reopen the settled choice.

**3. Build the pair table.** This takes the screen's `no_interaction` calls, unions them across both full-corpus runs, and resolves each written protein name to a reviewed UniProt accession. Run in order:

```bash
python -m flock.negatome_v3.literature.pairs union
python -m flock.negatome_v3.literature.pairs filter
python -m flock.negatome_v3.literature.pairs resolve
python -m flock.negatome_v3.literature.organisms          # writes the per-paper caches
python -m flock.negatome_v3.literature.pairs species
python -m flock.negatome_v3.literature.pairs finalise --upload
```

`filter` applies the deterministic drops that precede mapping — ungrounded pairs, engineered constructs, reagents and controls. `resolve` matches each name against every name UniProt records for a reviewed entry and gives each pair a `pair_status`. `organisms` scans each paper for the species it names and the abbreviations it defines, in one streaming pass over the block table. `species` then re-resolves each side using that evidence, which is what moves pairs into `resolved`; it resolves species **per side, never per pair**, since intersecting the two sides' candidate organisms mislabels host-pathogen negatives. `finalise` applies the reagent blocklist and the virus-virus drop, and writes the dated table plus its provenance.

**4. Pick an accession for the ambiguous names.** The pairs left at `needs_model_pick` have a side that resolves to several reviewed entries: either an orthologue question, where the candidates are one protein across organisms, or an identity question, where they are different proteins. Both are settled by what the paper says, so each side is sent once with the paragraph its excerpt came from and a numbered candidate list.

```bash
python -m flock.negatome_v3.literature.accessions inputs
python -m flock.negatome_v3.literature.accessions submit
python -m flock.negatome_v3.literature.accessions poll --wait
python -m flock.negatome_v3.literature.accessions collect
python -m flock.negatome_v3.literature.accessions retry   # only if collect reports failures
python -m flock.negatome_v3.literature.accessions apply --upload
```

`inputs` reduces those pairs to one row per (paper, name) side — the same name in the same paper means the same protein however many pairs it appears in — re-derives each side's candidates by re-running the resolver, and locates the containing paragraph in one pass over the block table. It writes `pick_inputs.parquet`, and the run's manifest records that file's name, size and modification time, so a resume against a rebuilt table is refused rather than mixing two sets of requests. `apply` must be given the same pair table the inputs were built from, which is what its `--source` default names; joining onto a different one would silently mismatch.

**The pick's configuration is settled:** Sonnet 5 at low effort, the containing paragraph as input, no confidence field, one pass with no replication. Excerpt to paragraph gained 11.5 points of accuracy for 1.29× the tokens; paragraph to full text a further 3.4 for 11.6×. Haiku 4.5 cannot do this below full text, at 41.2% accuracy and 47.1% abstention, which is why `retry` falls back to Sonnet 4.6 rather than cascading further down. Replication and confidence tiering were both measured and rejected — errors here are systematic rather than noise, and a second pass repeats the same wrong accession most of the time.

**5. Curate the resolved set.** This is the stage the benchmark's negatives actually come out of. Every paper carrying a resolved pair is sent as one self-contained packet — its pairs, a UniProt card per accession, how each accession was chosen, and the paper's text with methods dropped — and a tool-using model answers over a multi-turn loop it can spend on live UniProt lookups and on the methods section the packet withheld.

```bash
python -m flock.negatome_v3.literature.curation inputs --source <accessions table>
python -m flock.negatome_v3.literature.curation run --limit 20        # the pilot
python -m flock.negatome_v3.literature.curation status
python -m flock.negatome_v3.literature.curation run                   # the rest
python -m flock.negatome_v3.literature.curation apply --source <accessions table> --upload
```

`inputs` builds one row per paper, carrying the exact text the run will send, so what a verdict is grounded against cannot change under a resume; it records the pair table it was built from beside itself, and `apply` refuses a `--source` that disagrees. Two filters run there: pairs carrying one accession on both sides are dropped, since there is no A–B interaction on them to judge, and papers past `--max-chars` are cut with the cut marked in the text, since a few are longer than any request can hold.

`run` is also the retry: a record carrying an error is not treated as done, so re-running the same command re-attempts exactly those papers, and each invocation writes its own records file. An answer that covers only some of a packet's pairs, or names a pair the packet did not ask about, is an error rather than a short answer — a paper is answered whole or re-attempted whole. `--workers` sets how many papers are in flight, and `--max-cost` stops the invocation starting new papers once it has spent that much. `status` reports spend, turns, tool calls and what is left mid-run.

A paper whose verdicts plus thinking exceed `max_tokens` returns no answer at all, and unlike a refusal that is recoverable: re-run it under a larger `--max-tokens` (up to 128,000 on this model) and fold the result back with `apply --extra-run-id`. `max_tokens` is a `CurationConfig` field rather than a bare constant, so the larger budget resolves to its own `run_id` and a paper answered under it stays distinguishable from one answered under the settled value. `--extra-run-id` is repeatable and last-wins **across every paper the named run holds**, not only the ones that failed, so aiming it at an older pilot would replace that pilot's papers with their older answers rather than adding to them. The production run needed this for four pair-dense papers, re-answered at 64,000 tokens, which returned 63 of their 64 pairs.

`apply` writes the table locally whatever state the run is in, but `--upload` refuses one that is not finished: the dated file under `curation/v1/` is the canonical curated table, so publishing a pilot or a cost-capped invocation would make an incomplete pass current. It refuses on two counts, not one — any paper carrying an error, and any resolved pair left at `no_answer` or `not_judged`.

`--allow-incomplete` overrides that, for the one gap re-running cannot close: a paper the API refuses on bio-category grounds returns the same refusal every time, so the run is as finished as it will ever be. The production run ends with 14 such papers — Ebola, Marburg, anthrax, botulinum and the like, every one host-pathogen or host-toxin — accounting for 18 resolved pairs at `no_answer`, none of them usable negatives. The override does not weaken the check: the gap counts are written into the published provenance under `published_gaps`, and `parameters.allow_incomplete` records that it was used, so a table carrying gaps says so.

**A verdict whose quoted sentence is not a verbatim span of the text the model was shown is voided**, not corrected — the quote is the only part of an answer that can be checked against the paper. The check runs against the packet's body text plus the methods section *only if the model asked for it*, so a real sentence quoted from a section it never requested is recall rather than reading, and does not pass. Literal `\uXXXX` escapes are unescaped before comparing, since models write the six characters of an escape often enough to void real quotes. `apply` then recomputes `usable_as_negative` from the verdict's own fields rather than trusting the model's flag: correct mapping on both sides, a `non_interaction` relationship, a `full_length_pair` qualifier, and grounded evidence.

The stage runs Sonnet 5 at `xhigh` effort with the UniProt-lookup and methods tools. **A multi-turn tool loop cannot go through the Batch API**, so unlike the screen and the pick there is no 50% batch discount here, and cost tracks paper size rather than pair count — budget on papers. Two failure modes are handled rather than left to be found at scale: a paper that returns zero output tokens and no verdicts, billed for input with no error raised, is recorded as a failure and re-attempted; and duplicate verdicts for one pair are deduplicated before anything is written, keeping the stricter of the two where they disagree.

The rubric is `curate_v2`. A superseded rubric can be deleted rather than kept, since every run directory carries a frozen copy of the prompt it ran on and its records stay interpretable from that alone. Two of its `qualifier` values exist because a sentence can read as a clean negative while what is being denied is not a protein-protein interaction at all: `not_a_protein_pair`, where the partner is a promoter or other DNA region, an RNA, a metabolite or a drug, and `indirect_or_functional_only`, where the negative result is a knockdown, depletion or localisation observation rather than a binding test. Neither is a mapping error, so no earlier stage could have caught either, and the recomputed gate excludes both by already requiring `full_length_pair`.

One rule from the curated calibration set is already in the rubric and must stay: a bridged or indirect association is **not** an interaction for this benchmark. Where a paper shows no direct A–B interaction and the only positive it reports is mediated by a third protein, the pair is a genuine negative and is usable. It was the single biggest systematic disagreement between the agent and the curator, and the prompt reproduces the over-rejection at corpus scale unless it is stated.

Most of what this stage curates rests on at least one model-picked accession, which is why the packet states how each accession was chosen and the rubric asks for a mapping verdict on both sides rather than taking them as given.

### 3. PINDER (positive pairs)

```bash
python flock/compile_pinder.py
```

Downloads the raw PINDER metadata parquet, filters to biologically relevant interfaces (`label == BIO`) at resolution ≤4.5 Å, applies interface-quality filters (buried surface area ≥200 Å² **and** ≥4 intermolecular contacts), extracts UniProt pairs, deduplicates, **removes pairs involving a known engineered fusion tag**, and writes a CSV of directed `(Target, Partner)` pairs to the latest `pinder_pairs_v1_<date>.csv` in `FLOCK_S3`. `min_partners` defaults to **1**, which admits every pair and makes the output a symmetric edge list.

The interface-quality thresholds (`min_buried_sasa` and `min_intermolecular_contacts` in `flock.compile_pinder.filter_pinder`) drop questionable interfaces where two chains barely touch and are held together by a third chain inside a larger assembly — e.g. GNB1/OPRK1, whose best PINDER system buries only 74 Å² with a single contact because the receptor is held near Gβ by Gα. Filtering is per-system: a pair maps to many PINDER systems and is deduplicated downstream, so a pair survives if **any** of its systems clears both thresholds (it is judged by its best structure). See the "Interface-quality filter" section in [notebooks/pinder_eda.ipynb](notebooks/pinder_eda.ipynb) for the count this removes and the distributions motivating the thresholds.

The tag blocklist lives in `flock.compile_pinder.KNOWN_TAGS` and covers commonly co-crystallised fusion partners where the PDB interface is an artefact of the construct rather than a real biological interaction — GFP (P42212), DsRed/RFP (Q9U6Y8), GST (P08515), MBP/malE (P0AEX9), yeast SUMO/Smt3 (P40035), and human SUMO-1/2/3 (P63165, P61956, P55854). The filter is intentionally conservative and accepts one known false negative (P0AEX9 × P02916, a real *E. coli* maltose-transporter pair). See the "Engineered tag filter" section in [notebooks/pinder_eda.ipynb](notebooks/pinder_eda.ipynb) for the full analysis of pairs and targets this removes.

| Metric | Value |
|---|---|
| Raw PINDER entries | 2,319,564 |
| After BIO label filter | 1,180,362 |
| After resolution filter (≤4.5 Å) | 1,060,429 |
| After interface-quality filter (≥200 Å², ≥4 contacts) | 812,279 |
| Entries with both UniProts defined | 750,901 |
| Unique UniProt pairs | 44,805 |
| Pairs removed by engineered-tag filter | 43 |
| Unique pairs after tag filter (undirected) | 44,762 |
| Directed rows in output | 66,716 |

(Figures from the `pinder_pairs_v1_2026-08-18` build. 22,808 of the 44,762 pairs are homodimers and contribute one directed row each rather than two, which is why the row count is not exactly double; `assemble_flock` drops them from Flock.)

`min_partners` was **2** until the two-dataset re-scope and is now **1**. It was always inert for the cofolding benchmark, which requires 2 clean *positive* partners per target and so already excluded everything a lower `min_partners` would admit; what it did do was throw away the single-partner pairs a training set wants. The lever that actually widens the benchmark is the PINDER snapshot date.

### 3b. PPI3D (positive pairs, current PDB holdings)

```bash
python -m flock.fetch_ppi3d --cache-dir ppi3d_cache      # ~2.5 h, 104 windows
python -m flock.compile_ppi3d --cache-dir ppi3d_cache \
    --pdb-dir ppi3d_interfaces                            # ~22 h first run
```

PINDER's snapshot ends at 2024-02-07 and the project is no longer maintained, which
capped the leakage-free positive set. PPI3D indexes interfaces from current PDB
holdings and updates weekly, so it supplies the structures released since.

**Fetching.** PPI3D has no REST API; bulk download is a form POST that enqueues a
server-side job, polled for a gzipped CSV. `fetch_ppi3d` chunks the request by
release-date window, caches each window's CSV/criteria/log, and mirrors them to
`PPI3D_S3`. Windows widen for older data (yearly pre-2000, six-monthly to 2015,
quarterly after) — a quarter of recent PDB holds more structures than several years of
the 1980s. `--end` defaults to the snapshot date read from the server; asking beyond it
is silently rejected.

**Compiling.** `compile_ppi3d` keeps hetero interfaces, maps subunits to UniProt via
SIFTS, drops engineered-tag pairs, deduplicates to unique UniProt pairs, then classifies
one representative interface per pair with **PRODIGY-cryst**. Classification is last on
purpose: it costs one structure download each, and ordering it after the cheap filters
takes the work from ~152k interfaces to ~40k.

PPI3D publishes no biological-vs-crystallographic label and one is essential — among
interfaces shared with PINDER, XTAL outnumbers BIO. `prodigy-cryst` pins
`scikit-learn==0.22`, so it lives in a separate env driven as a subprocess:

```bash
CONDA_SUBDIR=osx-64 conda create -n prodigy -c conda-forge -y \
    python=3.8 "scikit-learn=0.22" "numpy=1.20" "biopython=1.79"
conda run -n prodigy python -m pip install --no-deps prodigy-cryst==1.0.1
```

**Republishing without re-running.** `min_partners` shapes only the directed expansion at
the very end, while everything upstream of it — the pull, the quality filters, the UniProt
mapping and above all the PRODIGY-cryst pass over ~39k structures — is unchanged by it.
`--from-interfaces` rebuilds the pair table from an interfaces table already in S3, which
carries the `pair` and `prodigy_class` columns that are the entire input to the pair step:

```bash
python -m flock.compile_ppi3d --cache-dir ppi3d_cache --from-interfaces
```

It takes an optional interfaces date and defaults to the latest. Verified against the
2026-08-04 tables: rebuilding at `min_partners=2` returns the published 27,505 directed
rows identically, which is what makes it safe to reach for when only the threshold moved.

Interface-quality thresholds are **not** calibrated separately: PPI3D's Voronoi contact
area is half buried SASA (Spearman 0.997, ratio 0.481), so its own ingestion floors
already encode PINDER's `buried_sasa >= 200` / `>= 4 contacts`. The filters remove 4
interfaces out of 151,726 — that is the finding, not an oversight.

**Check the SIFTS snapshot before re-running.** `PdbUniprotData` resolves to a dated
snapshot in shared S3, and a stale one drops the newest structures specifically — the
ones this source exists to reach. A four-month-old snapshot covered only 27.9% of 2026
releases; refreshing lifted it to 94.9% and added 1,466 pairs.

| Metric | Value |
|---|---|
| Raw interfaces pulled (1976-08-19 … 2026-07-29) | 151,726 |
| PDB entries | 27,887 |
| Interfaces released after 2024-02-07 | 48,112 (31.7%) |
| After quality filters | 151,722 |
| Both chains mapped to UniProt | 124,875 (82.3%) |
| After engineered-tag drop | 119,830 |
| Unique UniProt pairs classified | 40,319 |
| → PRODIGY-cryst BIO | 18,624 |
| → XTAL | 21,546 |
| → failed to classify | 149 |
| Directed rows in output | 37,248 |
| Pairs absent from PINDER | 4,332 |

(Interface-level figures from the `ppi3d_pairs_v1_2026-08-02` build against the `filteredforflock_v1_2026-07-31` pull. The directed row count is from `ppi3d_pairs_v1_2026-08-18`, republished from the same interfaces table with `--from-interfaces` at `min_partners=1`; PPI3D emits no homodimers, so it is exactly twice the undirected count.)

### 4. Assemble Flock v1

```bash
python flock/assemble_flock.py
```

Combines every pair from the negatome and from the union of the positive sources into a single CSV at `FLOCK_V1`, with a `Type` column (`Positive`/`Negative`) indicating the interaction type. Gene names are carried through from the source files.

**There is no intersection on shared targets.** Flock is a training set, so every pair from every source is kept and the pairs of targets carrying enough clean partners for evaluation are selected later, by the leakage-free build. `Target` and `Partner` are kept as column names for continuity with that build and the cofolding packages, but `Target` is now a labelling artifact rather than a filter: both labels are symmetric edge lists, so grouping by `Target` does not give a classification problem.

**Homodimers are dropped here**, over the assembled frame, rather than in the positive compilers — one line governs the whole dataset and each source table stays faithful to its upstream. They are all positives; the negative sources emit none.

Before writing the output, the assembler resolves cross-source conflicts: any negative pair whose sorted `(Target, Partner)` UniProt tuple also appears in the union of the positive sources (i.e., a `BIO` interface in any PDB, per PINDER or PPI3D) is dropped. This errs on the side of strictness — when a structural source lists a pair as biologically relevant, we don't claim it as a non-interaction regardless of what the PDB-derived Negatome's Cβ-Cβ ≤ 8 Å rule emitted. The Negatome source files (`negatome_pairs_v3_*.csv`, `pdb_v3_*.txt`) are not touched; only the assembled Flock dataset is filtered.

The conflict reflects a definitional mismatch between the two sources: PINDER's BIO label is set by PRODIGY-cryst's 5 Å heavy-atom NeighborSearch + classifier, which is stricter than Blohm et al.'s Cβ-Cβ ≤ 8 Å rule. Whether to change the Negatome rule to match PINDER's metric is a separate decision; until that's settled, the filter is the pragmatic resolution at the dataset level.

| Metric | Value |
|---|---|
| Proteins | 34,959 |
| Positive pairs (undirected) | 26,934 |
| Negative pairs (undirected) | 377,643 |
| Total pairs (undirected) | 404,577 |
| Directed rows in output | 809,154 |
| Homodimers | 0 |

(Figures from the `flock_v1_2026-08-18` build, over `negatome_pairs_v3_2026-08-18`, `pinder_pairs_v1_2026-08-18` and `ppi3d_pairs_v1_2026-08-18`.)

## Monomeric structure filtering

The EDA notebook (`notebooks/pinder_eda.ipynb`) investigated whether PINDER targets could be further filtered based on the availability of monomeric (unbound) structures in the PDB, using the SIFTS PDB–chain–UniProt mapping. Two criteria were evaluated:

- **Strict**: the target has at least one PDB structure containing none of its known binding partners, and ≥2 of its partners each have at least one structure without the target.
- **Relaxed**: for each specific pair (T, X), T has a structure not containing X, and X has a structure not containing T — allowing the pair-specific subset to be pruned rather than removing the target entirely.

The analysis found that the relaxed criterion does not recover substantially more targets than the strict one. More importantly, PDB metadata is a coarse proxy for structural availability — a UniProt accession may map to structures of poor quality, incorrect oligomeric state, or incomplete coverage. Monomeric structure curation is therefore better done manually at the next stage when actually gathering structures for benchmarking.

## Cofolding benchmark (leakage-free set)

Co-folding models (AlphaFold3, Boltz-2, Protenix, …) are trained on the PDB, so evaluating one on a pair whose complex it has already seen measures memorisation rather than generalisation. The cofolding benchmark carves a **leakage-free** subset out of Flock: pairs whose structural information was not available before a training cutoff, on which a model's prediction is a genuine test of generalisation.

**This is the second of the project's two datasets, and the evaluation one.** Everything gated lives here: both leakage rules and the per-target discrimination floor. Flock itself is ungated, so this build starts from the full pair set rather than from one already thinned to the same thresholds upstream.

A single cutoff of **2023-06-01** is used — Boltz-2's last training day, which also covers the earlier cutoffs of AlphaFold3 (2021-09-30) and Protenix. Leakage is keyed on PDB **`release_date`** (not deposition date — models train on what is publicly released) and the cutoff is inclusive (`release_date <= 2023-06-01`). The cutoff lives in `flock.cofolding_benchmark.CUTOFF_DATE`.

Every pair is annotated with two leakage flags:

- **`in_training_set`** — the exact pair was in a pre-cutoff PDB structure. For positives this is the release date of the pair's own PINDER complex; for negatives it is the release date of the raw PDB negatome's per-pair source structures (co-presence, not contact — a negative is two chains seen together but not interacting, which a model can still memorise). A source PDB missing from current holdings is treated as obsoleted, hence pre-cutoff, hence leaked. A literature-mined negative names no structure at all, so its co-presence entries come from the negative leakage table instead (below), unioned with the PDB negatome's rather than used only as a fallback: the date test is source-agnostic, so more evidence strictly sharpens it. Without that table every literature-only negative stays `NA` and is dropped from the clean pool.
- **`homolog_in_training_set`** — a homolog of the pair was in a pre-cutoff structure, detected via PINDER's **interface clusters**: the flag fires if any member of the pair's interface cluster is pre-cutoff. The cluster id encodes both monomer clusters, so the both-sides criterion is built in (a one-sided homolog match is *not* leakage — it still tests generalisation). The interface-cluster form of this check needs a real interacting complex as the query, so it applies to **positives only**. Negatives get their own homolog rule, computed from sequence identity rather than from interface clusters — see below.

  The two halves of the homolog check are scoped differently, deliberately. **Which clusters the query pair belongs to** is taken from the pair's *quality-passing* systems only — the same `compile_pinder.filter_pinder` thresholds (BIO, resolution, buried SASA, contacts) that made the pair a positive in the first place. A pair is a positive because of its biological interface, so that is the interface whose homologs we search for; a crystal-packing contact between the same two proteins must not drag in whatever clusters it happens to fall into. **Which cluster members count as training data** is left unscoped: a model trained on the PDB saw every pre-cutoff structure in a cluster, biological or not.

### Negative-side leakage

A negative carries no complex of its own to date-check unless its source happened to name one, and the literature-mined source never does — its evidence is a sentence. `flock/cofolding_benchmark/sequence_homology.py` computes the negative-side flags from the accessions alone, over both Negatome sources at once:

```bash
python -m flock.cofolding_benchmark.sequence_homology \
    --pairs <pdb source> <literature source> \
    --work-dir <dir> --upload
```

Publishes `negative_leakage_<version>_<date>.parquet` and its provenance JSON under the cofolding prefix, resolvable with `get_negative_leakage_path()`. The provenance names both source files, both reporting floors and both search floors, because this table decides which negatives reach the benchmark and a parquet alone carries none of that. `--reuse-hits` re-cuts the thresholds without searching again, but only against a hit TSV carrying the `.manifest.json` the search writes beside it: a hit file records only hits, so without the manifest there is no way to tell a query that matched nothing from one that was never searched, and the difference is a pair marked clean rather than unchecked.

Two rules, OR'd downstream:

- **Co-presence** — both accessions appear in the same pre-cutoff PDB entry, read from SIFTS. This is the direct counterpart of the source-PDB sets the structural sources supply for their own pairs, and the only way to see leakage for a pair whose source never named a structure. Not subsumed by the homology rule, so both are kept. It fires on 222,520 of 368,937 PDB-source negatives (60.3%) and 14 of 8,706 literature ones (0.16%) — the asymmetry is the sources themselves, since PDB negatives are drawn from entries that by construction hold both proteins. Its *marginal* contribution over the homology rule is 6,905 pairs (1.8%).
- **Sequence homology** — a pre-cutoff PDB entry holds chains matching **both** sides above an identity floor. Query sequences are full-length UniProt (the REST endpoint, not a local human-proteome FASTA, since these accessions are only about 46% human), searched with MMseqs2 against the unique protein chains of `pdb_seqres`.

**The identity floor is 0.40**, matching PPI3D's `cluster_data_40` — the only stated identity threshold anywhere in the pipeline. Query and target coverage floors are both 0.5. PINDER's `cluster_id` supplies no threshold to borrow, because it is not a sequence cut: over 400 sampled clusters, within-cluster identity runs from a 28% 10th percentile through a 44% median to a 92% 90th, and 12% of same-cluster pairs have no detectable alignment at all.

**The two matched chains must resolve to different SIFTS accessions**, and distinctness keys on the accession rather than the chain letter. Two sides that are mutual homologs — RAB27B and NRAS both matching HRAS in 6D56 — otherwise both match the same protein in the same entry and flag each other through it, which establishes nothing. That class is about a third of naive flags, and neither a chain-ID nor a sequence test removes it.

**What the rule does not establish**: that the two chains touch, that they touch through the interface the negative is about, or that the homology covers the interface rather than a shared unrelated domain. It is evidence a model could have memorised a related complex, not that the complex is the one under test — which is the right standard for excluding a pair from a generalisation test, and the wrong one for calling the pair a positive.

**`pdb_seqres` is a new external dataset.** The search needs the *sequence* of each PDB protein chain, and SIFTS (`PdbUniprotData`) carries only chain-to-accession mappings and residue ranges, so it cannot supply the database. `pdb_seqres.txt.gz` is fetched from RCSB (66 MB, ~1.15M records, collapsed to 174,411 unique protein sequences) and cached in `--work-dir`. It is the one input in this pipeline not resolved through `flock/paths.py`.

What it buys over a SIFTS-only route is the *deposited* chain sequence — the construct a model actually saw, truncations and mutations included — rather than the UniProt canonical. It does **not** buy coverage of unmapped chains: `build_accession_entry_index` drops any chain SIFTS cannot resolve to an accession, because the same-protein guard below is keyed on that accession, so the 9.3% of chains with no SIFTS mapping contribute nothing to the rule. Leakage supported only by such chains is missed, and the rule's reach is bounded by SIFTS either way. **Backlog:** fold the dataset into `flock/pdb_metadata.py` as a `PdbMetadata` subclass alongside `PdbUniprotData` and `PdbReleaseDateData`; and settle whether the seqres detour is reducible to comparing query sequences against the UniProt sequences of accessions SIFTS already places in pre-cutoff entries.

The search itself runs looser than either rule (identity 0.25, coverage 0.3) so one search serves every threshold: the floors are applied when the index is built rather than when the search runs, so a threshold can be re-cut without searching again. Entries with no known release date are excluded rather than defaulted to leaked — that default exists for obsoleted entries, which are old, whereas a missing date here overwhelmingly means a release newer than the cached snapshot.

Over all 377,643 negative pairs in the `flock_v1_2026-08-18` build:

| Rule | Pairs flagged |
|---|---|
| Co-presence in a pre-cutoff entry | 222,534 |
| Sequence homology at 40% identity, 0.5 coverage | 283,232 |
| Either | 290,137 |

### What is kept

The two labels treat an unknown flag differently, and deliberately. **Positives are kept only if both flags are explicitly False** — a positive's evidence is its own deposited complex, and if we cannot date it we cannot claim it is unseen. **Negatives are kept when the date flag is False and the homolog flag is not True.** The homolog rule is evidence *for* leakage rather than a property every negative can be scored on: a PDB negative whose sides no MMseqs2 hit reaches has no homolog evidence either way, and demanding an explicit False there would empty the clean negative pool. A positive homolog hit still disqualifies. This is the only asymmetry between the two labels' tests, and it exists because NA means "no such PDB entry" on one side and "cannot be checked" on the other.

`leakage_free_set` requires the leakage table to upload, and asserts it scores every Flock negative before using it. An unscored negative keeps NA on the homolog flag, which the clean mask admits, so a partial table would publish unchecked pairs as clean; `--no-upload` still allows a date-only build locally.

The leakage-free **target set** then keeps every target that still has at least **2 clean positive partners and 20 clean negative partners** after filtering. This is the only minimum on pair counts left in the project, and it is met from the full negatome rather than from a set already filtered to the same threshold upstream. The old contract that `min_clean_negatives` track a `min_negatives` in `compile_negatome` is void — that parameter no longer exists.

### What limits the size of this set

**Both sides now bind, and the negative side binds harder.** This reverses the earlier finding and the reversal is the point, so the old measurement is kept below.

Measured on the `flock_v1_2026-08-18` build: 1,328 proteins carry ≥2 clean positives and 1,680 carry ≥20 clean negatives, but only 335 carry both. So 993 of the 1,328 targets clearing the positive bar fail the negative one, and the two constraints are close to independent rather than nested.

What each rule costs, holding everything else fixed:

| Rule disabled | Leakage-free targets |
|---|---|
| none (the published build) | 335 |
| negatives' date rule | 346 |
| positives' date rule | 380 |
| positives' interface-cluster homolog rule | 434 |
| negatives' sequence-identity homolog rule | 446 |
| both negative rules | 472 |

The negatives' homolog rule is the single largest constraint in the build, worth 111 targets against the positive homolog rule's 99. That is a direct consequence of its reach: it flags 283,232 of 377,643 negative pairs, because the PDB negative pool is built from large assemblies and large assemblies have many homologs in the pre-cutoff PDB.

For contrast, the superseded measurement on the `flock_v1_2026-06-02` build, before the negative-side rules existed: 122 targets had ≥2 clean positives against 950 with ≥20 clean negatives, so 84% of targets clearing the positive bar already cleared the negative one, and dropping the negative date filter entirely moved the count by 5. **Do not quote that conclusion.** It was true of a build whose negatives were filtered on release date alone.

The ceiling above all of this is the PINDER snapshot. Its `release_date` maxes out at **2024-02-07**, so with a 2023-06-01 cutoff the clean-positive window is 8.2 months wide (10,274 PDB entries), and the *entire* PINDER snapshot contains only 999 leakage-free positive pairs. Meanwhile 38,869 PDB entries have been released since that snapshot and are invisible to the pipeline. Re-deriving PINDER-style dimers from post-2024-02 PDB is the single largest available lever — worth roughly 4x the current source window — and no threshold change comes close to it.

```bash
python -m flock.cofolding_benchmark.leakage_free_set
```

This is the single S3 writer for the benchmark. It loads the assembled Flock dataset, the PINDER metadata and interface `index.parquet`, the raw PDB negatome and the whole-PDB release-date table (`flock.pdb_release_dates.PdbReleaseDateData`), annotates both flags in memory (no intermediate file is persisted), and writes two versioned CSVs under the `cofolding_benchmark/` S3 prefix:

- **`leakage_free_targets_<version>_<date>.csv`** — one row per target: gene, clean positive/negative partner counts, the source-assembly chain-count range (`num_chains`, from PINDER `oligomeric_count`), the buried-surface-area range of the clean positive interfaces, and an example source PDB. `example_pdb` is taken from the first partner in **sorted** order; it used to come from set-iteration order, which made the column vary between processes under string hash randomisation, so the values in it move once on the next rebuild and are stable after that. Nothing else in the file depends on the order.
- **`leakage_free_pairs_<version>_<date>.csv`** — every clean positive and clean negative pair for those targets, carrying the leakage provenance flags.

Both files carry the usual `# key=value` provenance header (sources, the `min_clean_positives` / `min_clean_negatives` thresholds, and the leakage method). The PINDER `index.parquet` — which carries the interface `cluster_id`, absent from the metadata parquet — is the one extra input over the rest of the pipeline; it is paired by date with the PINDER metadata from the same release and resolved via `get_pinder_index_path()`.

`--negative-leakage <parquet>` supplies the table `sequence_homology` wrote. Without it every literature-mined negative stays NA on the date flag and no negative carries a homolog flag at all, so it is not optional for a real build.

**The negative homolog flag is written after `combine_homolog_flags`, and getting that order wrong fails silently.** Both interface-cluster steps rebuild `homolog_in_training_set` wholesale, overwriting every non-positive row with NA. Writing the negative flags before them leaves a correctly typed column full of plausible NAs that nothing downstream can tell apart from "never checked".

| Metric | Value |
|---|---|
| Leakage-free targets (≥2 clean positives, ≥20 clean negatives) | 335 |
| Clean positive pairs (undirected) | 783 |
| Clean negative pairs (undirected) | 11,203 |
| Total pairs (undirected) | 11,986 |
| Directed rows in the pairs file | 16,210 |

(Measured on the `flock_v1_2026-08-18` build, the first carrying the negative-side leakage rules, and published as `leakage_free_targets_v1_2026-08-18.csv` and `leakage_free_pairs_v1_2026-08-18.csv`. The build it replaces, `2026-08-04`, held 445 targets over 21,837 pair rows. The PINDER-only build gave 112 targets / 5,406 pairs and is still reproducible with `--no-ppi3d --flock-date 2026-07-31`.)

**43 of the 8,706 literature-mined negatives reach this set**, against zero under the old build. The number is small because the literature source is small and its pairs rarely touch a target carrying 20 clean negatives, but it is the figure the whole re-scope was for: the old gates put it at zero by construction.

Getting that number above zero needed one thing beyond the leakage rules themselves. `pair_in_training_set` returns None for an empty source set as well as a missing one, so a negative whose accessions genuinely share no pre-cutoff entry — which is every literature-mined negative, since its evidence is a sentence rather than a structure — could not be told apart from one that was never checked, and `clean_pair_masks` reads NA as not clean. `apply_negative_copresence_flags` writes the definite verdict the leakage table already holds. Without it the whole source is silently excluded and the target count reads 333 rather than 335.

Adding PPI3D grew the set **4.4x** when it was first added. Measured on the 2026-08-03 build, which held 1,110 clean positive pairs, surviving clean positives by source were **ppi3d 872**, pinder 179, pinder,ppi3d 59 — four in five are pairs PINDER never had. The leverage comes from the leakage filter, not the raw pool: PPI3D adds only ~20% more raw positive pairs (21,993 → 26,325), but nearly all of them are post-cutoff, whereas most PINDER positives are pre-cutoff and get filtered out. Raw-pool size is the wrong quantity to reason about when predicting the size of this set.

Both leakage flags combine across sources, and deliberately not in the same way:

- **`in_training_set`** asks which PDB entries contain the pair, which is source-agnostic — the PINDER and PPI3D source sets are **unioned** before the date test. Building this from PINDER alone silently drops every PPI3D-only positive, since a pair with no known source resolves to NA and NA is treated as not-clean.
- **`homolog_in_training_set`** asks, for a positive, what is in the pair's interface cluster, which only has meaning inside one resource's clustering. It is computed **per source** and reduced by `combine_homolog_flags`: any source saying leaked wins, otherwise any source saying clean with evidence, otherwise NA. A pair absent from a source's clusters yields NA from that source rather than a false "clean". The negatives' sequence-identity flag is written onto the same column afterwards and is not part of that reduction.

(Counts depend on the current state of the upstream sources — rerun the script to refresh them. The leakage-free target count in particular tightens whenever the PINDER interface-quality filter changes, since fewer clean positive partners push more targets below the ≥2 floor.)

### Leaked comparison arm

The leakage-free set measures generalisation; on its own it gives an absolute number with nothing to read it against. A co-folding score of 0.6 on unseen complexes only means something next to the score the same model gets on complexes it did see. `flock/cofolding_benchmark/leaked_set.py` builds that second arm, and the arm to compare it with.

**Read the two cohorts in `match_mode` differently, and quote the paired one.** For a **paired** target leakage really is the only variable: same protein, same per-target pair counts, leaked partners instead of clean ones. For a **substituted** target the protein changes and the match holds only complex size and pair capacity — MSA depth, family and fold, taxonomy, interface topology and monomer difficulty all move with it, and each of those shifts co-folding scores on its own. A score delta over the substituted cohort is a size- and length-matched control, not evidence about leakage, unless those covariates are stratified on as well. On the current build that makes the primary contrast 23 targets deep and the exploratory one 275; the `match_mode` column exists so the two can be split without rebuilding.

```bash
python -m flock.cofolding_benchmark.leaked_set \
    --negative-leakage negative_leakage.parquet \
    --work-dir leaked_set_work
```

**A pair is leaked on positive evidence: `in_training_set` is True OR `homolog_in_training_set` is True.** This is the mirror of `clean_pair_masks`, deliberately not its complement. A pair whose flags are NA on both sides — a positive with no datable complex, a negative no MMseqs2 hit reaches — was never checkable, and belongs to neither arm: "the model may have seen this" is not the claim a leaked arm is making. Both labels use the same rule here, so the asymmetry `clean_pair_masks` needs does not arise; asking for an explicit True removes the ambiguity that made it necessary.

Four things have to hold before the arms can be compared at all — necessary conditions, not sufficient ones; holding protein identity fixed is `match_mode`'s job and none of these speak to it. Each fails silently, so `assert_arms_comparable` checks all four rather than leaving them to the report:

- **Both arms sit under the same residue cap**, `--max-combined-residues` (default **1420**), counted over the whole complex (target + partner). A pair too large for one arm is absent from both, so neither is cheaper to run. This is the run cost, not a leakage rule, which is why it applies to the reference arm too — and why the reference arm is a *re-derivation*, `leakage_free_capped_*`, not the published `leakage_free_*` files. An accession with no resolvable UniProt length fails the cap: a pair that cannot be sized cannot be shown to fit.
- **The two arms share no undirected pair.** Guaranteed by construction — a pair is clean or leaked, never both — so a violation is a bug in the masks, and the assertion is there to catch it.
- **The arms hold the same number of targets.** A reference target with no eligible stand-in is dropped from the reference arm rather than left in it, because the reference arm is built from its own target list and an unmatched target would publish two arms of different sizes.
- **Every target in both arms clears 2 positives and 20 negatives**, the same floor as the published build. Below it a target cannot be ranked at all.

Matching then runs in three steps:

1. **The reference arm.** `clean_pair_masks` AND the cap, re-derived from scratch through the same `clean_partner_graphs` → `compute_leakage_free_targets` path. Capping only removes pairs, so its target set is necessarily a subset of the published 335; `check_against_published` raises if it is not, since that would mean the cap admitted a pair the uncapped build rejected.
2. **Target assignment.** Every reference target that still clears the floor on *leaked* pairs stands in for itself — same protein, leaked partners instead of clean ones, so leakage is the only variable. Where it cannot, the nearest unused leaked candidate by protein length substitutes, drawn only from accessions absent from the reference set. The split is logged and written per target as `match_mode`.
3. **Pair selection.** Per assigned target, the reference target's pair complex sizes are the distribution to hit and the leaked target's partners are the pool. `select_matched_partners` minimises the total absolute difference in complex size over the whole assignment (`scipy.optimize.linear_sum_assignment`) rather than greedily per pair, so the selected set reproduces the reference length distribution instead of clustering on its easiest members. **No randomness is involved** — sorted iteration and an optimal assignment — so there is no seed to record and a rerun on the same inputs gives the same files.

**Per-target counts are floors, not exact equalities, in both arms.** The positive partner graph is undirected, so a pair selected for one target also counts toward the other when both are targets; and a paired target whose leaked pool runs short keeps its pairing rather than being substituted away, which is the right trade but shrinks the arm. Both effects are reported — `n_pos_reference` against `n_pos_leaked` per target in the matching report, and requested-versus-achieved totals in the log — rather than hidden behind a guarantee the build cannot make.

**Source composition is reported, not matched.** Constraining complex size, per-target counts *and* source share at once over-determines the sampler, and the literature-mined negatives are two orders of magnitude rarer than the PDB ones. `summarise_arms` logs the `Positive_source` split and the Negatome `Pdb`/`Lit` split for each arm alongside the complex-size distributions and a two-sample KS statistic, so drift is visible without being silently corrected.

Five artifacts land under the `cofolding_benchmark/` prefix, uploaded unless `--no-upload`:

- **`leaked_targets_<version>_<date>.csv`** / **`leaked_pairs_<version>_<date>.csv`** — the leaked arm.
- **`leakage_free_capped_targets_<version>_<date>.csv`** / **`..._pairs_...`** — the arm it was matched to. Not the published `leakage_free_*` pair, which carries no cap.
- **`leaked_matching_report_<version>_<date>.csv`** — one row per reference target: the leaked target standing in for it, `match_mode`, requested and achieved counts for both labels, and each side's median complex size.

**Both arms carry the published schema**, so the whole downstream chain reads either one unchanged; the additions are appended columns only (`leakage_status` on the pairs, `matched_reference_target` and `match_mode` on `leaked_targets`). The annotation pass takes the arm by name rather than assuming one:

```bash
python -m flock.cofolding_benchmark.annotation.annotate --benchmark-set leaked
```

Whatever runs the co-folding models themselves has to keep the arms apart in its own output paths: the arms share no pairs, so joining the wrong one leaves every row unlabelled, and a result path keyed on only the model config and the target collides between arms at every path they have in common.

Measured on the `flock_v1_2026-08-18` build at a 1420-residue cap:

| Metric | leakage_free_capped | leaked |
|---|---|---|
| Targets | 298 | 298 |
| Undirected positive pairs | 679 | 974 |
| Undirected negative pairs | 9,992 | 12,484 |
| Directed rows | 14,129 | 14,402 |
| Complex size, mean / median | 612 / 569 | 574 / 523 |
| Positive sources | ppi3d 806, pinder 175, pinder,ppi3d 47 | pinder,ppi3d 587, pinder 355, ppi3d 138 |
| Negative sources | Pdb 9,976, Lit 16 | Pdb 12,484, Lit 0 |

Complex-size KS between the arms is D=0.072. **Every figure in that table is over undirected pairs**, complex size included. The arms carry a target on both ends at very different rates — 3,458 of the reference arm's 14,129 rows are a second copy of a pair, against 944 of the leaked arm's 14,402 — so per-row figures would compare the two CSVs' row multisets rather than the complexes they hold, which is what the GPU cost and the per-pair scores are over. The correction is small here (mean 613→612 and 577→574, KS 0.068→0.072) but it is not small by construction, and the double-counting rate is what decides that. Per-target counts match closely — 1,028 reference positives against 1,080 achieved with 1 target short, 13,101 reference negatives against 13,322 with 8 short — because the assignment costs capacity ahead of similarity. The totals run over rather than under for the reason the floors are floors: a pair selected for one target also counts toward the other when both are targets.

**Three numbers here are worth reading carefully.**

The **cap costs the leakage-free arm 37 of its 335 targets**, leaving 298. That is the price of comparability, not a filter on quality.

**Only 23 of the 298 targets are paired**; the other 275 are substituted. The diagnostic in the log says why: of the 298 reference targets, 44 carry ≥2 leaked positives but only 29 carry ≥20 leaked negatives, so the negative side is what makes pairing infeasible. This is a property of the data rather than a tuning failure — a protein reaches the leakage-free set precisely because its structures are recent, which is the same reason it has few leaked partners. Same-target pairing is the design to prefer and mostly not the design the data allows.

**The leaked arm holds ~28% more undirected pairs than the reference arm on nearly the same number of directed rows** (974 + 12,484 against 679 + 9,992, over 14,402 rows against 14,129). The per-target neighbourhoods match; what differs is how much the arms share pairs *between* their targets. The reference arm's 298 targets are densely interlinked, so many of its pairs appear twice in the directed file; the leaked arm's are mostly not. Per-target ranking is unaffected, since it reads each target's own neighbourhood — but the number of distinct complexes to fold, and so the GPU bill, is about a quarter higher on the leaked arm.

**Literature-mined negatives essentially cannot be leaked.** Sixteen reach the reference arm and none reach the leaked one, which follows from the source: its evidence is a sentence, so its co-presence flag fires on 14 of 8,706 pairs and its sides rarely carry a pre-cutoff homolog. The leaked arm is therefore a PDB-negative set in a way the reference arm is not quite.

Because `UniProt` length is not stored anywhere in Flock, `resolve_lengths` builds it from three sources in order of cost — the `--work-dir` cache, the published `protein_annotation_<version>_<date>.csv` under `DIVERSITY_S3` (~21k accessions, built over an older and smaller Flock), then the UniProt accessions endpoint for the remainder. The cache is why a rerun does not repeat the ~35k-accession lookup.

### Annotated set (AFDB pLDDT coverage)

```bash
python -m flock.cofolding_benchmark.annotation.annotate \
    --negative-leakage negative_leakage.parquet [--workers N]
```

Augments the leakage-free targets and pairs CSVs with **AlphaFold DB pLDDT≥65 coverage** for each protein — the fraction of residues that are actually observed (resolved) in the source PDB structures, as determined by SIFTS, whose corresponding AFDB prediction exceeds pLDDT 65. This measures how well AlphaFold models the benchmark-relevant portion of each protein, rather than its full sequence.

For each UniProt accession the script unions the observed residue numbers across all source PDB structures for that accession (PINDER `entry_id` for positives, PDB Negatome `PDB_Code` for negatives), downloads the AFDB model (all fragments), and computes the fraction. SIFTS files are pre-fetched once per unique PDB before the per-accession AFDB pass to avoid redundant S3 downloads.

Writes two new files under the `cofolding_benchmark/` prefix:
- **`leakage_free_targets_annotated_<version>_<date>.csv`** — one row per leakage-free target (335), adds `target_afdb_plddt65_obs`, `pos_partner_afdb_plddt65_obs_mean`, `neg_partner_afdb_plddt65_obs_mean`.
- **`leakage_free_pairs_annotated_<version>_<date>.csv`** — one row per row of the leakage-free pairs file, adds `target_afdb_plddt65_obs`, `partner_afdb_plddt65_obs`.

Positives' source structures come from PINDER `entry_id` **and** PPI3D `pdb_id`. Taking them from PINDER alone would leave most positives with no source structures at all and a meaningless coverage figure — on the 2026-08-03 build, 872 of 1,110 clean positive pairs were PPI3D-only.

Negatives get the same treatment for the same reason. `_build_acc_to_source_pdbs` takes the negative leakage table's shared-entry index alongside the PDB Negatome index, because every literature-mined negative is absent from the latter by construction: its evidence is a sentence, not a structure. It reads `copresence_entries`, which lists every PDB entry holding both sides **regardless of release date** — the date-filtered `copresence_entry` column would be empty here, since a pair only reaches the benchmark if it is clean, and clean means it has no pre-cutoff entry. Without the table those pairs resolve to no source PDBs, no SIFTS-observed residues and NaN coverage.

**Coverage is incomplete for the newest entries.** SIFTS fetches fail for recent releases whose SIFTS files are not yet published, and those accessions get no observed residues and so no coverage figure. On the 2026-08-18 build that leaves 959 of 16,210 rows without a target coverage figure and 1,036 without a partner one — 1,201 rows missing at least one, of which 1,088 are negative. This is the same upstream lag as the `PdbUniprotData` staleness and lands on the same structures. Re-running the annotation once SIFTS catches up fills them in; nothing upstream needs redoing.

Both files are resolved via `get_cofolding_benchmark_path('leakage_free_targets_annotated')` / `..._pairs_annotated`.

### Curated cofolding benchmark

The **curated set** predates the current build and has not been re-derived. It was cut from the 102-target leakage-free set of 2026-06-08, which contained many large-assembly proteins (ribosomes, photosystems) where the "positive" interactions are assembly-context artefacts and the negatives are trivial co-deposition bystanders. It is a manually selected subset of 13 targets spanning diverse PPI classes, all with genuine pairwise interfaces, good AFDB coverage, and tractable source assemblies. The set it was cut from now stands at 335 targets, so re-curating against the current build is an open job rather than a refresh of these thirteen.

| Target | Gene | n_pos | n_neg | AFDB coverage |
|--------|------|-------|-------|--------------|
| O94874 | UFL1 | 2 | 48 | 0.90 |
| P0CG48 | UBC | 3 | 153 | 0.95 |
| P62805 | H4C1 | 2 | 112 | 0.92 |
| P84233 | H32 | 2 | 29 | 0.87 |
| P06899 | H2BC11 | 2 | 80 | 0.95 |
| O60814 | H2BC12 | 2 | 33 | 0.95 |
| P02281 | H2B11 | 2 | 43 | 0.81 |
| P06897 | H2A1 | 2 | 46 | 1.00 |
| P0C0S8 | H2AC11 | 2 | 37 | 1.00 |
| P62873 | GNB1 | 6 | 84 | 1.00 |
| Q96N11 | INTS15 | 2 | 33 | 0.97 |
| Q15369 | ELOC | 2 | 46 | 0.89 |
| P04908 | H2AC4 | 2 | 72 | 0.99 |

Resolved via `get_cofolding_benchmark_path('curated_cofolding_benchmark_targets')` / `..._pairs`. Contains 847 pairs (31 positive, 816 negative); inherits all columns from the annotated set.

## Query pair scoring

`flock/npmi_score/` scores protein–protein pairs against a precomputed NPMI lookup table
over ESMC SAE features. Nothing is trained and no structures are involved: contacts are a
build-time input to the table, which is why scoring needs only sequences. It is the scorer
the paper reports on the leakage-free benchmark above, so the two ship and version
together.

### What you need

1. **An NPMI table.** Published alongside the paper. A CSV of `feature_i, feature_j, count,
   npmi` with a `# key=value` provenance header (the same format
   [flock/provenance.py](flock/provenance.py) writes) recording the codebook size and the
   feature space it was counted over.
2. **SAE features for your query proteins.** You produce these. The contract is in
   [flock/npmi_score/FEATURES.md](flock/npmi_score/FEATURES.md) and it is exacting — feature
   ids are only meaningful relative to the table, so a different normalisation gives
   plausible wrong numbers rather than an error. **Extract with `esm` pinned to exactly
   `3.3.0` and no `xformers` installed**, which is the environment the NPMI table was counted
   over; both are required and the second is not implied by the first, since ESM picks up
   fused attention kernels whenever they happen to be importable.
3. **A pair list.** A CSV with `id_a` and `id_b`, optionally `pair_id` and `label`.
   `pair_id` is generated positionally if absent.

Feature extraction needs ESMC and an SAE, which are deliberately **not** dependencies of
this repo: extraction is the caller's, and this package only reads the Parquet it produces.
That also means the `esm==3.3.0` pin cannot be enforced by installing this repo — it is on
you to pin it in the environment you extract in, and to record `esm_revision` and
`fused_kernels` in the Parquet so a mismatch is refused rather than scored.

### Running it

```bash
python -m flock.npmi_score.score_pairs \
    --pairs pairs.csv \
    --features features/ \
    --npmi-table npmi_table.csv \
    --output scores.csv
```

`--features` takes any mix of Parquet files, directories searched recursively, and globs.
`--config` defaults to [configs/npmi.yaml](configs/npmi.yaml), and `--min-count` puts a
floor on the accumulated count of a table pair (default 0.0, the paper's method as stated).

### Output

One row per input pair, in input order:

| Column | Meaning |
|---|---|
| `pair_id`, `id_a`, `id_b` | Carried through from the pair list |
| `score` | Mean NPMI over the selected residue pairs; `NaN` where undefined |
| `n_active_a`, `n_active_b` | Residues with at least one active feature |
| `n_selected` | *T*, the residue pairs the mean was taken over |
| `n_positive` | How many of those *T* carried a positive NPMI |
| `status` | `ok`, or why the pair has no score |
| `label` | Carried through where the pair list had one |

`n_positive` matters: a score whose selected residue pairs were mostly absent from the table
is otherwise indistinguishable from one where every pair carried a real value.

Pairs naming a protein with no features score `NaN` with a status saying so, rather than
being dropped or scored zero — zero would rank them as confident negatives.

### The method

For each residue pair, the best NPMI across the top-*k* active features of each side:

```
R(i, j) = max over a in top_k(i), b in top_k(j) of NPMI(a, b)
```

A pair the table does not hold reads as zero, since the table holds only NPMI above its
threshold and an absent pair is therefore not-positive rather than unknown.

The score is the mean of the *T* largest `R(i, j)`, where

```
T = max(1, min(t_max, floor(rho * min(|A_active|, |B_active|))))
```

The score is symmetric by construction: swapping the arguments transposes *R*, and both *T*
and the mean of the *T* largest values are invariant under that. These are equations 17 and
18 of the paper.

Parameters live in [configs/npmi.yaml](configs/npmi.yaml), tuned by the paper on the PPI3D
training set by AUROC:

| Parameter | Value | Meaning |
|---|---|---|
| `top_k` | 3 | Features per residue the max is taken over |
| `rho` | 0.25 | Fraction of the smaller protein's active residues to score |
| `t_max` | 70 | Cap on the residue pairs contributing to a score |

The table-construction parameters (`alpha`, `min_npmi`) are not here: the table is published
pre-built and its own provenance header records what it was built with. Features are loaded
at the top 15 per residue so a sweep over `top_k` is a prefix slice rather than a second
pass.

**The one failure mode that produces plausible wrong numbers rather than an error** is query
features extracted into a different feature space than the table was counted over. Where the
query Parquet records its own extraction provenance, `check_feature_space` compares it
against the table's header and refuses a mismatch. Where it records nothing — which an
extraction run outside this repo need not — the scorer warns, logs the feature space the
table expects, and proceeds. Recording at minimum `statistics_md5`, `sae_repo`, `layer` and
`codebook_size` is what converts that silent failure into a refusal; see
[FEATURES.md](flock/npmi_score/FEATURES.md).

### Tests

```bash
python -m pytest tests/
```

The repo's only test suite, covering the scoring maths, the feature loader, the
normalisation and the CLI's failure paths.

## Diversity analysis

> **Not in the repo.** `flock/diversity/` and the two notebooks described below do not exist
> on this or any other branch, so none of the commands in this section will run as written.
> What survives is the S3 side: `DIVERSITY_S3` and its three resolvers are in
> [flock/paths.py](flock/paths.py), and the tables themselves are published
> (`protein_annotation_v1`, `sequence_neighbours_v1`, `sequence_novelty_v1`, latest
> 2026-08-05). This section is kept as the design record for the package that produced
> them; treat it as a specification, not as documentation of code you can read.

Two questions the benchmark papers need answered with figures: whether the in-house Negatome v3 is genuinely more *diverse* than the published Negatome 2.0 set or merely 89x larger, and whether the leakage-free and curated splits survive their filtering as usable benchmarks. `flock/diversity/` builds the shared annotation those analyses need, and two notebooks consume it.

Because both notebooks need the same per-protein annotation, it is built once and versioned in S3 under `DIVERSITY_S3` (`data/flock/diversity/`) rather than cached per notebook — otherwise the ~21k-accession UniProt pull and the multi-hour AlphaFold DB download get duplicated and can drift apart.

### 1. Per-protein annotation

```bash
python -m flock.diversity.annotate
```

Collects the union of UniProt accessions across the three Negatome versions and the three benchmark stages (21,063 accessions), pulls every diversity axis in one batched UniProt request (organism, lineage, length, Pfam, InterPro, Gene3D, SUPFAM, protein families, sequence), maps Pfam families to clans using the InterPro `Pfam-A.clans.tsv` file, and derives a superkingdom column. Writes `protein_annotation_<version>_<date>.csv`.

Note that `xref_gene3d` carries CATH superfamilies and `xref_supfam` carries SCOP superfamilies — there is no valid `xref_cath` UniProt field. Note also that `flock.uniprot.fetch_uniprot_fields` passes `to_db='UniProtKB'`: UniProtMapper defaults to `UniProtKB-Swiss-Prot`, which silently drops unreviewed accessions, and most of the PDB-derived Negatome is unreviewed.

### 2. Sequence neighbours and novelty

```bash
python -m flock.diversity.sequence_space --work-dir sequence_space_work
```

Runs one all-vs-all MMseqs2 search at maximum sensitivity (`-s 7.5`) over every protein and derives two tables:

- `sequence_neighbours_<version>_<date>.csv` — the strongest hits per protein, supporting nearest-neighbour identity and the identity distribution. A protein whose nearest neighbour sits at 95% identity is a near-duplicate of something already in the set.
- `sequence_novelty_<version>_<date>.csv` — every protein's closest relative *within each dataset's protein set*. This is what separates "new sequence space" from "the same space, more densely sampled", which is the question a larger negatome has to answer.

Searches use `-c 0.8 --cov-mode 0` (80% mutual coverage) so a shared domain does not make two otherwise-unrelated multi-domain proteins look redundant, and maximum sensitivity because remote homologues at 25% identity are still benchmark redundancy. Results are cached in `--work-dir`, so a rerun after an interrupted upload does not repeat the search. Requires `mmseqs` on `PATH` (`brew install mmseqs2`) and the annotation table from step 1, which supplies the sequences.

Clustering itself lives in `flock/diversity/sequence_clusters.py`, which sweeps identity thresholds from 0.20 to 0.95 and caches MMseqs2 results under the notebook cache keyed by a hash of the sequence set. 0.30 is the primary threshold, the conventional homology boundary.

**Diversity is measured on sequence and family, not structure.** Fold-level clustering (Foldseek over AlphaFold DB models) was implemented and then dropped: it necessarily runs over *predicted* structures rather than experimental ones, and sequence identity is the axis that actually governs redundancy and training-set leakage for a co-folding benchmark. CATH-Gene3D and SCOP-SUPFAM cross-references are still fetched and carried in the exported per-protein table for reference, but no figure rests on them.

### 3. Notebooks

- **`notebooks/negatome_diversity.ipynb`** — `manual_v2` / `pdb_v2` / `pdb_v3` compared on Pfam family, sequence-cluster, sequence-novelty, length, taxonomic and family-pair diversity, plus a section attributing every `pdb_v2` pair absent from `pdb_v3` to the IntAct filter, obsoleted structures, or the TERN-2291 aggregation rule.
- **`notebooks/benchmark_split_diversity.ipynb`** — diversity through the `flock_v1` → `leakage_free` → `curated` funnel, the composition bias each filtering step introduces, and three trivial baselines measuring whether positives and negatives are separable on family or sequence-cluster identity alone, with no folding.

Both are size-controlled: because the sets differ by up to 370-fold, every metric is also **rarefied** to the smaller set's size over 20 replicates, and accumulation curves show whether diversity has saturated. Raw richness counts rise with dataset size regardless of diversity, so the rarefied figures are the ones to quote. The negatome notebook adds a second control, `pdb_v3_pre2014`, restricting v3 to pairs whose source structures all predate Negatome 2.0, which holds a decade of PDB growth fixed and isolates the methodology change.

Figures are exported to `notebooks/figures/` as SVG and PNG alongside the summary tables (`table1`-`table5`). That directory is gitignored: the committed notebook outputs already carry every figure inline, and the tables are regenerated by a re-run.

## Versioning

When reprocessing a source or rebuilding the combined dataset, bump the relevant version constant in [flock/__init__.py](flock/__init__.py) rather than overwriting existing S3 files:

```python
NEGATOME_PDB_VERSION = 'v3'  # the in-house PDB-derived Negatome source
NEGATOME_LIT_VERSION = 'v3'  # the in-house literature-mined Negatome source
FLOCK_VERSION        = 'v1'  # bump whenever any source changes
```

The negatome has two version constants because its sources move independently. `NEGATOME_LIT_VERSION` also sits in the `negatome/literature_v3/` prefix, so bumping it creates a new stage tree as well as a new source filename. The combined `negatome_pairs_<version>_<date>.csv` is versioned by `NEGATOME_PDB_VERSION`, which means adding or changing the literature source does not move its version string — the provenance header names the PDB source, the literature source and the IntAct table, and that is what distinguishes one build from another.

While `v1` is still under active development, its outputs are regenerated in place: a re-run writes a fresh date-stamped file (`resolve_latest_raw` always picks the newest date) and the superseded dated files are deleted from S3, so the version constant is **not** bumped for iterative changes like adjusting a filter threshold. Reserve a version bump for a frozen/published release where the older outputs must remain available.

## License

MIT — see [LICENSE](LICENSE).
