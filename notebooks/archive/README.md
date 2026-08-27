# Archived notebooks

Notebooks describing benchmark sets that the two-dataset re-scope (TERN-2710, merged as `1f5c0af`)
superseded. They are kept for their results; none of them describes the current state of Flock or the
leakage-free benchmark.

Each notebook opens with a banner naming the exact dated artifacts it ran against and how that set
relates to the current one. Outputs are preserved as they were last run.

| notebook | benchmark | last run |
| --- | --- | --- |
| `boltz_curated_full_length_eda.ipynb` | curated 13-target, Boltz-2 hosted API + ipSAE | 2026-06-16 |
| `boltz_leakage_free_eda.ipynb` | 102 targets of the former 445-target leakage-free set | 2026-06-08 |

`boltz_leakage_free_eda.ipynb` is the one that must not be re-run: the pair file it joined against has
been removed from S3, so its loader now resolves to the 2026-08-18 335-target set and would join June
scores onto an August spine.
