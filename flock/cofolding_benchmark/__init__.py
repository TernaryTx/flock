from __future__ import annotations

# Training-data cutoff for the co-folding leakage annotation. Keyed on PDB
# release_date (NOT deposition date). 2023-06-01 is Boltz-2's cutoff and is the
# latest among the co-folding models we target, so satisfying it auto-satisfies
# AF3, Protenix and ESMFold2 (all 2021-09-30 or earlier).
CUTOFF_DATE = '2023-06-01'
