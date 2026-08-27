from __future__ import annotations

# What must agree between the features a query was extracted into and the ones the NPMI
# table was counted over. It lives here rather than in schema.py because it is not a
# column contract: score.py compares against it, io.py reports it back to the user, and
# score_pairs.py decides on it whether there is anything to compare at all.
#
# Kept whole so a table's header can be reported back even though a query extracted
# outside this package carries no counterpart to compare against.
EXTRACTION_FEATURE_SPACE_KEYS = (
    'esmc_repo', 'esmc_revision', 'sae_repo', 'sae_revision',
    'transformers_revision', 'esm_revision', 'fused_kernels', 'statistics_md5',
    'image_commit', 'layer', 'codebook_size', 'top_k', 'threshold',
)
