# The feature contract

The scorer takes SAE features you produce yourself and looks up pairs of feature ids in a
precomputed NPMI table. **Feature ids are only meaningful relative to the table they are
looked up in.** If your features come from a different model, a different SAE, or a
different normalisation, the ids will not mean what the table counted, and every lookup
will be wrong — silently, with plausible-looking scores, not with an error.

This document is the contract. Follow it exactly, or the numbers are meaningless.

## The pipeline

```
sequence  →  ESMC-6B  →  layer-60 residual stream  →  SAE  →  raw activations
          →  IDF normalisation  →  threshold at 0.5  →  Parquet
```

| Component | Value |
|---|---|
| Backbone | `EvolutionaryScale/esmc-6b-2024-12` (ESMC-6B) |
| ESM package | **`esm==3.3.0`, pinned** — see below |
| `xformers` | **must not be installed** — see below |
| SAE | `biohub/ESMC-6B-sae-layer60-k64-codebook16384` |
| Layer | 60 — declared by the SAE checkpoint itself, not chosen |
| Codebook size | 16384 |
| SAE sparsity | top-k 64 (the SAE's own k, not the scorer's `top_k`) |
| Normalisation statistics | `max_idf_log10.pkl`, md5 `b75317aab86ba3eecd696b8d8ff2ba2e` |
| Active threshold | normalised activation `> 0.5` |

## Pin `esm` to 3.3.0, and do not install `xformers`

**The NPMI table was counted over features extracted with `esm==3.3.0` and no `xformers`
present. Reproduce both, or your activations are not the ones the table was built from.**

```bash
pip install 'esm==3.3.0'
pip uninstall -y xformers          # if anything pulled it in
python -c "import xformers" && echo 'STILL INSTALLED — uninstall it' || echo 'absent, correct'
```

Two separate requirements, and both matter:

- **The version pin.** `esm` owns the ESMC forward pass, so its version is part of the
  feature space in the same way the SAE checkpoint is. A different release can change the
  layer-60 residual stream the SAE reads, and a changed stream gives different active
  features under the same threshold. Nothing downstream can detect this: the ids stay in
  `[0, 16384)` and the scores stay plausible.
- **The absence of `xformers`.** This is not covered by the version pin. The ESM code path
  uses fused attention kernels *when they are importable*, so an `xformers` that some other
  package's resolution happened to pull into the environment silently changes the stream
  without any change to the pinned `esm` version. Absence is what is required, not a flag or
  a config setting.

Record `esm_revision` as `3.3.0` and `fused_kernels` as `none` in your Parquet metadata (see
[Optional: let the scorer check you](#optional-let-the-scorer-check-you)). The table's header
carries both, so recording them is what turns a mismatched environment into a refusal instead
of a plausible wrong number.

## Normalisation is the part people get wrong

**The normalisation statistics are not in the SAE checkpoint.** `layer_60.safetensors`
carries `idf` and `max` tensors, but every value in both is exactly `1.0` — they are
`register_buffer` defaults. Any built-in normalisation path that uses them is a silent
no-op returning the raw activation. Take **raw** activations from the SAE and normalise
against the published statistics yourself.

The statistics are public and unauthenticated:

```bash
aws s3 cp s3://esm-protein-atlas/v1/normalization/max_idf_log10.pkl . --no-sign-request
```

**Two files are published and they are not interchangeable.** `max_idf.pkl` is a natural-log
pass and `max_idf_log10.pkl` a separate base-10 pass over the corpus — not a conversion of
one another. Inverting both leaves the typical feature 2.3% apart, and 1,681 of the 16,384
`max` values differ outright. The per-feature multipliers differ by 0 to 5.25×, so the
threshold cannot be retuned to compensate. Under the natural-log file about 12 features per
residue clear the threshold; under log10 it is nearer 3.

**Use `max_idf_log10.pkl`.** Verify by md5 — nothing inside either file identifies it, and
the filenames are one token apart:

```bash
md5sum max_idf_log10.pkl   # b75317aab86ba3eecd696b8d8ff2ba2e
```

Never mix one file's `max` with the other's `idf`.

The normalisation and the active-feature cut are shipped as
[`normalise.py`](normalise.py) so you do not have to reimplement
them. It is numpy-only, with no torch dependency:

```python
from flock.npmi_score.normalise import active_features, load_statistics

max_per_feature, idf_per_feature, statistics_md5 = load_statistics('max_idf_log10.pkl')
residues, offsets, feature_ids, magnitudes = active_features(
    rows, raw_feature_ids, raw_magnitudes, max_per_feature, idf_per_feature, threshold=0.5,
)
```

which applies

```
normalised = raw_activation * idf[feature] / max[feature]
active     = normalised > 0.5
```

`load_statistics` refuses an all-ones file, so passing the checkpoint's own buffers by
mistake raises rather than producing unnormalised features.

The threshold of 0.5 is the paper's — *"Features with IDF-normalized activations > 0.5 were
considered active"* — and it applies to the **normalised** quantity. Against a raw
activation it keeps essentially everything and does nothing at all.

## The Parquet contract

One row per residue, with list columns. Not one row per active feature.

| Column | Type | Meaning |
|---|---|---|
| `interface_id` | `string` | Your protein id. Must match `id_a`/`id_b` in the pair list. |
| `side` | `string` | Unread by the scorer. Pin it to any single value. |
| `position` | `int32` | Residue position in the sequence. |
| `feature_ids` | `list<int32>` | Active feature ids for that residue, in `[0, 16384)`. |
| `magnitudes` | `list<float32>` | Their **normalised** activations, aligned to `feature_ids`. |

Rules:

- `magnitudes` holds the normalised value, not the raw one. The scorer ranks features per
  residue by this to take the top-k, so raw magnitudes silently change which features are
  selected.
- Residues with no active feature are **omitted**, not written with empty lists.
- Row order does not matter. The loader groups by protein and sorts by `position`.
- A protein must appear in exactly one file. Appearing in two is refused, since it means
  the files come from different runs and one would silently win.
- Split across as many files as you like — pass a directory, a glob, or explicit paths.

`flock.npmi_score.schema` ships `features_schema()` and `build_table()`, the same writers used to
produce the corpus features, so you can construct a conformant file directly rather than
matching the types by hand.

## Optional: let the scorer check you

If your Parquet carries key-value metadata for any of the feature-space keys, the scorer
compares them against the table's own header and **refuses a mismatch** instead of scoring:

```
statistics_md5, esmc_repo, esmc_revision, sae_repo, sae_revision,
transformers_revision, esm_revision, fused_kernels, layer, codebook_size,
top_k, threshold
```

`features_schema(provenance)` writes them for you. Recording at minimum `statistics_md5`,
`sae_repo`, `layer` and `codebook_size` is strongly recommended: it converts the one failure
mode that produces plausible wrong numbers into an error. With no metadata the scorer warns
and proceeds, because it has nothing to compare against.

## Reproducibility notes

- The SAE weights are float32. Running the backbone in bf16 rather than fp32 moves a small
  fraction of features across the 0.5 threshold — measured at about 0.4% — which shifts
  scores slightly without changing rankings materially. Record which you used.
- Fused attention and normalisation kernels (`transformer_engine`, `xformers`,
  `flash_attn`) change the layer-60 stream the SAE reads when present. `xformers` must be
  absent, as above; the other two were absent when the table was built and should be absent
  for you too. If you are chasing a discrepancy, check which are installed before checking
  anything else.

## Checklist

- [ ] `esm` is pinned to exactly `3.3.0`
- [ ] `import xformers` fails — it is not installed in the extraction environment
- [ ] Statistics file md5 is `b75317aab86ba3eecd696b8d8ff2ba2e`
- [ ] Activations taken **raw** from the SAE, then normalised in your own code
- [ ] Threshold `> 0.5` applied to the normalised value
- [ ] `magnitudes` column holds normalised, not raw, activations
- [ ] Residues with nothing active are omitted
- [ ] `interface_id` values match the ids in your pair list
- [ ] Each protein appears in exactly one Parquet file
