# Continuation classifier: long-region split experiment

Date: 2026-09-21
Code: `split_region_text` / `max_region_chars` / `filter_gibberish` /
`cross_region_negatives` in `continuation_data.py`; `--variant` /
`--data-variant` / `--max-chars` / `--filter-gibberish` /
`--cross-region-negatives` / `--max-length` / `--batch-size` in
`experiments/continuation_classifier.sh`.
Dataset: `ds-articlereconstruction-20260821`, seed 42, page-disjoint 70/15/15.

## What was tested

Long regions (>400 chars, 744 of 1,868) split into ordered sub-fragments
(`r_x#1`, `r_x#2`, ...) at sentence/whitespace boundaries (backward-snapped,
never exceeding 400 chars). This adds intra-region positives (A#1→A#2) and
extends cross-region chains (A#2→B), while preserving baseline cross-region
semantics exactly (`seq+1`, same article).

## Dataset gain

| dataset | train/val/test pairs | positives (intra + cross) | text len p50 / max |
|---|---|---|---|
| baseline (seed42-balanced) | 1,162 / 234 / 198 | 797 (0 + 797) | ~300 / 4,419 |
| split400 | 2,432 / 536 / 546 | 1,757 (960 + 797) | ~250 / 400 |
| split400-filtered | 2,428 / 532 / 520 | 1,740 (943 + 797) | ~250 / 400 |
| split400-cross | 2,432 / 536 / 546 | 1,757 (960 + 797) | ~250 / 400 |

- ×2.2 pairs from +960 intra-region positives (prediction was +958);
  cross-region positives unchanged, as designed.
- split400-filtered drops 38 gibberish regions (54 fragments, 34 pairs).
- split400-cross has the same counts and positives as split400, but negative
  examples must come from different source regions; same-region reversed/skip
  pairs are excluded.
- All split-pair texts ≤ 400 chars → worst-case pair ≈ 432 tokens → zero
  truncation at 512 (baseline: 13.5% of pairs truncated).

## Results

On the shared baseline test set (198 region-level pairs — same 14 pages):

| model | trained on | context | F1 | acc | AP | AUC |
|---|---|---:|---:|---:|---:|---:|
| DistilBERT | baseline | 512 | **0.748** | 0.727 | 0.800 | 0.792 |
| DistilBERT | split400 | 512 | 0.672 | 0.606 | 0.744 | 0.728 |
| DistilBERT | split400-filtered | 512 | 0.711 | 0.672 | 0.746 | 0.752 |
| DistilBERT | split400-cross | 512 | 0.672 | 0.616 | 0.745 | 0.733 |
| SEA-LION ModernBERT-300M | baseline | 512 | 0.766 | 0.737 | 0.808 | 0.827 |
| SEA-LION ModernBERT-300M | baseline context control | 8,192 | 0.769 | 0.758 | 0.738 | 0.796 |
| SEA-LION ModernBERT-300M | split400-filtered | 8,192 | 0.709 | 0.702 | 0.780 | 0.794 |
| SEA-LION ModernBERT-300M | split400-cross | 8,192 | **0.804** | **0.813** | **0.899** | **0.904** |

On the split400-cross test set (546 pairs, including intra-region positives):

| model / training data | F1 | acc | AP | AUC |
|---|---:|---:|---:|---:|
| DistilBERT baseline | 0.647 | 0.610 | 0.672 | 0.668 |
| DistilBERT split400 | 0.704 | 0.668 | 0.707 | 0.742 |
| DistilBERT split400-cross | 0.733 | 0.723 | 0.796 | 0.802 |
| SEA-LION split400-cross | **0.833** | **0.835** | **0.917** | **0.923** |

SEA-LION split400-cross also reached validation F1 0.8405 (AP 0.9079,
ROC-AUC 0.9167). The unsplit 8,192-context control reached validation F1
0.7583 and test F1 0.7692; training took approximately 6h37m because the long
baseline pairs required batch 8 after batch 16 exhausted MPS memory.

## Verdict

The initial split400 design was a negative result because it changed both the
positive and negative distributions. Restoring baseline-like negatives changed
the conclusion for SEA-LION:

- DistilBERT specializes to the split distribution but still regresses on the
  original region-level task.
- SEA-LION split400-cross improves on both distributions and establishes the
  best baseline-test result: F1 0.8042 versus 0.7658 for the existing SEA-LION
  baseline.
- Increasing context alone does not explain the gain: unsplit baseline data at
  8,192 context reached F1 0.7692, essentially unchanged from 0.7658 at 512,
  and had lower AP/AUC. The split positives plus cross-region-only negatives
  are the important change.
- Gibberish filtering helps the otherwise unrestricted split arm but is not
  needed for the best split400-cross result.

Residual control caveat: the unsplit 8,192 run used batch 8 while split400-cross
used batch 16; the existing best 512 baseline came from a 5-epoch LR schedule
whose best checkpoint was epoch 3, whereas the 8,192 control used a 3-epoch
schedule. These differences do not plausibly explain the much larger AP/AUC gap,
but an exact batch/schedule-matched rerun would remove the final confound.

Artifacts: `data/1_interim/ds-articlereconstruction-20260821/continuation/`
(`data-seed42-split400`, `data-seed42-split400-filtered`,
`data-seed42-split400-cross`, `models/*-split400*`, and the
`balanced8192` control). Baseline artifacts untouched.
