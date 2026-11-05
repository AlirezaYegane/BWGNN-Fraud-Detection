# COMP8240 A3: verified Ethereum sensitivities

The returned sensitivity batch is complete: 35 fits, 3,500 epochs and 280 validation/test primary metric values independently recomputed. Score/checkpoint hashes, validation threshold optimisation, first-best validation epochs, logits, probability scores, fixed scored populations, source snapshots and core protocol fingerprints match. The actual seven-configuration Windows model self-check passed. The audit did not execute checkpoint tensors or retrain models.

The batch took 1 hour 31 minutes 33 seconds. Together with the previously verified core batch, the evidence now comprises 110 scientific training fits and 11,000 epochs. Two-epoch toy software checks are excluded from these counts.

## Near-equal parameter budgets

The same graph, features, train/validation/test split, five initialisation seeds, 100 epochs and learning rate 0.01 are used throughout. Both graph models receive exactly one self-loop per node. All values are mean ± sample SD (ddof=1), n=5, in percent.

| Model | Width | Trainable parameters | Macro-F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| BWGNN, C=2 | 16 | 1,394 | 94.76 ± 0.39 | 98.67 ± 0.13 | 95.52 ± 0.38 |
| GCN | 66 | 1,388 | 89.88 ± 0.48 | 96.27 ± 0.14 | 89.78 ± 0.27 |
| MLP | 28 | 1,402 | 89.18 ± 0.29 | 96.46 ± 0.10 | 87.20 ± 0.78 |

BWGNN-minus-GCN Macro-F1 is 4.88 ± 0.80 percentage points; BWGNN-minus-MLP is 5.59 ± 0.50 pp. Each difference is positive in all five paired initialisations. Parameter counts range from 1,388 to 1,402, approximately a 1% spread. The observed advantage therefore persists at approximately matched low parameter budgets with a common loop policy for graph models.

This controls parameter count, not expressivity, architecture, computational cost or optimisation difficulty. It is not a complete tuning comparison or a causal demonstration of spectral filtering. Because the low-budget condition also changes BWGNN's loop policy relative to the core configuration, interpret it together with the separate loop sensitivity below.

## Default-width BWGNN: self-loop sensitivity

| Model graph | Parameters | Test Macro-F1 (%) | ROC-AUC (%) |
|---|---:|---:|---:|
| No added loops, original core C=2 | 17,858 | 95.15 ± 0.42 | 98.59 ± 0.22 |
| Exactly one loop per node, C=2 | 17,858 | 94.73 ± 0.14 | 98.59 ± 0.10 |

Adding loops changes test Macro-F1 by -0.42 ± 0.39 pp across paired seeds, negative in all five. Mean ROC-AUC change is approximately -0.0006 pp. Thus loop handling affects the thresholded metric in this configuration; do not describe it as universally improving or degrading every metric. BWGNN retains a strong result under the shared loop policy.

## Filter-order sensitivity

No added loops, width=64, seeds 0–4. Increasing C changes both filter order and trainable parameter count. Order is selected using mean validation Macro-F1 only; the original primary core result remains C=2.

| Order C | Parameters | Validation Macro-F1 (%) | Test Macro-F1 (%) | Test ROC-AUC (%) |
|---|---:|---:|---:|---:|
| 1 | 13,762 | 94.27 ± 0.47 | 94.65 ± 0.25 | 98.52 ± 0.07 |
| 2, core reference | 17,858 | 94.27 ± 0.13 | 95.15 ± 0.42 | 98.59 ± 0.22 |
| 3 | 21,954 | 94.25 ± 0.24 | 94.78 ± 0.17 | 98.69 ± 0.11 |
| 5, validation-selected | 30,146 | 94.36 ± 0.32 | 95.14 ± 0.30 | 98.61 ± 0.17 |

C=5 has the highest mean validation Macro-F1, exceeding C=2 by only 0.0877 percentage points. The C=5 minus C=2 test Macro-F1 difference is -0.0068 ± 0.6408 pp, positive in three of five pairs. It requires 30,146 rather than 17,858 parameters and substantially longer runs. Describe this as a small validation preference with essentially unchanged observed test performance, not a material improvement or evidence that larger C is generally better. Rounding makes C=1 and C=2 validation means appear equal; selection uses full precision.

## Overall interpretation

The five-seed core experiment addresses the marker's concern about single-seed extension results. These additional controls strengthen the finding that BWGNN's observed advantage is not confined to the original unequal parameter budgets or default difference in loop handling. They do not establish a causal mechanism, independent-graph generalisation or population-level fraud precision.

The Ethereum sample and split remain fixed. Negatives are unlabelled proxies selected through activity matching. Positives represent 20% of supervised roots by construction. Amazon remains an oracle stress test with paired graph/model seeds and a maximum-swap random control; the Ethereum sensitivities must not be presented as Amazon capacity controls.

## Reusable outputs and remaining work

CSV tables and publication-ready PDF/PNG figures accompany this review. `independent_sensitivity_audit.json` records the checks and paired comparisons. `order_selection_by_validation.json` records the secondary selection. No new training is required for the specified core and sensitivity design.

Remaining work: execute the secondary explanation-support evaluation without fabricating judge/human ratings, prepare the student-owned public repository, incorporate unit ethics resources, and compile/check the final ACL report against the rubric. The prepared explanation evaluation is an exploratory check of constructed descriptive texts and controls, not a claim that BWGNN generates natural-language explanations or that a judge can establish fraud.
