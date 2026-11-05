# COMP8240 A3: independently checked core results

The submitted core evidence contains 75 completed model fits: 15 Ethereum fits and 60 Amazon fits. All 7,500 epochs, 75 selected validation thresholds, 600 validation/test metric values, saved score/checkpoint fingerprints and 30 Amazon graph variants passed the independent audit. Audit work did not retrain a model. Original batch time: 2 hours 15 minutes 18 seconds.

## Ethereum: fixed sampled graph, five model initialisations

Mean ± sample SD, n=5, seeds 0–4. All scores are percentages. ROC-AUC and AP use float64 logit margins; AP is average precision, not trapezoidal PR-AUC.

| Model | Parameters | Macro-F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| BWGNN, C=2 | 17,858 | 95.15 ± 0.42 | 98.59 ± 0.22 | 95.76 ± 0.42 |
| GCN | 1,346 | 89.51 ± 0.85 | 96.06 ± 0.28 | 89.35 ± 0.69 |
| MLP | 5,506 | 89.34 ± 0.30 | 96.64 ± 0.12 | 86.36 ± 1.84 |

The prepared graph has 49,224 nodes, 18 features and 929,442 directed stored entries representing 464,721 undirected pairs. Train/validation/test scored roots number 2,330 / 1,153 / 2,342. Positives make up 20% of labelled roots by construction; negatives are activity-matched unlabelled accounts, not verified benign accounts. Features are standardised using training roots. The graph is fixed across seeds.

The default core architectures differ in capacity. BWGNN uses the loop-free base graph; GCN adds one self-loop per node. Therefore these scores alone do not establish that BWGNN's filters caused the performance difference. A separate 35-fit sensitivity kit tests near-equal parameter budgets, loop handling and filter orders. Those results are pending.

## Amazon: paired graph/model seeds

Both model and graph seed equal i for i=0,...,4. This is a paired design, not a crossed 5×5 experiment. The baseline graph is identical across all five initialisations. Levels are construction budgets, not measured mixing percentages. Mean ± sample SD, n=5.

| Condition | BWGNN Macro-F1 (%) | GCN Macro-F1 (%) |
|---|---:|---:|
| c=0 | 91.85 ± 0.45 | 68.25 ± 0.63 |
| c=0.2 | 92.28 ± 0.51 | 67.75 ± 0.47 |
| c=0.4 | 92.21 ± 0.18 | 67.25 ± 0.83 |
| c=0.6 | 92.75 ± 0.38 | 66.65 ± 1.32 |
| c=0.8 | 93.82 ± 0.41 | 66.58 ± 0.82 |
| Random control, maximum swap budget | 92.10 ± 0.38 | 68.39 ± 1.14 |

| Paired comparison | Mean ± sample SD (percentage points) |
|---|---:|
| BWGNN minus GCN at baseline | 23.60 ± 0.63 |
| BWGNN minus GCN at c=0.8 | 27.24 ± 0.82 |
| BWGNN minus GCN under random control | 23.71 ± 1.29 |
| Gap increase: c=0.8 minus baseline | 3.64 ± 1.01 |
| Gap increase: c=0.8 minus random control | 3.53 ± 1.13 |

The strongest intervention increases the gap over both the baseline and the random control in all five paired runs. The respective ranges are 2.59–5.02 and 1.90–5.08 percentage points. These are observed five-run consistency checks; no significance claim is made.

The random condition matches the number of swaps and edge multiplicity at c=0.8 only. There is no matched random condition for every intermediate level. Replaying all traces independently verified endpoints, degree/multiplicity preservation and the nested intervention prefixes. Spectral quadratic forms and their changes were independently recomputed. The baseline mean energy is 0.6112074751, dimensionless. The CSV reports both absolute change and mean per-feature relative change in percent; the latter is not the percentage change of the mean energy.

## Claims supported and limits

- The marker's explicit concern about single-seed extension results is now addressed with five repeated seeds under the stated design. This does not guarantee a mark or establish sampling robustness.
- Amazon construction uses validation and test labels as an oracle. It is a controlled structural stress test, not a deployable evasion attack or unseen-label generalisation result.
- Amazon variability combines construction and model initialisation. Ethereum variability concerns model initialisation on one sampled graph and split.
- Core model capacities are unequal. Increased spectral energy and an increased performance gap are associated observations, not proof of a causal spectral mechanism.
- New extension results use fresh post-update validation, restored checkpoints and validation-selected thresholds. Keep them distinct from the released reproduction's evaluation semantics.
- New tables use sample SD (ddof=1). Historical reproduction summaries retain the released implementation's population SD (ddof=0); label that distinction.

## Files

`all_model_results.csv`: 75 verified rows. `summary_by_condition.csv`: per-condition model summaries. `paired_model_gaps.csv`: per-seed Amazon differences. `graph_and_spectral_checks.csv`: independently replayed structural/spectral quantities. `independent_core_audit.json`: verification counts and paired summaries. Figures are derived from these CSVs without smoothing.

Still pending for the full assessment: sensitivity execution, the proposed secondary explanation-judging evaluation, a public student-owned reproducibility repository, ethics discussion using unit materials, and the compiled ACL final report with a full rubric check.
