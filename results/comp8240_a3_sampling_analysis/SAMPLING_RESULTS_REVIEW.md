# Ethereum construction sensitivity: independently verified results

The returned batch contains 18 new completed fits: construction seeds 1 and 2, each with model seeds 0–2 for BWGNN, GCN and MLP. Nine existing construction-seed0/model-seeds0–2 fits are reused as a reference. No reference fit was retrained. This comparison has 27 records but adds only 18 distinct scientific fits.

All 18 new score/checkpoint fingerprints, 1,800 epochs, validation threshold selections, first-best validation epochs, 144 validation/test metric values, fixed within-construction scored populations, source/protocol fingerprints and both prepared datasets passed verification. The saved Windows preprocessing and three-actual-model self-check logs also report PASS. This score audit did not retrain models or execute the saved checkpoint tensors.

The verified core, capacity/loop/order sensitivities and construction sensitivity now total **128 distinct extension fits and 12,800 epochs**. The original reproduction and toy checks remain outside that count.

## Sample quality

| Construction seed | Nodes | Undirected pairs | Stored bidirected entries | Labelled roots | Train / validation / test |
|---|---:|---:|---:|---:|---|
| 0, reference | 49,224 | 464,721 | 929,442 | 5,825 | 2,330 / 1,153 / 2,342 |
| 1 | 49,237 | 465,941 | 931,882 | 5,825 | 2,330 / 1,153 / 2,342 |
| 2 | 49,256 | 467,264 | 934,528 | 5,825 | 2,330 / 1,153 / 2,342 |

Each sample has 18 features and 1,165 positive roots plus 4,660 activity-matched noisy negatives. Graph symmetry, unique nonloop adjacency, split disjointness/coverage, two classes in each split, feature finiteness and training-root-only standardisation were independently checked. Raw features for nodes shared between constructions agree. The audit does not re-export all features from the original 13.55 million transactions; that pipeline is supported by the preserved preprocessing code, provenance and earlier feature checks.

## Results

Mean ± sample SD (ddof=1), three model initialisations per construction, all scores in percent. These seed0-reference means use seeds 0–2 only and must not replace the primary five-seed core summary.

| Construction seed | Model | Macro-F1 | ROC-AUC | Average precision |
|---|---|---:|---:|---:|
| 0 | BWGNN | 95.01 ± 0.30 | 98.47 ± 0.14 | 95.54 ± 0.39 |
| 0 | GCN | 89.77 ± 0.79 | 96.08 ± 0.26 | 89.48 ± 0.59 |
| 0 | MLP | 89.49 ± 0.30 | 96.70 ± 0.09 | 86.74 ± 1.83 |
| 1 | BWGNN | 95.07 ± 0.32 | 98.71 ± 0.11 | 95.72 ± 0.31 |
| 1 | GCN | 90.52 ± 0.55 | 96.42 ± 0.19 | 90.78 ± 0.64 |
| 1 | MLP | 90.00 ± 0.29 | 96.89 ± 0.10 | 89.77 ± 0.58 |
| 2 | BWGNN | 94.66 ± 0.14 | 98.62 ± 0.16 | 95.13 ± 0.09 |
| 2 | GCN | 89.77 ± 0.58 | 96.05 ± 0.19 | 87.93 ± 0.68 |
| 2 | MLP | 87.84 ± 0.19 | 95.76 ± 0.14 | 82.46 ± 0.14 |

The mean paired BWGNN-minus-GCN Macro-F1 gaps are 5.24 ± 0.73, 4.55 ± 0.35 and 4.89 ± 0.69 percentage points for construction seeds 0, 1 and 2. Each gap is positive in all three model-seed pairs. BWGNN also exceeds MLP in every corresponding pair. Its mean Macro-F1 spans 94.66–95.07%, supporting consistency within these three construction settings.

## Dependence and interpretation

The constructions are not independent datasets. Pairwise node Jaccard overlap is 0.2125–0.2138 and undirected-edge Jaccard overlap is 0.4167–0.4224. Importantly, **all 1,165 positive identities and all 468 held-out positive identities are shared across all three constructions**. The changed negative selection and context are therefore not evidence of generalisation to previously unseen positive accounts or a new source population.

The construction seed jointly changes activity-matched negative selection, root membership/assignment and sampled context. It does not isolate a pure topological treatment. All samples come from one static released snapshot and use label-informed selection, potentially noisy negatives and 20% positives among supervised roots. Report the within-construction three-seed variation separately; do not pool 27 records as independent replications.

These runs use the original default-width comparison: BWGNN 17,858 parameters, GCN 1,346 and MLP 5,506, with the original differing graph-model loop policies. Near-equal capacity and shared-loop results were tested separately on construction seed0; that sensitivity was not crossed with all three constructions. No significance or causal spectral claim follows from the observed gaps.

## Current completion state

The planned scientific training is complete. Remaining assessment work is to collect the actual LLM/human explanation-support ratings, verify the student-owned public repository and local repository-check receipt, incorporate unit ethics resources and write/visually check the final ACL report. There are no actual judge/human ratings in the prepared templates.
