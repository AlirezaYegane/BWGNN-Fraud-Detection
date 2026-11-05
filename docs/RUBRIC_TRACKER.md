# COMP8240 Assessment 3 - rubric and evidence tracker

Owner: Alireza Yegane. Initial audit: 3 November 2026.

This is the working checklist for the final report, not a grade prediction. Update each row when its evidence changes. Historical pilot results are not new robustness experiments.

## Official submission requirements

- Due: 6 November 2026, 23:55 Australia/Sydney, unless an approved extension applies.
- Submit a PDF compiled from LaTeX, using the ACL template, through iLearn.
- Typical length: 8-10 pages; maximum main body: 10 pages, excluding references.
- Appendices may supplement the report; the main body must stand alone for assessment.
- Include a footnote URL to the student's publicly accessible GitHub repository containing the scripts used. The original authors' repository alone does not satisfy this requirement.
- Link and cite the source paper and data appropriately; include actual data-format snippets.
- Where human annotation is performed, describe it and report agreement. Existing released/proxy labels must not be described as new human annotations.
- Explain the role of LLM-as-judge as requested in the brief. The submitted proposal, Section 7, already plans a secondary explanation-support evaluation: blinded structured local evidence, a fixed rating rubric, repeated judgements and a limited human comparison. Ground-truth fraud labels are withheld from the judge. Actual execution is pending; do not fabricate its use or replace numerical classification evaluation with LLM-generated fraud labels.
- Include ethical considerations using the unit resources, alongside relevant responsible-research sources.
- The update was graded 36/40, above the stated 32-mark default-project cap, and the submitted proposal includes novel-project justification. This is strong evidence for the 40-mark route. Cover all novel-project criteria in the final report; a separate formal project-category approval has not been supplied.

## Rubric mapping

| Criterion | Marks | Evidence already recovered | Remaining action | Status |
|---|---:|---|---|---|
| Chosen paper and justification | 4, novel only | Tang et al., ICML 2022; official source code; archived original sources | Explain the contribution and original evaluation; justify quality and feasibility using verified criteria, not invented rankings/citation counts | Partial |
| Original datasets | 4, novel only | Amazon and Yelp pipeline/source descriptions | Describe all original-paper datasets at an appropriate level, distinguish reproduced settings, show actual schema snippets and label meaning | Partial |
| Replication | 8 | Original Amazon 10-run log; Yelp 10-seed predictions; Amazon 5-seed diagnostics; source fingerprints | Write exact protocol and environment, compare with paper, explain remaining Amazon AUC gap and pre/post-update evaluation difference | Metrics rechecked; prose pending |
| Construction of new data | 4 | Ethereum acquisition/audit/preprocessing archives; Amazon compact data, swap traces and manifests | Explain provenance, features, root matching, graph projection, labels, invariants and oracle test-label use; show real examples | Evidence recovered; write-up pending |
| Results on new data | 6 | Complete 75-fit batch: Ethereum 3 models × 5 seeds; Amazon 6 conditions × 2 models × 5 paired seeds; predictions/checkpoints independently verified | 35-fit Ethereum sensitivity complete and independently audited; distinguish protocols and complete the secondary explanation evaluation | 128 training fits verified; explanation evaluation pending |
| Ethical considerations | 4 | Known public/proxy-label data and label-aware intervention | Original-paper ethics plus privacy/linkability, false accusations, proxy-label errors, licensing, controlled oracle use and transparent reporting; use unit materials | Pending |
| Code quality | 3 | Original sources and extension scripts recovered | Verify current local versions; readable structure, pinned environment, documented commands, small smoke test, public repository tested anonymously | Pending |
| New-data quality | 3 | Saved construction manifests and input arrays | Recheck splits, scaling, duplicates and graph invariants; document crawl/selection bias, noisy negatives and intervention validity | Partial |
| Report presentation | 4 | Official format requirements extracted | ACL LaTeX; <=10 main pages; readable figures/tables, correct citations, consistent units and a final visual check | Pending |

Total for a novel project: 40. Rows marked novel-only contribute 0 for a default project (remaining total 32).

## Work completed in this intake

- Read all six pages of the final-report brief and visually inspected its requirements/rubric.
- Recovered eight historical archives from the project, rather than treating presentation claims as raw evidence.
- Recomputed Macro-F1, ROC-AUC and average precision for 22 saved validation/test splits across 11 extension runs: all 66 values match their result records within 1e-12.
- Recomputed original-selection and alternative-validation-AUC summaries for five Amazon diagnostic seeds and ten Yelp seeds.
- Recomputed the Ethereum feature standardizer from training roots, reproduced all scaled features, and verified disjoint train/validation/test masks (2,330 / 1,153 / 2,342 roots).
- Recomputed the ten original Amazon runs from their rounded log. The archived unseeded log lacks raw predictions, so precision is limited to its printed values.
- No new model training has been performed in this intake.
- Do not describe the old audit's graph rebuild or model execution as newly performed work.

## Historical results: current evidence baseline

- Amazon, original released 40% homogeneous setting: Macro-F1 92.22 +/- 0.26%, ROC-AUC 96.56 +/- 1.86%, 10 unseeded runs; population SD matching the released implementation.
- Yelp, 1% heterogeneous setting: Macro-F1 66.95 +/- 0.62%, ROC-AUC 76.72 +/- 1.50%, seeds 0-9, validation-Macro-F1 selection; population SD.
- Amazon diagnostic, seeds 0-4: AUC 96.79 +/- 1.80% under validation-Macro-F1 selection; 97.76 +/- 0.27% using float64-margin validation-AUC selection. Probability-score AUC rounds to 97.75 +/- 0.28% in the latter condition. Keep ranking-score definitions explicit.
- Ethereum pilot: BWGNN 94.6631%, GCN 89.4437%, MLP 89.1755% Macro-F1; 17,858 / 1,346 / 5,506 trainable parameters respectively. One model seed on one sampled graph.
- Amazon camouflage: baseline BWGNN/GCN 92.2952/67.7259%, strongest intervention 93.5455/66.2373%; use the accompanying CSV for full-precision scores. Random control also increases the gap. One construction seed and one model seed.
- These pilots support exploratory findings; they do not establish a causal spectral advantage or performance at deployment prevalence.

## Priority execution sequence

1. COMPLETE: obtain and inspect the current Windows intake. Five critical source files match the audited originals after line-ending normalisation; the two extension files also match byte-for-byte. Prepared input/companion paths are confirmed. No new training results are present in this intake.
2. Lock extension evaluation: explicit metrics and ranking scores; validation-only checkpoint/threshold selection; restored parameters; common splits; no test-driven configuration selection. Preserve original reproduction semantics separately.
3. Meet the submitted proposal's specific design: camouflage levels 0, 0.2, 0.4, 0.6, 0.8, with five graph-construction seeds. A practical paired design uses graph seed i and model seed i for i=0,...,4, both BWGNN/GCN, plus the existing maximum-swap matched random control: 60 model fits. Its variability combines construction and model initialisation; it must not be called a crossed variance decomposition. For c=0 the graph is identical, with five model initialisations. Ethereum starts with five model seeds for the three existing models: 15 fits on the fixed sampled graph. COMPLETE: all 75 fits returned and independently verified; 7,500 epochs, 600 validation/test metric values and 30 graph variants checked. No extra training was performed during the audit.
4. Address unequal parameter counts with a documented capacity sensitivity and self-loop policy. Equal hidden width is not equal capacity. Do not retune on the test set.
5. Add graph-construction seed or sampling sensitivity if run time permits, reporting clearly whether variation concerns model initialisation, graph construction or data splits.
6. Assemble final tables/figures directly from verified prediction records, including Macro-F1, ROC-AUC and average precision with prevalence and uncertainty definitions.
7. Write the report alongside experiments; build the public reproducibility package after inspecting current code/data permissions.
8. Run the entire rubric checklist against the compiled PDF and public repository.

## Proposed main-body page budget

| Content | Target pages |
|---|---:|
| Abstract, introduction, paper justification | 1.0 |
| Original data and evaluation | 1.0 |
| Replication methods, results and diagnosis | 2.0 |
| New-data construction | 1.5 |
| New-data evaluation, results and sensitivities | 2.0 |
| Ethics, limitations, conclusion and reproducibility links | 1.5 |
| Total target | 9.0 |

References follow the main body; appendices contain extra configurations and evidence. Do not use a separate decorative cover page.

## Source anchors

- User-supplied Assessment-Project-Final-Report (1).pdf: pp.1-4 requirements; pp.5-6 rubric. The stray 'proposal' and PHTY footer text do not replace the explicit final-report instructions.
- ACL official template: https://github.com/acl-org/acl-style-files
- Source paper: https://proceedings.mlr.press/v162/tang22b.html
- Original authors' code: https://github.com/squareRoot3/Rethinking-Anomaly-Detection
- Responsible research: https://www.nhmrc.gov.au/about-us/publications/australian-code-responsible-conduct-research-2018

## Open inputs

- Current Windows intake ZIP received and inspected: 26,072,027 bytes, 166 ZIP entries, 193 indexed project files, zero collection omissions/errors. Source baseline de0631f039bbd19c1890b483cc01f1007f596af7; origin still points to the original authors' repository, not a student-owned public repository.
- Course 'Overall Goal and Assessment Criteria' page and any final-report announcements, particularly the secondary LLM-as-judge implementation guidance.
- Unit-provided ethics resources.
- Student repository URL and current visibility, if already created.

## Submitted documents and marker feedback received 3 November

- Read the submitted six-page proposal and eight-slide update.
- Marker: Ghazal Lalooha; grade 36/40; graded 22 October 2026. Feedback praises clarity, replication, paper comparison and dataset progress, and explicitly asks for more testing because the new experiments use one seed. Do not infer a criterion-by-criterion allocation of the lost four marks from this overall comment.
- Windows intake screenshot: Python is the bwgnn environment, 193 indexed files, 0 oversized evidence omissions, 0 read errors. This confirms the collector ran; it does not reveal current code hashes or result contents.
- Submitted proposal commitments: five camouflage levels and five independently seeded construction variants per nonzero level; measured global and fraud-local mixing; degree-preserving random control; spectral-energy analysis including per-feature relative change, with zero baselines handled separately.
- Existing pilot uses only three levels and one construction/training seed, and reports absolute spectral change. The final report must fulfil the broader proposed design or explicitly explain deviations. Report absolute and relative changes with their baselines rather than relabelling one as the other.
- Ethereum proposal includes activity-matched noisy negatives, transaction-derived features, neighbourhood-sampling fallback and validation of filter order. The other candidate datasets are alternatives, not additional mandatory experiments. T-Finance reproduction was conditional on feasibility.
- The proposal's explanation-judging module is a concrete planned component, not an unrelated requirement to dismiss as a template error.

## Core run kit prepared after intake audit

- Prepared Run-Assessment3-Core.ps1, run_assessment3_core.py and audited_inputs.json.
- Fixed design: 15 Ethereum fits plus 60 Amazon fits, 100 epochs each, sequential CPU workers, existing training scripts unchanged.
- Dataset fingerprints and all five critical source versions are checked before training; the original environment is required.
- Driver tested against actual 11-run archived predictions/checkpoints; old three-level pilot is rejected as an incomplete final-design batch.
- Synthetic orchestration fixtures verify a failure after 39 fits, partial evidence export, resume without redoing completed jobs, final 75-fit counts, five-replicate sample-SD summaries, and 30 paired model gaps. These are software tests, not new scientific training results.
- COMPLETE: the user executed the 75-fit core kit and returned COMP8240_A3_Core_Evidence.zip. Independent verification passed. Capacity/self-loop/order sensitivity is now complete; explanation judging, public repository and final report remain pending.

## Core completion and next sensitivity stage

- Core batch core_20261103T072708_683589Z: 75 verified fits, 7,500 epochs; elapsed 2h 15m 18s. All score/checkpoint fingerprints and 75 validation threshold selections checked.
- Independently replayed all 30 Amazon graphs, preserving weighted degrees, multiplicities and nested prefixes; recomputed spectral quantities and mixing.
- Ethereum Macro-F1 (sample SD, n=5): BWGNN 95.15 ± 0.42%, GCN 89.51 ± 0.85%, MLP 89.34 ± 0.30%. Fixed sampled graph and split.
- Amazon BWGNN-minus-GCN gap: baseline 23.60 ± 0.63 pp; strongest oracle condition 27.24 ± 0.82 pp; maximum-budget random control 23.71 ± 1.29 pp. Paired strongest-minus-control change 3.53 ± 1.13 pp, positive in all five pairs.
- The feedback on single-seed extension testing is addressed by this design; no claim of independent dataset or split robustness.
- Prepared 35 additional Ethereum fits only: 15 near-equal-budget fits (graph models share one self-loop/node), 5 default-width BWGNN loop fits, 15 C=1/3/5 fits. Reuse the existing five C=2 fits. Filter order is selected by mean validation Macro-F1; primary core remains C=2. Parameter count changes with order and must be reported.
- Sensitivity software checks pass on actual core references and simulated failure/resume. Actual seven-configuration model self-check runs in Windows before training; sensitivity scientific results remain pending.
- Full analysis: COMP8240_A3_Core_Analysis.zip. Next runnable package: COMP8240_Assessment3_Sensitivity_RunKit.zip.

## Sensitivity evidence received and independently verified, 4 November

- Completed 35 scientific fits and 3,500 epochs in 1h 31m 33s. Combined core plus sensitivity: 110 fits, 11,000 epochs. Actual Windows model self-check passed. No training performed during the independent audit.
- Recomputed 280 validation/test primary metric values and all 35 validation thresholds; checked score/checkpoint/source/protocol hashes, logit margins, probabilities and core population agreement.
- Near-equal budgets, shared graph loop policy: BWGNN 1,394 parameters, Macro-F1 94.76 ± 0.39%; GCN 1,388, 89.88 ± 0.48%; MLP 1,402, 89.18 ± 0.29%. Mean paired BWGNN-minus-GCN gap 4.88 ± 0.80 pp, positive in all five seeds. Sample SD throughout.
- Default-width BWGNN with one loop/node: 94.73 ± 0.14% Macro-F1 versus core no-loop 95.15 ± 0.42%; paired difference -0.42 ± 0.39 pp.
- C=5 selected by mean validation Macro-F1 94.3625%, versus C=2 94.2747%. Test Macro-F1: C=5 95.1424% versus C=2 95.1492%; essentially unchanged. C=5 uses 30,146 parameters versus 17,858. Original primary core remains C=2. Do not sell this as a meaningful test gain.
- All planned core/sensitivity training is finished. No additional training has been requested for this fixed design.
- Analysis: COMP8240_A3_Sensitivity_Analysis.zip. Explanation support judging, public student repository, unit ethics resources and final report remain outstanding.

## Secondary explanation-support study prepared, not executed

- Frozen eight cases from actual primary Ethereum BWGNN C2/width64/seed0 test predictions, selected without true labels from predicted-class × threshold-distance strata.
- 24 rule-written texts: descriptive explanations and controlled unsupported-attribution/intent variants. No claim of model-native explanation generation or causal attribution.
- Three 24-item judge runs prepared with separately shuffled orders; use the same LLM in three fresh chats and retain all real responses. No labels, addresses or node IDs are in judge input.
- Twelve human ratings fixed in advance, covering four cases across the selection strata; human review before LLM responses. One author review must be reported as such, not independent expert annotation.
- All rating templates remain empty. There are no LLM/human evaluation results yet. Planned reporting: per-run distributions, pairwise exact agreement/Cohen kappa, human-versus-run agreement, and majority ties.
- Package: COMP8240_A3_Explanation_Study.zip; instructions START_HERE_FA.md. Unit ethics materials and student GitHub URL remain missing inputs for the final report.

## Tonight closure audit, 4 November

- Re-read the reattached brief (same SHA as earlier), proposal and update; reviewed submission-2020.pdf as an old example only. Official brief says 40 raw marks/40%, not 30. Current ACL/LLM/ethics instructions take precedence over the old example.
- Corrected an omitted open commitment: update slide 8 promises sampling sensitivity. Existing 110 fits vary initialisation/capacity/loops/order but reuse one Ethereum construction.
- Prepared an 18-new-fit construction check: two fresh construction seeds × three model seeds × three models, with nine existing seed0-construction reference fits reused. This jointly varies negative matching, root assignment and sampled context; not independent datasets or pure topology. Not executed yet.
- Prepared COMP8240_A3_Repository.zip with pinned dependencies, source bootstrap, unchanged extension sources, actual saved-score verifier, dataset-quality checks, sampling driver with resume, genuine rating analyser, README/data instructions, public-file exclusions and attribution.
- Offline build checks pass: all 110 actual saved fits, source fingerprints, four software workflow tests, real dataset invariants and actual preprocessing toy self-check. Windows Torch/DGL runtime checks are deliberately pending.
- Recomputed eligible cross-class and known-FB/all-nonloop-pair fractions for all 30 Amazon variants, alongside existing fraud-local metrics. Unknown-context labels remain excluded from known-FB numerators; do not misdescribe that denominator as fully labelled global mixing.
- Tonight blockers: Windows repository smoke check; actual 18-fit sampling evidence; actual 3×24 LLM ratings plus 12 human ratings; student-owned public repository URL checked signed out; unit ethics resources/current project criteria page. Tomorrow: ACL report, <=10 main pages, required source/data/repo links, real data snippets and final rubric/visual check.

- Tonight public staging rehearsal passes: git ignores raw arrays, checkpoints, generated upstream files and local runtime receipts; immutable Python line endings are specified. Publishing script prepares a local commit and requires explicit -Push for remote publication. No repository has been published or remotely verified in this session.
- LLM analyser also verifies frozen case/prompt fingerprints, fixed item sets, actual metadata and fresh-chat declarations. Synthetic software fixtures remain separate from genuine study output.

## Sampling completion, 4 November

- Returned COMP8240_A3_Sampling_Evidence.zip: 18 new fits complete, 1,800 epochs. Verified score/checkpoint/source/protocol fingerprints, all validation thresholds and 144 validation/test metric values; two actual prepared datasets checked for graph/split/scaler invariants. Nine existing reference fits reused, not retrained. Total distinct extension fits: 128; total epochs: 12,800.
- Actual Windows preprocessing and all-three-model CPU self-check logs report PASS. Full repository wrapper receipt remains a separate requested local artefact.
- Three-seed per-construction Macro-F1 means (BWGNN / GCN / MLP): seed0 95.01 / 89.77 / 89.49%; seed1 95.07 / 90.52 / 90.00%; seed2 94.66 / 89.77 / 87.84%. BWGNN exceeds both baselines in all nine construction/model-seed pairs.
- The primary construction0 five-seed summary remains 95.15 ± 0.42% BWGNN; three-seed reference values here use seeds0–2 for comparability. Do not replace or mix these summaries.
- Pairwise node Jaccard 0.2125–0.2138 and edge Jaccard 0.4167–0.4224. All 1,165 positive identities and all 468 held-out positive identities are shared. This is joint negative/context construction sensitivity on one snapshot, not independent-dataset or new-positive generalisation.
- Updated code package adds read-only verify_sampling.py; preserves archived training drivers and their hashes. Complete-Tonight.ps1 now verifies existing sampling output by default and requires explicit -RunSampling to start a new batch. No further scientific training is currently planned.
- Remaining: actual 3×24 LLM ratings and 12 author ratings with metadata; public student repository URL and local wrapper receipt; unit ethics materials/current overall project criteria; final ACL report and final visual/rubric check.
