# COMP8240 Assessment 3: current readiness tracker

Updated 5 November 2026. This is the current tracker. Copies under the three training-analysis result folders are historical audit snapshots, not current task lists. This checklist records evidence and remaining work; it does not award or predict marks.

## Completed code and evidence

- Student-owned public repository: https://github.com/AlirezaYegane/BWGNN-Fraud-Detection . Anonymous access and the published README were checked on 5 November.
- Original reproduction: ten archived unseeded Amazon runs, ten YelpChi seeds and five Amazon selection diagnostics. Original semantics and population SD remain separate from extension protocols and sample SD.
- Extensions: 75 core fits + 35 capacity/loop/order fits + 18 new construction fits = 128 distinct fits, 12,800 epochs. Nine construction-0 references are reused, not additional training.
- Actual Windows Python 3.8.20 runtime receipt: `WINDOWS_RUNTIME_VALIDATION.json`. It covers preprocessing, original-loop instrumentation, three-model CPU checks, 75 core fits and 35 sensitivity fits. The 18-fit construction and 72-rating judge audits are separate.
- Actual Gemini study: three single-turn API responses, 24 texts each, 72 ratings; raw requests/responses and independent audit included. Pairwise agreement is 100%, unweighted kappa 1.00, on these simple dependent controls.
- No independent human review was supplied. The prior AI-assisted draft is not independent human annotation; no human–LLM agreement is claimed. Explain this departure from the proposal in the report.
- Frozen prompts, casepacks and original manifests remain unchanged. Their `prepared_not_executed` status records the historical plan. Completed execution is under `results/explanation_judge/`.
- Raw API evidence has byte-preserving Git attributes. `verify_public_results.py` rechecks its fingerprints and reconstructs the published summaries without network access, model training or API calls.

## Official requirements and remaining report work

The supplied final-report brief specifies 40 raw marks for the novel route, 40% weighting, ACL LaTeX, a compiled PDF submitted through iLearn, and a maximum of 10 main-body pages excluding references. Due: 6 November 2026 at 23:55 Australia/Sydney, unless an approved extension applies. The main body must stand alone; appendices are supplementary.

| Criterion | Marks | Evidence available | Remaining report action |
|---|---:|---|---|
| Paper description and justification | 4 | ICML source paper, original code and provenance | Accurate abstract-level explanation, original evaluation and several verified selection criteria |
| Original datasets | 4 | Original paper and dataset sources | Describe every original-paper dataset; identify only the reproduced settings; show real schema snippets and source links |
| Replication | 8 | Original runs, diagnostics, environment and fingerprints | Explain execution, paper-versus-reproduction comparison and observed discrepancies without test-driven selection |
| New-data construction | 4 | Ethereum acquisition/features/matching/sampling and Amazon traces | Explain processing, source bias, label semantics, actual examples and oracle access to eligible validation/test labels |
| New-data results | 6 | Core, sensitivities, resampling and judge evidence complete | Present uncertainty, paired comparisons, protocol differences, limitations and the human-review deviation |
| Ethics | 4 | Data/label/privacy and oracle limitations identified | Obtain unit resources; discuss original and new work with actual citations |
| Code quality | 3 | Public repository, pinned environment, bootstrap, offline and runtime checks | Perform final independent-checkout verification and link the public repository in a report footnote |
| New-data quality | 3 | Splits, train-only scaler, degree/loop/multiplicity checks and overlap audits | Explain quality controls and remaining proxy-label and snapshot limitations |
| Report presentation | 4 | Tables/figures and source records ready | Write ACL report, verify all citations/numbers/links and inspect the compiled PDF |

## Scientific boundaries to retain

- Ethereum negatives are sampled unlabelled accounts, not confirmed benign. The 20% positive share among labelled roots is constructed.
- The three Ethereum constructions share the released snapshot and all positive identities, including 468 held-out positives. They are not independent datasets or tests of transfer to new positives. Keep primary n=5 and per-construction n=3 summaries separate.
- Amazon rewiring is a label-aware oracle using eligible validation/test labels. Graph and model seeds are paired, not crossed; the random control matches the maximum swap budget only.
- Capacity matching does not establish equal expressivity; performance gaps do not establish a causal spectral mechanism. C=2 remains primary; C=5 is a secondary validation-selected sensitivity.
- Judge outputs measure support judgments on eight cases and 24 rule-written texts, not classification accuracy, causal explanation faithfulness or independent human agreement.

## Open inputs for the report

- Unit ethics resources.
- The Overall Goal and Assessment Criteria page and any later guidance on LLM/human evaluation or submission.
- Final ACL manuscript and final rubric/visual audit. Neither additional training nor a new judge batch is currently required by the fixed experimental design.
