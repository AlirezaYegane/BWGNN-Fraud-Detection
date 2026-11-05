# Detailed reproducibility notes

Run the commands below from the repository root. The front-page overview is in [README.md](../README.md).

## Environment

Verified execution platform: Windows PowerShell, Python 3.8.20, NumPy 1.23.5, PyTorch 1.9.0, DGL 0.8.1, SciPy 1.10.1, scikit-learn 1.3.2, SymPy 1.13.3 and NetworkX 3.1. New-data training uses CPU, 16 threads. Do not switch the reported experiments to a newer environment or GPU mid-protocol.

Use the existing `bwgnn` environment when available. For a separate clean environment, `environment.yml` records the pinned stack:

```powershell
conda env create -f environment.yml
conda activate bwgnn-a3
```

The historical wheel availability on every operating system has not been tested; the documented target is Windows/Python 3.8.20. Record any installation failure rather than silently upgrading libraries.

## Original source bootstrap

From this repository directory:

```powershell
python bootstrap_upstream.py
```

This downloads the exact three original files at the audited commit and verifies their LF-normalised SHA256 fingerprints. They are ignored by Git; attribution is in `THIRD_PARTY.md`. If the original project already exists locally, avoid a network download:

```powershell
python bootstrap_upstream.py --from-project "C:\Users\Asus\Documents\COMP8240\Rethinking-Anomaly-Detection"
```

The original files are never patched. Run scientific scripts from the project directory so their original-source checks can locate those files.

## Marker quick checks

A standard-library-only public check is available before installing the ML stack or downloading upstream code: `python verify_public_results.py`. It checks the published per-fit/aggregate tables, byte-exact raw judge evidence, and original replication logs/validation selections. It does not check private classifier prediction arrays. See [MARKER_GUIDE.md](MARKER_GUIDE.md) for the separate verification levels.

```powershell
python verify_submission.py --runtime
```

This runs actual small preprocessing, metric/instrumentation and three-model CPU checks. It does not download full datasets or repeat the scientific training. To verify complete saved evidence, extract its ZIPs into separate directories, then run:

```powershell
python verify_submission.py --core "PATH_TO_EXTRACTED_CORE" --sensitivity "PATH_TO_EXTRACTED_SENSITIVITY"
```

Recorded Windows paths can be relocated; the verifier resolves the evidence's directory layout. Saved checkpoint files are fingerprinted. The score verifier recomputes metrics from NPZ arrays; it does not claim to execute checkpoints or infer their correctness from filenames.

## Original reproduction

Amazon/YelpChi download automatically through DGL when first needed. An existing DGL cache can be reused. These commands preserve the released evaluation semantics:

```powershell
python main.py --dataset amazon --train_ratio 0.4 --hid_dim 64 --order 2 --homo 1 --epoch 100 --run 10
python audit_bwgnn.py --dataset yelp --train_ratio 0.01 --hid_dim 64 --order 2 --homo 0 --epoch 100 --seeds 0 1 2 3 4 5 6 7 8 9
python audit_bwgnn.py --dataset amazon --train_ratio 0.4 --hid_dim 64 --order 2 --homo 1 --epoch 100 --seeds 0 1 2 3 4
```

The archived ten-run Amazon experiment is unseeded; an exact rerun of its individual values is not promised. Historical summaries use population SD (ddof=0), matching the released implementation. Extension summaries use sample SD (ddof=1).

## Data construction and general reproduction

See `docs/DATA_AND_COMMANDS.md` for raw-data provenance, fingerprints, preprocessing and training commands. Prepared inputs are not committed. Fresh reconstructions have new provenance manifests; use the general commands rather than pretending their byte hashes must equal the archived run's immutable companion reports.

`run_assessment3_core.py` and `run_sensitivity.py` are the exact archived orchestration drivers. Their `audited_inputs.json` deliberately locks the original prepared inputs and companions; they are appropriate for continuing the audited local project, not for bypassing provenance checks after a fresh reconstruction.

## Completed construction-sensitivity check and reproduction

For the existing audited Windows project:

```powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-Sampling-Check.ps1
```

Completed and verified: 18 additional fits, construction seeds 1/2 × model seeds 0/1/2 × BWGNN/GCN/MLP. Nine seed0-construction reference fits are reused from the core batch. Each fit uses 100 epochs, width64, C=2, learning rate0.01 and the original loop policies. State and partial evidence support resume using `-Resume "PRINTED_BATCH_DIRECTORY"`. `-PreflightOnly` checks the required inputs without starting training.

This changes matched negative selection and sampled context, and may change root membership/split assignment. It is a **joint construction sensitivity** on the same released snapshot, not an isolated sampling-only causal experiment or three independent datasets. Report model-seed variation within each construction separately. Keep all outcomes, including worse results; do not choose construction seeds using test scores.

## Secondary explanation evaluation

Completed on 5 November 2026 using the three frozen prompts, independent single-turn Gemini 3.6 Flash API calls and the recorded bounded-overload retry policy. All 72 ratings and their raw responses are in `results/explanation_judge/`. The separate audit script checks source/input/output fingerprints, case-specific citations, distinct prompt orders and response IDs, exact agreement and Cohen kappa. All four frozen casepack/prompt files also match regeneration from the primary saved Ethereum predictions and dataset.

Each run gives eight supported descriptive texts, eight partly_supported attribution controls and eight speculative intent controls. Every pair of runs agrees on all 24 texts (kappa1.00). Read `results/explanation_judge/JUDGE_RESULTS_REVIEW.md` for limits: eight cases, simple rule-written templates, one model, repeated dependent judgments, no causal explanation faithfulness or classifier-accuracy claim. The earlier author-review draft was AI-assisted; no independent human–LLM agreement is reported.

The frozen files in `explanation_study/` preserve the original prepared plan and blank templates; their historical prepared_not_executed status is not the current execution status. The actual protocol and recovery-policy addendum are in the completed evidence. Do not run another judge batch to reproduce a preferred answer. To verify the returned ZIP without API calls, use:

```powershell
python results/explanation_judge/audit_judge_evidence.py --evidence "PATH_TO_GEMINI_JUDGE_EVIDENCE_ZIP" --output "LOCAL_JUDGE_CHECKS"
```

## Scope and interpretation

- Primary core configuration remains C=2. C=5 was selected only for a secondary sensitivity by mean validation Macro-F1 and has essentially unchanged observed test Macro-F1.
- Ethereum uses label-informed sampling, a static snapshot, train-only standardisation and noisy proxy negatives; supervised-root prevalence is 20% by construction.
- Amazon rewiring uses all eligible labels, including validation/test labels, as an oracle. It is not a realistic unseen-label attack. Its random control matches the largest swap budget only. Graph and model seeds are paired, not crossed.
- Parameter-matched performance is not proof of identical model expressivity or of a causal spectral mechanism.
- New-data validation uses fresh post-update logits and restores the selected parameters; original reproduction retains its released semantics.

## Public repository and submission

The public project repository is https://github.com/AlirezaYegane/BWGNN-Fraud-Detection. The final ACL report is prepared separately and references this repository. Keep raw addresses, credentials, full data arrays, model checkpoints and local runtime evidence out of ordinary Git commits; use the documented data-access path.

The final ACL report is prepared as a separate deliverable and was revised after AI-assisted review. It discloses AI-assisted code/document preparation and the actual LLM judging. Unit-specific ethics resources and later lecturer clarifications remain to be checked; I take responsibility for the submitted work. No grade is guaranteed by a runnable repository.

To run the software workflow tests without full training:

```powershell
python -m unittest discover -s tests -v
```

For a new empty repository created in your own GitHub account, prepare a local commit using `Publish-Repository.ps1 -RepositoryUrl "YOUR_HTTPS_REPOSITORY_URL"`. Review the staged summary. Add `-Push` only when you intend to publish to that repository. Existing origins are not replaced and force-push is not used. Git must already have a valid author identity and authentication. Keep Python line endings as specified in `.gitattributes`; the provenance-locked scripts are byte-checked.

To verify the completed sampling evidence without training:

```powershell
python verify_sampling.py --sampling "EXTRACTED_SAMPLING_DIRECTORY" --core "EXTRACTED_CORE_DIRECTORY" --baseline-dataset "ORIGINAL_SEED0_DATASET.npz" --output "LOCAL_SAMPLING_CHECKS"
```

The baseline dataset option adds overlap comparisons against construction0; its recorded fingerprint must match. The other two sampled datasets are included in the sampling evidence ZIP. `Complete-Tonight.ps1` now checks existing results by default; it starts a new 18-fit batch only with explicit `-RunSampling`. The unchanged `Run-Sampling-Check.ps1` remains available for deliberate reproduction.

## Historical API execution code

`gemini_api/run_gemini_judge.py` and `gemini_api/recovery/` are the actual executed runner and TLS/overload recovery code. They preserve frozen inputs and save full responses, usage and hashes. Successful execution is already complete; no further API requests are needed. Its original README describes the pre-execution state; the completed result review and evidence supersede that status.

`openai_api/` records an attempted alternative after Gemini overload. My OpenAI request returned credit_balance_exhausted and produced no judgment. I declined paid API credit and returned to the verified Gemini Free Tier project. GPT-5.4 is not the source of the reported judge results. Its offline software checks are not study findings. Preserve the failed OpenAI evidence separately if available; it was not supplied in the completed Gemini ZIP.

## Public original-replication records

`results/replication/` preserves the ten-run Amazon log and the 15 seeded summary/epoch record pairs. `python verify_replication.py` verifies all 31 original-file fingerprints, first-best validation selections, the 55 derived selection rows and four population-SD summaries. Those selectors reuse the same trajectories. They are separate from the 128 extension fits. Original ten-run prediction arrays were not retained; seeded selected-score arrays remain local.

The Gemini evidence is per-case de-identified feature, degree and prediction information, not guaranteed anonymous aggregate data. Temperature was omitted in the recorded requests. The unsuccessful OpenAI route is an additional attempted external disclosure, with no GPT judgments; its original raw error ZIP was not supplied with the successful Gemini evidence.
