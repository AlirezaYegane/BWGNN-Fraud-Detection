# Checking this project

All commands below run from the repository root. My student repository is https://github.com/AlirezaYegane/BWGNN-Fraud-Detection . The model and training data are attributed in [THIRD_PARTY.md](../THIRD_PARTY.md).

## 1. Public-package check: no downloads or model fitting

With Python 3.8 or later, using only its standard library:

```powershell
python verify_public_results.py --output validation_receipt_public_local.json
```

This checks Python 3.8 syntax, complete seed inventories, all four metric means/sample SDs against the published per-fit tables, the nine reused construction references, Amazon paired gaps, original replication log/epoch/selection summaries, and all three raw Gemini responses and their recorded hashes/citations/ratings/agreement. It makes zero network/API requests. A receipt from the final package check is in `docs/PUBLIC_PACKAGE_VALIDATION.json`.

This is a consistency check of included evidence. It does not rerun model checkpoints or recompute classifier metrics from prediction arrays that are excluded from Git.

## 2. Actual small runtime checks

The pinned stack and source bootstrap are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). After installing that environment and obtaining the three unchanged upstream files:

```powershell
python bootstrap_upstream.py
python verify_submission.py --runtime
```

If the original audited project already exists locally, `python bootstrap_upstream.py --from-project "PATH_TO_ORIGINAL_PROJECT"` avoids the source download. The runtime command runs actual toy preprocessing and CPU model checks in Python 3.8.20. It does not repeat the scientific training. The historical actual Windows receipt is `docs/WINDOWS_RUNTIME_VALIDATION.json`.

## 3. Full classifier evidence and data reconstruction

Raw address data, NPZ prediction arrays and checkpoints are excluded from ordinary Git commits; they are not silently downloaded by the public check. The full-evidence verification commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Source downloads, provenance, acquisition-manifest fields and general reconstruction commands are in [DATA_AND_COMMANDS.md](DATA_AND_COMMANDS.md). Ethereum source access may require a Kaggle account and acceptance of the source terms. The source graph is about 1.26 GB before loading and has substantially higher RAM requirements than the toy checks.

The archived orchestration drivers retain the original input fingerprints and owner-machine defaults. They are evidence of the fixed runs, not a one-command fresh-checkout dataset download. For newly constructed data use the general training commands and record its new provenance rather than bypassing hash locks.

## 4. Study status and preserved artifacts

`docs/RUBRIC_TRACKER.md` is the current status. The trackers under the three training-result directories are historical snapshots. Original frozen prompts/manifests and blank review templates preserve the plan; completed Gemini API evidence is separate. No independent human agreement is claimed. The frozen judge-evidence and historical replication raw-record subtrees is exempt from Git line-ending conversion so its recorded SHA256 values remain meaningful across Windows and Linux.

No extra model fitting or API calls are necessary to inspect the published findings. The final ACL report is prepared separately.

## 5. Original replication, separately from extensions

`python verify_replication.py` checks the original log and seeded records using only the standard library. The evidence directory explains its exact coverage and the precision limit of the original ten-run console figures. The three Amazon diagnostic selectors reuse five training trajectories, not fifteen. No public-check receipt claims access to the excluded prediction arrays or execution of model checkpoints.

The separate ACL report has been prepared and revised after review. Course-specific ethics resources and later lecturer guidance remain final compliance inputs; this repository does not certify those unseen requirements.
