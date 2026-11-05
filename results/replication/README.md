# Original reproduction records

These are the historical records behind the report's replication table and Amazon diagnostic. I added them after the independent audit so a reader can inspect this part of the project from a public checkout.

| Record | Training trajectories | Selection and precision |
|---|---:|---|
| Amazon 40%, homogeneous, original log | 10, unseeded | Original validation Macro-F1; final test figures printed to two decimal percentage points |
| YelpChi 1%, heterogeneous | 10, seeds 0–9 | Original validation Macro-F1; full-precision per-seed summaries |
| Amazon diagnostic | 5, seeds 0–4 | Same trajectories observed under three validation selectors; these are not 15 separate fits |

`raw/` preserves the original 10-run UTF-16 console log and 15 pairs of `epochs.csv`/`summary.json` byte for byte. `source_manifest.json` records their hashes and original archive-member names. The Git attributes exclude these raw records from line-ending conversion. The source-member names preserve the two original seed-0 batches and their follow-ups; no seed is counted twice.

`per_run_results.csv` is derived from those records. `summary.json` reproduces the four report/diagnostic aggregates, using population SD (ddof=0). Blank original-log margin AUC and threshold fields mean not retained, not zero. The log contains ten complete 100-epoch blocks; original prediction arrays are not available. Seeded selected-score NPZs remain local and are not part of this public check.

```powershell
python verify_replication.py
python verify_public_results.py
```

The first command validates the byte hashes, ten final log records, complete seeded epoch inventories, first-best validation selection and derived per-run/aggregate figures. It does not independently recompute F1/AUC from private predictions or rerun training. Epoch records include the original loop's per-epoch test observations; test scores do not select a record in this check. Alternative validation-AUC selections are diagnostics only, not replacements for the reproduction table.

`audit_bwgnn.py` retains the released training loop and observes its pre-update logits; the extension drivers use their separate restored-checkpoint protocol. The model and source attribution remain in `THIRD_PARTY.md`.
