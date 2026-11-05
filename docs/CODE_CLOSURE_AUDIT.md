# Code closure audit — 5 November 2026

## Changes

- Replaced obsolete active task instructions with the completed execution status; labelled the three old analysis trackers as historical snapshots.
- Added a marker guide separating public consistency checks, actual pinned-runtime smoke checks, and full classifier-evidence verification. No private data access is implied by the public check.
- Added `verify_public_results.py`, a Python 3.8-compatible standard-library command with zero network requests, training or API calls.
- Corrected Git line-ending conversion of frozen Gemini evidence. Every published raw-evidence file now has exactly the original acquisition ZIP bytes; `.gitattributes` disables text conversion for that subtree. No response wording, ratings, scientific model code or result-table values were changed.

## Validation

- Public check PASS: 75 core fits, 35 sensitivity fits, 18 new construction fits and nine reused references; means/sample SDs for all four reported metrics; paired Amazon gaps; three byte-exact raw Gemini responses and all 72 ratings/citations/agreement.
- 25 existing software tests PASS: workflow 4, Gemini runner 6, overload recovery 5, historical OpenAI runner 10. These are software fixtures, not new scientific observations.
- Failure detection checked on temporary copies: an incorrect aggregate Macro-F1 is rejected; an altered raw response is rejected by its recorded hash.
- `PUBLIC_PACKAGE_VALIDATION.json` records the host and precise public-check scope. It does not replace the separately supplied actual Windows/Python 3.8.20 receipt in `WINDOWS_RUNTIME_VALIDATION.json`.
- Fresh Git index checkout with `core.autocrlf=true` PASS: all 45 raw-evidence files equal the acquisition ZIP bytes, the public check passes, and local upstream bootstrap/source verification passes. Active documentation links resolve.
- Full 128-fit scientific evidence was verified in the earlier independent audits. This closure does not retrain models or claim to rerun the Windows ML stack on this host.

## Outside code closure

The ACL report, unit ethics resources, latest overall assessment guidance and final PDF/rubric audit remain report-stage work. There is no independent human annotation, so the report must disclose that limitation and the departure from the proposal. Code readiness is not an award of the complete 40 report/project marks.
