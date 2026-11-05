# COMP8240: Gemini API factual-support judging

Three independent single-turn REST requests, one fixed model ID, 24 items/request. The original frozen eight cases, explanation texts, rubric and three shuffled orders are unchanged. This package does not retrain BWGNN or install SDKs; Python 3.8+ standard library only.

## Run on the existing Windows environment

```powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Run-Gemini-Judge.ps1
```

Paste the Gemini API key at the hidden prompt and press Enter. The launcher sets a process-local environment variable and clears it afterward. It does not put the key in a source file, command-line argument or results. Google auth keys and standard restricted Gemini keys use the documented `x-goog-api-key` header; the script does not enforce an obsolete key prefix.

Default model: `gemini-3.6-flash`, a stable production model in the documentation checked 5 November 2026. The model ID is locked for the three requests; its returned `modelVersion`, response IDs, usage metadata and safe response headers are retained. A GET model-metadata preflight precedes the three generation requests. Account availability, quota and billing must be established by the real request; no success is fabricated from an offline test.

The structured JSON configuration uses `generationConfig.responseFormat.text` as documented. Output limit: 8,192 tokens/request. No temperature, top-p or top-k control is claimed: those fields are unsupported/ignored on Gemini 3.6+ according to the current documentation. Provider defaults and possible variation remain limitations. No tools, browsing, prior responses, fraud labels or human-review draft are included in the judge requests.

Completion message: `GEMINI JUDGE COMPLETE: 3 runs, 72 verified ratings.` Upload the printed **COMP8240_A3_Gemini_Judge_Evidence.zip**.

## Failure, partial evidence and resume

All requests and raw responses are retained. The script checks all 24 unique IDs, label enums, nonempty reasons, same-case evidence citations and normal STOP completion. It never requests a replacement just because an answer is undesirable or invalid. Network failures produce partial evidence and stop. Send that evidence instead of launching many new batches. If only later runs have not started, `-Resume "PRINTED_BATCH_DIRECTORY"` skips completed calls; an already attempted but unverified run requires inspection rather than an automatic extra generation.

The evidence contains request bodies without credentials, full returned JSON, extracted ratings, timestamps, token usage, hashes, all 72 ratings in CSV and pairwise exact agreement/unweighted Cohen kappa. It does not compute independent-human agreement: the earlier 12-item draft was AI-assisted, so such a claim would be incorrect. An independently completed human review could be analysed separately if later obtained.

## Reproduction and method reporting

This is protocol version2: REST calls replace manual fresh chats, and a structured-output schema is added before execution. The frozen item content, rubric and presentation orders remain unchanged. Three calls to one model are repeated judgments, not three independent models. Constructed control texts are not model-native or causal explanations. Report the actual returned versions/settings and actual outputs; make no accuracy or mechanism claim from agreement alone.

Offline preflight (no key or API calls): `Run-Gemini-Judge.ps1 -DryRun`. Software tests: `python -m unittest discover -s tests -v`. Test fixtures are synthetic software checks, not study findings. Live API execution happens on the user's computer.

Official documentation checked:
- https://aistudio.google.com/docs/api-key
- https://aistudio.google.com/docs/models/gemini-3.6-flash
- https://aistudio.google.com/api-reference/generate-content/generate-content
- https://aistudio.google.com/api-reference/generate-content

Keys shared in chat should be replaced; Google auth-key leak enforcement may reject an exposed key. The script stores sanitised errors and never echoes the key.
