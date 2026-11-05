# COMP8240: OpenAI API explanation-support judging

Run `Run-OpenAI-Judge.ps1` with Windows PowerShell. Enter the full OpenAI secret key at its hidden prompt. A tracking ID or masked dashboard value cannot authenticate an API request. The launcher uses the existing `miniconda3/envs/bwgnn/python.exe`; it installs no SDK and changes no training dependency. It clears the process environment key when finished.

The fixed model snapshot is `gpt-5.4-2026-03-05`, with reasoning effort `medium`, no custom temperature or seed, maximum output 16,384 tokens per request, strict JSON Schema, and `store: false`. There are three new single-turn Responses API requests, 24 ratings each. No tools, web search, conversation history, previous response ID, fraud labels, or human-review draft are supplied. A model-metadata GET precedes generation. A valid key and actual account access/credit must be confirmed by real execution; an offline software test does not establish them.

Completion: `OPENAI JUDGE COMPLETE: 3 runs, 72 verified ratings.` Send the printed `COMP8240_A3_OpenAI_Judge_Evidence.zip`. An API error stops the batch and produces partial evidence. Do not repeatedly launch new batches. The optional `-Resume "BATCH_DIRECTORY"` checks existing completed responses and skips them, but refuses to replace any already attempted, unverified generation. No automatic retry, fallback model, or answer-based rerun is implemented.

The Windows SSL workaround is included: an existing certifi, pip-vendored certifi or Conda PEM CA bundle is explicitly loaded. Certificate and hostname verification remain enabled. The CA fingerprint and OpenSSL version are recorded. API authentication uses a Bearer header, never a URL parameter, request body, or logged command-line argument. Evidence exports are checked for the current key bytes.

## Protocol version3 and provider change

The eight cases, 24 explanation variants, rubric and three shuffled presentation orders retain their original SHA256 fingerprints. OpenAI replaces Gemini after the user's repeated explicit HTTP 503 UNAVAILABLE responses, before any judge answer was received. This is a provider change for service availability, not selection of a model based on favorable judgments. The default previous-batch path matches the observed Windows run. Its JSON/text/code evidence is copied and fingerprinted into the new export; its files are left intact. The script refuses the switch if an active run, verified Gemini judgment, or raw Gemini generation response is found in the current run folders. It also checks that the frozen study inputs match. The failed requests are not counted as judge ratings or as a comparison against Gemini.

The report must disclose the provider change and actual model, settings, response IDs, timestamps and usage. Three calls to one fixed snapshot are repeated judgments, not three independent models; a snapshot is not a guarantee of deterministic outputs. The constructed variants are rule-written explanation/control texts, not BWGNN-native or causal explanations. AI-assisted preparation and use of an OpenAI judge can introduce related biases. Agreement does not establish correctness or model faithfulness. The earlier 12-item review draft was AI-assisted and is not treated as independent human validation.

The results contain raw response JSON, request bodies without credentials, actual returned model IDs, response IDs, token usage, safe request-ID headers, SHA256 fingerprints, 72 CSV ratings if complete, and ID-aligned pairwise exact agreement and unweighted Cohen kappa. Refusals, incomplete output, wrong model snapshots, duplicate IDs and wrong-case evidence citations are retained as failures, never replaced just to obtain a valid or desirable answer.

## Offline verification

`Run-OpenAI-Judge.ps1 -DryRun` checks frozen inputs without a key or any network request. `python -m unittest discover -s tests -v` runs synthetic software checks. The validation receipt records ten passing software tests, not actual study findings. Real judging occurs on the user's computer.

Official OpenAI documentation checked 5 November 2026:
- https://developers.openai.com/api/docs/models/gpt-5.4
- https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses
- https://developers.openai.com/api/docs/guides/reasoning

TLS background:
- https://github.com/python/cpython/issues/151504
- https://github.com/certifi/python-certifi
