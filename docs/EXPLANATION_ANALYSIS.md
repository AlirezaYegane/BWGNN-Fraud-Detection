# Completed explanation-support evaluation

The reported results are the three actual Gemini 3.6 Flash single-turn API responses under `results/explanation_judge/evidence/`, not the blank templates in the original study directory. Read [the completed audit and limits](../results/explanation_judge/JUDGE_RESULTS_REVIEW.md).

From the repository root, check all published tables and the raw judge evidence without credentials, downloads or model training:

```powershell
python verify_public_results.py
```

The standalone raw-evidence auditor also accepts the original unchanged ZIP:

```powershell
python results/explanation_judge/audit_judge_evidence.py --evidence "ORIGINAL_GEMINI_EVIDENCE.zip" --output "LOCAL_JUDGE_CHECKS"
```

There are 24 unique rule-written texts from eight cases and three repeated judgments, not 72 independent examples. Agreement is repeated-judge consistency on simple controls. The prior human-review draft was AI-assisted; no independent human–LLM agreement is computed. Any later author review after seeing these results would be post-LLM and non-blinded.

## Historical combined human/LLM analyser

`analyse_explanation_study.py` remains available for the originally planned protocol when genuinely completed human ratings and metadata are supplied. Its software tests use synthetic fixtures. Neither those fixtures nor empty templates are findings. That historical analyser was not used to manufacture human comparisons for the completed Gemini pilot. Do not fill missing human ratings with generated text or retrospectively label them independent.

## What left the local environment

The frozen pack contains per-case de-identified degree statistics, selected standardised features and model scores. It excludes addresses and true fraud labels, but linkage to the released graph may remain possible. These are not aggregate-only or guaranteed anonymous records. Temperature is not specified in the API requests; the response schema restricts ratings to the three stated categories. I also attempted the same study through the OpenAI runner; quota exhaustion returned no judgment, and the original raw failure ZIP is not included. The submitted report states these limits.
