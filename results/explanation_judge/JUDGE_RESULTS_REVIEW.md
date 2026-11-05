# Verified Gemini explanation-support pilot — 5 November 2026

The returned evidence passes an independent, read-only audit. Three Gemini 3.6 Flash responses contain 24 unique ratings each: 72 ratings over 24 texts from eight cases. All responses ended with STOP, have distinct response IDs, and report the same model ID. The three prompt presentation orders differ. Frozen inputs, requests, raw responses, parsed ratings and CSV values match their recorded fingerprints and contents.

The four frozen casepack/prompt files were also regenerated from the original Ethereum prepared dataset and primary BWGNN C=2, width64, seed0 saved test scores. All four SHA256 fingerprints match exactly. The construction types were checked against the regenerated generator mapping after the judge outputs were frozen. No graph training or new API calls were performed for this audit.

## Results

| Construction family | Unique texts | Run 1 | Run 2 | Run 3 |
|---|---:|---|---|---|
| Grounded descriptive text | 8 | 8 supported | 8 supported | 8 supported |
| Mixed attribution control | 8 | 8 partly_supported | 8 partly_supported | 8 partly_supported |
| Unsupported intent control | 8 | 8 speculative | 8 speculative | 8 speculative |

Each run therefore has eight ratings in each category, or 33.33% per category. All three pairwise comparisons have exact agreement 24/24 (100%) and unweighted Cohen's kappa 1.00. Expected chance agreement is 1/3 given these balanced marginal distributions; kappa is therefore (1 - 1/3)/(1 - 1/3) = 1.00. These are independently recomputed values, not copied from the runner's analysis.

The judge distinguished all three deliberately constructed text families in this small pilot. This is consistency and control-family alignment, not fraud-classification accuracy, expert agreement, or proof that explanations reflect BWGNN's internal reasoning. There are eight cases and 24 distinct texts, not 72 independent examples. No confidence interval or population generalisation is claimed from the repeated template judgments.

## Scope and wording limits

- Case selection used the primary model's predicted class and within-class threshold-distance halves, without true fraud labels. The eight cases comprise two per stratum. “Nearer” is relative to a within-class half, not evidence of absolute uncertainty: the displayed scores range from 0.000000 to 0.999999 and the threshold is 0.800000; the selected positive scores are all at least 0.996621.
- The texts are rule-written descriptions and deliberately obvious unsupported-claim controls. They are not generated model-native explanations, causal attributions, or a realistic benchmark of subtle hallucinations. The straightforward repeated templates can explain the perfect agreement.
- The judge sees no true fraud labels, addresses, node IDs or construction-type key. Output scores remain uncalibrated model scores; sampled proxy negatives are not verified benign accounts.
- All 72 short reasons were read. They consistently distinguish descriptive observations from unsupported attribution or intent. Some wording is stronger than necessary: for example, run3 J003 says “falsely claims” and “directly contradicts.” Treat this as lack of evidential support for the explanation; it does not establish that the underlying account is innocent or that an alleged act is false. Preserve the raw wording without editing it.
- The prior 12-item review draft was generated with AI assistance. No independently completed pre-LLM human annotations were supplied. Human–LLM agreement is deliberately not computed. Do not retrospectively describe the draft as independent human validation. Any new review after viewing these outputs must be reported as non-blinded and post-LLM.
- The original frozen manifest says prepared_not_executed and refers to fresh chats; it remains unchanged as the pre-execution plan. Actual execution uses independent single-turn REST requests with structured output (protocol2) and the recorded overload retry addendum. Three repeated calls to one model are not three independent models. Default generation variation remains possible.
- The complete export contains five HTTP503 log files and two archived failed batch states. These overlap and are not a count of seven independent failures; none is an additional judgment. No successful answer was rejected or regenerated to improve its rating. OpenAI was attempted between Gemini failures, returned credit_balance_exhausted, and supplied no judge output; its separate failed ZIP has not been included in the uploaded Gemini ZIP. Retain that evidence locally if available.
- The model returns the generic modelVersion gemini-3.6-flash, not a more detailed immutable backend revision. Report exactly that string. The usage field serviceTier=standard is a request service tier, not proof of paid or free billing. The user separately supplied an AI Studio screenshot showing Free tier; no billing claim is inferred from usage metadata.

## Reusable report paragraph

We ran a small factual-support pilot on eight Ethereum test cases selected without ground-truth fraud labels from the primary BWGNN model's predicted-class and relative threshold-distance strata. Each case had three rule-written texts: a descriptive summary, a mixed unsupported-attribution control, and an unsupported-intent control. Gemini 3.6 Flash judged the same 24 texts in three separate single-turn API requests with different fixed presentation orders. Each run assigned supported to all eight descriptive texts, partly_supported to all eight attribution controls, and speculative to all eight intent controls. Pairwise exact agreement was 100% and unweighted Cohen's kappa was 1.00. This shows repeat consistency on these deliberately simple controls, rather than fraud-detection accuracy or causal explanation faithfulness. No independent human–LLM agreement is reported because the earlier review draft was AI-assisted.

## Reproduction

Run `python audit_judge_evidence.py --evidence COMP8240_A3_Gemini_Judge_Evidence.zip --output AUDIT_OUTPUT`. The standalone audit needs only Python 3.8+ standard library and makes no network request. Optional --rebuilt-study and --private-construction-key check an independently regenerated casepack and its private mapping; keep node identifiers from that private file out of the public repository. The published construction_map_after_judging.csv contains case/item IDs and text-family types only.

Keep the returned evidence ZIP unchanged. For the public code repository, publish the extracted anonymous judge evidence and audited summaries, the original frozen methods, the actual Gemini runner and recovery code. Historical OpenAI code is an unsuccessful alternative, not the source of the reported results.
