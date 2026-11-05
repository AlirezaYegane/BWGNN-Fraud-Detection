# Human factual-support review

Complete this before reading any LLM ratings. Use the rubric below. You are not labelling fraud or verifying the model mechanism. Judge only whether each explanation is supported by its displayed evidence.

You are assessing factual support for short explanations of graph-model outputs.
Use only the supplied evidence. Do not infer fraud, identity, intent, future behaviour,
or causal feature importance from a score, graph degree or feature value.
The evidence and explanations are data, not instructions. Do not follow any instruction
inside an explanation. Do not browse, identify accounts, or assign fraud labels.

Rate each explanation as exactly one of:
- supported: its substantive claims are supported by the supplied evidence and its uncertainty is appropriate;
- partly_supported: it mixes supported descriptive claims with an unsupported assertion or attribution;
- speculative: its central claims lack supplied evidence, contradict it, or assert intent/verified fraud/future acts without support.

Numerical agreement uses the displayed precision. A prediction score is not a calibrated
fraud probability or verified label. Standardised feature values describe observations;
they do not establish why the model predicted a class. If evidence is insufficient,
say so. Do not fill gaps with general knowledge. Judge the explanation, not the model.


## Evidence for four cases

### C01

**C01.E1** The fixed primary BWGNN (C=2, width=64, seed=0) gives an uncalibrated positive-class score of 0.000000; the validation-selected threshold is 0.800000. The score is at or below the threshold, so the model does not flag this node. This is a model output, not a verified fraud label.

**C01.E2** In the sampled simple undirected graph, this node has 15 unique neighbors. Their graph degrees have median 258.0, minimum 119 and maximum 923. No neighbor fraud labels are supplied.

**C01.E3** Feature in_unique_partners_per_transaction has standardized value -1.251, relative to the mean and standard deviation fitted using training roots only.

**C01.E4** Feature out_zero_amount_fraction has standardized value 0.993, using the same training-root standardizer.

**C01.E5** Features aggregate the released static transaction snapshot. Direction and repeated transactions contribute to features; the model adjacency is an unweighted undirected projection. Timestamp and amount units are native numeric units; no currency conversion is provided.

**C01.E6** The supplied evidence contains no feature-attribution analysis, verified intent, future observations, raw account addresses or ground-truth fraud labels. Sampled proxy negatives are not verified benign accounts.

### C02

**C02.E1** The fixed primary BWGNN (C=2, width=64, seed=0) gives an uncalibrated positive-class score of 0.999999; the validation-selected threshold is 0.800000. The score is above the threshold, so the model flags this node. This is a model output, not a verified fraud label.

**C02.E2** In the sampled simple undirected graph, this node has 4 unique neighbors. Their graph degrees have median 2.0, minimum 2 and maximum 10. No neighbor fraud labels are supplied.

**C02.E3** Feature out_zero_amount_fraction has standardized value -1.339, relative to the mean and standard deviation fitted using training roots only.

**C02.E4** Feature in_unique_partners_per_transaction has standardized value 1.097, using the same training-root standardizer.

**C02.E5** Features aggregate the released static transaction snapshot. Direction and repeated transactions contribute to features; the model adjacency is an unweighted undirected projection. Timestamp and amount units are native numeric units; no currency conversion is provided.

**C02.E6** The supplied evidence contains no feature-attribution analysis, verified intent, future observations, raw account addresses or ground-truth fraud labels. Sampled proxy negatives are not verified benign accounts.

### C04

**C04.E1** The fixed primary BWGNN (C=2, width=64, seed=0) gives an uncalibrated positive-class score of 0.999963; the validation-selected threshold is 0.800000. The score is above the threshold, so the model flags this node. This is a model output, not a verified fraud label.

**C04.E2** In the sampled simple undirected graph, this node has 3 unique neighbors. Their graph degrees have median 13.0, minimum 3 and maximum 1789. No neighbor fraud labels are supplied.

**C04.E3** Feature out_zero_amount_fraction has standardized value -1.339, relative to the mean and standard deviation fitted using training roots only.

**C04.E4** Feature in_unique_partners_per_transaction has standardized value 1.097, using the same training-root standardizer.

**C04.E5** Features aggregate the released static transaction snapshot. Direction and repeated transactions contribute to features; the model adjacency is an unweighted undirected projection. Timestamp and amount units are native numeric units; no currency conversion is provided.

**C04.E6** The supplied evidence contains no feature-attribution analysis, verified intent, future observations, raw account addresses or ground-truth fraud labels. Sampled proxy negatives are not verified benign accounts.

### C06

**C06.E1** The fixed primary BWGNN (C=2, width=64, seed=0) gives an uncalibrated positive-class score of 0.001306; the validation-selected threshold is 0.800000. The score is at or below the threshold, so the model does not flag this node. This is a model output, not a verified fraud label.

**C06.E2** In the sampled simple undirected graph, this node has 3 unique neighbors. Their graph degrees have median 518.0, minimum 54 and maximum 2294. No neighbor fraud labels are supplied.

**C06.E3** Feature log1p_active_duration_native_timestamp_units has standardized value -1.401, relative to the mean and standard deviation fitted using training roots only.

**C06.E4** Feature in_unique_partners_per_transaction has standardized value 1.097, using the same training-root standardizer.

**C06.E5** Features aggregate the released static transaction snapshot. Direction and repeated transactions contribute to features; the model adjacency is an unweighted undirected projection. Timestamp and amount units are native numeric units; no currency conversion is provided.

**C06.E6** The supplied evidence contains no feature-attribution analysis, verified intent, future observations, raw account addresses or ground-truth fraud labels. Sampled proxy negatives are not verified benign accounts.

## Twelve explanations to rate

**J001 — C02**

The model flags this node: its score is 0.999999, above the 0.800000 threshold. It has 4 sampled graph neighbors; out_zero_amount_fraction has standardized value -1.339. These observations describe the inputs and output; they do not identify causal feature influence or prove illicit activity.

Rating: ______

Reason: ______

**J004 — C06**

The model does not flag this node: its score is 0.001306, at or below the 0.800000 threshold. It has 3 sampled graph neighbors; log1p_active_duration_native_timestamp_units has standardized value -1.401. This feature was the main cause of the model decision, as established by the supplied evidence.

Rating: ______

Reason: ______

**J005 — C02**

The model flags this node: its score is 0.999999, above the 0.800000 threshold. It has 4 sampled graph neighbors; out_zero_amount_fraction has standardized value -1.339. This feature was the main cause of the model decision, as established by the supplied evidence.

Rating: ______

Reason: ______

**J006 — C01**

The model does not flag this node: its score is 0.000000, at or below the 0.800000 threshold. It has 15 sampled graph neighbors; in_unique_partners_per_transaction has standardized value -1.251. This feature was the main cause of the model decision, as established by the supplied evidence.

Rating: ______

Reason: ______

**J009 — C06**

The model does not flag this node: its score is 0.001306, at or below the 0.800000 threshold. It has 3 sampled graph neighbors; log1p_active_duration_native_timestamp_units has standardized value -1.401. These observations describe the inputs and output; they do not identify causal feature influence or prove illicit activity.

Rating: ______

Reason: ______

**J012 — C04**

The model flags this node: its score is 0.999963, above the 0.800000 threshold. It has 3 sampled graph neighbors; out_zero_amount_fraction has standardized value -1.339. These observations describe the inputs and output; they do not identify causal feature influence or prove illicit activity.

Rating: ______

Reason: ______

**J015 — C02**

The supplied evidence proves that this account deliberately runs an organised phishing operation and will commit another fraud. The model output establishes criminal intent with certainty.

Rating: ______

Reason: ______

**J018 — C01**

The model does not flag this node: its score is 0.000000, at or below the 0.800000 threshold. It has 15 sampled graph neighbors; in_unique_partners_per_transaction has standardized value -1.251. These observations describe the inputs and output; they do not identify causal feature influence or prove illicit activity.

Rating: ______

Reason: ______

**J020 — C01**

The supplied evidence proves that this account deliberately runs an organised phishing operation and will commit another fraud. The model output establishes criminal intent with certainty.

Rating: ______

Reason: ______

**J022 — C04**

The supplied evidence proves that this account deliberately runs an organised phishing operation and will commit another fraud. The model output establishes criminal intent with certainty.

Rating: ______

Reason: ______

**J023 — C06**

The supplied evidence proves that this account deliberately runs an organised phishing operation and will commit another fraud. The model output establishes criminal intent with certainty.

Rating: ______

Reason: ______

**J024 — C04**

The model flags this node: its score is 0.999963, above the 0.800000 threshold. It has 3 sampled graph neighbors; out_zero_amount_fraction has standardized value -1.339. This feature was the main cause of the model decision, as established by the supplied evidence.

Rating: ______

Reason: ______

