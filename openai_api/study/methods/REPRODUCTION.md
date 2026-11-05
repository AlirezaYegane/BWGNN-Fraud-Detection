# Rebuild the frozen casepack

The script does not run judges. Run it with the audited Ethereum dataset, primary core Ethereum seed0 C2 score file and feature_descriptions.json from the preprocessing evidence. It requires NumPy only. Use a NEW empty output directory; the generator writes blank rating forms. Do not point it at completed ratings.

```text
python prepare_explanation_study.py --dataset DATASET_NPZ --scores PRIMARY_CORE_SEED0_SCORES_NPZ --feature-descriptions FEATURE_DESCRIPTIONS_JSON --output NEW_OUTPUT_DIRECTORY --private-key PRIVATE_MAPPING_JSON
```

The private mapping records sampled local IDs and constructed text types; never provide it to the judge, and do not consult it before human annotation. It contains no fraud labels. Dataset and score fingerprints are recorded in study_manifest.json. The frozen prompts, not a regenerated rendering, are the inputs for this assessment. Labels do not drive sampling. Variants are controlled rule-written texts, not causal model explanations.
