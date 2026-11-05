# Analyse actual explanation-support ratings

After completing the human review and the three fresh-chat runs, save each actual JSON response and complete human_ratings.csv and run_metadata.csv. Do not use empty templates as results.

```text
python analyse_explanation_study.py --study explanation_study --responses RUN1.json RUN2.json RUN3.json --human HUMAN.csv --metadata METADATA.csv --output EXPLANATION_RESULTS.json
```

The analyser rejects missing/duplicated item IDs, blank ratings, incomplete metadata and altered frozen case/prompt inputs. A software fixture is tested in tests/test_workflow.py; its synthetic ratings are never included as study findings. Per-run rating proportions, pairwise judge agreement, unweighted Cohen kappa, human-versus-run agreement and majority ties are reported. Templates are constructed controls, cases are dependent, and the one author reviewer is not an independent expert panel.
