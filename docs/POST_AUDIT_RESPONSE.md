# Response to AI-assisted report review

Prepared 5 November 2026 by Alireza Yegane with AI assistance. I used its factual findings to revise the report and improve access to the evidence. I have not added new training runs or human annotations.

| Review point | Revision and verification |
|---|---|
| D5: relate the extensions to the replication | Section 6.2 now directly compares the unmodified Amazon extension with the ten-run reproduction and the paper's GCN baseline. It also contrasts the primary paired Ethereum gap (5.64 ± 1.12 pp) with Amazon (23.60 ± 0.63 pp), without implying that dataset differences identify a cause. |
| D1: oracle interpretation | The results and conclusion now state that the rewired structure carries validation/test-label information. Structural label leakage is an alternative explanation; the maximum-budget random control does not exclude it. |
| D2: data construction | Section 4 identifies the whole released unlabelled candidate pool, induced-subgraph adjacency, sampling without replacement, all 18 feature components and the untested crawl-position shortcut. |
| D3: original evaluation | The report no longer calls the extension an implicit correction of erroneous original results. BWGNN has no dropout/normalisation layers; the original reused logits describe pre-update parameters. The observation wrapper and reuse of five diagnostic trajectories are explicit. |
| D4: privacy and judge settings | Per-case de-identified evidence replaces the inaccurate aggregate-only/anonymity wording. Possible linkage, omitted temperature, JSON categories and the failed OpenAI route are disclosed. A separate audit script is not an independent human rater. |
| D6: references and graph counts | Added CARE-GNN, the two review-corpus sources and GCN, plus actual data-release footnotes. Bidirectional relation entries plus loops explain 9,569,592 stored Amazon entries; the remaining 19,184 unique-pair discrepancy is retained as unresolved. |
| D7: paper justification | Added a numerical paper-baseline comparison. I did not invent a Google Scholar count. The supplied brief presents citation counts among possible selection considerations, rather than a standalone mandatory number. |
| D8: ethics | Expanded data-rights, expert-label provenance, re-identification, access-versus-licensing and dual-use discussion. No consent, ethics approval or exemption is invented. Unit-specific resources remain an open input. |
| D9: clarity and layout | The spectral caption now identifies Amazon graph seeds; API capitalisation and diagnostic wording are corrected. Both ACL and cover/contents variants were rebuilt and inspected. |
| P1: public replication evidence | Added the unchanged ten-run log and 15 seeded epoch/summary pairs, derived selection/aggregate tables and a standard-library verifier. Original ten-run prediction arrays remain unavailable. |
| P4: stale documentation | Updated the current tracker, report status, first-person account of API attempts, and Ethereum-only scaling statement. Historical process scripts and frozen templates remain at their original paths to preserve relative-path execution and evidence hashes; they are not advertised as completed human review. |

## What the checks establish

The public check passes on this Linux/Python 3.12 host and parses every source against Python 3.8 syntax. It recomputes published means/SDs and original validation selections and verifies the raw Gemini evidence. The existing Windows/Python 3.8.20 receipt is historical and is not rewritten as if it covered this new verifier. The installation script runs the updated public check again locally before staging or publishing.

All 25 existing orchestration/API tests pass. Negative checks on temporary copies reject an altered historical file and a changed derived reproduction score. None of these checks retrains the scientific models or re-executes checkpoint tensors. Training sources and the 45 frozen Gemini evidence files are unchanged from the published experimental baseline.

## Remaining limits and final actions

1. Apply this evidence/documentation patch and retain the new local verification receipt; publish normally without force-pushing.
2. Check the unit ethics materials, the Overall Goal and Assessment Criteria page and later lecturer instructions when supplied. The report does not claim to have reviewed them.
3. Read the final manuscript personally and submit the compiled ACL PDF under the actual course instructions. The plain version contains eight manuscript pages plus references; the optional cover/contents version has ten pages before references.

No predicted full mark is attached to these revisions. Proxy negatives, one source snapshot, shared positive identities and no independent human evaluation remain scientific limitations.

## Second review: final wording and provenance corrections

The draft/report review was AI-assisted, not an independent human assessment. I checked its factual suggestions against saved records; I removed its estimated mark from the repository.

- N1: The report footnote and self-reference now distinguish experimental-results commit `3e70f6d` from original-replication evidence commit `7699123`. These are stable source snapshots, not a claim that either hash is the latest documentation commit.
- N2: Section 10 explicitly discloses AI-assisted draft/repository review. README and current documentation use the same description.
- N3: Section 7 describes the threshold-distance halves directly; it no longer uses an undefined “Nearer” label.
- N4: Section 9 distinguishes the earlier Windows check from the completed updated Linux/Python 3.12.14 check and its actual receipt.
- N5: The abstract now states that oracle rewiring carries test-label information.
- N6: Section 3.2 reports the observed early-checkpoint pattern, including printed-value ties and the distinction between association and a causal explanation. `results/replication/checkpoint_diagnostic.json` retains all ten runs' printed validation maxima; it is derived from the unchanged log in evidence commit `7699123`. It does not recover missing full-precision epoch selection or original prediction arrays.

Only documentation and this derived diagnostic are changed in the second update. Training drivers, verifiers, scientific result tables, historical validation receipts, and frozen raw evidence remain unchanged. The public verification was rerun on the revised checkout; no training or API generation was repeated.
