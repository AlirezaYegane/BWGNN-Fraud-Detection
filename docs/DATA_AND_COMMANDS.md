# Data provenance and execution

## Ethereum

Source: XBlock Ethereum Phishing Transaction Network, https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network . The originally downloaded archive has SHA256 `1bd86d022840e84a52c60caaea9ffbf621d46a9c4c1914e09432f566fe34f95f`; selected entry `Ethereum Phishing Transaction Network/MulDiGraph.pkl` has SHA256 `155fbbd738a7855a194a3c975711412b48d6d483f6387e47638832d999dc8d14` and 1,257,580,711 bytes. Download from the source and follow its terms. Public access does not establish a redistribution licence.

The source contains 2,973,489 addresses, 13,551,303 directed transactions and 1,165 released phishing labels. `isp=0` means unlabelled phishing status, not confirmed benign. The raw pickle must be fingerprinted against the recorded source before loading. `audit_ethereum.py` uses a Windows supervisor and an explicit acquisition manifest containing `raw_file`, `raw_sha256` and `raw_bytes`. The original local acquisition manifest is kept with the downloaded data, not committed publicly. If relocating the raw input, update only its local path while keeping size/hash checks.

```powershell
python audit_ethereum.py --manifest "PATH_TO_ACQUISITION_MANIFEST.json" --output "data\ethereum_audit"
python prepare_ethereum.py --audit-report "PATH_TO_GRAPH_AUDIT.json" --seed 0 --negative-ratio 4 --fanout 8 --hops 2 --max-nodes 50000 --max-edges 2000000 --output "processed_results"
python train_ethereum.py --dataset "PATH_TO_PREPARED_DATASET.npz" --models bwgnn gcn mlp --seeds 0 1 2 3 4 --order 2 --hidden 64 --epochs 100 --lr 0.01 --threads 16 --output "training_results"
```

The audit prints its actual directory and preprocessing prints its prepared dataset path. Use those concrete paths. A copied audit report must point to the actual compact-array directory; all exported compact files are hash checked.

Features retain all released directed transactions, parallel events and self-transactions. The GNN base adjacency collapses multiplicity and direction and removes loops. GCN adds one loop/node; original core BWGNN does not. Eighteen features encode counts, amounts, unique partners, duration, direction balance and activity rate. Native amount/timestamp units are retained, without an ETH/wei conversion claim.

The archived prepared sample has 49,224 nodes and 929,442 bidirected stored edges. It has 5,825 labelled roots (20% positives), including train/validation/test counts 2,330/1,153/2,342. Features are scaled using training roots only; context labels are -1 and excluded from supervision. Prepared NPZ SHA256: `e1c261e30ddc70266829f5eebae65fab0c3e67facc69b8da7bc2adfd4a39bbd9`.

## Amazon

Amazon uses the DGL FraudDataset benchmark (Musical Instruments users, 25 features). Nodes 0–3304 lack supervised labels in the released benchmark. Homogeneous conversion retains relation multiplicities and the original self-loops; do not substitute a silently simplified graph.

```powershell
python audit_amazon_camouflage.py --output "audit_results"
```

The printed directory contains `amazon_compact.npz` and `amazon_graph_audit.json`. The archived compact SHA256 is `31be189110b98a8d39f46b99e261b2ed4ac454aba15af1d3ed7b6bd9d13c4334`.

For each paired construction/model seed, run all five levels plus the automatic maximum-budget random control:

```powershell
0..4 | ForEach-Object {
    python run_amazon_camouflage.py --compact "PATH_TO_AMAZON_COMPACT.npz" --levels 0 0.2 0.4 0.6 0.8 --graph-seed $_ --training-seed $_ --epochs 100 --threads 16
    if ($LASTEXITCODE -ne 0) { throw "Amazon run failed." }
}
```

This is a label-informed oracle test: equal-multiplicity FF and BB pairs become FB pairs, preserving weighted and unique degrees. Four distinct eligible endpoints, no new duplicates/loops, unchanged features/labels/splits and nested swap prefixes are checked. The random control matches the largest swap count and multiplicity sequence only.

## Metrics

Checkpoint and decision threshold are selected using validation Macro-F1 only. Probability must be strictly greater than the chosen grid threshold to predict positive; first threshold/epoch wins ties. Test is scored after restoring the chosen parameters. ROC-AUC, average precision and trapezoidal PR-AUC use the float64 difference of stored float32 logits; average precision and PR trapezoidal area are different quantities. Do not pool original reproduction and extension statistics without identifying their protocol differences.
