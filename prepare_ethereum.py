"""Reproducible XBlock pilot: transaction features, matched noisy negatives,
bounded neighbourhood sampling, and a train-only feature standardizer.
Python 3.8+, NumPy, SciPy, scikit-learn. No model training or source edits.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SOURCE_HASHES = {
    "main.py": "1642e10e5c799ebd303bb20d66a991a55363276b35855996f5615eba0fd1ea13",
    "dataset.py": "229b5e5101c63879264802f909e64396a49be694a8bb4815dfea201f9b78d1fb",
    "BWGNN.py": "0ac45b1104f80fa17c50cd5e3bf9aa6241e008e6b57fdab9366ec69f80cb2f84",
}
FEATURES = [
    "log1p_in_transaction_count", "log1p_out_transaction_count",
    "log1p_in_unique_partners", "log1p_out_unique_partners",
    "log1p_in_amount_sum", "log1p_out_amount_sum",
    "log1p_in_mean_amount", "log1p_out_mean_amount",
    "log1p_in_max_amount", "log1p_out_max_amount",
    "in_zero_amount_fraction", "out_zero_amount_fraction",
    "log1p_active_duration_native_timestamp_units",
    "transaction_direction_balance_out_minus_in",
    "amount_direction_balance_out_minus_in",
    "in_unique_partners_per_transaction", "out_unique_partners_per_transaction",
    "log1p_transactions_per_active_timestamp_unit",
]
ARRAY_NAMES = ["labels", "source", "target", "amount", "timestamp", "in_transaction_count",
               "out_transaction_count", "in_unique_partners", "out_unique_partners"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_original_sources(project):
    result = {}
    for name, expected in SOURCE_HASHES.items():
        content = (project / name).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise RuntimeError("Original source changed: " + name)
        result[name] = actual
    return result


def load_compact(report):
    if report.get("status") != "audit_complete" or not report.get("schema_usable_for_preprocessing"):
        raise RuntimeError("The graph audit does not approve this schema for preprocessing.")
    compact = Path(report["compact_directory"])
    for name, record in report["export_files"].items():
        path = compact / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError("Compact-file fingerprint mismatch: " + name)
    data = {name: np.load(str(compact / (name + ".npy")), mmap_mode="r", allow_pickle=False) for name in ARRAY_NAMES}
    n, m = report["node_count"], report["transaction_count"]
    for name, array in data.items():
        expected = m if name in ("source", "target", "amount", "timestamp") else n
        if array.shape != (expected,):
            raise RuntimeError("Unexpected array shape: " + name)
    observed = {str(int(v)): int(c) for v, c in zip(*np.unique(data["labels"], return_counts=True))}
    if observed != report["isp_counts"] or set(observed) != {"0", "1"}:
        raise RuntimeError("Released-label counts disagree with the audit.")
    return data


def activity_bins(data):
    transactions = data["in_transaction_count"] + data["out_transaction_count"]
    # Sum of directional unique-partner counts; not a union of both directions.
    partners = data["in_unique_partners"] + data["out_unique_partners"]
    tx_bin = np.floor(np.log2(1.0 + transactions)).astype(np.int16)
    partner_bin = np.floor(np.log2(1.0 + partners)).astype(np.int16)
    return transactions, partners, tx_bin, partner_bin


def match_negatives(data, ratio, seed):
    labels = data["labels"]
    positives = np.flatnonzero(labels == 1)
    candidates = np.flatnonzero(labels == 0)
    if len(positives) < 6 or len(candidates) < ratio * len(positives):
        raise RuntimeError("Insufficient positives or negative candidates for the pilot.")
    transactions, partners, tx_bin, partner_bin = activity_bins(data)
    stride = int(partner_bin.max()) + 1
    code = tx_bin.astype(np.int32) * stride + partner_bin
    order = np.argsort(code[candidates], kind="stable")
    sorted_ids = candidates[order]
    group_codes, starts, counts = np.unique(code[sorted_ids], return_index=True, return_counts=True)
    rng = np.random.RandomState(seed)
    groups = {}
    used = {}
    for value, start, count in zip(group_codes, starts, counts):
        group = sorted_ids[int(start):int(start + count)]
        rng.shuffle(group)
        groups[int(value)] = group
        used[int(value)] = 0
    positive_order = positives.copy()
    rng.shuffle(positive_order)
    rows = []
    negatives = []
    distances = []
    for positive in positive_order:
        a, b = int(tx_bin[positive]), int(partner_bin[positive])
        ranked = sorted(groups, key=lambda key: (abs(key // stride - a) + abs(key % stride - b), key))
        needed = ratio
        for key in ranked:
            available = len(groups[key]) - used[key]
            take = min(available, needed)
            if not take:
                continue
            distance = abs(key // stride - a) + abs(key % stride - b)
            for negative in groups[key][used[key]:used[key] + take]:
                negative = int(negative)
                negatives.append(negative)
                distances.append(distance)
                rows.append((int(positive), negative, a, b, key // stride, key % stride, distance,
                             int(transactions[positive]), int(transactions[negative]),
                             int(partners[positive]), int(partners[negative])))
            used[key] += take
            needed -= take
            if needed == 0:
                break
        if needed:
            raise RuntimeError("Matching exhausted negative candidates.")
    negatives = np.asarray(negatives, dtype=np.int64)
    if len(np.unique(negatives)) != len(negatives) or np.any(labels[negatives] != 0):
        raise RuntimeError("Negative-sampling uniqueness/label check failed.")
    metrics = {}
    for name, values in (("log1p_transaction_count", np.log1p(transactions)),
                         ("log1p_directional_unique_partner_sum", np.log1p(partners))):
        p, q = values[positives], values[negatives]
        pooled = np.sqrt((p.var(ddof=0) + q.var(ddof=0)) / 2)
        metrics[name] = {"positive_mean": float(p.mean()), "negative_mean": float(q.mean()),
                         "standardized_mean_difference": float((p.mean() - q.mean()) / pooled) if pooled > 0 else None}
    summary = {"positive_count": len(positives), "sampled_negative_count": len(negatives),
               "negative_to_positive_ratio": ratio, "seed": seed,
               "exact_bin_match_fraction": float(np.mean(np.asarray(distances) == 0)),
               "maximum_bin_distance": int(max(distances)),
               "distance_counts": {str(k): v for k, v in sorted(collections.Counter(distances).items())},
               "activity_balance": metrics,
               "method": "Without replacement; floor(log2(1+in/out transaction sum)) and floor(log2(1+directional unique-partner sum)); nearest nonempty bin by Manhattan distance when exact bin exhausted; ties by bin code; shuffled positive order and candidate pools."}
    return positives, negatives, rows, summary


def simple_undirected_adjacency(data):
    n = len(data["labels"])
    src, dst = data["source"], data["target"]
    if src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n:
        raise RuntimeError("Transaction endpoint out of bounds.")
    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    keys = low.astype(np.int64)
    keys *= n
    keys += high
    del low, high
    keys = np.unique(keys)
    left, right = keys // n, keys % n
    keep = left != right
    left = left[keep].astype(np.int32)
    right = right[keep].astype(np.int32)
    del keys, keep
    pairs = len(left)
    row = np.concatenate((left, right))
    col = np.concatenate((right, left))
    adjacency = csr_matrix((np.ones(len(row), dtype=np.uint8), (row, col)), shape=(n, n))
    adjacency.sort_indices()
    if adjacency.nnz != 2 * pairs or not adjacency.has_canonical_format:
        raise RuntimeError("Undirected projection invariant failed.")
    return adjacency, pairs


def bounded_neighbourhood(adjacency, roots, fanout, hops, max_nodes, seed):
    n = adjacency.shape[0]
    roots = np.unique(roots)
    if len(roots) > max_nodes:
        raise RuntimeError("Node cap is smaller than the supervised root set.")
    seen = np.zeros(n, dtype=bool)
    seen[roots] = True
    frontier = roots
    rng = np.random.RandomState(seed)
    history = []
    for hop in range(1, hops + 1):
        lists = []
        for node in frontier:
            neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
            if len(neighbors) > fanout:
                neighbors = rng.choice(neighbors, size=fanout, replace=False)
            if len(neighbors):
                lists.append(neighbors)
        candidates = np.unique(np.concatenate(lists)) if lists else np.empty(0, dtype=np.int32)
        candidates = candidates[~seen[candidates]]
        remaining = max_nodes - int(seen.sum())
        capped = len(candidates) > remaining
        if capped:
            candidates = np.sort(rng.choice(candidates, size=remaining, replace=False))
        seen[candidates] = True
        history.append({"hop": hop, "frontier_count": len(frontier), "new_nodes": len(candidates),
                        "node_cap_applied": bool(capped), "selected_total": int(seen.sum())})
        frontier = candidates
        if not len(frontier) or seen.sum() == max_nodes:
            break
    return np.flatnonzero(seen).astype(np.int32), history


def raw_features(data, node_ids, chunk_size=1000000):
    n = len(data["labels"])
    src, dst, amount, timestamp = (data[k] for k in ("source", "target", "amount", "timestamp"))
    in_sum = np.zeros(n); out_sum = np.zeros(n)
    in_max = np.zeros(n); out_max = np.zeros(n)
    in_zero = np.zeros(n); out_zero = np.zeros(n)
    first = np.full(n, np.inf); last = np.full(n, -np.inf)
    for start in range(0, len(src), chunk_size):
        stop = min(start + chunk_size, len(src))
        u, v, a, t = src[start:stop], dst[start:stop], amount[start:stop], timestamp[start:stop]
        if not np.isfinite(a).all() or not np.isfinite(t).all() or np.any(a < 0):
            raise RuntimeError("Invalid transaction attributes.")
        in_sum += np.bincount(v, weights=a, minlength=n)
        out_sum += np.bincount(u, weights=a, minlength=n)
        in_zero += np.bincount(v[a == 0], minlength=n)
        out_zero += np.bincount(u[a == 0], minlength=n)
        np.maximum.at(in_max, v, a); np.maximum.at(out_max, u, a)
        np.minimum.at(first, u, t); np.minimum.at(first, v, t)
        np.maximum.at(last, u, t); np.maximum.at(last, v, t)
        if len(src) > chunk_size:
            print("FEATURE TRANSACTIONS: {:,}/{:,}".format(stop, len(src)), flush=True)
    idx = node_ids
    inc = data["in_transaction_count"][idx].astype(np.float64)
    outc = data["out_transaction_count"][idx].astype(np.float64)
    inp = data["in_unique_partners"][idx].astype(np.float64)
    outp = data["out_unique_partners"][idx].astype(np.float64)
    iv, ov = in_sum[idx], out_sum[idx]
    span = np.zeros(len(idx))
    active = np.isfinite(first[idx]) & np.isfinite(last[idx])
    span[active] = last[idx][active] - first[idx][active]
    if np.any(span < 0):
        raise RuntimeError("Negative activity duration.")
    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)
    features = np.column_stack([
        np.log1p(inc), np.log1p(outc), np.log1p(inp), np.log1p(outp), np.log1p(iv), np.log1p(ov),
        np.log1p(ratio(iv, inc)), np.log1p(ratio(ov, outc)),
        np.log1p(in_max[idx]), np.log1p(out_max[idx]),
        ratio(in_zero[idx], inc), ratio(out_zero[idx], outc), np.log1p(span),
        ratio(outc - inc, outc + inc), ratio(ov - iv, ov + iv),
        ratio(inp, inc), ratio(outp, outc), np.log1p((inc + outc) / (span + 1.0)),
    ])
    if features.shape != (len(idx), len(FEATURES)) or not np.isfinite(features).all():
        raise RuntimeError("Invalid feature matrix.")
    return features


def make_dataset(data, ratio, seed, fanout, hops, max_nodes, max_edges):
    print("Matching potentially noisy negatives by activity...", flush=True)
    positives, negatives, rows, matching = match_negatives(data, ratio, seed)
    roots = np.sort(np.concatenate((positives, negatives)))
    root_y = data["labels"][roots].astype(np.int64)
    print("SUPERVISED ROOTS: positives={} negatives={}".format(len(positives), len(negatives)), flush=True)
    # Fixed stratified node split, matching the original 40%/remaining 33:67 protocol.
    train, remainder = train_test_split(roots, train_size=0.4, stratify=root_y, random_state=2)
    val, test = train_test_split(remainder, test_size=0.67, stratify=data["labels"][remainder], random_state=2)
    print("Building an unweighted undirected simple projection for BWGNN's symmetric Laplacian...", flush=True)
    adjacency, full_pairs = simple_undirected_adjacency(data)
    print("Sampling equal-rule neighbourhoods around all positive and negative roots...", flush=True)
    selected, history = bounded_neighbourhood(adjacency, roots, fanout, hops, max_nodes, seed)
    subgraph = adjacency[selected][:, selected].tocsr()
    subgraph.sort_indices()
    del adjacency
    if subgraph.nnz > max_edges:
        raise RuntimeError("Sample has {:,} bidirected edges, above the {:,} pilot limit. Send the output to adjust the sampling budget.".format(subgraph.nnz, max_edges))
    if subgraph.diagonal().any() or (subgraph != subgraph.T).nnz or not subgraph.has_canonical_format:
        raise RuntimeError("Sampled graph is not simple, loop-free, and symmetric.")
    print("SAMPLED GRAPH: nodes={} bidirected_edges={}".format(len(selected), subgraph.nnz), flush=True)
    raw = raw_features(data, selected)
    targets = np.full(len(selected), -1, dtype=np.int64)
    root_idx = np.searchsorted(selected, roots)
    if not np.array_equal(selected[root_idx], roots):
        raise RuntimeError("Supervised roots were lost during sampling.")
    targets[root_idx] = root_y
    masks = {}
    for name, global_ids in (("train", train), ("val", val), ("test", test)):
        mask = np.zeros(len(selected), dtype=bool)
        mask[np.searchsorted(selected, global_ids)] = True
        if set(np.unique(targets[mask])) != {0, 1}:
            raise RuntimeError("Both classes are required in every split.")
        masks[name] = mask
    mask_sum = sum(mask.astype(np.int8) for mask in masks.values())
    if np.any(mask_sum > 1) or not np.array_equal(mask_sum == 1, targets >= 0):
        raise RuntimeError("Split overlap/coverage invariant failed.")
    scaler = StandardScaler().fit(raw[masks["train"]])
    scaled = scaler.transform(raw).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise RuntimeError("Feature standardization produced nonfinite values.")
    coo = subgraph.tocoo()
    payload = {"source": coo.row.astype(np.int32), "target": coo.col.astype(np.int32),
               "global_node_id": selected, "labels": targets, "features": scaled, "raw_features": raw,
               "standardizer_mean": scaler.mean_, "standardizer_scale": scaler.scale_,
               "train_mask": masks["train"], "val_mask": masks["val"], "test_mask": masks["test"]}
    split_summary = {name: {"total": int(mask.sum()), "positive": int((targets[mask] == 1).sum()),
                            "noisy_negative": int((targets[mask] == 0).sum())} for name, mask in masks.items()}
    summary = {"matching": matching, "sampling": {"seed": seed, "fanout": fanout, "requested_hops": hops,
               "max_nodes": max_nodes, "max_bidirected_edges": max_edges, "hop_history": history},
               "full_projection_undirected_pairs": full_pairs, "sampled_node_count": len(selected),
               "sampled_bidirected_edge_count": int(subgraph.nnz), "sampled_undirected_pair_count": int(subgraph.nnz // 2),
               "sampled_isolated_nodes": int((np.diff(subgraph.indptr) == 0).sum()),
               "supervised_node_count": len(roots), "context_node_count": int((targets < 0).sum()),
               "sampled_supervised_positive_fraction": float(root_y.mean()), "split": split_summary,
               "split_seed": 2, "feature_count": len(FEATURES), "feature_names": FEATURES,
               "training_constant_features": [FEATURES[i] for i in np.flatnonzero(scaler.var_ == 0)],
               "invariants": {"root_retention": True, "split_disjointness_and_coverage": True,
                              "two_classes_per_split": True, "simple_symmetric_loop_free_graph": True,
                              "finite_features": True, "train_only_standardizer": True}}
    return payload, rows, summary


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, text):
        self.terminal.write(text); self.log.write(text); self.log.flush()
        return len(text)
    def flush(self):
        self.terminal.flush(); self.log.flush()


def run(args):
    report_path = Path(args.audit_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output).resolve() / ("ethereum_pilot_" + stamp)
    output.mkdir(parents=True)
    started = time.monotonic()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    success = False
    print("OUTPUT DIRECTORY: " + str(output), flush=True)
    with (output / "console.log").open("w", encoding="utf-8") as log:
        sys.stdout = Tee(original_stdout, log); sys.stderr = Tee(original_stderr, log)
        try:
            source_hashes = verify_original_sources(Path.cwd())
            print("Verifying every compact-file fingerprint...", flush=True)
            data = load_compact(report)
            payload, rows, summary = make_dataset(data, args.negative_ratio, args.seed, args.fanout,
                                                  args.hops, args.max_nodes, args.max_edges)
            dataset_path = output / "dataset.npz"
            np.savez_compressed(str(dataset_path), **payload)
            with (output / "root_matches.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["positive_global_id", "negative_global_id", "positive_tx_bin", "positive_partner_bin",
                                 "negative_tx_bin", "negative_partner_bin", "bin_distance", "positive_transactions",
                                 "negative_transactions", "positive_directional_partner_sum", "negative_directional_partner_sum"])
                writer.writerows(rows)
            summary.update({
                "status": "preprocessing_complete", "environment": {"python": platform.python_version(),
                    "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                "source_audit_report": str(report_path), "source_audit_sha256": sha256(report_path),
                "raw_graph_sha256": report["raw_sha256"], "original_source_hashes": source_hashes,
                "runner_sha256": sha256(Path(__file__)), "dataset_sha256": sha256(dataset_path),
                "graph_semantics": "Unweighted undirected simple projection: direction and parallel multiplicity are collapsed only for GNN adjacency, and self-loops removed. Both directions are stored. All released directed transactions, including zero amounts, parallel events and self-loops, contribute to node features.",
                "feature_protocol": "Transaction-derived only; aggregates from the complete released static snapshot, with no label-derived features. Log1p on counts, amounts, duration and rate; train-root-only StandardScaler. Amount and timestamp units are retained as native numerical units; no currency-unit assumption.",
                "label_protocol": "All released isp=1 nodes are positive roots; sampled isp=0 roots are potentially noisy negatives. Context nodes have label=-1 and no supervised mask.",
                "evaluation_scope": "Transductive static-snapshot pilot, with label-informed balanced root selection and neighbourhood sampling around both classes. Not a temporal forecast, not deployment-population evaluation, and not a leakage-free estimate of generalisation to an independently selected graph.",
                "metric_interpretation": "The pilot supervised positive prevalence is 1/(1+negative_ratio), unlike the native 0.039% released-positive prevalence. PR metrics and thresholds apply to the sampled noisy-negative protocol; they do not estimate native-population precision.",
                "elapsed_seconds": time.monotonic() - started,
            })
            save_json(output / "pipeline_manifest.json", summary)
            save_json(output / "feature_descriptions.json", {"ordered_names": FEATURES,
                "zero_denominator_rule": "Ratio features are zero when their denominator is zero.",
                "active_duration": "Maximum minus minimum timestamp across all incident transactions; inactive duration zero.",
                "activity_rate": "Total in/out transactions divided by (native timestamp span + 1), followed by log1p.",
                "unique_partner_ratio": "Unique directed partners divided by the corresponding transaction count.",
                "self_loop_rule": "A self transaction contributes once to each in/out aggregate.",
                "context_rule": "Context node attributes are visible transductively; context targets are excluded from supervision.",
                "standardization_fit_global_ids_sha256": hashlib.sha256(payload["global_node_id"][payload["train_mask"]].tobytes()).hexdigest()})
            print("PREPROCESSING COMPLETE", flush=True)
            print("Matching exact-bin fraction: {:.2%}".format(summary["matching"]["exact_bin_match_fraction"]), flush=True)
            print("Split counts: " + json.dumps(summary["split"]), flush=True)
            print("Feature count: " + str(len(FEATURES)), flush=True)
            print("Dataset: " + str(dataset_path), flush=True)
            success = True
        except BaseException as error:
            save_json(output / "pipeline_error.json", {"type": type(error).__name__, "message": str(error),
                                                        "traceback": traceback.format_exc()})
            traceback.print_exc()
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
    evidence = output / "ethereum_preprocessing_evidence.zip"
    with zipfile.ZipFile(evidence, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != evidence:
                z.write(path, path.name)
        z.write(report_path, "source_graph_audit.json")
        z.write(Path(__file__).resolve(), "prepare_ethereum.py")
    print("EVIDENCE ZIP: " + str(evidence), flush=True)
    if not success:
        raise RuntimeError("Preprocessing did not complete. Send the evidence ZIP and output.")


def self_check():
    n = 30
    source = list(range(n)); target = [(i + 1) % n for i in range(n)]
    amount = [float(i % 4) for i in range(n)]; timestamp = [float(i + 100) for i in range(n)]
    for i in range(0, n, 5):
        source.append(i); target.append((i + 1) % n); amount.append(2.0); timestamp.append(float(i + 200))
    source.append(0); target.append(0); amount.append(0.0); timestamp.append(99.0)
    source, target = np.asarray(source, dtype=np.int32), np.asarray(target, dtype=np.int32)
    amount, timestamp = np.asarray(amount), np.asarray(timestamp)
    labels = np.zeros(n, dtype=np.int8); labels[:6] = 1
    data = {"labels": labels, "source": source, "target": target, "amount": amount, "timestamp": timestamp,
            "in_transaction_count": np.bincount(target, minlength=n), "out_transaction_count": np.bincount(source, minlength=n),
            "in_unique_partners": np.asarray([len(set(source[target == i])) for i in range(n)]),
            "out_unique_partners": np.asarray([len(set(target[source == i])) for i in range(n)])}
    measured = raw_features(data, np.arange(n), chunk_size=7)
    # Independent per-node feature calculation without bincount/at aggregation.
    expected = []
    for i in range(n):
        incoming, outgoing = np.flatnonzero(target == i), np.flatnonzero(source == i)
        inc, outc = len(incoming), len(outgoing)
        inp, outp = len(set(source[incoming])), len(set(target[outgoing]))
        iv, ov = float(sum(amount[incoming])), float(sum(amount[outgoing]))
        incident = sorted(set(incoming.tolist() + outgoing.tolist()))
        span = float(max(timestamp[incident]) - min(timestamp[incident])) if incident else 0.0
        divide = lambda a, b: a / b if b else 0.0
        expected.append([np.log1p(inc), np.log1p(outc), np.log1p(inp), np.log1p(outp),
            np.log1p(iv), np.log1p(ov), np.log1p(divide(iv, inc)), np.log1p(divide(ov, outc)),
            np.log1p(max(amount[incoming], default=0)), np.log1p(max(amount[outgoing], default=0)),
            divide(sum(amount[incoming] == 0), inc), divide(sum(amount[outgoing] == 0), outc), np.log1p(span),
            divide(outc - inc, outc + inc), divide(ov - iv, ov + iv), divide(inp, inc), divide(outp, outc),
            np.log1p((inc + outc) / (span + 1))])
    assert np.allclose(measured, expected, rtol=1e-12, atol=1e-12)
    altered = dict(data); altered["labels"] = 1 - labels
    assert np.array_equal(raw_features(altered, np.arange(n)), measured)
    one, rows, summary = make_dataset(data, 2, 0, 2, 2, 24, 1000)
    two, rows2, summary2 = make_dataset(data, 2, 0, 2, 2, 24, 1000)
    for name in one:
        assert np.array_equal(one[name], two[name]), name
    assert rows == rows2 and summary == summary2
    assert len(rows) == 12 and len(set(row[1] for row in rows)) == 12
    assert one["train_mask"].sum() + one["val_mask"].sum() + one["test_mask"].sum() == 18
    train_raw = one["raw_features"][one["train_mask"]]
    assert np.allclose(one["standardizer_mean"], train_raw.mean(axis=0))
    assert np.array_equal(one["labels"] >= 0, one["train_mask"] | one["val_mask"] | one["test_mask"])
    edges = set(zip(one["source"].tolist(), one["target"].tolist()))
    assert all(u != v and (v, u) in edges for u, v in edges)
    print("SELF-CHECK PASS: independently calculated features, label independence, deterministic matching/sampling, split coverage, symmetric topology, train-only standardizer.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report")
    parser.add_argument("--output", default="processed_results")
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fanout", type=int, default=8)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=50000)
    parser.add_argument("--max-edges", type=int, default=2000000)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    if not args.audit_report:
        parser.error("--audit-report is required")
    if min(args.negative_ratio, args.fanout, args.hops, args.max_nodes, args.max_edges) <= 0:
        parser.error("Sampling parameters must be positive")
    run(args)


if __name__ == "__main__":
    main()
