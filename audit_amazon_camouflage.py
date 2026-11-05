"""Audit the unchanged released Amazon multigraph before camouflage rewiring.

No training or rewiring occurs here. Compact data preserve all reciprocal
parallel entries and node self-loop multiplicities. IDs below 3305 remain
context-only, as in the released supervised evaluation.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split

SOURCES = {
    "main.py": "1642e10e5c799ebd303bb20d66a991a55363276b35855996f5615eba0fd1ea13",
    "dataset.py": "229b5e5101c63879264802f909e64396a49be694a8bb4815dfea201f9b78d1fb",
    "BWGNN.py": "0ac45b1104f80fa17c50cd5e3bf9aa6241e008e6b57fdab9366ec69f80cb2f84",
}
EXPECTED = {
    "source_ids": "b55b78751d5bf930933f19253e792c473660579bb81a989715664caaa41397b3",
    "target_ids": "3c0131d26255ff474c0133dce28160ca4024a786433aacb2305e455312c071c9",
    "features": "5b3330d3ea888dfc07cb9b60881dbd902e77629dd270e12bb414eedf43112f29",
    "labels": "9d951f4326d38e72e33191aa7017df8b336bc40b1547a8ec919d4c0c454482ad",
    "train_ids": "08469961fb7d2cb8277e1f6d7e3cbb7d344a89036e5774c227eb5fcbbe2d008a",
    "val_ids": "2b1d8b3ef7712e21424e0fe6ad8784e3edc486f4a00f5a232324a7e1ef1eacaf",
    "test_ids": "ca5b60051bc3e0785e71c352b966a9903083fa7a6d44ae8243a0453eb49e78c6",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(str(value.shape).encode())
    h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def compact_graph(source, target, n):
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if source.shape != target.shape or source.ndim != 1 or not len(source):
        raise RuntimeError("Invalid edge arrays.")
    if min(source.min(), target.min()) < 0 or max(source.max(), target.max()) >= n:
        raise RuntimeError("Endpoint outside node range.")
    keys, counts = np.unique(source * n + target, return_counts=True)
    u, v = keys // n, keys % n
    upper, lower, diagonal = u < v, u > v, u == v
    lower_reverse = v[lower] * n + u[lower]
    permutation = np.argsort(lower_reverse)
    if not np.array_equal(keys[upper], lower_reverse[permutation]):
        raise RuntimeError("Graph has asymmetric non-loop adjacency; intervention needs a separate directed protocol.")
    if not np.array_equal(counts[upper], counts[lower][permutation]):
        raise RuntimeError("Reciprocal multiplicities differ; intervention needs a separate directed protocol.")
    pu, pv = u[upper].astype(np.int32), v[upper].astype(np.int32)
    multiplicity = counts[upper].astype(np.int32)
    loops = np.zeros(n, dtype=np.int32)
    loops[u[diagonal]] = counts[diagonal]
    degree = (np.bincount(pu, weights=multiplicity, minlength=n)
              + np.bincount(pv, weights=multiplicity, minlength=n)).astype(np.int64) + loops
    unique_degree = np.bincount(pu, minlength=n) + np.bincount(pv, minlength=n)
    if not np.array_equal(degree, np.bincount(source, minlength=n)):
        raise RuntimeError("Compact out-degree verification failed.")
    if not np.array_equal(degree, np.bincount(target, minlength=n)):
        raise RuntimeError("Compact in-degree verification failed.")
    if 2 * int(multiplicity.sum()) + int(loops.sum()) != len(source):
        raise RuntimeError("Compact edge-entry verification failed.")
    return {"pair_source": pu, "pair_target": pv, "multiplicity": multiplicity,
            "loop_multiplicity": loops, "weighted_degree": degree,
            "unique_nonloop_degree": unique_degree}, {
        "edge_entries": len(source), "unique_directed_endpoint_pairs_including_loops": len(keys),
        "undirected_nonloop_pairs": len(pu), "nonloop_entries_per_direction": int(multiplicity.sum()),
        "self_loop_entries": int(loops.sum()), "nodes_with_self_loops": int((loops > 0).sum()),
        "additional_parallel_directed_entries": int(len(source) - len(keys)),
        "nonloop_isolated_nodes": int((unique_degree == 0).sum()),
        "symmetric_adjacency": True, "equal_reciprocal_multiplicity": True,
        "pair_multiplicity_histogram": {str(int(w)): int((multiplicity == w).sum()) for w in np.unique(multiplicity)},
        "loop_multiplicity_histogram": {str(int(w)): int((loops == w).sum()) for w in np.unique(loops)},
        "degree_verified_against_both_original_endpoint_arrays": True,
    }


def self_check():
    # Two reciprocal pairs of different multiplicity, repeated loops and an isolate.
    u = np.asarray([0, 0, 1, 1, 1, 2, 0, 0, 2])
    v = np.asarray([1, 1, 0, 0, 2, 1, 0, 0, 2])
    data, report = compact_graph(u, v, 4)
    assert np.array_equal(data["multiplicity"], [2, 1])
    assert np.array_equal(data["loop_multiplicity"], [2, 0, 1, 0])
    assert np.array_equal(data["weighted_degree"], [4, 3, 2, 0])
    assert np.array_equal(data["unique_nonloop_degree"], [1, 2, 1, 0])
    assert report["additional_parallel_directed_entries"] == 3
    for a, b in [(u[:-1], np.r_[v[:-2], 3]), (u[1:], v[1:])]:
        try:
            compact_graph(a, b, 4)
        except RuntimeError:
            continue
        raise AssertionError("Asymmetric graph was accepted.")
    print("SELF-CHECK PASS: parallel edges, reciprocal multiplicities, repeated loops, isolates and original endpoint degrees.")


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.stream.flush(); self.log.flush()


def audit(output):
    import torch
    import dgl
    import sklearn
    started = time.monotonic()
    hashes = {}
    for name, expected in SOURCES.items():
        hashes[name] = hashlib.sha256(Path(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if hashes[name] != expected:
            raise RuntimeError("Original source changed: " + name)
    from dataset import Dataset
    g = Dataset("amazon", True).graph
    n = g.num_nodes()
    source, target = g.edges(order="eid")
    source, target = source.numpy(), target.numpy()
    features = g.ndata["feature"].numpy()
    labels = g.ndata["label"].numpy()
    if n != 11944 or len(source) != 9569592:
        raise RuntimeError("Amazon dimensions differ from the verified benchmark.")
    fingerprints = {"source_ids": array_hash(source), "target_ids": array_hash(target),
                    "features": array_hash(features), "labels": array_hash(labels)}
    eligible = np.arange(3305, n, dtype=np.int64)
    train, rest = train_test_split(eligible, train_size=0.4, stratify=labels[eligible], random_state=2)
    val, test = train_test_split(rest, test_size=0.67, stratify=labels[rest], random_state=2)
    masks = {}
    for name, ids in (("train", train), ("val", val), ("test", test)):
        mask = np.zeros(n, dtype=bool); mask[ids] = True
        masks[name + "_mask"] = mask
        fingerprints[name + "_ids"] = array_hash(np.flatnonzero(mask))
    if fingerprints != EXPECTED:
        raise RuntimeError("Graph, features, labels or splits differ from the previously verified Amazon benchmark.")
    print("BENCHMARK FINGERPRINTS: PASS", flush=True)
    compact, report = compact_graph(source, target, n)
    u, v, w = compact["pair_source"], compact["pair_target"], compact["multiplicity"]
    known = (u >= 3305) & (v >= 3305)
    ff = known & (labels[u] == 1) & (labels[v] == 1)
    bb = known & (labels[u] == 0) & (labels[v] == 0)
    fb = known & (labels[u] != labels[v])
    buckets = {}
    for weight in np.unique(w):
        nf, nb = int((ff & (w == weight)).sum()), int((bb & (w == weight)).sum())
        buckets[str(int(weight))] = {"fraud_fraud_pairs": nf, "benign_benign_pairs": nb,
                                     "same_multiplicity_swap_upper_bound": min(nf, nb)}
    m0 = min(int(ff.sum()), int(bb.sum()))
    report.update({"status": "amazon_structure_audit_complete", "nodes": n, "feature_count": features.shape[1],
        "original_source_hashes": hashes, "benchmark_array_fingerprints": fingerprints,
        "eligible_supervised_nodes": len(eligible), "context_only_nodes": 3305,
        "eligible_class_counts": {"positive": int(labels[eligible].sum()), "negative": int((labels[eligible] == 0).sum())},
        "supervised_splits": {name: {"total": int(mask.sum()), "positive": int(labels[mask].sum())}
                              for name, mask in masks.items()},
        "eligible_pair_counts": {"FF": int(ff.sum()), "BB": int(bb.sum()), "FB": int(fb.sum()),
                                  "at_least_one_context_endpoint": int((~known).sum())},
        "eligible_weighted_pair_counts": {"FF": int(w[ff].sum()), "BB": int(w[bb].sum()), "FB": int(w[fb].sum())},
        "M0_unique_eligible_pairs": m0, "eligible_multiplicity_buckets": buckets,
        "same_multiplicity_total_swap_upper_bound": sum(x["same_multiplicity_swap_upper_bound"] for x in buckets.values()),
        "candidate_targets": {str(c): int(np.floor(c * m0)) for c in (0.0, 0.2, 0.4, 0.6, 0.8)},
        "eligible_cross_class_pair_fraction": float(fb.sum() / known.sum()),
        "known_FB_pairs_over_all_nonloop_pairs": float(fb.sum() / len(u)),
        "fraud_local_unique_mixing": float(fb.sum() / (2 * ff.sum() + fb.sum())),
        "fraud_local_weighted_mixing": float(w[fb].sum() / (2 * w[ff].sum() + w[fb].sum())),
        "compact_semantics": "Each u<v pair stores original equal reciprocal multiplicity. Node loop multiplicities remain explicit. No simplification or rewiring has occurred; pair order is canonical, not original DGL edge-ID order.",
        "intervention_constraints": "Swap only eligible FF/BB pairs with equal original multiplicity, four distinct endpoints and absent new pairs. This preserves both unique non-loop degree and the original multigraph in/out degree. Keep all original loops, context incident pairs, features, labels and splits fixed. The upper bound ignores duplicate-pair rejection and is not an achieved swap count.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "dgl": dgl.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__},
        "runner_sha256": digest(Path(__file__)),
    })
    compact.update({"features": features, "labels": labels, "global_node_id": np.arange(n, dtype=np.int32), **masks})
    payload = output / "amazon_compact.npz"
    np.savez_compressed(str(payload), **compact)
    report["compact_sha256"] = digest(payload)
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "amazon_graph_audit.json", report)
    print("AUDIT COMPLETE: nodes={} pairs={} loop_entries={}".format(n, len(u), report["self_loop_entries"]), flush=True)
    print("Eligible FF={} BB={} FB={} M0={}".format(ff.sum(), bb.sum(), fb.sum(), m0), flush=True)
    print("Multiplicity buckets: " + json.dumps(buckets), flush=True)
    print("COMPACT DATA: " + str(payload), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="audit_results")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output).resolve() / ("amazon_structure_" + stamp)
    output.mkdir(parents=True)
    print("OUTPUT DIRECTORY: " + str(output), flush=True)
    success = False
    with (output / "console.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                audit(output); success = True
            except Exception as error:
                write_json(output / "audit_error.json", {"type": type(error).__name__, "message": str(error),
                                                         "traceback": traceback.format_exc()})
                traceback.print_exc()
    evidence = output / "amazon_structure_evidence.zip"
    with zipfile.ZipFile(evidence, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(output.iterdir()):
            if file.is_file() and file != evidence:
                z.write(file, file.name)
        z.write(Path(__file__).resolve(), "audit_amazon_camouflage.py")
    print("EVIDENCE ZIP: " + str(evidence), flush=True)
    if not success:
        raise RuntimeError("Amazon structure audit failed. Send the evidence ZIP and output.")
    print("AMAZON STRUCTURE STAGE: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
