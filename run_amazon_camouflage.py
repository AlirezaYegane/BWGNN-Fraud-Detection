"""One-seed Amazon camouflage pilot with a matched random-rewiring control.

Keeps released features, supervised splits, loops, reciprocal multiplicities,
both multigraph degrees and unique non-loop degrees. Uses the separately
verified train_ethereum.py trainer for fresh-logit validation selection.
"""
import argparse
import contextlib
import csv
import datetime
import gc
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
from scipy import sparse

TRAINER_HASH = "3283fd3c7351f7c2dc788711333895f2c98eeaa14b6cca809bb2f07365145330"
COMPACT_HASH = "31be189110b98a8d39f46b99e261b2ed4ac454aba15af1d3ed7b6bd9d13c4334"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
    h.update(memoryview(a).cast("B")); return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def canonical(a, b, n):
    return min(a, b) * n + max(a, b)


def eligible_masks(data):
    u, v, y = data["pair_source"], data["pair_target"], data["labels"]
    known = (u >= data.get("eligible_start", 3305)) & (v >= data.get("eligible_start", 3305))
    return known, known & (y[u] == 1) & (y[v] == 1), known & (y[u] == 0) & (y[v] == 0), known & (y[u] != y[v])


def generate_swaps(data, target, seed, control_weights=None):
    """Each original pair is retired at most once; new pairs cannot be reused."""
    u, v, w = data["pair_source"], data["pair_target"], data["multiplicity"]
    n = len(data["labels"]); rng = np.random.RandomState(seed)
    keys = u.astype(np.int64) * n + v
    if not np.all(keys[1:] > keys[:-1]):
        raise RuntimeError("Input pairs must be canonical and sorted.")
    known, ff, bb, _ = eligible_masks(data)
    removed, added, patches = set(), set(), []
    rejected = {"shared_endpoint": 0, "existing_new_pair": 0, "insufficient_pool": 0, "exhausted_candidate_attempt_limit": 0}
    attempts = 0
    def present(key):
        if key in added: return True
        if key in removed: return False
        j = np.searchsorted(keys, key)
        return j < len(keys) and keys[j] == key
    pools = {}
    for weight in np.unique(w):
        ids = np.flatnonzero((known if control_weights is not None else bb) & (w == weight))
        rng.shuffle(ids); pools[int(weight)] = [ids, len(ids)]
    if control_weights is not None:
        steps = list(map(int, control_weights))
    else:
        fraud = np.flatnonzero(ff); rng.shuffle(fraud)
        steps = fraud.tolist()
    for step in steps:
        if len(patches) >= target: break
        weight = int(step if control_weights is not None else w[step])
        pool, size = pools[weight]
        need = 2 if control_weights is not None else 1
        if size < need:
            rejected["insufficient_pool"] += 1
            if control_weights is not None: break
            continue
        for retry in range(500):
            attempts += 1
            if control_weights is None:
                first = int(step); pos_second = int(rng.randint(size)); second = int(pool[pos_second])
                positions = [pos_second]
            else:
                pos_first = int(rng.randint(size)); pos_second = int(rng.randint(size - 1))
                if pos_second >= pos_first: pos_second += 1
                first, second = int(pool[pos_first]), int(pool[pos_second])
                positions = [pos_first, pos_second]
            a, b, c, d = int(u[first]), int(v[first]), int(u[second]), int(v[second])
            if rng.randint(2): a, b = b, a
            if rng.randint(2): c, d = d, c
            if len({a, b, c, d}) != 4:
                rejected["shared_endpoint"] += 1; continue
            new1, new2 = canonical(a, c, n), canonical(b, d, n)
            if present(new1) or present(new2):
                rejected["existing_new_pair"] += 1; continue
            removed.update([int(keys[first]), int(keys[second])]); added.update([new1, new2])
            patches.append([a, b, c, d, weight, first, second])
            # Remove sampled originals from the active pool without replacement.
            for position in sorted(positions, reverse=True):
                size -= 1; pool[position] = pool[size]
            pools[weight][1] = size
            break
        else:
            rejected["exhausted_candidate_attempt_limit"] += 1
            if control_weights is not None:
                # Stop rather than silently lose the matched multiplicity sequence.
                break
        if patches and len(patches) % 1000 == 0:
            print("GENERATED {} valid swaps ({} attempts)".format(len(patches), attempts), flush=True)
    return np.asarray(patches, dtype=np.int64).reshape(-1, 7), {
        "requested_swaps": int(target), "achieved_swaps": len(patches), "candidate_attempts": attempts,
        "rejected_attempts": rejected, "per_candidate_attempt_limit": 500,
        "protocol": "Matched label-agnostic random swap" if control_weights is not None else "Eligible FF+BB -> FB+FB",
        "same_multiplicity_constraint": True, "each_original_pair_used_at_most_once": True,
    }


def apply_compact(data, patches):
    u, v, w = data["pair_source"].copy(), data["pair_target"].copy(), data["multiplicity"]
    start = data.get("eligible_start", 3305)
    touched = set()
    for a, b, c, d, weight, first, second in patches:
        if len({int(a), int(b), int(c), int(d)}) != 4 or min(a, b, c, d) < start:
            raise RuntimeError("Invalid swap endpoints.")
        if first in touched or second in touched:
            raise RuntimeError("Original pair was reused.")
        touched.update([int(first), int(second)])
        if w[first] != weight or w[second] != weight:
            raise RuntimeError("Swap multiplicities differ.")
        if {int(u[first]), int(v[first])} != {int(a), int(b)} or {int(u[second]), int(v[second])} != {int(c), int(d)}:
            raise RuntimeError("Swap does not match its original pair.")
        u[first], v[first] = min(a, c), max(a, c)
        u[second], v[second] = min(b, d), max(b, d)
    n = len(data["labels"]); keys = u.astype(np.int64) * n + v
    if np.any(u >= v) or len(np.unique(keys)) != len(keys):
        raise RuntimeError("Variant has a loop or duplicate pair.")
    unique = np.bincount(u, minlength=n) + np.bincount(v, minlength=n)
    weighted = (np.bincount(u, weights=w, minlength=n) + np.bincount(v, weights=w, minlength=n)).astype(np.int64) + data["loop_multiplicity"]
    if not np.array_equal(unique, data["unique_nonloop_degree"]) or not np.array_equal(weighted, data["weighted_degree"]):
        raise RuntimeError("Per-node degrees changed.")
    variant = dict(data); variant["pair_source"], variant["pair_target"] = u, v
    return variant


def mixing(data):
    known, ff, bb, fb = eligible_masks(data); w = data["multiplicity"]
    nf, nb, nc = int(ff.sum()), int(bb.sum()), int(fb.sum())
    return {"FF_unique_pairs": nf, "BB_unique_pairs": nb, "FB_unique_pairs": nc,
            "eligible_cross_class_pair_fraction": float(nc / known.sum()),
            "known_FB_pairs_over_all_nonloop_pairs": float(nc / len(w)),
            "fraud_local_unique_mixing_among_eligible_pairs": float(nc / (2 * nf + nc)),
            "fraud_local_weighted_mixing_among_eligible_pairs": float(w[fb].sum() / (2 * w[ff].sum() + w[fb].sum())),
            "fraud_context_neighbor_pairs_excluded_from_mixing": int(((~known) & ((data["labels"][data["pair_source"]] == 1) | (data["labels"][data["pair_target"]] == 1))).sum())}


def spectral_energy(data):
    n = len(data["labels"]); u, v, w = data["pair_source"], data["pair_target"], data["multiplicity"]
    loop_ids = np.flatnonzero(data["loop_multiplicity"])
    adjacency = sparse.coo_matrix((np.r_[w, w, data["loop_multiplicity"][loop_ids]].astype(np.float64),
                                  (np.r_[u, v, loop_ids], np.r_[v, u, loop_ids])), shape=(n, n)).tocsr()
    x = data["features"].astype(np.float64)
    degree = data["weighted_degree"].astype(np.float64)
    if np.any(degree <= 0): raise RuntimeError("Spectral protocol requires positive degrees.")
    normalized = x / np.sqrt(degree)[:, None]
    denominator = np.sum(x * x, axis=0)
    numerator = denominator - np.sum(normalized * adjacency.dot(normalized), axis=0)
    if np.any(numerator < -1e-10 * np.maximum(denominator, 1)):
        raise RuntimeError("Negative normalized Laplacian energy.")
    values = np.divide(np.maximum(numerator, 0), denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return values, normalized, denominator


def spectral_delta(patches, normalized, denominator):
    if not len(patches): return np.zeros_like(denominator)
    a, b, c, d, w = patches[:, :5].T
    change = 2 * np.sum(w[:, None] * (normalized[a] * normalized[b] + normalized[c] * normalized[d]
                                   - normalized[a] * normalized[c] - normalized[b] * normalized[d]), axis=0)
    return np.divide(change, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def apply_original_eids(source, target, order, sorted_keys, patches, n):
    u, v = source.copy(), target.copy()
    changed = []
    for a, b, c, d, weight, _, _ in patches:
        for old_a, old_b, new_a, new_b in [(a, b, a, c), (b, a, c, a), (c, d, b, d), (d, c, d, b)]:
            left, right = np.searchsorted(sorted_keys, int(old_a) * n + int(old_b), side="left"), np.searchsorted(sorted_keys, int(old_a) * n + int(old_b), side="right")
            ids = order[left:right]
            if len(ids) != weight or not np.all(source[ids] == old_a) or not np.all(target[ids] == old_b):
                raise RuntimeError("Original EID multiplicity does not match swap trace.")
            u[ids], v[ids] = new_a, new_b
            changed.extend(ids.tolist())
    if len(changed) != len(set(changed)):
        raise RuntimeError("An original directed edge ID was changed twice.")
    for original, variant in [(source, u), (target, v)]:
        if not np.array_equal(np.bincount(original, minlength=n), np.bincount(variant, minlength=n)):
            raise RuntimeError("Original directed endpoint degree changed.")
    return u, v


def save_trace(path, patches):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(["a", "b", "c", "d", "multiplicity", "original_pair_id_1", "original_pair_id_2"])
        writer.writerows(patches.tolist())


def self_check():
    n = 8; u = np.asarray([0, 0, 2, 2, 4, 4, 6], dtype=np.int32); v = np.asarray([1, 3, 3, 7, 5, 7, 7], dtype=np.int32)
    w = np.asarray([2, 1, 1, 1, 2, 1, 1], dtype=np.int32); loops = np.ones(n, dtype=np.int32)
    rng = np.random.RandomState(12)
    data = {"pair_source": u, "pair_target": v, "multiplicity": w, "loop_multiplicity": loops,
            "weighted_degree": (np.bincount(u, weights=w, minlength=n) + np.bincount(v, weights=w, minlength=n)).astype(np.int64) + loops,
            "unique_nonloop_degree": np.bincount(u, minlength=n) + np.bincount(v, minlength=n),
            "labels": np.asarray([1, 1, 1, 1, 0, 0, 0, 0]), "features": rng.normal(size=(n, 3)), "eligible_start": 0}
    patches, _ = generate_swaps(data, 2, 0)
    assert len(patches) == 2
    variant = apply_compact(data, patches)
    original_mix, changed_mix = mixing(data), mixing(variant)
    assert changed_mix["FF_unique_pairs"] == original_mix["FF_unique_pairs"] - 2
    assert changed_mix["FB_unique_pairs"] == original_mix["FB_unique_pairs"] + 4
    energy, normalized, denom = spectral_energy(data)
    other, _, _ = spectral_energy(variant)
    assert np.allclose(other - energy, spectral_delta(patches, normalized, denom), atol=1e-12, rtol=0)
    su = np.r_[np.repeat(u, w), np.repeat(v, w), np.arange(n)]
    sv = np.r_[np.repeat(v, w), np.repeat(u, w), np.arange(n)]
    shuffle = rng.permutation(len(su)); su, sv = su[shuffle], sv[shuffle]
    keys = su * n + sv; order = np.argsort(keys)
    nu, nv = apply_original_eids(su, sv, order, keys[order], patches, n)
    expected = np.zeros((n, n), dtype=np.int64)
    np.add.at(expected, (nu, nv), 1)
    direct = np.zeros_like(expected)
    direct[variant["pair_source"], variant["pair_target"]] = w; direct += direct.T; np.fill_diagonal(direct, 1)
    assert np.array_equal(expected, direct)
    root = 1 / np.sqrt(data["weighted_degree"])
    original_dense = np.zeros((n, n), dtype=np.float64)
    original_dense[u, v] = w; original_dense += original_dense.T; np.fill_diagonal(original_dense, loops)
    laplacian = np.eye(n) - root[:, None] * original_dense * root[None, :]
    dense_energy = np.sum(data["features"] * laplacian.dot(data["features"]), axis=0) / np.sum(data["features"] ** 2, axis=0)
    assert np.allclose(energy, dense_energy, atol=1e-12, rtol=0)
    control, _ = generate_swaps(data, len(patches), 0, patches[:, 4])
    assert len(control) == len(patches) and np.array_equal(control[:, 4], patches[:, 4])
    apply_compact(data, control)
    flipped = dict(data); flipped["labels"] = 1 - data["labels"]
    label_invariant, _ = generate_swaps(flipped, len(patches), 0, patches[:, 4])
    assert np.array_equal(label_invariant, control)
    again, _ = generate_swaps(data, 2, 0); assert np.array_equal(again, patches)
    print("SELF-CHECK PASS: deterministic swaps, random control, weighted/unique degrees, original EIDs, parallel entries, loops and independent spectral change.")


class Tee:
    def __init__(self, stream, log): self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value); self.log.write(value); self.log.flush(); return len(value)
    def flush(self): self.stream.flush(); self.log.flush()


def training_backend():
    import dgl
    if digest(Path("train_ethereum.py")) != TRAINER_HASH:
        raise RuntimeError("Verified training helper is missing or changed.")
    import train_ethereum as trainer
    trainer.verify_sources()
    saved_builder = trainer.build_model
    def common_builder(name, graph, in_features, hidden, order_c):
        # Existing Amazon loops are retained once for both models, not duplicated for GCN.
        if name == "gcn": graph = dgl.remove_self_loop(graph)
        return saved_builder(name, graph, in_features, hidden, order_c)
    trainer.build_model = common_builder
    return trainer


def model_self_check():
    import torch
    import dgl
    trainer = training_backend(); torch.set_num_threads(2)
    n = 10; u = np.arange(n); v = (u + 1) % n
    graph = dgl.graph((torch.from_numpy(np.r_[u, v, u, 0, 1]), torch.from_numpy(np.r_[v, u, u, 1, 0])), num_nodes=n)
    data = {"features": np.random.RandomState(0).normal(size=(n, 25)).astype(np.float32),
            "labels": np.asarray([0, 1] * 4 + [0, 0]), "global_node_id": np.arange(n),
            "train_mask": np.isin(u, [0, 1, 2, 3]), "val_mask": np.isin(u, [4, 5]), "test_mask": np.isin(u, [6, 7])}
    model = trainer.build_model("gcn", graph, 25, 8, 2)
    a, b = model.graph.edges(); loops = a[a == b].numpy()
    assert np.array_equal(np.sort(loops), u)
    assert model.graph.num_edges() == graph.num_edges()
    with tempfile.TemporaryDirectory() as directory:
        for name in ("bwgnn", "gcn"):
            trainer.train_one(name, 0, graph, data, argparse.Namespace(hidden=8, order=2, epochs=2, lr=0.01), Path(directory))
    print("MODEL SELF-CHECK PASS: both actual models on a multigraph; exactly one GCN self-loop per node, validation selection and checkpoint restoration.")


def experiment(args, output):
    import torch
    import dgl
    trainer = training_backend()
    source_hashes = trainer.verify_sources()
    compact_path = Path(args.compact).resolve()
    if digest(compact_path) != COMPACT_HASH:
        raise RuntimeError("Amazon compact fingerprint changed.")
    audit = json.loads((compact_path.parent / "amazon_graph_audit.json").read_text(encoding="utf-8-sig"))
    if audit["status"] != "amazon_structure_audit_complete" or audit["compact_sha256"] != COMPACT_HASH:
        raise RuntimeError("Amazon audit report is invalid.")
    with np.load(str(compact_path), allow_pickle=False) as f: data = {k: f[k] for k in f.files}
    for name in ("features", "labels"):
        if array_hash(data[name]) != audit["benchmark_array_fingerprints"][name]:
            raise RuntimeError("Node arrays changed.")
    # The baseline keeps every original edge ID and every original loop.
    from dataset import Dataset
    original_graph = Dataset("amazon", True).graph
    source, target = original_graph.edges(order="eid"); source, target = source.numpy(), target.numpy()
    if array_hash(source) != audit["benchmark_array_fingerprints"]["source_ids"] or array_hash(target) != audit["benchmark_array_fingerprints"]["target_ids"]:
        raise RuntimeError("Original Amazon edge ordering or endpoints changed.")
    if not np.array_equal(original_graph.ndata["feature"].numpy(), data["features"]) or not np.array_equal(original_graph.ndata["label"].numpy(), data["labels"]):
        raise RuntimeError("Compact and released node arrays disagree.")
    torch.set_num_threads(args.threads)
    raw_keys = source * len(data["labels"]) + target
    order = np.argsort(raw_keys); sorted_keys = raw_keys[order]; del raw_keys
    _, ff, bb, _ = eligible_masks(data); m0 = min(int(ff.sum()), int(bb.sum()))
    requested = int(np.floor(max(args.levels) * m0))
    patches, generation = generate_swaps(data, requested, args.graph_seed)
    random_patches, random_generation = generate_swaps(data, len(patches), args.graph_seed, patches[:, 4])
    if len(random_patches) != len(patches) or not np.array_equal(random_patches[:, 4], patches[:, 4]):
        raise RuntimeError("Random control could not match the achieved camouflage count and multiplicity sequence.")
    baseline_energy, normalized, denominator = spectral_energy(data)
    conditions = [("camo_c{:g}".format(c).replace(".", "p"), c, patches[:min(int(np.floor(c * m0)), len(patches))], "camouflage") for c in args.levels]
    conditions.append(("random_control", max(args.levels), random_patches, "random_control"))
    training_args = argparse.Namespace(hidden=64, order=2, epochs=args.epochs, lr=0.01)
    all_results, graph_records = [], []
    fixed_hashes = {k: array_hash(data[k]) for k in ("features", "labels", "train_mask", "val_mask", "test_mask", "loop_multiplicity")}
    for tag, level, swaps, kind in conditions:
        print("\n=== CONDITION {} swaps={} ===".format(tag, len(swaps)), flush=True)
        folder = output / tag; folder.mkdir()
        save_trace(folder / "swap_trace.csv", swaps)
        variant = apply_compact(data, swaps)
        mix = mixing(variant)
        if kind == "camouflage":
            initial = mixing(data)
            if mix["FF_unique_pairs"] != initial["FF_unique_pairs"] - len(swaps) or mix["BB_unique_pairs"] != initial["BB_unique_pairs"] - len(swaps) or mix["FB_unique_pairs"] != initial["FB_unique_pairs"] + 2 * len(swaps):
                raise RuntimeError("Camouflage class-pair counts failed.")
        values, _, _ = spectral_energy(variant)
        if not np.allclose(values - baseline_energy, spectral_delta(swaps, normalized, denominator), atol=1e-10, rtol=0):
            raise RuntimeError("Full-matrix and swap-delta spectral calculations disagree.")
        with (folder / "spectral_features.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f); writer.writerow(["feature", "baseline_Shigh", "variant_Shigh", "absolute_change", "relative_change_if_baseline_nonzero"])
            for j in range(len(values)):
                writer.writerow([j, baseline_energy[j], values[j], values[j] - baseline_energy[j],
                                 (values[j] - baseline_energy[j]) / baseline_energy[j] if baseline_energy[j] > 1e-12 else ""])
        record = {"condition": tag, "type": kind, "c_intervention_parameter": level,
                  "requested_swaps": int(np.floor(level * m0)) if kind == "camouflage" else len(patches),
                  "achieved_swaps": len(swaps), "graph_seed": args.graph_seed, "M0": m0,
                  "multiplicity_histogram_of_swaps": {str(int(w)): int((swaps[:, 4] == w).sum()) for w in np.unique(swaps[:, 4])},
                  "mixing": mix, "weighted_and_unique_degrees_preserved": True,
                  "loop_multiplicities_preserved": True, "fixed_array_hashes": fixed_hashes,
                  "spectral_definition": "Per original feature x: x.T (I-D^-1/2 A D^-1/2) x / (x.T x), where A retains the original multigraph multiplicities and one original self-loop per node.",
                  "spectral_absolute_change_mean": float((values - baseline_energy).mean()),
                  "swap_trace_sha256": digest(folder / "swap_trace.csv")}
        if len(swaps):
            su, sv = apply_original_eids(source, target, order, sorted_keys, swaps, len(data["labels"]))
            graph = dgl.graph((torch.from_numpy(su), torch.from_numpy(sv)), num_nodes=len(data["labels"]))
            record["variant_original_EID_source_hash"], record["variant_original_EID_target_hash"] = array_hash(su), array_hash(sv)
            del su, sv
        else:
            graph = original_graph
            record["variant_original_EID_source_hash"], record["variant_original_EID_target_hash"] = array_hash(source), array_hash(target)
        if graph.num_edges() != len(source): raise RuntimeError("Directed edge entry count changed.")
        if not np.array_equal(graph.in_degrees().numpy(), data["weighted_degree"]) or not np.array_equal(graph.out_degrees().numpy(), data["weighted_degree"]):
            raise RuntimeError("Actual training graph degree check failed.")
        record["actual_DGL_in_out_degrees_verified"] = True
        write_json(folder / "graph_record.json", record); graph_records.append(record)
        print("FRAUD-LOCAL ELIGIBLE MIXING: {:.4f}%".format(100 * mix["fraud_local_unique_mixing_among_eligible_pairs"]), flush=True)
        for name in ("bwgnn", "gcn"):
            result = trainer.train_one(name, args.training_seed, graph, data, training_args, folder)
            result["condition"] = tag
            if name == "gcn":
                result["architecture"] = "Pilot two-layer GraphConv, symmetric normalization, input->64 ReLU->2. Original Amazon self-loops retained once; existing loops removed before helper adds exactly one per node."
            run_tag = "{}_seed{}{}".format(name, args.training_seed, "_C2" if name == "bwgnn" else "")
            write_json(folder / run_tag / "result.json", result)
            all_results.append(result); write_json(output / "results.json", all_results)
            gc.collect()
        del graph, variant; gc.collect()
        if any(array_hash(data[k]) != value for k, value in fixed_hashes.items()):
            raise RuntimeError("A fixed array changed during the experiment.")
    write_json(output / "camouflage_manifest.json", {
        "status": "camouflage_pilot_complete", "levels": args.levels, "graph_seed": args.graph_seed,
        "training_seed": args.training_seed, "epochs": args.epochs, "completed_runs": len(all_results),
        "models": ["bwgnn", "gcn"], "compact_sha256": COMPACT_HASH, "trainer_sha256": TRAINER_HASH,
        "runner_sha256": digest(Path(__file__)), "original_source_hashes": source_hashes,
        "generation": generation, "random_generation": random_generation, "graphs": graph_records,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "dgl": dgl.__version__, "threads": args.threads, "device": "cpu"},
        "selection_protocol": "Fresh post-update validation logits and strict validation Macro-F1 improvement; validation threshold grid; restore selected parameters; compute test metrics after selection. Same protocol for all new comparisons. The original benchmark reproduction remains separate.",
        "generation_protocol": "Nested camouflage prefixes from one graph seed. Random control uses eligible original pairs without labels and matches the achieved maximum-camouflage swap count and multiplicity sequence when feasible. Original pair used at most once. Original directed edge IDs retained; only selected endpoints are replaced.",
        "limitations": "One graph seed and one model seed: exploratory pilot, no confidence interval or robustness claim. Label-aware synthetic intervention includes labels across all supervised splits and is an oracle stress test, not an implementable attacker or label-independent graph preprocessing. Relation types are not retained in the homogeneous model. Camouflage removes FF/BB and creates FB edges simultaneously. Initial eligible fraud-local mixing is already high. Graph connectivity or centrality beyond degree is not controlled. Different model capacities do not isolate a causal spectral mechanism.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact")
    parser.add_argument("--output", default="camouflage_results")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--model-self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check: self_check(); return
    if args.model_self_check: model_self_check(); return
    if not args.compact: parser.error("--compact is required")
    if args.levels != sorted(set(args.levels)) or args.levels[0] != 0 or not 0 < max(args.levels) <= 1:
        parser.error("Use ascending distinct levels including 0 and a positive level <=1.")
    if args.epochs < 1 or args.threads < 1 or min(args.graph_seed, args.training_seed) < 0:
        parser.error("Invalid training settings.")
    output = Path(args.output).resolve() / ("amazon_camouflage_" + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True); print("OUTPUT DIRECTORY: " + str(output), flush=True)
    success = False
    with (output / "console.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                experiment(args, output); success = True
            except Exception as error:
                write_json(output / "experiment_error.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
                traceback.print_exc()
    evidence = output / "amazon_camouflage_evidence.zip"
    with zipfile.ZipFile(evidence, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file != evidence: z.write(file, str(file.relative_to(output)).replace("\\", "/"))
        for path, name in [(Path(__file__).resolve(), "run_amazon_camouflage.py"), (Path("train_ethereum.py"), "train_ethereum.py"),
                           (Path(args.compact), "input_amazon_compact.npz"), (Path(args.compact).parent / "amazon_graph_audit.json", "input_amazon_audit.json")]:
            if path.is_file(): z.write(path, name)
        for name in ("main.py", "dataset.py", "BWGNN.py"):
            if Path(name).is_file(): z.write(name, "original_sources/" + name)
    print("EVIDENCE ZIP: " + str(evidence), flush=True)
    if not success: raise RuntimeError("Pilot did not complete. Send evidence ZIP and complete output.")
    print("AMAZON CAMOUFLAGE PILOT: COMPLETE", flush=True)


if __name__ == "__main__": main()
