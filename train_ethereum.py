"""CPU pilot comparison on the verified Ethereum sample.

BWGNN uses the unchanged released class. GCN and MLP are explicit pilot
baselines, not claims to reproduce the paper's baseline implementations.
Checkpoints are selected only by validation Macro-F1; test is scored once
after restoring each validation-selected checkpoint.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np
import sklearn
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

SOURCE_HASHES = {
    "main.py": "1642e10e5c799ebd303bb20d66a991a55363276b35855996f5615eba0fd1ea13",
    "dataset.py": "229b5e5101c63879264802f909e64396a49be694a8bb4815dfea201f9b78d1fb",
    "BWGNN.py": "0ac45b1104f80fa17c50cd5e3bf9aa6241e008e6b57fdab9366ec69f80cb2f84",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def verify_sources():
    hashes = {}
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256(Path(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError("Original BWGNN source changed: " + name)
        hashes[name] = actual
    return hashes


def choose_threshold(y, probability):
    best_f1, best_threshold = -1.0, None
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(y, probability > threshold, average="macro", labels=[0, 1], zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = float(score), float(threshold)
    return best_f1, best_threshold


def logit_margin(logits):
    logits64 = np.asarray(logits, dtype=np.float64)
    return logits64[:, 1] - logits64[:, 0]


def metrics(y, probability, margin, threshold):
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Metrics require both labelled classes, without context labels.")
    if not np.isfinite(probability).all() or not np.isfinite(margin).all():
        raise RuntimeError("Nonfinite prediction scores.")
    prediction = probability > threshold
    precision, recall, _ = precision_recall_curve(y, margin)
    return {
        "macro_f1": float(f1_score(y, prediction, average="macro", labels=[0, 1], zero_division=0)),
        "fraud_f1": float(f1_score(y, prediction, zero_division=0)),
        "fraud_precision": float(precision_score(y, prediction, zero_division=0)),
        "fraud_recall": float(recall_score(y, prediction, zero_division=0)),
        "roc_auc_margin64": float(roc_auc_score(y, margin)),
        "roc_auc_probability32": float(roc_auc_score(y, probability)),
        "average_precision_margin64": float(average_precision_score(y, margin)),
        "pr_auc_trapezoid_margin64": float(auc(recall, precision)),
        "positive_prevalence": float(y.mean()),
        "probability_exactly_zero_count": int((probability == 0).sum()),
        "probability_exactly_one_count": int((probability == 1).sum()),
        "confusion_matrix_labels_0_1": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def load_dataset(path):
    pipeline_path = path.parent / "pipeline_manifest.json"
    meta = json.loads(pipeline_path.read_text(encoding="utf-8-sig"))
    if meta.get("status") != "preprocessing_complete" or sha256(path) != meta["dataset_sha256"]:
        raise RuntimeError("Dataset or preprocessing record failed verification.")
    with np.load(str(path), allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    y, x = data["labels"], data["features"]
    n = len(y)
    if x.shape != (n, meta["feature_count"]) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise RuntimeError("Invalid model feature matrix.")
    counts = np.zeros(n, dtype=np.int8)
    for name in ("train", "val", "test"):
        mask = data[name + "_mask"]
        if mask.shape != (n,) or mask.dtype != bool or set(np.unique(y[mask])) != {0, 1}:
            raise RuntimeError("Invalid split: " + name)
        counts += mask
    if np.any(counts > 1) or not np.array_equal(counts == 1, y >= 0):
        raise RuntimeError("Overlapping splits or supervised context nodes.")
    fit = StandardScaler().fit(data["raw_features"][data["train_mask"]])
    if not np.allclose(fit.mean_, data["standardizer_mean"], rtol=1e-10, atol=1e-12):
        raise RuntimeError("Standardizer was not fitted to the training roots.")
    if not np.allclose(fit.scale_, data["standardizer_scale"], rtol=1e-10, atol=1e-12):
        raise RuntimeError("Standardizer scales do not match training roots.")
    if not np.array_equal(((data["raw_features"] - data["standardizer_mean"]) / data["standardizer_scale"]).astype(np.float32), x):
        raise RuntimeError("Standardized features disagree with their source matrix.")
    u, v = data["source"], data["target"]
    if u.shape != v.shape or u.min() < 0 or v.min() < 0 or u.max() >= n or v.max() >= n or np.any(u == v):
        raise RuntimeError("Invalid graph endpoints.")
    keys = u.astype(np.int64) * n + v
    if len(np.unique(keys)) != len(keys) or not np.array_equal(np.sort(keys), np.sort(v.astype(np.int64) * n + u)):
        raise RuntimeError("Graph must be simple and bidirected.")
    return data, meta


def build_model(name, graph, in_features, hidden, order):
    import torch
    from torch import nn
    if name == "bwgnn":
        from BWGNN import BWGNN
        return BWGNN(in_features, hidden, 2, graph, d=order)
    if name == "gcn":
        import dgl
        from dgl.nn.pytorch import GraphConv
        class PilotGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.graph = dgl.add_self_loop(graph)
                self.first = GraphConv(in_features, hidden, norm="both", activation=torch.relu)
                self.second = GraphConv(hidden, 2, norm="both")
            def forward(self, features):
                return self.second(self.graph, self.first(self.graph, features))
        return PilotGCN()
    if name == "mlp":
        return nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(),
                             nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    raise ValueError("Unknown model: " + name)


def seed_everything(seed):
    import torch
    import dgl
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dgl.seed(seed)
    if hasattr(dgl, "random") and hasattr(dgl.random, "seed"):
        dgl.random.seed(seed)


def train_one(name, seed, graph, data, args, output):
    import torch
    import torch.nn.functional as F
    seed_everything(seed)
    model = build_model(name, graph, data["features"].shape[1], args.hidden, args.order)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.from_numpy(data["features"])
    y = torch.from_numpy(data["labels"]).long()
    train_mask = torch.from_numpy(data["train_mask"])
    val_ids = np.flatnonzero(data["val_mask"])
    test_ids = np.flatnonzero(data["test_mask"])
    train_y = y[train_mask]
    weight = float((train_y == 0).sum().item() / (train_y == 1).sum().item())
    class_weight = torch.tensor([1.0, weight], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    tag = "{}_seed{}_C{}".format(name, seed, args.order) if name == "bwgnn" else "{}_seed{}".format(name, seed)
    run = output / tag
    run.mkdir()
    print("\n=== {} parameters={} class_weight={:.6f} ===".format(tag, parameters, weight), flush=True)
    started = time.monotonic()
    best = -1.0
    best_state = best_epoch = best_threshold = best_val = None
    fieldnames = ["epoch", "train_loss", "threshold", "selected_by_validation_macro_f1",
                  "val_macro_f1", "val_auc_margin64", "val_auc_probability32", "val_average_precision", "val_pr_auc_trapezoid"]
    with (run / "epochs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            train_logits = model(x)
            loss = F.cross_entropy(train_logits[train_mask], y[train_mask], weight=class_weight)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite training loss in " + tag)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                # Fresh post-update eval-mode logits, aligned with the saved parameters.
                val_logits = model(x)[val_ids]
                val_probability = val_logits.softmax(dim=1)[:, 1].numpy()
                val_margin = logit_margin(val_logits.numpy())
            val_f1, threshold = choose_threshold(data["labels"][val_ids], val_probability)
            val_result = metrics(data["labels"][val_ids], val_probability, val_margin, threshold)
            selected = val_f1 > best
            if selected:
                best = val_f1
                best_epoch, best_threshold, best_val = epoch, threshold, val_result
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            writer.writerow({"epoch": epoch, "train_loss": float(loss.item()), "threshold": threshold,
                             "selected_by_validation_macro_f1": int(selected), "val_macro_f1": val_f1,
                             "val_auc_margin64": val_result["roc_auc_margin64"],
                             "val_auc_probability32": val_result["roc_auc_probability32"],
                             "val_average_precision": val_result["average_precision_margin64"],
                             "val_pr_auc_trapezoid": val_result["pr_auc_trapezoid_margin64"]})
            f.flush()
            if selected or epoch % 10 == 0 or epoch == args.epochs - 1:
                print("Epoch {} loss={:.5f} val_MF1={:.4f} best={:.4f}".format(epoch, loss.item(), val_f1, best), flush=True)
            del train_logits, loss, val_logits
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(x)
        final_probability = final_logits.softmax(dim=1)[:, 1].numpy()
        final_logits = final_logits.numpy()
    final_margin = logit_margin(final_logits)
    restored_val = metrics(data["labels"][val_ids], final_probability[val_ids], final_margin[val_ids], best_threshold)
    if not np.isclose(restored_val["macro_f1"], best_val["macro_f1"], rtol=0, atol=1e-12):
        raise RuntimeError("Restored checkpoint does not reproduce selected validation Macro-F1.")
    if not np.isclose(restored_val["roc_auc_margin64"], best_val["roc_auc_margin64"], rtol=0, atol=1e-10):
        raise RuntimeError("Restored checkpoint does not reproduce validation ranking.")
    # Test labels/scores are evaluated only here, after validation selection.
    test_result = metrics(data["labels"][test_ids], final_probability[test_ids], final_margin[test_ids], best_threshold)
    score_arrays = {"selected_epoch": np.asarray(best_epoch), "threshold": np.asarray(best_threshold)}
    for split, ids in (("val", val_ids), ("test", test_ids)):
        score_arrays[split + "_local_id"] = ids
        score_arrays[split + "_global_id"] = data["global_node_id"][ids]
        score_arrays[split + "_labels"] = data["labels"][ids]
        score_arrays[split + "_logits"] = final_logits[ids]
        score_arrays[split + "_probability32"] = final_probability[ids]
        score_arrays[split + "_margin64"] = final_margin[ids]
    score_file = run / "selected_scores.npz"
    np.savez_compressed(str(score_file), **score_arrays)
    checkpoint = run / "selected_parameters.pt"
    torch.save(best_state, str(checkpoint))
    result = {"model": name, "seed": seed, "order": args.order if name == "bwgnn" else None,
              "hidden": args.hidden, "epochs": args.epochs, "learning_rate": args.lr,
              "class_weight_positive": weight, "trainable_parameter_count": parameters,
              "selected_epoch": best_epoch, "selected_threshold": best_threshold,
              "validation": restored_val, "test": test_result, "elapsed_seconds": time.monotonic() - started,
              "checkpoint_sha256": sha256(checkpoint), "scores_sha256": sha256(score_file),
              "selection": "Strictly improving validation Macro-F1; threshold grid 0.05:0.05:0.95, strict probability>threshold, first threshold/epoch wins ties.",
              "evaluation": "Fresh post-update eval-mode validation logits; restore exact selected parameters; score test once after epoch selection.",
              "architecture": {"bwgnn": "Unchanged released homogeneous BWGNN; no graph self-loops added.",
                               "gcn": "Pilot 2-layer GraphConv: input->hidden ReLU->2; symmetric normalization; one self-loop per node added to the common base graph.",
                               "mlp": "Pilot feature-only MLP: input->hidden ReLU->hidden ReLU->2."}[name]}
    write_json(run / "result.json", result)
    print("TEST {} epoch={} threshold={:.2f} MF1={:.2f}% ROC-margin={:.2f}% AP={:.2f}% PR-trap={:.2f}%".format(
        tag, best_epoch, best_threshold, 100 * test_result["macro_f1"], 100 * test_result["roc_auc_margin64"],
        100 * test_result["average_precision_margin64"], 100 * test_result["pr_auc_trapezoid_margin64"]), flush=True)
    del model, best_state, optimizer
    return result


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, value):
        self.terminal.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self):
        self.terminal.flush(); self.log.flush()


def run(args):
    import torch
    import dgl
    path = Path(args.dataset).resolve()
    data, pipeline = load_dataset(path)
    sources = verify_sources()
    torch.set_num_threads(args.threads)
    graph = dgl.graph((torch.from_numpy(data["source"].astype(np.int64)),
                       torch.from_numpy(data["target"].astype(np.int64))), num_nodes=len(data["labels"]))
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output).resolve() / ("ethereum_training_" + stamp)
    output.mkdir(parents=True)
    print("OUTPUT DIRECTORY: " + str(output), flush=True)
    original_out, original_err = sys.stdout, sys.stderr
    results = []
    success = False
    with (output / "console.log").open("w", encoding="utf-8") as log:
        sys.stdout = Tee(original_out, log); sys.stderr = Tee(original_err, log)
        try:
            print("CPU PILOT: nodes={} base_edges={} features={} threads={}".format(len(data["labels"]), graph.num_edges(),
                  data["features"].shape[1], args.threads), flush=True)
            for model in args.models:
                for seed in args.seeds:
                    results.append(train_one(model, seed, graph, data, args, output))
                    write_json(output / "results.json", results)
            write_json(output / "training_manifest.json", {
                "status": "training_complete", "dataset_sha256": sha256(path),
                "preprocessing_manifest_sha256": sha256(path.parent / "pipeline_manifest.json"),
                "runner_sha256": sha256(Path(__file__)), "original_source_hashes": sources,
                "environment": {"python": platform.python_version(), "torch": torch.__version__, "dgl": dgl.__version__,
                                "numpy": np.__version__, "sklearn": sklearn.__version__, "device": "cpu", "threads": args.threads},
                "models": args.models, "seeds": args.seeds, "bwgnn_order": args.order,
                "metric_definitions": {"roc_auc_margin64": "ROC-AUC using float64 difference of float32 class logits; avoids softmax saturation ties.",
                    "roc_auc_probability32": "ROC-AUC using the float32 positive probability, retained for comparison.",
                    "average_precision_margin64": "Non-interpolated average precision, reported separately from trapezoidal PR-AUC.",
                    "pr_auc_trapezoid_margin64": "Trapezoidal area under the precision-recall curve."},
                "limitations": "Pilot results on one fixed label-informed sampled graph with potentially noisy negatives and 20% supervised positives. Standard baselines have different parameter counts. Results do not isolate a causal spectral advantage or estimate deployment-population precision.",
                "graph_semantics": pipeline["graph_semantics"], "evaluation_scope": pipeline["evaluation_scope"],
                "protocol_change_from_original_reproduction": "The original benchmark audit retains the released pre-update-logit evaluation. This new-data extension evaluates fresh post-update logits so selected scores and saved parameters agree; original source files remain unchanged.",
                "planned_runs": len(args.models) * len(args.seeds), "completed_runs": len(results),
            })
            success = True
        except BaseException as error:
            write_json(output / "training_error.json", {"type": type(error).__name__, "message": str(error),
                                                       "traceback": traceback.format_exc()})
            traceback.print_exc()
        finally:
            sys.stdout, sys.stderr = original_out, original_err
    evidence = output / "ethereum_training_evidence.zip"
    with zipfile.ZipFile(evidence, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file != evidence:
                z.write(file, str(file.relative_to(output)).replace("\\", "/"))
        z.write(path, "input_dataset.npz")
        z.write(path.parent / "pipeline_manifest.json", "preprocessing_manifest.json")
        z.write(Path(__file__).resolve(), "train_ethereum.py")
        for name in SOURCE_HASHES:
            z.write(Path(name), "original_sources/" + name)
    print("EVIDENCE ZIP: " + str(evidence), flush=True)
    if not success:
        raise RuntimeError("Training did not complete. Send the evidence ZIP and output.")
    print("ETHEREUM TRAINING: COMPLETE", flush=True)


def metric_self_check():
    labels = np.asarray([0, 1, 0, 1])
    probability = np.asarray([0.12, 0.88, 0.27, 0.73], dtype=np.float32)
    f1, threshold = choose_threshold(labels, probability)
    assert f1 == 1.0 and np.isclose(threshold, 0.3)
    result = metrics(labels, probability, np.asarray([-2.0, 2.0, -1.0, 1.0]), threshold)
    assert result["macro_f1"] == result["roc_auc_margin64"] == result["average_precision_margin64"] == 1.0
    saturated = metrics(np.asarray([0, 1]), np.asarray([1.0, 1.0], dtype=np.float32), np.asarray([80., 81.]), 0.5)
    assert saturated["roc_auc_probability32"] == 0.5 and saturated["roc_auc_margin64"] == 1.0
    best, selected = -1., None
    for epoch, score in enumerate([0.5, 0.7, 0.7, 0.6]):
        if score > best:
            best, selected = score, epoch
    assert selected == 1
    print("METRIC SELF-CHECK PASS: threshold grid/ties, ranking saturation, distinct AP/PR-AUC definitions.")


def self_check(args):
    import torch
    import dgl
    metric_self_check()
    verify_sources()
    torch.set_num_threads(2)
    rng = np.random.RandomState(0)
    n = 10
    u = np.arange(n, dtype=np.int64); v = (u + 1) % n
    graph = dgl.graph((torch.from_numpy(np.r_[u, v]), torch.from_numpy(np.r_[v, u])), num_nodes=n)
    data = {"features": rng.normal(size=(n, 18)).astype(np.float32), "labels": np.asarray([0, 1] * 4 + [-1, -1]),
            "global_node_id": np.arange(n, dtype=np.int32),
            "train_mask": np.isin(np.arange(n), [0, 1, 2, 3]), "val_mask": np.isin(np.arange(n), [4, 5]),
            "test_mask": np.isin(np.arange(n), [6, 7])}
    fixture_args = argparse.Namespace(hidden=8, order=2, epochs=2, lr=0.01)
    with tempfile.TemporaryDirectory() as directory:
        for name in ("bwgnn", "gcn", "mlp"):
            result = train_one(name, 0, graph, data, fixture_args, Path(directory))
            assert 0 <= result["test"]["macro_f1"] <= 1
            folder = next(p for p in Path(directory).iterdir() if p.name.startswith(name + "_"))
            with np.load(str(folder / "selected_scores.npz"), allow_pickle=False) as scores:
                assert np.array_equal(scores["test_labels"], np.asarray([0, 1]))
                assert len(scores["test_global_id"]) == 2
    print("SELF-CHECK PASS: all three actual models, optimizer updates, checkpoint restoration, masked context exclusion, prediction export.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--output", default="training_results")
    parser.add_argument("--models", nargs="+", choices=("bwgnn", "gcn", "mlp"), default=["bwgnn", "gcn", "mlp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--metric-self-check", action="store_true")
    args = parser.parse_args()
    if args.metric_self_check:
        metric_self_check(); return
    if args.self_check:
        self_check(args); return
    if not args.dataset:
        parser.error("--dataset is required")
    if min(args.order, args.hidden, args.epochs, args.lr, args.threads) <= 0:
        parser.error("Training settings must be positive")
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate models/seeds are not allowed")
    run(args)


if __name__ == "__main__":
    main()
