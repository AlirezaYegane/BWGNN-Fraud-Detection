"""Observe the archived BWGNN training loop without editing its source files.

Python 3.8 compatible. Run beside the original main.py, dataset.py and BWGNN.py.
Model selection always uses validation data; test metrics are diagnostic only.
"""
import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

EXPECTED_SOURCES = {
    "main.py": "1642e10e5c799ebd303bb20d66a991a55363276b35855996f5615eba0fd1ea13",
    "dataset.py": "229b5e5101c63879264802f909e64396a49be694a8bb4815dfea201f9b78d1fb",
    "BWGNN.py": "0ac45b1104f80fa17c50cd5e3bf9aa6241e008e6b57fdab9366ec69f80cb2f84",
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def score_diagnostics(labels, probability, logits):
    # Subtract in float64 to retain distinctions in the already-produced logits.
    margin = logits.astype(np.float64)[:, 1] - logits.astype(np.float64)[:, 0]
    return {
        "auc_prob32": float(roc_auc_score(labels, probability)),
        "auc_margin64": float(roc_auc_score(labels, margin)),
        "prob_exact_zero_count": int(np.count_nonzero(probability == 0)),
        "prob_exact_one_count": int(np.count_nonzero(probability == 1)),
        "unique_probability_count": int(np.unique(probability).size),
        "unique_margin_count": int(np.unique(margin).size),
    }, margin


def instrument_train(source):
    """Add precisely one observation call; leave original statements intact."""
    original = ast.parse(source)
    modified = copy.deepcopy(original)
    matches = 0
    hook = ast.parse(
        "_audit_observer.observe(e, labels, logits, probs, train_mask, val_mask, "
        "test_mask, f1, thres, tmf1, tauc, best_f1)"
    ).body[0]
    for node in ast.walk(modified):
        if isinstance(node, ast.For):
            replacement = []
            for statement in node.body:
                replacement.append(statement)
                if (isinstance(statement, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "tauc"
                                for t in statement.targets)):
                    replacement.append(copy.deepcopy(hook))
                    matches += 1
            node.body = replacement
    if matches != 1:
        raise RuntimeError("Expected exactly one test-AUC assignment in the original loop.")
    ast.fix_missing_locations(modified)
    # Confirm the only syntax-tree change is the inserted observation call.
    restored = copy.deepcopy(modified)
    for node in ast.walk(restored):
        if isinstance(node, ast.For):
            node.body = [s for s in node.body if not (
                isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Attribute)
                and isinstance(s.value.func.value, ast.Name)
                and s.value.func.value.id == "_audit_observer"
            )]
    if ast.dump(restored) != ast.dump(original):
        raise RuntimeError("Training-loop preservation check failed.")
    return modified


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()


class Observer:
    def __init__(self, directory):
        self.directory = directory
        self.best = {}
        self.count = 0
        self.split_hashes = None
        self.file = (directory / "epochs.csv").open("w", newline="", encoding="utf-8")
        self.writer = None

    def observe(self, epoch, labels, logits, probs, train_mask, val_mask, test_mask,
                val_mf1, threshold, test_mf1, test_auc, previous_best_f1):
        y = labels.detach().cpu().numpy()
        tr = train_mask.detach().cpu().numpy().astype(bool)
        v = val_mask.detach().cpu().numpy().astype(bool)
        t = test_mask.detach().cpu().numpy().astype(bool)
        raw = logits.detach().cpu().numpy()
        p = probs.detach().cpu().numpy()[:, 1]
        if self.split_hashes is None:
            if np.any(tr & v) or np.any(tr & t) or np.any(v & t):
                raise RuntimeError("Overlapping training, validation or test masks.")
            self.split_hashes = {
                "train_ids": array_hash(np.flatnonzero(tr)),
                "validation_ids": array_hash(np.flatnonzero(v)),
                "test_ids": array_hash(np.flatnonzero(t)),
            }
        vd, vm = score_diagnostics(y[v], p[v], raw[v])
        td, tm = score_diagnostics(y[t], p[t], raw[t])
        if not np.isclose(td["auc_prob32"], test_auc, rtol=0, atol=1e-12):
            raise RuntimeError("Diagnostic AUC differs from the original loop's AUC.")
        row = {
            "epoch_zero_based": int(epoch), "val_macro_f1": float(val_mf1),
            "threshold": float(threshold), "test_macro_f1": float(test_mf1),
            "selected_by_original_val_f1": bool(val_mf1 > previous_best_f1),
        }
        row.update({"val_" + k: value for k, value in vd.items()})
        row.update({"test_" + k: value for k, value in td.items()})
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()
        candidates = {
            "val_macro_f1": (float(val_mf1), bool(val_mf1 > previous_best_f1)),
            "val_auc_prob32": (vd["auc_prob32"], None),
            "val_auc_margin64": (vd["auc_margin64"], None),
        }
        for selector, (value, original_selected) in candidates.items():
            selected = original_selected if original_selected is not None else (
                selector not in self.best or value > self.best[selector][selector]
            )
            if selected:
                self.best[selector] = dict(row)
                np.savez_compressed(
                    self.directory / ("scores_selected_by_" + selector + ".npz"),
                    epoch_zero_based=np.array(epoch), threshold=np.array(threshold),
                    val_ids=np.flatnonzero(v), val_labels=y[v],
                    val_prob32=p[v], val_logits=raw[v], val_margin64=vm,
                    test_ids=np.flatnonzero(t), test_labels=y[t],
                    test_prob32=p[t], test_logits=raw[t], test_margin64=tm,
                )
        self.count += 1

    def close(self):
        self.file.close()


def legacy_evidence(project):
    path = project / "amazon_40_bwgnn_10runs.log"
    if not path.exists():
        return {"status": "Previous ten-run log not present in this folder."}
    raw = path.read_bytes()
    text = raw.decode("utf-16") if b"\x00" in raw[:100] else raw.decode("utf-8-sig")
    records = re.findall(r"Test: REC ([\d.]+) PRE ([\d.]+) MF1 ([\d.]+) AUC ([\d.]+)", text)
    epochs = re.findall(r"^Epoch (\d+),", text, re.MULTILINE)
    result = {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
              "run_count": len(records), "epoch_line_count": len(epochs),
              "all_epoch_blocks_complete": bool(epochs) and len(epochs) % 100 == 0
              and all(epochs[j:j+100] == [str(i) for i in range(100)]
                      for j in range(0, len(epochs), 100)),
              "per_run_rounded_results": records}
    if records:
        mf1 = np.array([float(r[2]) for r in records])
        auc = np.array([float(r[3]) for r in records])
        result["statistics_from_rounded_percentages"] = {
            "mf1_mean": float(mf1.mean()), "mf1_std_ddof0": float(mf1.std(ddof=0)),
            "auc_mean": float(auc.mean()), "auc_std_ddof0": float(auc.std(ddof=0)),
        }
    return result


def self_check():
    y = np.array([0, 1, 0, 1])
    raw = np.array([[0, 20], [0, 30], [0, 25], [0, 35]], dtype=np.float32)
    diagnostic, _ = score_diagnostics(y, np.ones(4, dtype=np.float32), raw)
    assert diagnostic["auc_prob32"] == 0.5
    assert diagnostic["auc_margin64"] == 1.0
    source = "def train():\n    for e in range(100):\n        tauc = roc_auc_score(y, p)\n        if best_f1 < f1:\n            best_f1 = f1\n"
    compile(instrument_train(source), "<self-check>", "exec")
    print("SELF-CHECK PASS: metric diagnostic and training-loop preservation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--dataset", choices=["amazon", "yelp"], default="amazon")
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--homo", type=int, choices=[0, 1], default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, default=Path("audit_results"))
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.epoch < 1 or not 0 < args.train_ratio < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive epochs, a valid training ratio, and distinct seeds.")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must lie in [0, 2**32).")
    project = Path(__file__).resolve().parent
    if Path.cwd().resolve() != project:
        raise RuntimeError("Run this script from the original project folder.")
    hashes = {}
    for name, expected in EXPECTED_SOURCES.items():
        hashes[name] = hashlib.sha256((project / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if hashes[name] != expected:
            raise RuntimeError("Source differs from the supplied archive: " + name)
    import torch
    import dgl
    import main as official
    official.dataset_name = args.dataset
    tree = instrument_train(inspect.getsource(official.train))
    exec(compile(tree, "<observed-original-train>", "exec"), official.__dict__)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = args.output.resolve() / (args.dataset + "_homo" + str(args.homo) + "_" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    print("OUTPUT DIRECTORY:", out)
    with (out / "console.log").open("w", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            metadata = {
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "python": sys.version, "executable": sys.executable,
                "torch": torch.__version__, "dgl": dgl.__version__, "numpy": np.__version__,
                "source_sha256_normalized_newlines": hashes,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "device": "cpu", "torch_threads": torch.get_num_threads(),
                "split_random_state": 2, "second_split_test_size": 0.67,
                "evaluation_semantics": "Original pre-optimizer-step logits are retained.",
                "selection_rule": "Strict validation Macro-F1 improvement; first threshold on ties.",
                "seed_protocol": "Seed Python, NumPy, PyTorch and DGL immediately before each model.",
                "scope": "Seeded diagnostic extension; does not replace previous unseeded reproduction.",
            }
            write_json(out / "metadata.json", metadata)
            if args.dataset == "amazon":
                write_json(out / "legacy_log_audit.json", legacy_evidence(project))
            graph = official.Dataset(args.dataset, bool(args.homo)).graph
            features, labels = graph.ndata["feature"], graph.ndata["label"]
            if not torch.isfinite(features).all().item():
                raise RuntimeError("Non-finite node features found.")
            if set(labels.unique().tolist()) != {0, 1}:
                raise RuntimeError("Expected binary labels with both classes present.")
            edge_fingerprints = []
            for relation in graph.canonical_etypes:
                src, dst = graph.edges(etype=relation, order="eid")
                edge_fingerprints.append({
                    "relation": list(relation), "edge_entries": graph.num_edges(etype=relation),
                    "source_ids_hash": array_hash(src.numpy()),
                    "destination_ids_hash": array_hash(dst.numpy()),
                })
                del src, dst
            metadata["graph"] = {
                "nodes": graph.num_nodes(),
                "edge_entries": sum(e["edge_entries"] for e in edge_fingerprints),
                "edge_fingerprints": edge_fingerprints,
                "features": list(features.shape), "feature_hash": array_hash(features.numpy()),
                "label_hash": array_hash(labels.numpy()),
                "eligible_labelled_nodes": len(labels) - 3305 if args.dataset == "amazon" else len(labels),
                "edge_count_note": "DGL entries; may include parallel edges, reverse entries and self-loops.",
            }
            write_json(out / "metadata.json", metadata)
            summary = []
            for seed in args.seeds:
                run_dir = out / ("seed_" + str(seed))
                run_dir.mkdir(exist_ok=False)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if hasattr(dgl, "seed"):
                    dgl.seed(seed)
                model_class = official.BWGNN if args.homo else official.BWGNN_Hetero
                model = model_class(features.shape[1], args.hid_dim, 2, graph, d=args.order)
                observer = Observer(run_dir)
                official._audit_observer = observer
                print("\n=== DIAGNOSTIC SEED", seed, "===")
                try:
                    mf1, auc = official.train(model, graph, args)
                finally:
                    observer.close()
                if observer.count != args.epoch or "val_macro_f1" not in observer.best:
                    raise RuntimeError("Incomplete epoch observations or no selected epoch.")
                selected = observer.best["val_macro_f1"]
                if not np.isclose(selected["test_macro_f1"], mf1, rtol=0, atol=1e-12):
                    raise RuntimeError("Selected F1 does not match the original training loop.")
                if not np.isclose(selected["test_auc_prob32"], auc, rtol=0, atol=1e-12):
                    raise RuntimeError("Selected AUC does not match the original training loop.")
                result = {"seed": seed, "observed_epochs": observer.count,
                          "split_hashes": observer.split_hashes, "selections": observer.best}
                write_json(run_dir / "summary.json", result)
                summary.append(result)
                write_json(out / "summary.json", summary)
                print("AUDIT RESULT: seed={} epoch={} MF1={:.4f}% AUC-prob32={:.4f}% AUC-margin64={:.4f}%".format(
                    seed, selected["epoch_zero_based"], mf1 * 100,
                    auc * 100, selected["test_auc_margin64"] * 100))
            print("\nAUDIT COMPLETE")
            print("RESULTS DIRECTORY:", out)
            print("Next: inspect all validation-selected epochs; never select by test metrics.")


if __name__ == "__main__":
    main()
