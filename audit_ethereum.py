"""Audit XBlock's released MultiDiGraph and export every transaction to NumPy.

Python 3.8+, NetworkX, NumPy. Run via the parent supervisor on Windows.
Original BWGNN sources and raw data are never modified.
"""
import argparse
import builtins
import collections
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

GIB = 1024 ** 3


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class ReleasedGraphUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("networkx.classes.multidigraph", "MultiDiGraph"):
            import networkx as nx
            return nx.MultiDiGraph
        if module == "builtins" and name in ("dict", "set", "list", "tuple", "str", "int", "float", "bytes"):
            return getattr(builtins, name)
        raise pickle.UnpicklingError("Unexpected pickle class: {}.{}".format(module, name))


def audit_export(raw_file, output, recorded_sha):
    import networkx as nx
    import numpy as np
    started = time.monotonic()
    print("Checking raw-file SHA256...", flush=True)
    actual_sha = sha256(raw_file)
    if actual_sha.lower() != recorded_sha.lower():
        raise RuntimeError("Raw SHA256 differs from the acquisition record.")
    print("Loading the released graph under the supervisor's memory limit...", flush=True)
    with Path(raw_file).open("rb") as f:
        graph = ReleasedGraphUnpickler(f).load()
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("Expected a NetworkX MultiDiGraph.")
    # Use the stored adjacency dictionaries directly, including legacy files.
    nodes = graph.__dict__.get("_node")
    successors = graph.__dict__.get("_succ", graph.__dict__.get("_adj"))
    predecessors = graph.__dict__.get("_pred")
    if not all(isinstance(x, dict) for x in (nodes, successors, predecessors)):
        raise RuntimeError("Unexpected graph state: " + repr(sorted(graph.__dict__)))
    n = len(nodes)
    if n == 0 or n >= np.iinfo(np.int32).max:
        raise RuntimeError("Unsupported node count: {}".format(n))
    m = sum(len(keys) for neighbors in successors.values() for keys in neighbors.values())
    if m == 0:
        raise RuntimeError("The graph contains no transactions.")
    if shutil.disk_usage(str(output)).free < (78 * n + 24 * m + 256 * 1024 * 1024):
        raise RuntimeError("Insufficient free disk space for the compact transaction export.")
    print("GRAPH LOADED: nodes={} transactions={}".format(n, m), flush=True)
    compact = output / "compact"
    compact.mkdir()
    ids = {}
    labels = np.lib.format.open_memmap(str(compact / "labels.npy"), mode="w+", dtype=np.int8, shape=(n,))
    labels[:] = -1
    label_counts = collections.Counter()
    node_keys = collections.Counter()
    invalid_labels = 0
    address_digest = hashlib.sha256()
    with (compact / "node_addresses.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, (address, attrs) in enumerate(nodes.items()):
            if not isinstance(address, str):
                raise TypeError("Expected string address identifiers.")
            ids[address] = i
            line = json.dumps(address, ensure_ascii=True) + "\n"
            f.write(line)
            address_digest.update(line.encode("utf-8"))
            node_keys.update(attrs.keys())
            value = attrs.get("isp")
            if value is not None and isinstance(value, (int, float, bool, np.integer, np.floating)) and value in (0, 1):
                labels[i] = int(value)
                label_counts[str(int(value))] += 1
            elif "isp" not in attrs:
                label_counts["missing"] += 1
            else:
                invalid_labels += 1
                label_counts["invalid"] += 1
    labels.flush()
    src = np.lib.format.open_memmap(str(compact / "source.npy"), mode="w+", dtype=np.int32, shape=(m,))
    dst = np.lib.format.open_memmap(str(compact / "target.npy"), mode="w+", dtype=np.int32, shape=(m,))
    amounts = np.lib.format.open_memmap(str(compact / "amount.npy"), mode="w+", dtype=np.float64, shape=(m,))
    stamps = np.lib.format.open_memmap(str(compact / "timestamp.npy"), mode="w+", dtype=np.float64, shape=(m,))
    in_counts = np.zeros(n, dtype=np.int64)
    out_counts = np.zeros(n, dtype=np.int64)
    in_partners = np.zeros(n, dtype=np.int64)
    out_partners = np.zeros(n, dtype=np.int64)
    edge_keys = collections.Counter()
    value_types = {"amount": collections.Counter(), "timestamp": collections.Counter()}
    missing = collections.Counter()
    invalid = collections.Counter()
    negative_amounts = 0
    zero_amounts = 0
    numeric_ranges = {"amount": [math.inf, -math.inf], "timestamp": [math.inf, -math.inf]}
    pairs = loops = parallel_pairs = parallel_extras = 0
    count = 0
    next_progress = 1000000
    for u, neighbors in successors.items():
        uid = ids[u]
        out_partners[uid] = len(neighbors)
        for v, keys in neighbors.items():
            vid = ids[v]
            multiplicity = len(keys)
            if multiplicity == 0:
                raise RuntimeError("Empty endpoint-pair dictionary.")
            if predecessors[v].get(u) is not keys:
                raise RuntimeError("Successor/predecessor edge dictionaries disagree.")
            pairs += 1
            in_partners[vid] += 1
            out_counts[uid] += multiplicity
            in_counts[vid] += multiplicity
            if uid == vid:
                loops += multiplicity
            if multiplicity > 1:
                parallel_pairs += 1
                parallel_extras += multiplicity - 1
            src[count:count + multiplicity] = uid
            dst[count:count + multiplicity] = vid
            for attrs in keys.values():
                edge_keys.update(attrs.keys())
                for name, array in (("amount", amounts), ("timestamp", stamps)):
                    value = attrs.get(name)
                    value_types[name][type(value).__name__] += 1
                    number = math.nan
                    if name not in attrs:
                        missing[name] += 1
                    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                        try:
                            number = float(value)
                        except (ValueError, OverflowError):
                            pass
                        if not math.isfinite(number):
                            invalid[name] += 1
                            number = math.nan
                    else:
                        invalid[name] += 1
                    array[count] = number
                    if math.isfinite(number):
                        numeric_ranges[name][0] = min(numeric_ranges[name][0], number)
                        numeric_ranges[name][1] = max(numeric_ranges[name][1], number)
                        if name == "amount":
                            negative_amounts += int(number < 0)
                            zero_amounts += int(number == 0)
                count += 1
            if count >= next_progress:
                print("EXPORTED: {:,}/{:,} transactions".format(count, m), flush=True)
                next_progress = (count // 1000000 + 1) * 1000000
    if count != m or int(in_counts.sum()) != m or int(out_counts.sum()) != m:
        raise RuntimeError("Transaction-count invariant failed.")
    if pairs + parallel_extras != m or int(in_partners.sum()) != pairs:
        raise RuntimeError("Endpoint-pair invariant failed.")
    isolated = int(np.count_nonzero((in_counts == 0) & (out_counts == 0)))
    for name, arr in (("in_transaction_count", in_counts), ("out_transaction_count", out_counts),
                      ("in_unique_partners", in_partners), ("out_unique_partners", out_partners)):
        np.save(str(compact / (name + ".npy")), arr, allow_pickle=False)
    for array in (src, dst, amounts, stamps):
        array.flush()
    report = {
        "status": "audit_complete", "raw_sha256": actual_sha,
        "environment": {"python": platform.python_version(), "networkx": nx.__version__, "numpy": np.__version__},
        "graph_type": "MultiDiGraph", "directed": True, "node_count": n,
        "transaction_count": m, "ordered_endpoint_pair_count": pairs,
        "self_loop_transaction_count": loops, "parallel_endpoint_pair_count": parallel_pairs,
        "additional_parallel_transactions": parallel_extras, "isolated_node_count": isolated,
        "isp_counts": dict(label_counts), "invalid_label_count": invalid_labels,
        "label_interpretation": "isp=1: released phishing label; isp=0: not labelled phishing, not verified benign; missing/invalid stored as -1.",
        "node_attribute_counts": dict(node_keys), "edge_attribute_counts": dict(edge_keys),
        "edge_value_types": {k: dict(v) for k, v in value_types.items()},
        "missing_edge_values": dict(missing), "invalid_edge_values": dict(invalid),
        "negative_amount_count": negative_amounts, "zero_amount_count": zero_amounts,
        "numeric_ranges": {k: v if math.isfinite(v[0]) else None for k, v in numeric_ranges.items()},
        "proposal_reference_counts": {"nodes": 2973489, "transactions": 13551303, "phishing": 1165},
        "matches_proposal_counts": n == 2973489 and m == 13551303 and label_counts.get("1", 0) == 1165,
        "node_order": "original pickle node dictionary insertion order; JSONL line i maps to node i",
        "address_order_sha256": address_digest.hexdigest(),
        "transaction_order": "original successor, neighbor, edge-key dictionary insertion order",
        "export_semantics": "Every directed transaction retained, including parallel edges and self-loops. Numeric attributes exported as float64; units are not inferred.",
        "schema_usable_for_preprocessing": invalid_labels == 0 and not missing and not invalid and negative_amounts == 0,
        "compact_directory": str(compact.resolve()), "elapsed_seconds": time.monotonic() - started,
        "invariants": {"in_out_transaction_sums": True, "pair_plus_parallel_count": True, "successor_predecessor_identity": True},
    }
    # Verify the exported endpoints independently, in bounded chunks.
    verified_in = np.zeros(n, dtype=np.int64)
    verified_out = np.zeros(n, dtype=np.int64)
    for start in range(0, m, 1000000):
        stop = min(start + 1000000, m)
        verified_out += np.bincount(src[start:stop], minlength=n)
        verified_in += np.bincount(dst[start:stop], minlength=n)
    if not np.array_equal(verified_in, in_counts) or not np.array_equal(verified_out, out_counts):
        raise RuntimeError("Exported endpoint degree checks failed.")
    report["invariants"]["exported_endpoint_degrees"] = True
    report["export_files"] = {}
    print("Checking compact-file fingerprints...", flush=True)
    for path in sorted(compact.iterdir()):
        report["export_files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "graph_audit.json", report)
    print("AUDIT COMPLETE: nodes={} transactions={} phishing={} unlabelled={}".format(
        n, m, label_counts.get("1", 0), label_counts.get("0", 0)), flush=True)
    print("REPORT: " + str(output / "graph_audit.json"), flush=True)
    print("COMPACT DATA: " + str(compact), flush=True)
    return report


def windows_memory():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    return memory.total_phys, memory.avail_phys


def attach_memory_job(process, limit):
    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", ctypes.c_uint32), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                    ("active", ctypes.c_uint32), ("affinity", ctypes.c_size_t),
                    ("priority", ctypes.c_uint32), ("scheduling", ctypes.c_uint32)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", IO)] + [(name, ctypes.c_size_t) for name in
            ("process_limit", "job_limit", "peak_process", "peak_job")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel.SetInformationJobObject.restype = ctypes.c_int
    kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.AssignProcessToJobObject.restype = ctypes.c_int
    kernel.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = Extended()
    info.basic.flags = 0x100 | 0x2000  # PROCESS_MEMORY, KILL_ON_JOB_CLOSE
    info.process_limit = int(limit)
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    if not kernel.AssignProcessToJobObject(job, int(process._handle)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    return kernel, job, Extended


def supervisor(args):
    if os.name != "nt":
        raise RuntimeError("Production supervisor requires Windows; use --self-check for portable fixture validation.")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    raw_file = Path(manifest["raw_file"])
    if not raw_file.is_file() or raw_file.stat().st_size != int(manifest["raw_bytes"]):
        raise RuntimeError("Raw file missing or size changed.")
    total, available = windows_memory()
    reserve = 3 * GIB
    limit = min(26 * GIB, available - reserve)
    if limit < 10 * GIB:
        raise RuntimeError("Less than 13 GiB RAM is available. Close unused applications and retry.")
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(args.output).resolve() / ("ethereum_audit_" + stamp)
    output.mkdir(parents=True)
    print("OUTPUT DIRECTORY: " + str(output), flush=True)
    print("FREE RAM: {:.2f} GiB; WORKER COMMIT LIMIT: {:.2f} GiB".format(available / GIB, limit / GIB), flush=True)
    worker_log = output / "console.log"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--manifest", str(manifest_path), "--output", str(output)]
    job = None
    process = None
    stopped = None
    started = time.monotonic()
    guard = {"initial_available_gib": available / GIB, "total_ram_gib": total / GIB,
             "worker_commit_limit_gib": limit / GIB, "minimum_free_ram_gib": 3.0, "job_limit_attached": False}
    try:
        with worker_log.open("wb") as log, worker_log.open("r", encoding="utf-8", errors="replace") as reader:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            kernel, job, limit_structure = attach_memory_job(process, limit)
            guard["job_limit_attached"] = True
            process.stdin.write(b"GO\n")
            process.stdin.close()
            heartbeat = time.monotonic()
            while process.poll() is None:
                text = reader.read()
                if text:
                    print(text, end="", flush=True)
                _, free = windows_memory()
                if free < reserve:
                    stopped = "System available physical memory fell below 3 GiB."
                    process.terminate()
                    break
                if time.monotonic() - heartbeat >= 15:
                    print("[MONITOR] elapsed={:.0f}s free_ram={:.2f} GiB".format(time.monotonic() - started, free / GIB), flush=True)
                    heartbeat = time.monotonic()
                time.sleep(0.5)
            process.wait()
            print(reader.read(), end="", flush=True)
        guard["worker_exit_code"] = process.returncode
        guard["stopped_reason"] = stopped
    except BaseException as error:
        guard["supervisor_error"] = str(error)
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        if job is not None:
            peak = limit_structure()
            if kernel.QueryInformationJobObject(job, 9, ctypes.byref(peak), ctypes.sizeof(peak), None):
                guard["peak_process_commit_gib"] = peak.peak_process / GIB
            kernel.CloseHandle(job)
        guard["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "memory_guard.json", guard)
        evidence = output / "ethereum_audit_evidence.zip"
        with zipfile.ZipFile(evidence, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in ("graph_audit.json", "memory_guard.json", "console.log", "worker_error.json"):
                path = output / name
                if path.exists():
                    z.write(path, name)
            z.write(manifest_path, "acquisition_manifest.json")
            preflight = manifest_path.parent / "ethereum_preflight.json"
            if preflight.is_file():
                z.write(preflight, "ethereum_preflight.json")
            z.write(Path(__file__).resolve(), "audit_ethereum.py")
        print("EVIDENCE ZIP: " + str(evidence), flush=True)
    if stopped or process.returncode != 0 or not (output / "graph_audit.json").exists():
        raise RuntimeError("Audit did not complete. Send the evidence ZIP and output; partial compact files must not be used.")
    print("SUPERVISED AUDIT: PASS", flush=True)


def self_check():
    import networkx as nx
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        graph = nx.MultiDiGraph()
        graph.add_nodes_from([("A", {"isp": 1}), ("B", {"isp": 0}), ("C", {}), ("D", {"isp": 0})])
        graph.add_edge("A", "B", amount=0, timestamp=10)
        graph.add_edge("A", "B", amount=2, timestamp=11)
        graph.add_edge("B", "A", amount=3, timestamp=12)
        graph.add_edge("A", "A", amount=5, timestamp=13)
        raw = root / "graph.pkl"
        with raw.open("wb") as f:
            pickle.dump(graph, f, protocol=3)
        result = audit_export(raw, root, sha256(raw))
        assert result["node_count"] == 4 and result["transaction_count"] == 4
        assert result["ordered_endpoint_pair_count"] == 3
        assert result["additional_parallel_transactions"] == 1
        assert result["self_loop_transaction_count"] == 1
        assert result["isolated_node_count"] == 2
        compact = root / "compact"
        assert np.load(str(compact / "labels.npy")).tolist() == [1, 0, -1, 0]
        exported = list(zip(np.load(str(compact / "source.npy")).tolist(), np.load(str(compact / "target.npy")).tolist(),
                            np.load(str(compact / "amount.npy")).tolist(), np.load(str(compact / "timestamp.npy")).tolist()))
        assert exported == [(0, 1, 0, 10), (0, 1, 2, 11), (0, 0, 5, 13), (1, 0, 3, 12)]
        assert result["isp_counts"] == {"1": 1, "0": 2, "missing": 1}
    print("SELF-CHECK PASS: directed transactions, parallel edges, self-loops, labels, node IDs, endpoint degree verification.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--output", default="audit_results")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.manifest:
        parser.error("--manifest is required")
    if args.worker:
        if sys.stdin.readline().strip() != "GO":
            raise RuntimeError("Worker requires the supervisor handshake.")
        output = Path(args.output)
        try:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
            audit_export(Path(manifest["raw_file"]), output, manifest["raw_sha256"])
        except BaseException as error:
            write_json(output / "worker_error.json", {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
            raise
    else:
        supervisor(args)


if __name__ == "__main__":
    main()
