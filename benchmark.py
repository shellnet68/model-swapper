#!/usr/bin/env python3
"""Compare local-file reloads with pinned-RAM switching on the same backend.

Offline local-checkpoint benchmark. Keeps the OS file cache and uses one
Python process; it does not measure a separate model-serving engine.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from warm_cache import WarmCache

MIB = 1024 * 1024
PROMPT = "The capital of France is"


def load_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).eval().requires_grad_(False)


class Reload:
    """Rebuild and load a model on every alternating request."""
    def __init__(self, paths):
        self.paths = paths
        self.model = None

    def release(self):
        if self.model is not None:
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

    @contextmanager
    def acquire(self, name):
        self.release()
        self.model = load_model(self.paths[name]).to("cuda")
        torch.cuda.synchronize()
        with torch.inference_mode():
            yield self.model

    def close(self):
        self.release()


def rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return None


def guard_headroom():
    free, _ = torch.cuda.mem_get_info()
    if free < 4096 * MIB:
        raise RuntimeError("Less than 4 GiB GPU memory free; stopping shared-host experiment")


def infer(model, inputs, count, pad_id):
    return model.generate(
        **inputs, do_sample=False, max_new_tokens=count, min_new_tokens=count,
        use_cache=True, pad_token_id=pad_id,
    )


def measure(manager, name, inputs, tokenizer, reference, method, reference_logits, *, expected_warm=False):
    guard_headroom()
    source = manager.state(name) if isinstance(manager, WarmCache) else "reload"
    if expected_warm and source != "warm":
        raise AssertionError(f"{method}/{name}: expected warm weights, found {source}; refusing mislabeled timing")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with manager.acquire(name) as model:
        torch.cuda.synchronize()
        ready = time.perf_counter()
        one = infer(model, inputs, 1, tokenizer.eos_token_id)
        torch.cuda.synchronize()
        first = time.perf_counter()
        # Separate eight-token probe; do not count it as the same request.
        gen_start = time.perf_counter()
        output = infer(model, inputs, 8, tokenizer.eos_token_id)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        ids = output[0].tolist()
        first_ids = one[0].tolist()
        # Outside timed probes: stronger than matching only argmax tokens.
        logits = model(**inputs).logits[:, -1].cpu()
        torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
    if ids != reference or first_ids[-1] != reference[inputs["input_ids"].shape[1]]:
        raise AssertionError(f"{method}/{name}: output differs from normal loaded model")
    record = {
        "method": method, "model": name,
        "source": source,
        "ready_ms": (ready - started) * 1000,
        "ready_plus_one_token_ms": (first - started) * 1000,
        "one_token_ms": (first - ready) * 1000,
        "eight_token_ms": (finished - gen_start) * 1000,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "rss_mib": rss_mib(),
        "pinned_weight_mib": manager.ram_bytes / MIB if isinstance(manager, WarmCache) else 0,
        "pinned_allocator_estimate_mib": manager.pinned_allocator_estimate_bytes / MIB if isinstance(manager, WarmCache) else 0,
        "correct": True, "output_ids": ids,
        "logits_equal": True,
        "completion": tokenizer.decode(ids[inputs["input_ids"].shape[1]:]),
    }
    # Drop the caller's model reference before the next switch.
    del model, one, output
    return record


def summary(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["method"], row["model"])].append(row)
    result = []
    for (method, model), rows in groups.items():
        row = {"method": method, "model": model, "n": len(rows)}
        for metric in ("ready_ms", "ready_plus_one_token_ms", "eight_token_ms"):
            values = sorted(r[metric] for r in rows)
            row[metric + "_p50"] = statistics.median(values)
            row[metric + "_p95"] = values[math.ceil(0.95 * len(values)) - 1]
        row["peak_allocated_mib"] = max(r["peak_allocated_mib"] for r in rows)
        row["peak_reserved_mib"] = max(r["peak_reserved_mib"] for r in rows)
        row["max_rss_mib"] = max(r["rss_mib"] for r in rows)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs=2, action="append", metavar=("NAME", "LOCAL_PATH"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu-budget-mib", type=int, default=768)
    parser.add_argument("--ram-budget-mib", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-port", type=int, help="Expose cache metrics on localhost during each warm method")
    parser.add_argument("--metrics-dir", type=Path, help="Directory for per-method Prometheus snapshots")
    args = parser.parse_args()
    if len(args.model) != 2 or len({name for name, _ in args.model}) != 2 or args.repeats < 1:
        parser.error("Provide two different model names and at least one repeat")
    if args.output.exists():
        parser.error("Output already exists; choose a new file to preserve prior measurements")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.metrics_port is not None and not 1 <= args.metrics_port <= 65535:
        parser.error("Metrics port must be between 1 and 65535")
    metrics_dir = args.metrics_dir or args.output.parent / (args.output.stem + "-metrics")
    if metrics_dir.exists():
        parser.error("Metrics directory already exists; preserve the earlier evidence")
    metrics_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    # Caps this process's PyTorch allocator, not all CUDA/context memory.
    torch.cuda.set_per_process_memory_fraction(0.05)
    paths = {name: str(Path(path).resolve()) for name, path in args.model}
    tokenizers, inputs, references = {}, {}, {}
    reference_logits = {}
    identities = {}
    for name, path in paths.items():
        p = Path(path)
        files = {}
        for weight_file in sorted(p.glob("*.safetensors")):
            digest = hashlib.sha256()
            with weight_file.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
            files[weight_file.name] = digest.hexdigest()
        identities[name] = {"path": path, "safetensors_sha256": files}
        tokenizers[name] = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        inputs[name] = {k: v.to("cuda") for k, v in tokenizers[name](PROMPT, return_tensors="pt").items()}
    # Independent references come from normal checkpoint loading, not our cache.
    baseline = Reload(paths)
    try:
        for name in paths:
            guard_headroom()
            with baseline.acquire(name) as model:
                references[name] = infer(model, inputs[name], 8, tokenizers[name].eos_token_id)[0].tolist()
                reference_logits[name] = model(**inputs[name]).logits[:, -1].cpu()
                torch.cuda.synchronize()
            del model
    finally:
        baseline.close()
    records, metric_snapshots = [], {}
    methods = (("reload", None, True), ("warm_release", 1, True),
               ("warm_reuse", 1, False), ("warm_reuse_2_streams", 2, False))
    for method, streams, release_on_switch in methods:
        manager = Reload(paths) if streams is None else WarmCache(
            gpu_budget_bytes=args.gpu_budget_mib * MIB,
            ram_budget_bytes=args.ram_budget_mib * MIB, streams=streams,
            release_on_switch=release_on_switch,
        )
        metrics_server = metrics_thread = None
        try:
            if isinstance(manager, WarmCache):
                for name, path in paths.items():
                    manager.register(name, lambda path=path: load_model(path))
                if args.metrics_port is not None:
                    from prometheus_client import start_http_server
                    metrics_server, metrics_thread = start_http_server(
                        args.metrics_port, addr="127.0.0.1", registry=manager.metrics.registry,
                    )
                    print("METRICS_READY", method, args.metrics_port, flush=True)
            # Untimed calibration primes CUDA kernels and both models/RAM copies.
            for name in paths:
                measure(manager, name, inputs[name], tokenizers[name], references[name], method, reference_logits[name])
            for repeat in range(args.repeats):
                for name in paths:
                    row = measure(manager, name, inputs[name], tokenizers[name], references[name], method,
                                  reference_logits[name], expected_warm=isinstance(manager, WarmCache))
                    row["repeat"] = repeat
                    records.append(row)
                    print(json.dumps(row), flush=True)
        finally:
            try:
                manager.close()
                if isinstance(manager, WarmCache):
                    snapshot = metrics_dir / (method + ".prom")
                    snapshot.write_bytes(manager.metrics.render())
                    metric_snapshots[method] = str(snapshot)
            finally:
                if metrics_server is not None:
                    metrics_server.shutdown()
                    metrics_server.server_close()
                    metrics_thread.join()
                del manager
                gc.collect()
                torch.cuda.empty_cache()
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                     "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
                     "transformers": version("transformers"),
                     "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                "--format=csv,noheader"], text=True, timeout=5).strip()},
        "configuration": {"dtype": "float32", "attention": "eager", "prompt": PROMPT,
                          "generated_tokens": 8, "repeats": args.repeats, "order": list(paths),
                          "gpu_weight_budget_mib": args.gpu_budget_mib, "ram_weight_budget_mib": args.ram_budget_mib,
                          "methods": [m[0] for m in methods],
                          "allocator_fraction": 0.05, "cpu_threads": 2},
        "limits": ["OS file cache is allowed; this is not a physical cold-disk measurement",
                   "Not a direct llama-swap benchmark; same-process Transformers reload baseline",
                   "One-token and eight-token probes are separate generations",
                   "Fixed method order after calibration; not randomised or a statistical significance test",
                   "Weight budgets exclude CUDA context, activations and allocator overhead"],
        "models": identities, "reference_ids": references,
        "reference_logits_sha256": {name: hashlib.sha256(t.numpy().tobytes()).hexdigest()
                                    for name, t in reference_logits.items()},
        "metric_snapshots": metric_snapshots,
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in ([Path(__file__).with_name(name) for name in
                          ("warm_cache.py", "metrics.py", "benchmark.py", "pyproject.toml", "uv.lock", "ARCHITECTURE.md")]
                         + sorted(Path(__file__).parent.joinpath("tests").glob("test_*.py")))
        },
        "records": records, "summary": summary(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
