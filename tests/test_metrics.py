"""Prometheus metrics for WarmCache — real CUDA, no mocked transfers.

See ARCHITECTURE.md Metrics table for the model_swapper_* contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from prometheus_client.parser import text_string_to_metric_families
from torch import nn

from metrics import CacheMetrics
from warm_cache import BusyError, BudgetError, UnknownModelError, WarmCache

ACTIVATIONS_TOTAL = "model_swapper_activations_total"
ACTIVATION_DURATION = "model_swapper_activation_duration_seconds"
REJECTIONS_TOTAL = "model_swapper_rejections_total"
GPU_WEIGHT = "model_swapper_gpu_weight_bytes"
PINNED_WEIGHT = "model_swapper_pinned_weight_bytes"
GPU_WEIGHT_BUDGET = "model_swapper_gpu_weight_budget_bytes"
PINNED_WEIGHT_BUDGET = "model_swapper_pinned_weight_budget_bytes"
MODELS = "model_swapper_models"
LEASES_ACTIVE = "model_swapper_leases_active"
PINNED_ALLOCATOR_ESTIMATE = "model_swapper_pinned_allocator_estimate_bytes"
CUDA_ALLOCATED = "model_swapper_process_cuda_allocated_bytes"
CUDA_RESERVED = "model_swapper_process_cuda_reserved_bytes"

UNKNOWN_MODEL = "not-a-registered-model"


class Affine(nn.Module):
    def __init__(self, weights, bias, offset):
        super().__init__()
        self.weight = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, x):
        return x @ self.weight.T + self.bias + self.offset


@pytest.fixture(autouse=True)
def cuda_required():
    assert torch.cuda.is_available(), "Run these tests in the CUDA lab, not with fake transfers"
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.05)


@pytest.fixture
def metrics():
    return CacheMetrics(gpu_budget_bytes=16, ram_budget_bytes=32)


def parse_render(blob: bytes):
    return list(text_string_to_metric_families(blob.decode()))


def sample_map(families):
    out: dict[tuple[str, frozenset], float] = {}
    for family in families:
        for sample in family.samples:
            key = (sample.name, frozenset(sample.labels.items()))
            out[key] = sample.value
    return out


def sum_counter(metric_name: str, match: dict[str, str], families) -> float:
    total = 0.0
    required = frozenset(match.items())
    for (name, labels), value in sample_map(families).items():
        if name != metric_name:
            continue
        if required <= labels:
            total += value
    return total


def sum_histogram_count(match: dict[str, str], families) -> float:
    return sum_counter(f"{ACTIVATION_DURATION}_count", match, families)


def gauge(name: str, labels: dict[str, str] | None, families) -> float:
    label_set = frozenset((labels or {}).items())
    return sample_map(families).get((name, label_set), 0.0)


def checkpoints(tmp_path: Path):
    paths = {}
    for name, weight, bias, offset in (
        ("a", [[2.0, 3.0]], [5.0], [7.0]),
        ("b", [[4.0, 1.0]], [-1.0], [2.0]),
    ):
        path = tmp_path / f"{name}.pt"
        torch.save(
            {"weights": torch.tensor(weight), "bias": torch.tensor(bias), "offset": torch.tensor(offset)},
            path,
        )
        paths[name] = path
    return paths


def register(cache: WarmCache, name: str, path: Path):
    def load():
        return Affine(**torch.load(path, weights_only=True, map_location="cpu"))

    cache.register(name, load)


def answer(cache: WarmCache, name: str) -> float:
    with cache.acquire(name) as model:
        return model(torch.tensor([[1.0, 2.0]], device="cuda")).item()


def process_cuda_bytes():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def cache_with_metrics(metrics: CacheMetrics, **kwargs):
    kwargs.setdefault("gpu_budget_bytes", metrics.gpu_budget_bytes)
    kwargs.setdefault("ram_budget_bytes", metrics.ram_budget_bytes)
    return WarmCache(metrics=metrics, **kwargs)


def test_registry_is_isolated(metrics):
    other = CacheMetrics(gpu_budget_bytes=8, ram_budget_bytes=16)
    metrics.activation("a", "cold", "success", 0.01)
    fam = parse_render(metrics.render())
    assert sum_counter(
        ACTIVATIONS_TOTAL,
        {"model": "a", "source": "cold", "outcome": "success"},
        fam,
    ) == 1.0
    assert (
        sum_counter(
            ACTIVATIONS_TOTAL,
            {"model": "a", "source": "cold", "outcome": "success"},
            parse_render(other.render()),
        )
        == 0.0
    )
    assert metrics.registry is not other.registry


def test_unknown_acquire_rejects_without_exporting_model_name(tmp_path, metrics):
    paths = checkpoints(tmp_path)
    cache = cache_with_metrics(metrics)
    try:
        register(cache, "a", paths["a"])
        with pytest.raises(UnknownModelError):
            with cache.acquire(UNKNOWN_MODEL):
                pytest.fail("unknown model must be rejected before activation")
        text = metrics.render().decode()
        assert UNKNOWN_MODEL not in text
        fam = parse_render(metrics.render())
        assert sum_counter(REJECTIONS_TOTAL, {"reason": "unknown"}, fam) == 1.0
        assert sum_counter(ACTIVATIONS_TOTAL, {"model": UNKNOWN_MODEL}, fam) == 0.0
    finally:
        cache.close()


def test_cold_warm_hot_activation_and_histogram_counts(tmp_path, metrics):
    paths = checkpoints(tmp_path)
    cache = cache_with_metrics(metrics)
    try:
        register(cache, "a", paths["a"])
        register(cache, "b", paths["b"])

        assert answer(cache, "a") == 20.0
        fam = parse_render(metrics.render())
        assert sum_counter(ACTIVATIONS_TOTAL, {"model": "a", "source": "cold", "outcome": "success"}, fam) == 1.0
        assert sum_histogram_count({"model": "a", "source": "cold", "outcome": "success"}, fam) == 1.0

        assert answer(cache, "b") == 7.0
        fam = parse_render(metrics.render())
        assert sum_counter(ACTIVATIONS_TOTAL, {"model": "b", "source": "cold", "outcome": "success"}, fam) == 1.0
        assert sum_counter(ACTIVATIONS_TOTAL, {"source": "warm", "outcome": "success"}, fam) == 0.0

        assert answer(cache, "a") == 20.0
        fam = parse_render(metrics.render())
        assert sum_counter(ACTIVATIONS_TOTAL, {"model": "a", "source": "warm", "outcome": "success"}, fam) == 1.0
        assert sum_histogram_count({"model": "a", "source": "warm", "outcome": "success"}, fam) == 1.0

        with cache.acquire("a") as model:
            assert model(torch.tensor([[1.0, 2.0]], device="cuda")).item() == 20.0
        fam = parse_render(metrics.render())
        assert sum_counter(ACTIVATIONS_TOTAL, {"model": "a", "source": "hot", "outcome": "success"}, fam) == 1.0
        assert sum_histogram_count({"model": "a", "source": "hot", "outcome": "success"}, fam) == 1.0
    finally:
        cache.close()


def test_busy_rejection_increments_rejections_total(tmp_path, metrics):
    paths = checkpoints(tmp_path)
    cache = cache_with_metrics(metrics)
    try:
        register(cache, "a", paths["a"])
        register(cache, "b", paths["b"])
        with cache.acquire("a"):
            with pytest.raises(BusyError):
                with cache.acquire("b"):
                    pytest.fail("busy lease must reject, not evict")
        fam = parse_render(metrics.render())
        assert sum_counter(REJECTIONS_TOTAL, {"reason": "busy"}, fam) == 1.0
        assert sum_counter(ACTIVATIONS_TOTAL, {"model": "a", "source": "cold", "outcome": "success"}, fam) >= 1.0
    finally:
        cache.close()


def test_activation_budget_failure_counts(tmp_path):
    metrics = CacheMetrics(gpu_budget_bytes=15, ram_budget_bytes=32)
    paths = checkpoints(tmp_path)
    cache = cache_with_metrics(metrics)
    try:
        register(cache, "a", paths["a"])
        with pytest.raises(BudgetError):
            answer(cache, "a")
        fam = parse_render(metrics.render())
        assert sum_counter(
            ACTIVATIONS_TOTAL,
            {"model": "a", "source": "cold", "outcome": "error"},
            fam,
        ) == 1.0
        assert sum_histogram_count({"model": "a", "source": "cold", "outcome": "error"}, fam) == 1.0
        assert sum_counter(ACTIVATIONS_TOTAL, {"model": "a", "source": "cold", "outcome": "success"}, fam) == 0.0
        assert gauge(GPU_WEIGHT, None, fam) == 0.0
    finally:
        cache.close()


def test_weight_tier_lease_and_allocator_estimate_via_real_cache(tmp_path, metrics):
    paths = checkpoints(tmp_path)
    torch.cuda.empty_cache()
    baseline_allocated, baseline_reserved = process_cuda_bytes()
    cache = cache_with_metrics(metrics)
    try:
        register(cache, "a", paths["a"])
        register(cache, "b", paths["b"])
        fam = parse_render(metrics.render())
        assert gauge(GPU_WEIGHT_BUDGET, None, fam) == 16.0
        assert gauge(PINNED_WEIGHT_BUDGET, None, fam) == 32.0

        answer(cache, "a")
        with cache.acquire("b") as model:
            fam = parse_render(metrics.render())
            logical_hot = gauge(GPU_WEIGHT, None, fam)
            allocated = gauge(CUDA_ALLOCATED, None, fam)
            reserved = gauge(CUDA_RESERVED, None, fam)
            assert logical_hot == 16.0
            assert reserved >= allocated >= logical_hot
            assert allocated == process_cuda_bytes()[0]
            assert reserved == process_cuda_bytes()[1]
            assert gauge(PINNED_WEIGHT, None, fam) == 32.0
            estimate = gauge(PINNED_ALLOCATOR_ESTIMATE, None, fam)
            assert estimate >= gauge(PINNED_WEIGHT, None, fam)
            assert gauge(LEASES_ACTIVE, None, fam) == 1.0
            assert gauge(MODELS, {"tier": "hot"}, fam) == 1.0
            assert gauge(MODELS, {"tier": "warm"}, fam) == 1.0
            assert gauge(MODELS, {"tier": "cold"}, fam) == 0.0
            assert model(torch.tensor([[1.0, 2.0]], device="cuda")).item() == 7.0

        fam = parse_render(metrics.render())
        assert gauge(LEASES_ACTIVE, None, fam) == 0.0
        assert gauge(GPU_WEIGHT, None, fam) == 16.0
        assert gauge(PINNED_WEIGHT, None, fam) == 32.0

        cache.close()
        fam = parse_render(metrics.render())
        assert gauge(LEASES_ACTIVE, None, fam) == 0.0
        assert gauge(GPU_WEIGHT, None, fam) == 0.0
        assert gauge(PINNED_WEIGHT, None, fam) == 0.0
        assert gauge(PINNED_ALLOCATOR_ESTIMATE, None, fam) == 0.0
        assert gauge(CUDA_ALLOCATED, None, fam) == baseline_allocated
        assert gauge(CUDA_RESERVED, None, fam) == baseline_reserved
        assert gauge(MODELS, {"tier": "warm"}, fam) == 0.0
        assert gauge(MODELS, {"tier": "hot"}, fam) == 0.0
        assert gauge(MODELS, {"tier": "cold"}, fam) == 2.0
    finally:
        cache.close()
