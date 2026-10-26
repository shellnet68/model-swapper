"""Prometheus instrumentation for the model-swapper warm cache."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_ACTIVATIONS_TOTAL = "model_swapper_activations_total"
_ACTIVATION_DURATION = "model_swapper_activation_duration_seconds"
_REJECTIONS_TOTAL = "model_swapper_rejections_total"
_GPU_WEIGHT = "model_swapper_gpu_weight_bytes"
_PINNED_WEIGHT = "model_swapper_pinned_weight_bytes"
_GPU_WEIGHT_BUDGET = "model_swapper_gpu_weight_budget_bytes"
_PINNED_WEIGHT_BUDGET = "model_swapper_pinned_weight_budget_bytes"
_MODELS = "model_swapper_models"
_LEASES_ACTIVE = "model_swapper_leases_active"
_PINNED_ALLOCATOR_ESTIMATE = "model_swapper_pinned_allocator_estimate_bytes"
_PROCESS_CUDA_ALLOCATED = "model_swapper_process_cuda_allocated_bytes"
_PROCESS_CUDA_RESERVED = "model_swapper_process_cuda_reserved_bytes"

_PROCESS_CUDA_HELP = (
    "Current process PyTorch CUDA allocator statistics only; "
    "not NVML, driver context, or non-PyTorch GPU memory."
)

_PINNED_ALLOCATOR_HELP = (
    "Estimated active pinned allocation bytes rounded per PyTorch 2.5 host "
    "allocator, excludes cached freed blocks and page granularity; not physical usage."
)

_TIERS = ("cold", "warm", "hot")
_LABEL_MODEL = "model"
_LABEL_SOURCE = "source"
_LABEL_OUTCOME = "outcome"
_LABEL_REASON = "reason"
_LABEL_TIER = "tier"


def _activation_labels(model: str, source: str, outcome: str) -> dict[str, str]:
    return {_LABEL_MODEL: model, _LABEL_SOURCE: source, _LABEL_OUTCOME: outcome}


class CacheMetrics:
    def __init__(self, gpu_budget_bytes: int, ram_budget_bytes: int):
        self.gpu_budget_bytes = gpu_budget_bytes
        self.ram_budget_bytes = ram_budget_bytes
        self.registry = CollectorRegistry()

        self._activations_total = Counter(
            _ACTIVATIONS_TOTAL,
            "Readiness attempts, split by cold/warm/hot source and success/error.",
            [_LABEL_MODEL, _LABEL_SOURCE, _LABEL_OUTCOME],
            registry=self.registry,
        )
        self._activation_duration = Histogram(
            _ACTIVATION_DURATION,
            "Time to make a model ready, excluding caller inference.",
            [_LABEL_MODEL, _LABEL_SOURCE, _LABEL_OUTCOME],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300),
        )
        self._rejections_total = Counter(
            _REJECTIONS_TOTAL,
            "Rejected operations by bounded reason: busy, unknown, or admin_busy.",
            [_LABEL_REASON],
            registry=self.registry,
        )
        self._gpu_weight = Gauge(
            _GPU_WEIGHT,
            "Logical unique tensor bytes of the hot model.",
            registry=self.registry,
        )
        self._pinned_weight = Gauge(
            _PINNED_WEIGHT,
            "Logical retained pinned tensor bytes.",
            registry=self.registry,
        )
        self._gpu_weight_budget = Gauge(
            _GPU_WEIGHT_BUDGET,
            "Configured logical GPU tensor budget.",
            registry=self.registry,
        )
        self._pinned_weight_budget = Gauge(
            _PINNED_WEIGHT_BUDGET,
            "Configured logical pinned tensor budget.",
            registry=self.registry,
        )
        self._models = Gauge(
            _MODELS,
            "Registered entries in each state.",
            [_LABEL_TIER],
            registry=self.registry,
        )
        self._leases_active = Gauge(
            _LEASES_ACTIVE,
            "Whether a caller currently holds the lease.",
            registry=self.registry,
        )
        self._pinned_allocator_estimate = Gauge(
            _PINNED_ALLOCATOR_ESTIMATE,
            _PINNED_ALLOCATOR_HELP,
            registry=self.registry,
        )
        self._process_cuda_allocated = Gauge(
            _PROCESS_CUDA_ALLOCATED,
            _PROCESS_CUDA_HELP,
            registry=self.registry,
        )
        self._process_cuda_reserved = Gauge(
            _PROCESS_CUDA_RESERVED,
            _PROCESS_CUDA_HELP,
            registry=self.registry,
        )

        self._gpu_weight_budget.set(gpu_budget_bytes)
        self._pinned_weight_budget.set(ram_budget_bytes)

    def activation(self, model: str, source: str, outcome: str, seconds: float) -> None:
        labels = _activation_labels(model, source, outcome)
        self._activations_total.labels(**labels).inc()
        self._activation_duration.labels(**labels).observe(seconds)

    def reject(self, reason: str) -> None:
        self._rejections_total.labels(**{_LABEL_REASON: reason}).inc()

    def update(
        self,
        states: list[str],
        gpu_bytes: int,
        pinned_bytes: int,
        active: bool,
        *,
        pinned_allocator_estimate_bytes: int = 0,
        cuda_allocated_bytes: int = 0,
        cuda_reserved_bytes: int = 0,
    ) -> None:
        counts = {tier: 0 for tier in _TIERS}
        for state in states:
            counts[state] = counts.get(state, 0) + 1
        for tier in _TIERS:
            self._models.labels(**{_LABEL_TIER: tier}).set(counts[tier])
        self._gpu_weight.set(gpu_bytes)
        self._pinned_weight.set(pinned_bytes)
        self._leases_active.set(1 if active else 0)
        self._pinned_allocator_estimate.set(pinned_allocator_estimate_bytes)
        self._process_cuda_allocated.set(cuda_allocated_bytes)
        self._process_cuda_reserved.set(cuda_reserved_bytes)

    def render(self) -> bytes:
        return generate_latest(self.registry)
