"""Real CUDA tests with local fixtures and actual model transfers."""
from pathlib import Path

import pytest
import torch
from torch import nn

from warm_cache import BusyError, BudgetError, WarmCache


class Affine(nn.Module):
    def __init__(self, weights, bias, offset):
        super().__init__()
        self.weight = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        # Real transformer models also have non-persistent rotary buffers.
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, x):
        return x @ self.weight.T + self.bias + self.offset


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 1, bias=False)
        self.second = nn.Linear(2, 1, bias=False)
        self.first.weight.data.copy_(torch.tensor([[2.0, 3.0]]))
        self.second.weight = self.first.weight

    def forward(self, x):
        return self.first(x) + self.second(x)


@pytest.fixture(autouse=True)
def cuda_required():
    assert torch.cuda.is_available(), "Run these tests in the CUDA lab, not with fake transfers"
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.05)


def checkpoints(tmp_path):
    paths = {}
    for name, weight, bias, offset in (
        ("a", [[2.0, 3.0]], [5.0], [7.0]),
        ("b", [[4.0, 1.0]], [-1.0], [2.0]),
        ("c", [[1.0, 1.0]], [0.0], [0.0]),
    ):
        path = tmp_path / f"{name}.pt"
        torch.save({"weights": torch.tensor(weight), "bias": torch.tensor(bias), "offset": torch.tensor(offset)}, path)
        paths[name] = path
    return paths


def register(cache, name, path):
    def load():
        return Affine(**torch.load(path, weights_only=True, map_location="cpu"))
    cache.register(name, load)


def answer(cache, name):
    with cache.acquire(name) as model:
        return model(torch.tensor([[1.0, 2.0]], device="cuda")).item()


def test_warm_switch_keeps_outputs_and_nonpersistent_buffers_without_checkpoint(tmp_path):
    """Reloading from disk or losing the rotary-style buffer breaks this."""
    paths = checkpoints(tmp_path)
    cache = WarmCache(gpu_budget_bytes=16, ram_budget_bytes=32)
    try:
        register(cache, "a", paths["a"])
        register(cache, "b", paths["b"])
        assert answer(cache, "a") == 20.0
        paths["a"].unlink()
        assert answer(cache, "b") == 7.0
        assert cache.state("a") == "warm"
        assert answer(cache, "a") == 20.0
        assert cache.state("b") == "warm"
        assert cache.gpu_bytes == 16
        assert cache.ram_bytes == 32
    finally:
        cache.close()


def test_ram_budget_drops_least_recently_used_checkpoint(tmp_path):
    """Retaining the wrong waiting model causes a missing-file error on B."""
    paths = checkpoints(tmp_path)
    cache = WarmCache(gpu_budget_bytes=16, ram_budget_bytes=32)
    try:
        for name, path in paths.items():
            register(cache, name, path)
        assert answer(cache, "a") == 20.0
        assert answer(cache, "b") == 7.0
        paths["b"].unlink()
        assert answer(cache, "c") == 3.0
        assert cache.state("a") == "cold"
        assert answer(cache, "b") == 7.0
        paths["a"].unlink()
        with pytest.raises(FileNotFoundError):
            answer(cache, "a")
        assert cache.ram_bytes <= 32
    finally:
        cache.close()


def test_active_lease_is_never_evicted(tmp_path):
    """A competing request must not replace weights used by the current caller."""
    paths = checkpoints(tmp_path)
    cache = WarmCache(gpu_budget_bytes=16, ram_budget_bytes=32)
    try:
        register(cache, "a", paths["a"])
        register(cache, "b", paths["b"])
        with cache.acquire("a") as model:
            with pytest.raises(BusyError):
                with cache.acquire("b"):
                    pytest.fail("A busy model was evicted")
            assert model(torch.tensor([[1.0, 2.0]], device="cuda")).item() == 20.0
        assert answer(cache, "b") == 7.0
    finally:
        cache.close()


@pytest.mark.parametrize("gpu,ram", [(15, 32), (16, 15)])
def test_model_over_budget_is_rejected_before_any_gpu_allocation(tmp_path, gpu, ram):
    """Omitting buffer bytes or using > instead of >= breaks the boundary."""
    paths = checkpoints(tmp_path)
    cache = WarmCache(gpu_budget_bytes=gpu, ram_budget_bytes=ram)
    try:
        register(cache, "a", paths["a"])
        with pytest.raises(BudgetError):
            answer(cache, "a")
        assert cache.gpu_bytes == 0
        assert cache.ram_bytes == 0
        assert cache.state("a") == "cold"
    finally:
        cache.close()


def test_tied_weights_fit_once_and_survive_warm_restore():
    """Duplicating tied embeddings breaks the byte budget or parameter sharing."""
    cache = WarmCache(gpu_budget_bytes=8, ram_budget_bytes=16)
    try:
        cache.register("a", Tied)
        cache.register("b", Tied)
        assert answer(cache, "a") == 16.0
        assert answer(cache, "b") == 16.0
        with cache.acquire("a") as model:
            assert model.first.weight is model.second.weight
            assert model(torch.tensor([[1.0, 2.0]], device="cuda")).item() == 16.0
        assert cache.gpu_bytes == 8
        assert cache.ram_bytes == 16
    finally:
        cache.close()
