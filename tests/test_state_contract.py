"""State-binding, allocator and cleanup regression cases."""
import threading

import pytest
import torch
from torch import nn
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM

from warm_cache import BusyError, UnsupportedModelError, WarmCache
from metrics import CacheMetrics


@pytest.fixture(autouse=True)
def cuda():
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(0.05)
    torch.set_num_threads(2)


class ParameterBufferAlias(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("mirror", self.weight, persistent=False)

    def forward(self):
        return (self.weight + self.mirror).sum()


class ShadowAlias(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0]))
        self.original = self.scale

    def forward(self, x):
        return x * self.scale + self.original


def test_buffer_alias_keeps_registration_and_persistence():
    cache = WarmCache(gpu_budget_bytes=16, ram_budget_bytes=32)
    cache.register("a", ParameterBufferAlias)
    cache.register("b", lambda: nn.Linear(1, 1))
    try:
        for _ in range(2):
            with cache.acquire("a") as model:
                assert dict(model.named_parameters()).keys() == {"weight"}
                assert dict(model.named_buffers()).keys() == {"mirror"}
                assert model.state_dict().keys() == {"weight"}
                assert "mirror" in model._non_persistent_buffers_set
                assert model.mirror.data_ptr() == model.weight.data_ptr()
                assert model().item() == 6.0
            with cache.acquire("b"):
                pass
    finally:
        cache.close()


def test_plain_tensor_alias_moves_with_registered_buffer():
    cache = WarmCache(gpu_budget_bytes=16, ram_budget_bytes=32)
    cache.register("a", ShadowAlias)
    cache.register("b", lambda: nn.Linear(1, 1))
    try:
        for _ in range(2):
            with cache.acquire("a") as model:
                assert model.original.device.type == "cuda"
                assert model.original.data_ptr() == model.scale.data_ptr()
                assert model(torch.ones(1, device="cuda")).item() == 4.0
            with cache.acquire("b"):
                assert model.original.device.type == "meta"
    finally:
        cache.close()


@pytest.mark.parametrize("container", [False, True])
def test_unregistered_tensor_state_is_rejected(container):
    def load():
        model = nn.Linear(1, 1)
        value = torch.ones(1)
        model.hidden = {"nested": [value]} if container else value
        return model
    cache = WarmCache(gpu_budget_bytes=128, ram_budget_bytes=128)
    cache.register("a", load)
    try:
        with pytest.raises(UnsupportedModelError):
            with cache.acquire("a"):
                pytest.fail("Unregistered tensor state was accepted")
        assert cache.state("a") == "cold"
        assert cache.ram_bytes == 0
    finally:
        cache.close()


def dynamic_llama():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                   num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                   max_position_embeddings=16, rope_scaling={"rope_type": "dynamic", "factor": 2.0},
                   bos_token_id=1, eos_token_id=2, pad_token_id=0))


def test_dynamic_rope_change_discards_snapshot_and_reloads_fresh_state():
    cache = WarmCache(gpu_budget_bytes=2**20, ram_budget_bytes=2**21)
    cache.register("a", dynamic_llama)
    cache.register("b", lambda: nn.Linear(1, 1))
    try:
        with cache.acquire("a") as model:
            model(torch.arange(24, device="cuda").reshape(1, 24) % 32)
        with cache.acquire("b"):
            pass
        assert cache.state("a") == "cold"
        inputs = torch.arange(20).reshape(1, 20) % 32
        with torch.inference_mode():
            expected = dynamic_llama().eval()(inputs).logits
        with cache.acquire("a") as model:
            actual = model(inputs.to("cuda")).logits.cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    finally:
        cache.close()


def test_new_runtime_buffer_invalidates_cached_structure():
    cache = WarmCache(gpu_budget_bytes=128, ram_budget_bytes=256)
    cache.register("a", lambda: nn.Linear(1, 1))
    cache.register("b", lambda: nn.Linear(1, 1))
    try:
        with cache.acquire("a") as model:
            model.register_buffer("new_cache", torch.ones(1, device="cuda"))
        with cache.acquire("b"):
            pass
        assert cache.state("a") == "cold"
    finally:
        cache.close()


def test_register_refuses_to_race_an_active_lease():
    cache = WarmCache(gpu_budget_bytes=128, ram_budget_bytes=256)
    cache.register("a", lambda: nn.Linear(1, 1))
    errors = []
    def register_other():
        try:
            cache.register("b", lambda: nn.Linear(1, 1))
        except BaseException as error:
            errors.append(error)
    try:
        with cache.acquire("a"):
            thread = threading.Thread(target=register_other)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert len(errors) == 1 and isinstance(errors[0], BusyError)
    finally:
        cache.close()


@pytest.mark.parametrize("family", ["llama", "gpt_neox"])
def test_normal_generation_preserves_a_real_warm_snapshot(family):
    """A false-positive invalidation would quietly turn warm swaps into reloads."""
    def load():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(77)
            if family == "llama":
                return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                    max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))
            return GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
                bos_token_id=1, eos_token_id=2, pad_token_id=0))
    cache = WarmCache(gpu_budget_bytes=2**20, ram_budget_bytes=2**21)
    cache.register("a", load)
    cache.register("b", lambda: nn.Linear(1, 1))
    try:
        with cache.acquire("a") as model:
            expected = model.generate(torch.tensor([[1, 4, 8]], device="cuda"),
                                     max_new_tokens=3, do_sample=False).cpu()
        with cache.acquire("b"):
            pass
        assert cache.state("a") == "warm"
        with cache.acquire("a") as model:
            actual = model.generate(torch.tensor([[1, 4, 8]], device="cuda"),
                                   max_new_tokens=3, do_sample=False).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        cache.close()


@pytest.mark.parametrize("release", [True, False])
def test_allocator_policy_changes_actual_driver_segment_releases(release):
    """Ignoring release_on_switch must change this resource-visible result."""
    torch.cuda.empty_cache()
    cache = WarmCache(gpu_budget_bytes=64 * 2**20, ram_budget_bytes=128 * 2**20,
                      release_on_switch=release)
    cache.register("a", lambda: nn.Linear(4096, 4096, bias=False))
    cache.register("b", lambda: nn.Linear(4096, 4096, bias=False))
    try:
        with cache.acquire("a"):
            pass
        before = torch.cuda.memory_stats()["segment.all.freed"]
        with cache.acquire("b"):
            pass
        freed = torch.cuda.memory_stats()["segment.all.freed"] - before
        assert (freed > 0) if release else (freed == 0)
    finally:
        cache.close()


class FailingMetrics(CacheMetrics):
    fail = False

    def update(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("injected metrics failure")
        return super().update(*args, **kwargs)


@pytest.mark.parametrize("body_fails", [False, True])
def test_metrics_cleanup_failure_does_not_lose_lease_or_mask_inference_error(body_fails):
    """An instrumentation fault must not permanently lock the cache."""
    metrics = FailingMetrics(128, 256)
    cache = WarmCache(gpu_budget_bytes=128, ram_budget_bytes=256, metrics=metrics)
    cache.register("a", lambda: nn.Linear(1, 1))
    original = ValueError("injected caller failure")
    try:
        with pytest.raises(ValueError if body_fails else RuntimeError) as caught:
            with cache.acquire("a") as model:
                model(torch.ones(1, 1, device="cuda")).sum().item()
                metrics.fail = True
                if body_fails:
                    raise original
        if body_fails:
            assert caught.value is original
        metrics.fail = False
        with cache.acquire("a") as model:
            assert model.weight.device.type == "cuda"
    finally:
        metrics.fail = False
        cache.close()


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_none_binding_invalidates_snapshot_even_after_original_tensor_dies(kind):
    class DropsState(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "parameter":
                self.value = nn.Parameter(torch.ones(1))
            else:
                self.register_buffer("value", torch.ones(1))

        def forward(self, x):
            result = self.value + x
            self.value = None
            return result

    cache = WarmCache(gpu_budget_bytes=128, ram_budget_bytes=256)
    cache.register("a", DropsState)
    cache.register("b", lambda: nn.Linear(1, 1))
    try:
        with cache.acquire("a") as model:
            assert model(torch.ones(1, device="cuda")).item() == 2
        with cache.acquire("b"):
            pass
        assert cache.state("a") == "cold"
        with cache.acquire("a") as model:
            assert model.value.item() == 1
    finally:
        cache.close()


@pytest.mark.parametrize("delete", [False, True])
def test_removed_plain_tensor_alias_invalidates_snapshot(delete):
    class DropsAlias(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.shadow = self.weight.detach()

        def forward(self, x):
            result = self.shadow + x
            if delete:
                del self.shadow
            else:
                self.shadow = None
            return result

    cache = WarmCache(gpu_budget_bytes=128, ram_budget_bytes=256)
    cache.register("a", DropsAlias)
    cache.register("b", lambda: nn.Linear(1, 1))
    try:
        with cache.acquire("a") as model:
            assert model(torch.ones(1, device="cuda")).item() == 2
        with cache.acquire("b"):
            pass
        assert cache.state("a") == "cold"
        with cache.acquire("a") as model:
            assert model.shadow.item() == 1
    finally:
        cache.close()


def test_metrics_cleanup_failure_inside_outer_exception_handler_is_not_swallowed():
    metrics = FailingMetrics(128, 256)
    cache = WarmCache(gpu_budget_bytes=128, ram_budget_bytes=256, metrics=metrics)
    cache.register("a", lambda: nn.Linear(1, 1))
    try:
        try:
            raise ValueError("already handled caller error")
        except ValueError:
            with pytest.raises(RuntimeError, match="injected metrics failure"):
                with cache.acquire("a") as model:
                    assert model.weight.device.type == "cuda"
                    metrics.fail = True
        metrics.fail = False
        with cache.acquire("a") as model:
            assert model.weight.device.type == "cuda"
    finally:
        metrics.fail = False
        cache.close()


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
def test_filling_initial_none_slot_invalidates_snapshot(kind):
    class FillsState(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(1))
            if kind == "parameter":
                self.register_parameter("value", None)
            else:
                self.register_buffer("value", None)

        def forward(self, x):
            value = x + 1
            self.value = nn.Parameter(value, requires_grad=False) if kind == "parameter" else value
            return self.anchor + x

    cache = WarmCache(gpu_budget_bytes=128, ram_budget_bytes=256)
    cache.register("a", FillsState)
    cache.register("b", lambda: nn.Linear(1, 1))
    try:
        with cache.acquire("a") as model:
            assert model(torch.ones(1, device="cuda")).item() == 2
        with cache.acquire("b"):
            pass
        assert cache.state("a") == "cold"
        with cache.acquire("a") as model:
            assert model.value is None
    finally:
        cache.close()
