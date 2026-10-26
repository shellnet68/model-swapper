"""Architecture parity and explicit rejection of unsupported tensor state."""
import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel, ViTConfig, ViTModel, WhisperConfig, WhisperModel

from warm_cache import UnsupportedModelError, WarmCache


@pytest.fixture(autouse=True)
def real_cuda():
    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_per_process_memory_fraction(0.05)
    torch.set_num_threads(2)


def task_model(task):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        if task == "cnn":
            return nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        if task == "embedding":
            return BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                       num_attention_heads=2, intermediate_size=32, max_position_embeddings=32))
        if task == "vision":
            return ViTModel(ViTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                     intermediate_size=32, image_size=16, patch_size=8))
        return WhisperModel(WhisperConfig(vocab_size=32, num_mel_bins=4, d_model=16,
                            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                            decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                            max_source_positions=10, max_target_positions=16,
                            pad_token_id=0, bos_token_id=1, eos_token_id=2))


def task_output(model, task, device):
    if task == "cnn":
        return model(torch.arange(64, device=device, dtype=torch.float32).reshape(1, 1, 8, 8) / 64)
    if task == "embedding":
        return model(input_ids=torch.tensor([[1, 3, 5, 2]], device=device)).last_hidden_state
    if task == "vision":
        return model(pixel_values=torch.linspace(0, 1, 768, device=device).reshape(1, 3, 16, 16)).last_hidden_state
    return model(input_features=torch.linspace(0, 1, 80, device=device).reshape(1, 4, 20),
                 decoder_input_ids=torch.tensor([[1, 3]], device=device)).last_hidden_state


@pytest.mark.parametrize("task", ["cnn", "embedding", "vision", "speech"])
def test_task_architecture_matches_normal_cpu_model_after_warm_restore(task):
    """Dropping buffers or hard-coding a causal-LM layout breaks task outputs."""
    reference = task_model(task).eval()
    with torch.inference_mode():
        expected = task_output(reference, task, "cpu")
    cache = WarmCache(gpu_budget_bytes=4 * 2**20, ram_budget_bytes=8 * 2**20)
    try:
        cache.register(task, lambda: task_model(task))
        cache.register("other", lambda: nn.Linear(2, 1))
        for _ in range(2):
            with cache.acquire(task) as model:
                actual = task_output(model, task, "cuda").cpu()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            with cache.acquire("other") as other:
                other(torch.ones(1, 2, device="cuda"))
    finally:
        cache.close()


class TaggedParameter(nn.Parameter):
    """A stand-in for parameter subclasses carrying custom inference state."""


def unsupported(kind):
    model = nn.Module()
    if kind == "parameter_subclass":
        model.register_parameter("weight", TaggedParameter(torch.ones(2)))
    elif kind == "quantized":
        model.register_buffer("weight", torch.quantize_per_tensor(torch.ones(2), 0.1, 0, torch.qint8))
    elif kind == "sparse":
        model.register_buffer("weight", torch.sparse_coo_tensor(torch.tensor([[0]]), torch.ones(1), (2,)))
    else:
        storage = torch.arange(3, dtype=torch.float32)
        model.register_parameter("first", nn.Parameter(storage[:2]))
        model.register_parameter("second", nn.Parameter(storage[1:]))
    return model


@pytest.mark.parametrize("kind", ["parameter_subclass", "quantized", "sparse", "shared_view"])
def test_unsupported_storage_is_rejected_before_gpu_or_pinned_allocation(kind):
    """Silently replacing custom/aliased state can change model semantics."""
    cache = WarmCache(gpu_budget_bytes=4096, ram_budget_bytes=4096)
    try:
        cache.register(kind, lambda: unsupported(kind))
        with pytest.raises(UnsupportedModelError):
            with cache.acquire(kind):
                pytest.fail("Unsupported model was exposed for inference")
        assert cache.state(kind) == "cold"
        assert cache.gpu_bytes == 0
        assert cache.ram_bytes == 0
    finally:
        cache.close()


@pytest.mark.parametrize("release", [True, False])
def test_close_returns_allocations_in_both_allocator_policies(release):
    """Retaining models or cached device blocks after close leaks resources."""
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    cache = WarmCache(gpu_budget_bytes=4096, ram_budget_bytes=4096, release_on_switch=release)
    cache.register("a", lambda: nn.Linear(2, 1))
    cache.register("b", lambda: nn.Linear(2, 1))
    try:
        for name in ("a", "b", "a"):
            with cache.acquire(name) as model:
                model(torch.ones(1, 2, device="cuda")).sum().item()
    finally:
        cache.close()
    assert torch.cuda.memory_allocated() <= before
    assert torch.cuda.memory_reserved() <= reserved_before
