"""Inference-only whole-model CUDA cache with bounded logical weight storage.

A lease owns one GPU model. Ordinary immutable registered tensors can be kept
in pinned RAM. Changed runtime bindings invalidate the snapshot and reload on
next use.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Callable
from weakref import ref

import torch
from torch import nn

from metrics import CacheMetrics


class BudgetError(RuntimeError):
    """Model tensor bytes exceed a configured logical weight budget."""


class BusyError(RuntimeError):
    """Another caller holds the cache lease."""


class UnsupportedModelError(ValueError):
    """Tensor state cannot be preserved safely by this cache."""


class UnknownModelError(KeyError):
    """The requested model has not been registered."""


@dataclass
class _Entry:
    loader: Callable[[], nn.Module]
    model: nn.Module | None = None
    pinned: dict[str, torch.Tensor] | None = None
    aliases: dict[str, str] = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    shadows: list = field(default_factory=list)
    layout: list = field(default_factory=list)
    live_refs: dict = field(default_factory=dict)
    shadow_refs: dict = field(default_factory=dict)
    nbytes: int = 0
    pin_estimate: int = 0


def _tensors(model):
    # state_dict() omits non-persistent buffers, such as rotary frequencies.
    return dict(list(model.named_parameters(remove_duplicate=False))
                + list(model.named_buffers(remove_duplicate=False)))


def _key(tensor):
    return (tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype, tensor.is_conj(), tensor.is_neg())


def _unique_state(tensors):
    unique, aliases, seen, storage_views = {}, {}, {}, {}
    for name, tensor in tensors.items():
        if type(tensor) not in (torch.Tensor, nn.Parameter):
            raise UnsupportedModelError(f"{name}: tensor subclasses need a dedicated adapter")
        if (tensor.device.type != "cpu" or tensor.layout != torch.strided
                or tensor.is_quantized or tensor.is_conj() or tensor.is_neg()):
            raise UnsupportedModelError(f"{name}: expected ordinary dense CPU tensor state")
        key = _key(tensor) if tensor.numel() else (name,)
        if tensor.numel():
            storage = tensor.untyped_storage().data_ptr()
            if storage in storage_views and storage_views[storage] != key:
                raise UnsupportedModelError(f"{name}: distinct views of shared storage are unsupported")
            storage_views[storage] = key
        if key in seen:
            aliases[name] = seen[key]
        else:
            seen[key] = name
            unique[name] = tensor
    if not unique:
        raise UnsupportedModelError("No registered tensor state to cache")
    return unique, aliases


def _get(binding):
    owner, name, kind = binding
    return (owner._parameters if kind == "parameter" else owner._buffers).get(name)


def _set(binding, tensor):
    owner, name, kind = binding
    if kind == "parameter":
        owner._parameters[name] = tensor if isinstance(tensor, nn.Parameter) else nn.Parameter(tensor, requires_grad=False)
    else:
        # Assigning a Parameter through setattr would change buffer registration.
        owner._buffers[name] = tensor.detach()


def _retie(entry, canonical=None):
    for name, original in entry.aliases.items():
        if canonical is None or original == canonical:
            _set(entry.bindings[name], _get(entry.bindings[original]))
    for owner, attribute, original in entry.shadows:
        if canonical is None or original == canonical:
            vars(owner)[attribute] = _get(entry.bindings[original]).detach()


def _contains_tensor(value, seen=None):
    if isinstance(value, (torch.Tensor, torch.ScriptObject)):
        return True
    if not isinstance(value, (dict, list, tuple)):
        return False
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    return any(_contains_tensor(item, seen) for item in (value.values() if isinstance(value, dict) else value))


def _metadata(model, tensors, aliases):
    bindings, shadows, layout = {}, [], []
    canonical_keys = {_key(t): aliases.get(name, name) for name, t in tensors.items() if t.numel()}
    for name in tensors:
        path, _, attribute = name.rpartition(".")
        owner = model.get_submodule(path)
        kind = "parameter" if attribute in owner._parameters else "buffer"
        bindings[name] = (owner, attribute, kind)
    for module in model.modules():
        if (getattr(module, "_hf_hook", None) is not None
                or type(module).__module__.startswith("torch._dynamo")):
            raise UnsupportedModelError("Compiled/offloaded modules need a dedicated adapter")
        layout.append((module, frozenset(module._parameters), frozenset(module._buffers),
                       tuple((k, id(v)) for k, v in module._modules.items()),
                       frozenset(module._non_persistent_buffers_set),
                       tuple(k for k, v in module._parameters.items() if v is None),
                       tuple(k for k, v in module._buffers.items() if v is None)))
        for attribute, value in vars(module).items():
            if attribute in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(value, torch.Tensor):
                if (type(value) not in (torch.Tensor, nn.Parameter) or value.device.type != "cpu"
                        or value.layout != torch.strided or value.is_quantized or value.is_conj() or value.is_neg()):
                    raise UnsupportedModelError(f"{attribute}: unsupported plain tensor attribute")
                original = canonical_keys.get(_key(value)) if value.numel() else None
                if original is None:
                    raise UnsupportedModelError(f"{attribute}: register tensor state as a parameter or buffer")
                shadows.append((module, attribute, original))
            elif (_contains_tensor(value) or type(value).__module__.startswith("transformers.cache_utils")
                  or isinstance(value, torch.cuda.CUDAGraph)):
                raise UnsupportedModelError(f"{attribute}: opaque or container tensor state is unsupported")
    return bindings, shadows, layout


def _changed(entry):
    for module, parameters, buffers, children, persistence, none_params, none_buffers in entry.layout:
        if (module.training or module._parameters.keys() != parameters
                or module._buffers.keys() != buffers
                or tuple((k, id(v)) for k, v in module._modules.items()) != children
                or module._non_persistent_buffers_set != persistence
                or any(module._parameters[name] is not None for name in none_params)
                or any(module._buffers[name] is not None for name in none_buffers)):
            return True
    for name, binding in entry.bindings.items():
        original = entry.live_refs[name]()
        if original is None or _get(binding) is not original:
            return True
    for owner, attribute, _ in entry.shadows:
        original = entry.shadow_refs[(id(owner), attribute)]()
        if original is None or vars(owner).get(attribute) is not original:
            return True
    for module, *_ in entry.layout:
        for attribute, value in vars(module).items():
            if attribute in ("_parameters", "_buffers", "_modules"):
                continue
            if isinstance(value, torch.Tensor):
                key = (id(module), attribute)
                if key not in entry.shadow_refs:
                    return True
            elif (_contains_tensor(value) or type(value).__module__.startswith("transformers.cache_utils")
                  or isinstance(value, torch.cuda.CUDAGraph)):
                return True
    return False


def _clear_model(entry):
    # Dirty entries are never reused. Clear both current and old module owners,
    # including modules replaced by a forward pass, before dropping metadata.
    modules = {module for module, *_ in entry.layout}
    if entry.model is not None:
        modules.update(entry.model.modules())
    for module in modules:
        for name in module._parameters:
            module._parameters[name] = None
        for name in module._buffers:
            module._buffers[name] = None
        for name, value in list(vars(module).items()):
            if name not in ("_parameters", "_buffers", "_modules") and _contains_tensor(value):
                vars(module)[name] = None


class WarmCache:
    def __init__(self, *, gpu_budget_bytes, ram_budget_bytes, streams=1,
                 device="cuda:0", release_on_switch=False, metrics=None):
        if gpu_budget_bytes <= 0 or ram_budget_bytes <= 0 or streams <= 0:
            raise ValueError("Budgets and stream count must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This cache requires CUDA")
        self.gpu_budget_bytes = gpu_budget_bytes
        self.ram_budget_bytes = ram_budget_bytes
        self.release_on_switch = release_on_switch
        self.metrics = metrics if metrics is not None else CacheMetrics(gpu_budget_bytes, ram_budget_bytes)
        if (self.metrics.gpu_budget_bytes, self.metrics.ram_budget_bytes) != (gpu_budget_bytes, ram_budget_bytes):
            raise ValueError("Metrics budgets must match the cache budgets")
        self._streams = [torch.cuda.Stream(device=self.device) for _ in range(streams)]
        self._entries, self._warm = {}, OrderedDict()
        self._hot, self._ram_bytes = None, 0
        self._lease = Lock()
        self._refresh_metrics(False)

    @property
    def ram_bytes(self):
        return self._ram_bytes

    @property
    def gpu_bytes(self):
        return self._entries[self._hot].nbytes if self._hot is not None else 0

    @property
    def pinned_allocator_estimate_bytes(self):
        return sum(entry.pin_estimate for entry in self._warm.values())

    def state(self, name):
        entry = self._entries[name]
        return "hot" if name == self._hot else "warm" if entry.pinned is not None else "cold"

    def _refresh_metrics(self, active):
        self.metrics.update(
            [self.state(name) for name in self._entries], self.gpu_bytes, self.ram_bytes, active,
            pinned_allocator_estimate_bytes=self.pinned_allocator_estimate_bytes,
            cuda_allocated_bytes=torch.cuda.memory_allocated(self.device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(self.device),
        )

    def register(self, name, loader):
        if not self._lease.acquire(blocking=False):
            self.metrics.reject("admin_busy")
            raise BusyError("Cannot register while the cache is leased")
        try:
            if name in self._entries:
                raise ValueError(f"Already registered: {name}")
            self._entries[name] = _Entry(loader=loader)
            self._refresh_metrics(False)
        finally:
            self._lease.release()

    def _forget(self, name):
        entry = self._entries[name]
        _clear_model(entry)
        self._warm.pop(name, None)
        self._ram_bytes -= entry.nbytes
        entry.model, entry.pinned = None, None
        entry.aliases, entry.bindings, entry.shadows, entry.layout = {}, {}, [], []
        entry.live_refs, entry.shadow_refs = {}, {}
        entry.nbytes, entry.pin_estimate = 0, 0

    def _demote_hot(self):
        if self._hot is None:
            return
        torch.cuda.synchronize(self.device)
        name, entry = self._hot, self._entries[self._hot]
        self._hot = None
        if _changed(entry):
            self._forget(name)
            torch._C._host_emptyCache()
        else:
            entry.model.to_empty(device="meta")
            _retie(entry)
        if self.release_on_switch:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def _prepare(self, name, entry):
        model = entry.loader()
        if not isinstance(model, nn.Module):
            raise UnsupportedModelError("Loaders must return a torch.nn.Module")
        model.eval().requires_grad_(False)
        tensors = _tensors(model)
        unique, aliases = _unique_state(tensors)
        bindings, shadows, layout = _metadata(model, tensors, aliases)
        nbytes = sum(t.numel() * t.element_size() for t in unique.values())
        if nbytes > self.gpu_budget_bytes or nbytes > self.ram_budget_bytes:
            raise BudgetError(f"{name} needs {nbytes} logical tensor bytes")
        self._demote_hot()
        evicted = False
        while self._ram_bytes + nbytes > self.ram_budget_bytes:
            self._forget(next(iter(self._warm)))
            evicted = True
        if evicted:
            torch._C._host_emptyCache()
        entry.model, entry.aliases = model, aliases
        entry.bindings, entry.shadows, entry.layout = bindings, shadows, layout
        del tensors
        pinned, estimate = {}, 0
        try:
            while unique:
                key, tensor = unique.popitem()
                buffer = torch.empty_like(tensor, device="cpu", pin_memory=True)
                buffer.copy_(tensor.detach())
                pinned[key] = buffer
                size = tensor.numel() * tensor.element_size()
                estimate += 1 << (size - 1).bit_length() if size else 0
                # Release each source as its pinned copy becomes ready, rather
                # than keeping two full host models alive through preparation.
                _set(entry.bindings[key], torch.empty_like(tensor, device="meta"))
                _retie(entry, canonical=key)
        except BaseException:
            _clear_model(entry)
            entry.model = None
            entry.aliases, entry.bindings, entry.shadows, entry.layout = {}, {}, [], []
            pinned.clear()
            torch._C._host_emptyCache()
            raise
        entry.pinned, entry.nbytes, entry.pin_estimate = pinned, nbytes, estimate
        self._warm[name] = entry
        self._ram_bytes += nbytes

    def _activate(self, name):
        entry = self._entries[name]
        if self._hot == name:
            self._warm.move_to_end(name)
            return entry.model
        if entry.pinned is None:
            self._prepare(name, entry)
        else:
            self._demote_hot()
        try:
            destinations = {}
            for key, source in entry.pinned.items():
                target = torch.empty_like(source, device=self.device)
                _set(entry.bindings[key], target)
                destinations[key] = _get(entry.bindings[key])
            _retie(entry)
            current = torch.cuda.current_stream(self.device)
            for stream in self._streams:
                stream.wait_stream(current)
            for index, (key, source) in enumerate(entry.pinned.items()):
                with torch.cuda.stream(self._streams[index % len(self._streams)]):
                    destinations[key].copy_(source, non_blocking=True)
            for stream in self._streams:
                stream.synchronize()
        except BaseException as error:
            try:
                torch.cuda.synchronize(self.device)
            except BaseException as cleanup_error:
                error.add_note(f"CUDA synchronization during cleanup failed: {cleanup_error}")
            try:
                entry.model.to_empty(device="meta")
                _retie(entry)
            except BaseException as cleanup_error:
                error.add_note(f"Model-state cleanup failed: {cleanup_error}")
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            except BaseException as cleanup_error:
                error.add_note(f"CUDA allocator cleanup failed: {cleanup_error}")
            raise
        entry.live_refs = {key: ref(_get(binding)) for key, binding in entry.bindings.items()}
        entry.shadow_refs = {(id(owner), attribute): ref(vars(owner)[attribute])
                             for owner, attribute, _ in entry.shadows}
        self._hot = name
        self._warm.move_to_end(name)
        return entry.model

    @contextmanager
    def acquire(self, name):
        if name not in self._entries:
            self.metrics.reject("unknown")
            raise UnknownModelError(name)
        if not self._lease.acquire(blocking=False):
            self.metrics.reject("busy")
            raise BusyError("The cache is leased; retry after the current request")
        source, started = self.state(name), perf_counter()
        primary_error = None
        try:
            with torch.inference_mode():
                try:
                    model = self._activate(name)
                except BaseException:
                    self.metrics.activation(name, source, "error", perf_counter() - started)
                    raise
                self.metrics.activation(name, source, "success", perf_counter() - started)
                self._refresh_metrics(True)
                yield model
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                self._refresh_metrics(False)
            except BaseException as metrics_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Metrics refresh during cleanup failed: {metrics_error}")
            finally:
                self._lease.release()

    def close(self):
        if not self._lease.acquire(blocking=False):
            self.metrics.reject("admin_busy")
            raise BusyError("Cannot close while a caller holds the cache lease")
        try:
            self._demote_hot()
            for name in list(self._warm):
                self._forget(name)
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            # PyTorch 2.5.1 exposes only this private host-cache cleanup API.
            # The locked version is deliberate; tests exercise this path.
            torch._C._host_emptyCache()
            self._refresh_metrics(False)
        finally:
            self._lease.release()
