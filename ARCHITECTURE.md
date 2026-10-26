# Architecture: model-swapper

| Boundary | Contract |
|---|---|
| Inference models on the GPU per cache | One active model |
| Concurrent leases per cache | One; overlap raises `BusyError` |
| Model implementation | A caller-supplied CPU `torch.nn.Module` loader |
| Tensor state | Immutable, ordinary dense registered parameters/buffers |
| Native runtime | PyTorch on one CUDA device |

The cache manages **where model weights live**. It does not implement tokenization, generation, image processing or speech decoding. Those remain the caller's responsibility. It is not an API proxy or hardware GPU partition.

## Request flow

One `with cache.acquire(name) as model:` block, as implemented by `WarmCache.acquire`, `_activate`, `_prepare` and `_demote_hot` in [warm_cache.py](warm_cache.py). The application owns task inputs and `forward`/`generate`; the cache only decides where the weights live. The diagram is tall; use GitHub's diagram expand/zoom controls if labels render small.

```mermaid
flowchart TD
    Start(["acquire(name)"]) --> Known{"Registered name?"}
    Known -->|"no"| Unknown["UnknownModelError<br/>reject: unknown, no lease taken"]
    Known -->|"yes"| Free{"Lease free?<br/>non-blocking"}
    Free -->|"no"| Busy["BusyError<br/>reject: busy, no lease taken"]
    Free -->|"yes"| State{"State of name"}
    State -->|"hot"| Reuse["Reuse GPU model<br/>no copy"]
    State -->|"warm"| DemoteW["Demote current hot model"]
    State -->|"cold"| Load["Call loader<br/>fresh CPU nn.Module"]
    Load --> Check{"Supported tensors<br/>within both budgets?"}
    Check -->|"no"| Reject["Activation error<br/>hot model untouched"]
    Reject --> Finally["finally: refresh metrics,<br/>release lease"]
    Finally --> Raise["Error reaches caller"]
    Check -->|"yes"| DemoteC["Demote current hot model"]
    DemoteC --> Evict["Evict LRU pinned entries<br/>only if RAM budget needs it"]
    Evict --> Pin["Copy unique tensors to pinned RAM<br/>CPU source tensors become meta"]
    Pin --> Alloc["Allocate GPU tensors<br/>on caller stream"]
    DemoteW --> Alloc
    Alloc --> Copy["Transfer streams wait,<br/>then copy pinned to GPU"]
    Copy -->|"error: back to meta"| Finally
    Copy --> Sync["Synchronize transfer streams"]
    Sync --> Yield
    Reuse --> Yield["Yield model inside lease<br/>torch.inference_mode"]
    Yield --> Infer["Caller runs inference"]
    Infer --> Release["Release lease<br/>model stays hot"]
```

Validation and copy failures are shown. Other activation or inference errors also release the lease through `finally`.

- **Refusals happen first.** An unregistered name is refused before the lease is touched, so it reports `unknown` even while another caller holds the lease. A held lease is an immediate `BusyError`, not a queue. Neither refusal acquires the lease, so neither has anything to release.
- **Hot reuse** returns the live GPU model with no copy and no checkpoint access.
- **Warm activation** never calls the loader. It demotes the current hot model, allocates GPU tensors and copies the retained pinned snapshot.
- **Cold activation** runs the loader and validates tensor support and logical GPU/RAM budgets **before** demoting the current hot model. A rejected model therefore leaves the hot model in place; the error is recorded and the lease is released in `acquire`'s `finally` before the error reaches the caller. Only after validation are least recently used pinned entries evicted, and only while the new snapshot would exceed the RAM budget. Evicted entries become cold.
- **Demotion** (`_demote_hot`) synchronises the device, then checks the hot model's bindings and module layout. Unchanged: tensors become `meta` and the pinned copy stays, so the model is warm. Changed: the snapshot and module are discarded, so the model is cold. Neither path copies GPU weights back to RAM. With `release_on_switch=True`, unused CUDA allocator blocks are then released.
- **Stream synchronisation.** Transfer streams wait for allocations made on the caller's current stream, copy asynchronously from pinned memory, and are synchronised before the model is yielded. If allocation or copying fails, the model returns to `meta` tensors and the same `finally` releases the lease before the error is raised to the caller.
- **Inside the lease** the caller uses the model under `torch.inference_mode()`. Readiness metrics are recorded before the yield, so they exclude caller inference. Leaving the block releases the lease even if metrics refresh fails; it does not demote the hot model.

## State transitions

```mermaid
stateDiagram-v2
    [*] --> Cold: register CPU loader
    Cold --> Hot: load, validate, pin, copy to GPU
    Hot --> Warm: unchanged model; another model needs GPU
    Warm --> Hot: copy retained pinned weights to GPU
    Warm --> Cold: RAM LRU eviction
    Hot --> Cold: close
    Hot --> Cold: registered bindings or module layout changed
    Warm --> Cold: close
```

HOT means one model has materialised GPU tensors. Its pinned CPU copy is retained. WARM means pinned weights and empty model structure remain, without that model's active GPU parameters. COLD means only the loader remains; checkpoint loading is needed again.

The warm cache's RAM budget includes the pinned copy of the hot model. The least recently used pinned entry is dropped when loading another model would exceed that budget. A model exceeding either logical tensor budget is rejected before allocating its pinned/GPU weights.

## Immutable snapshots avoid unnecessary copying

Inference does not change supported model weights. Once their CPU snapshot exists, there is no reason to copy them from GPU to RAM at each demotion. The cache frees the GPU tensors, keeps the model's empty structure, and restores from the original pinned copy later.

In-place weight edits remain unsupported. If a forward replaces/adds/removes registered tensor bindings or changes module layout, demotion discards that snapshot and module; the next request loads a fresh CPU model. This prevents dynamic rotary-position buffers from being combined with stale Python-side cache metadata. While a model remains hot, it keeps its live state.

Callers must not retain GPU tensor references or use the model after the lease ends. A loader must return a fresh, fully initialised CPU model, not an object previously handed to this cache.

## Tensor identity and compatibility

The initial snapshot includes both named parameters and named buffers with duplicate names retained. This includes non-persistent buffers, which `state_dict()` alone omits. Exact tied tensors are stored once and their aliases are reattached after restoration.

Bindings cache the owning module, attribute name and parameter/buffer kind. This avoids repeatedly walking module trees and resolving names during warm activation. GPU allocation is done once per unique tensor, rather than allocating tied embeddings twice and repairing them afterward.

Direct plain tensor attributes that exactly alias registered tensors are tracked as shadow bindings. They move to GPU during activation and to metadata-only tensors during demotion. Other plain tensors, tensor containers and opaque runtime caches are rejected. Weak references compare live tensor objects without relying on reusable Python object IDs or retaining old GPU allocations.

Each alias keeps its original registration kind. Rebinding writes the correct parameter or buffer dictionary, so a non-persistent buffer cannot accidentally become a parameter.

The cache explicitly rejects unsupported tensor/parameter subclasses, quantized tensors, sparse layouts, conjugate/negative views, non-CPU loader state, distinct views sharing storage, compiled/offload-hook modules and opaque cache state. Those cases can require custom runtime metadata or storage semantics that ordinary tensor replacement cannot preserve. They fail with `UnsupportedModelError` instead of silently changing the model.

“Model-agnostic” means the cache does not hard-code a causal-LM architecture. It does **not** mean every Python module or model engine is proven compatible. GGUF/llama.cpp, CTranslate2/faster-whisper and Strata are separate engines, not loaders for this interface. Tensor attributes used by `forward` must be registered state or supported direct aliases of it; arbitrary unregistered state is not migrated.

## Streams and allocator policy

Each activation allocates destinations on the caller's CUDA stream. Transfer streams wait for those allocations, copy from pinned RAM, and finish before the model is yielded. Demotion synchronises device work before replacing model tensors with metadata-only tensors.

`release_on_switch=False` retains PyTorch's reusable allocator blocks between switches. This can avoid repeated driver allocation/free calls. It also means CUDA memory reserved by the allocator can remain high while a model is not hot. Logical weight bytes and reserved CUDA bytes are different.

`release_on_switch=True` flushes unused allocator blocks after each demotion. It trades more allocation work for earlier release of unused reservations. Both policies release cache-owned model state and unused allocator blocks on `close()`.

One transfer stream is the conservative default. More streams cannot create PCIe bandwidth. The benchmark includes a two-stream variant so that its benefit, or lack of benefit, is measured rather than assumed.

Pinned preparation releases each CPU source tensor after its pinned copy is ready, rather than retaining two full host models through the entire copy. Loader work and transient allocations still sit outside the logical weight budget. PyTorch 2.5.1 rounds pinned allocations; the estimated rounded active bytes are exposed separately from logical tensor bytes.

Eviction and close flush unused host allocator blocks using `torch._C._host_emptyCache()`. This is a private API verified in the locked PyTorch 2.5.1 runtime. A PyTorch upgrade must recheck it. It operates on the process's unused pinned-memory cache, not on another process or still-referenced model data.

## Metrics

`metrics.py` owns a fresh `CollectorRegistry` for each cache. There is no global registry collision when an application creates multiple caches. The application can expose this registry with `prometheus_client.start_http_server`, or use `cache.metrics.render()` in its existing metrics endpoint.

| Metric | Meaning |
|---|---|
| `model_swapper_activations_total{model,source,outcome}` | Readiness attempts, split by cold/warm/hot source and success/error |
| `model_swapper_activation_duration_seconds{model,source,outcome}` | Time to make a model ready, excluding caller inference |
| `model_swapper_rejections_total{reason}` | Bounded `busy`, `unknown` or `admin_busy` rejections; unknown IDs do not become labels |
| `model_swapper_gpu_weight_bytes` | Logical unique tensor bytes of the hot model |
| `model_swapper_pinned_weight_bytes` | Logical retained pinned tensor bytes |
| `model_swapper_pinned_allocator_estimate_bytes` | Estimated active rounded host allocations; excludes unused cached blocks and page granularity |
| `model_swapper_process_cuda_allocated_bytes` | Current process's PyTorch-allocated CUDA tensor memory |
| `model_swapper_process_cuda_reserved_bytes` | Current process's PyTorch-reserved CUDA allocator memory |
| `model_swapper_gpu_weight_budget_bytes` | Configured logical GPU tensor budget |
| `model_swapper_pinned_weight_budget_bytes` | Configured logical pinned tensor budget |
| `model_swapper_models{tier}` | Registered entries in each state |
| `model_swapper_leases_active` | Whether a caller currently holds the lease |

Registered model IDs are bounded by the application. Do not put prompts, request IDs, paths or arbitrary user strings into metric labels. These memory gauges are not NVML process memory, CUDA context size or total host RSS. Histogram samples include startup/calibration whenever those activations pass through the cache; they are not automatically the benchmark's measured-only subset.

## Benchmark boundaries

`benchmark.py` loads two staged local Transformers causal LMs through the same backend and precision. It compares checkpoint reload, release-on-switch warm caching, allocator-reuse warm caching and allocator reuse with two transfer streams. Each method is calibrated before timed alternating activations. Every timed warm sample must actually start in the warm state; unexpected reloads abort the run. Output tokens and last-token logits must exactly match independently loaded references for the test prompt.

It records model and source hashes, timings, memory readings and raw samples. It can expose localhost-only metrics during warm methods and saves per-method Prometheus snapshots after cleanup. Data and profiling traces belong outside the repository.

The reload baseline allows the OS file cache and runs in one Python process. It is not a direct llama-swap measurement, physical cold-disk test, large-model scaling test or production concurrency benchmark. GPU sharing can affect timing. More model families passing output-parity tests establishes architecture coverage, not pretrained task accuracy.

## Deployment and safety

Nothing installs itself as a service. The benchmark's allocator fraction and free-memory guard are soft controls, not hardware isolation. The library's budgets exclude temporary activations, allocator rounding, CUDA context memory and Python objects.

The lease is released even if metrics cleanup fails. Secondary cleanup failures are attached to the original exception rather than replacing it. Fatal CUDA context errors, such as illegal memory access or driver loss, require a fresh process; the cache does not claim to recover a broken CUDA context.

Run experiments on a dedicated GPU where possible. Close the cache explicitly, keep model files and results outside Git, and verify the live environment before sharing a GPU. This library does not replace an application's serving engine or admission policy.

## Deferred multi-GPU roadmap

**Status: documented for later work; not implemented or multi-GPU tested.** No delivery date or scaling result is claimed.

| Boundary | Current evidence | Proposed next scope |
|---|---|---|
| GPU count exercised | One V100, 2026-10-26 | At least two physical GPUs for acceptance |
| Device selection | `WarmCache(device="cuda:0")` accepts one CUDA device | One worker process and cache per assigned GPU |
| Concurrent inference | One lease per cache | Independent leases on different GPU workers |
| Model size | Each model must fit one GPU | Same restriction; no combined GPU memory |
| Routing | Application chooses a cache | Capacity-aware dispatch across workers |
| Metrics | Isolated registry per cache | Worker/device identity plus dispatch and health signals |

### Proposed architecture

The diagram shows a proposed two-worker deployment, not an existing API. The number of workers would follow available GPUs.

```mermaid
flowchart TB
    App["Application requests"] --> Dispatch["Dispatcher<br/>compatibility and capacity checks"]
    Dispatch --> W0["Worker process 0<br/>one cache and one lease"]
    Dispatch --> W1["Worker process 1<br/>one cache and one lease"]
    W0 --> G0["Assigned GPU 0<br/>one whole active model"]
    W1 --> G1["Assigned GPU 1<br/>one whole active model"]
    W0 -.-> RAM["Host RAM admission<br/>sum of worker budgets and overhead"]
    W1 -.-> RAM
    W0 -.-> Metrics["Metrics collection<br/>worker and device identity"]
    W1 -.-> Metrics
    Dispatch -.-> Metrics
```

- **Separate processes, explicit devices.** Start workers with a CUDA-safe process start method, not a fork after CUDA initialisation. Each owns its CUDA context, cache, streams and registry. Resolve physical assignments explicitly: device visibility can make a worker's local `cuda:0` refer to a different physical GPU.
- **Whole models only.** Register a fresh CPU loader in each eligible worker. Do not pass live CUDA modules between workers. Separate `WarmCache` instances on different devices are a possible starting point, but that arrangement has not been validated by the existing suite.
- **Dispatch to available capacity.** Check model/device compatibility and reserve a worker's capacity before dispatch. Prefer an eligible idle worker with the model hot, then warm, before a cold load. Stale status alone must not allow overlapping leases. Busy capacity remains an explicit refusal unless a bounded queue is designed separately.
- **Account for the entire host.** Independent workers can duplicate pinned model snapshots. Admit configured RAM budgets together, with room for loaders, allocator rounding and process overhead. Monitor actual host use and locked-memory limits; logical budgets are not hard memory isolation.
- **Leave room beyond weights.** GPU admission must account for request activations, generation caches, context memory and allocator reservations. A model passing the current logical-weight budget is not proof that its request fits.
- **Keep failures local.** Stop routing to an unhealthy worker. A broken CUDA context requires replacing that worker process; another healthy worker must remain usable. Do not silently replay a failed request.

### Proposed metrics

These are requirements for future integration, not new metric names or endpoints available today.

| Signal | Required meaning |
|---|---|
| Worker/device identity | Stable configured worker and GPU assignment, attached by the application or scrape configuration |
| Existing activation and memory metrics | Preserve per-worker, per-model cold/warm/hot timing and weight/allocator measurements |
| Worker health and capacity | Whether a worker can accept a lease, plus failures and restarts |
| Dispatch decisions and refusals | Selected worker and bounded reason values; never request IDs or prompts as labels |
| End-to-end request duration | Separate dispatch, model readiness and inference time; first-token latency for streaming text workloads |
| Physical GPU and host memory | Complement cache gauges with an appropriate GPU exporter and host measurements |

Do not add per-worker CUDA memory gauges together and call that available combined memory for one model. Keep logical pinned weights, estimated pinned allocation, process RSS and physical GPU measurements distinct. If multiple workers expose the same model, preserve worker identity rather than merging their registries without labels.

### Acceptance before advertising support

1. **Real hardware:** run on at least two physical GPUs. Record GPU models, interconnect, runtime versions, precision, workload and source revision. Single-GPU tests or mocks alone do not pass this gate.
2. **Correct device and parity:** exercise cold, warm and hot paths on each worker. Check weights, buffers and request tensors use its assigned device. Compare outputs with independent same-device loads using a documented precision-appropriate correctness check.
3. **Independent progress:** run overlapping requests on different workers and show concurrent GPU work. A busy lease on one worker must not block the other. Same-worker overlap must remain refused without corrupting state.
4. **Capacity boundaries:** test unequal GPU capacities, an unsupported or oversized model, and a host RAM budget exhausted by duplicated snapshots. Admission failure must leave existing healthy models usable.
5. **Failure isolation:** terminate a worker during a request. Report that request's failure, keep the other worker serving, and verify clean replacement without stale reservations or leaked worker processes.
6. **Metrics correctness:** scrape workers during switching, overlap and failure. Verify distinct identities, readiness versus total request duration, error counts and memory accounting. No prompts, credentials or private checkpoint paths may enter labels or published results.
7. **Measured scaling:** compare one-worker and multi-worker runs with the same model mix, request load and budgets. Report throughput, median/p95 latency, output checks and peak RAM/VRAM. Do not promise linear scaling or a speedup before measurement.
8. **Clean shutdown:** stop admission, finish or explicitly fail active work, then close caches and exit workers. Check process exit and GPU memory recovery.

### Explicit non-goals for this extension

- Splitting one model across GPUs. Tensor parallelism, which divides a model's computation across devices, needs a suitable inference engine and a separate integration.
- Combining GPU memory into a transparent larger pool, sharing live CUDA tensors between workers, or automatically migrating an active model.
- Shared pinned snapshots, GPU-to-GPU copies, multi-node scheduling and NVMe/GDS storage tiers.
- Expanding the existing model-format contract. Quantized, compiled, distributed and engine-specific models remain outside this cache interface.

Implement and test the worker boundary first, then dispatch and metric aggregation. Keep the current single-device path working throughout. Record real multi-GPU evidence before changing the README's supported-capabilities claim.
