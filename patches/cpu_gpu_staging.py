from __future__ import annotations

from typing import MutableMapping, Sequence

import torch


def _new_cpu_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    pin_memory: bool,
) -> torch.Tensor:
    if pin_memory:
        try:
            return torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(shape, device="cpu", dtype=dtype)


def _cache_get(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    key: str,
) -> object | None:
    if stage_cache is not None:
        return stage_cache.get(key)
    if cache_owner is not None:
        return getattr(cache_owner, key, None)
    return None


def _cache_set(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    key: str,
    value: object,
) -> None:
    if stage_cache is not None:
        stage_cache[key] = value
    elif cache_owner is not None:
        setattr(cache_owner, key, value)


# [STAGE-RING-FIX 2026-07-02] Host-side WAR guard for the reused PINNED staging
# buffers: `gpu.copy_(cpu_stage, non_blocking=True)` returns before the H2D
# completes, so rewriting the same cached cpu_stage on the next call (e.g.
# per-layer rapid-fire staging under one cache_name) raced the in-flight
# transfer and delivered torn/stale values to the consumer kernel. Guard: one
# completion event per cache_name; wait on it (query-first: the common case is
# already-complete, ~1us) before the host rewrites the pinned buffer.
def _wait_stage_h2d_evt(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    key: str,
) -> None:
    evt = _cache_get(stage_cache=stage_cache, cache_owner=cache_owner, key=key)
    if evt is None:
        return
    # [GUARD-NO-SWALLOW] query/synchronize 失败时继续=在 H2D 在飞时覆写
    # pinned 缓冲（本守卫要防的 WAR 撕裂本身），必须炸。
    if not evt.query():
        evt.synchronize()


def _record_stage_h2d_evt(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    key: str,
    device: torch.device,
) -> None:
    if device.type != "cuda":
        return
    # [GUARD-NO-SWALLOW] record 失败时静默返回=下次 wait 无事件可等（守卫
    # 整体失效、无声放行覆写），必须炸。
    evt = torch.cuda.Event(enable_timing=False)
    evt.record(torch.cuda.current_stream(device=device))
    _cache_set(stage_cache=stage_cache, cache_owner=cache_owner, key=key, value=evt)


def cached_sequence_to_device(
    values: Sequence[int | bool],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    cache_name: str,
    stage_cache: MutableMapping[str, object] | None = None,
    cache_owner: object | None = None,
    out: torch.Tensor | None = None,
    reuse_unchanged: bool = False,
) -> torch.Tensor:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    values_tuple = tuple(values)
    length = int(len(values_tuple))
    if device.type != "cuda":
        result = torch.as_tensor(values_tuple, device=device, dtype=dtype)
        if out is not None and out.device == device and out.dtype == dtype and int(out.numel()) >= length:
            out[:length].copy_(result)
            return out if int(out.numel()) == length else out[:length]
        return result

    capacity = max(1, length)
    gpu_stage = out
    gpu_stage_reused = (
        isinstance(gpu_stage, torch.Tensor)
        and gpu_stage.device == device
        and gpu_stage.dtype == dtype
        and int(gpu_stage.numel()) >= capacity
    )
    if (
        not isinstance(gpu_stage, torch.Tensor)
        or gpu_stage.device != device
        or gpu_stage.dtype != dtype
        or int(gpu_stage.numel()) < capacity
    ):
        gpu_key = f"_cpu_gpu_stage_{cache_name}_gpu"
        cached_gpu = _cache_get(
            stage_cache=stage_cache,
                cache_owner=cache_owner,
                key=gpu_key,
            )
        gpu_stage_reused = True
        if (
            not isinstance(cached_gpu, torch.Tensor)
            or cached_gpu.device != device
            or cached_gpu.dtype != dtype
            or int(cached_gpu.numel()) < capacity
        ):
            gpu_stage_reused = False
            old_cap = int(cached_gpu.numel()) if isinstance(cached_gpu, torch.Tensor) else 0
            if isinstance(cached_gpu, torch.Tensor) and cached_gpu.device.type == "cuda":
                # [STAGING-SWAP-UAF-GUARD 2026-07-07] R1 单点收口:容量换代弃旧
                # GPU stage 前对三个潜在消费流 record_stream——rebuild/selector
                # 链在主流 drain 与 refresh_stream off-loop 双上下文交替,旧
                # stage 可能仍是另一流未决 copy/kernel 的 src(seq_lens/slot 两
                # 雷同款前提被打破);直接 GC 让 allocator 按创建流序复用/解映射
                # = 脏读/illegal。冷事件(容量翻倍增长),零热路径开销;pinned
                # CPU 侧由 CachingHostAllocator 自动挂事件,无需守卫。
                _guard_streams = []
                _rs = getattr(cache_owner, "refresh_stream", None)
                if _rs is not None:
                    _guard_streams.append(_rs)
                for _cand in (
                    torch.cuda.current_stream(),
                    torch.cuda.default_stream(),
                ):
                    if all(_cand != s for s in _guard_streams):
                        _guard_streams.append(_cand)
                for _s in _guard_streams:
                    cached_gpu.record_stream(_s)
            capacity = max(capacity, old_cap * 2, 1)
            cached_gpu = torch.empty((capacity,), device=device, dtype=dtype)
            _cache_set(
                stage_cache=stage_cache,
                cache_owner=cache_owner,
                key=gpu_key,
                value=cached_gpu,
            )
        gpu_stage = cached_gpu

    values_key = f"_cpu_gpu_stage_{cache_name}_values"
    if bool(reuse_unchanged) and out is None:
        cached_values = _cache_get(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=values_key,
        )
        if gpu_stage_reused and cached_values == values_tuple:
            return gpu_stage if int(gpu_stage.numel()) == length else gpu_stage[:length]

    cpu_key = f"_cpu_gpu_stage_{cache_name}_cpu"
    cpu_stage = _cache_get(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=cpu_key,
    )
    if (
        not isinstance(cpu_stage, torch.Tensor)
        or cpu_stage.device.type != "cpu"
        or cpu_stage.dtype != dtype
        or int(cpu_stage.numel()) < capacity
    ):
        cpu_stage = _new_cpu_tensor(
            (capacity,),
            dtype=dtype,
            pin_memory=True,
        )
        _cache_set(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=cpu_key,
            value=cpu_stage,
        )

    _wait_stage_h2d_evt(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=f"{cpu_key}_evt",
    )
    # [STAGE-FILL-BULK 2026-07-06] 逐元素 Python setitem（本函数是布局/payload
    # 小张量 staging 的公共地板，capture eager 步每层 20-40 次调用）换 C 层一次
    # 构造+整块拷贝：值/位置逐位等价（torch.as_tensor 对 bool/int 的转换与原
    # 逐元素 bool()/int() 同语义），META-STAGE-BULK（d63a188）同款。
    if length > 0:
        cpu_stage[:length].copy_(torch.as_tensor(values_tuple, dtype=dtype))
    if length > 0:
        gpu_stage[:length].copy_(cpu_stage[:length], non_blocking=True)
        _record_stage_h2d_evt(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=f"{cpu_key}_evt",
            device=device,
        )
    if out is None:
        _cache_set(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=values_key,
            value=values_tuple,
        )
    return gpu_stage if int(gpu_stage.numel()) == length else gpu_stage[:length]


def cached_cpu_tensor_to_device(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    cache_name: str,
    stage_cache: MutableMapping[str, object] | None = None,
    cache_owner: object | None = None,
    reuse_unchanged: bool = False,
) -> torch.Tensor:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if tensor.device == device and tensor.dtype == dtype:
        return tensor
    if tensor.device.type != "cpu":
        return tensor.to(device=device, dtype=dtype)
    if device.type != "cuda":
        return tensor.to(device=device, dtype=dtype)

    shape = tuple(int(v) for v in tensor.shape)
    key = (str(device.type), int(device.index) if device.index is not None else -1, dtype, shape)
    key_name = f"_cpu_gpu_stage_{cache_name}_key"
    cpu_name = f"_cpu_gpu_stage_{cache_name}_cpu"
    gpu_name = f"_cpu_gpu_stage_{cache_name}_gpu"
    cached_key = _cache_get(stage_cache=stage_cache, cache_owner=cache_owner, key=key_name)
    cpu_stage = _cache_get(stage_cache=stage_cache, cache_owner=cache_owner, key=cpu_name)
    gpu_stage = _cache_get(stage_cache=stage_cache, cache_owner=cache_owner, key=gpu_name)
    if (
        cached_key != key
        or not isinstance(cpu_stage, torch.Tensor)
        or not isinstance(gpu_stage, torch.Tensor)
        or cpu_stage.device.type != "cpu"
        or gpu_stage.device != device
        or cpu_stage.dtype != dtype
        or gpu_stage.dtype != dtype
        or tuple(cpu_stage.shape) != shape
        or tuple(gpu_stage.shape) != shape
    ):
        cpu_stage = _new_cpu_tensor(shape, dtype=dtype, pin_memory=True)
        gpu_stage = torch.empty(shape, device=device, dtype=dtype)
        _cache_set(stage_cache=stage_cache, cache_owner=cache_owner, key=key_name, value=key)
        _cache_set(stage_cache=stage_cache, cache_owner=cache_owner, key=cpu_name, value=cpu_stage)
        _cache_set(stage_cache=stage_cache, cache_owner=cache_owner, key=gpu_name, value=gpu_stage)
    values_name = f"_cpu_gpu_stage_{cache_name}_values"
    if bool(reuse_unchanged):
        values_tuple = tuple(int(v) for v in tensor.reshape(-1).tolist())
        cached_values = _cache_get(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=values_name,
        )
        if cached_values == values_tuple:
            return gpu_stage
    else:
        values_tuple = None
    _wait_stage_h2d_evt(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=f"{cpu_name}_evt",
    )
    cpu_stage.copy_(tensor)
    gpu_stage.copy_(cpu_stage, non_blocking=True)
    _record_stage_h2d_evt(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=f"{cpu_name}_evt",
        device=device,
    )
    if values_tuple is not None:
        _cache_set(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=values_name,
            value=values_tuple,
        )
    return gpu_stage
