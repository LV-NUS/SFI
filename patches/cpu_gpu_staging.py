from __future__ import annotations

from typing import MutableMapping, Sequence

import torch


_STAGE_H2D_QUARANTINED = object()
_STAGE_GPU_VALUE_INVALID = object()


def _annotate_stage_h2d_error(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    # Python 3.10 compatibility for the runtime environment.  Keep the
    # original exception type/message stable and attach diagnostics out of
    # band rather than replacing the causative copy/record failure.
    setattr(error, "_stage_h2d_note", note)


def _new_cpu_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch.empty(
        shape,
        device="cpu",
        dtype=dtype,
        pin_memory=bool(pin_memory),
    )
    if pin_memory and not tensor.is_pinned():
        raise RuntimeError("CUDA H2D staging source must use pinned CPU memory")
    return tensor


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


def _stage_ring_slot_key(base_key: str, slot: int) -> str:
    return base_key if slot == 0 else f"{base_key}_ring{slot}"


def _current_cuda_stream_contract(
    device: torch.device,
) -> tuple[object, tuple[int, int]]:
    stream = torch.cuda.current_stream(device=device)
    raw_stream = getattr(stream, "cuda_stream", None)
    if raw_stream is None:
        raise RuntimeError("CUDA staging stream must expose cuda_stream")
    device_index = (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )
    return stream, (device_index, int(raw_stream))


def _stage_pool_state(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    base_key: str,
    stream_contract: tuple[int, int],
) -> dict[str, object]:
    """Return the one-stream FIFO source pool for ``base_key``.

    The stable GPU destination is cached under the same logical cache name.
    Binding the whole cache entry to the actual ``(device, cudaStream_t)`` is
    therefore required: per-stream CPU sources alone would not make concurrent
    writes to that shared GPU destination safe.
    """

    if stage_cache is None and cache_owner is None:
        raise ValueError(
            "CUDA H2D staging requires a persistent stage_cache or cache_owner"
        )
    state_key = f"{base_key}_pool"
    state = _cache_get(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=state_key,
    )
    if state is None:
        state = {
            "stream": stream_contract,
            "slots": [base_key],
            "next": 0,
            "next_slot": 1,
        }
        _cache_set(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=state_key,
            value=state,
        )
        return state
    if not isinstance(state, dict):
        raise RuntimeError("CUDA H2D staging pool state is invalid")
    if state.get("stream") != stream_contract:
        raise RuntimeError(
            "CUDA H2D staging cache cannot share one GPU destination across "
            "different device/stream owners"
        )
    slots = state.get("slots")
    if not isinstance(slots, list) or not slots:
        raise RuntimeError("CUDA H2D staging pool slots are invalid")
    if not isinstance(state.get("next"), int) or not isinstance(
        state.get("next_slot"), int
    ):
        raise RuntimeError("CUDA H2D staging pool cursor is invalid")
    return state


def _validate_unchanged_stage_stream_owner(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    base_key: str,
    device: torch.device,
) -> None:
    """Validate a no-copy reuse against the destination's stream owner.

    Same-stream reuse is ordered by the CUDA stream even while the original
    H2D is pending.  A different stream has no such dependency and therefore
    must not consume the shared destination.  This check performs no event
    query, wait, synchronization, or device transfer.
    """

    _stream, stream_contract = _current_cuda_stream_contract(device)
    state = _cache_get(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=f"{base_key}_pool",
    )
    if state is None:
        raise RuntimeError(
            "CUDA H2D staging reuse has no stream-ownership proof"
        )
    _stage_pool_state(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        base_key=base_key,
        stream_contract=stream_contract,
    )


def _acquire_stage_ring_slot(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    base_key: str,
    stream_contract: tuple[int, int],
) -> str:
    """Acquire one FIFO pinned source without scanning or blocking the host.

    The GPU destination remains single-buffered so callers retain a stable
    data pointer (including CUDA-graph inputs).  One device/stream owns the
    pool, so its cursor is always the oldest submitted source.  A single event
    query either proves that source reusable or measures real pressure and
    grows exactly one newest slot.  There is no empirical cap, linear scan, or
    host synchronization fallback.
    """

    state = _stage_pool_state(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        base_key=base_key,
        stream_contract=stream_contract,
    )
    slots = state["slots"]
    while True:
        oldest_index = int(state.get("next", 0)) % len(slots)
        oldest_key = slots[oldest_index]
        if not isinstance(oldest_key, str):
            raise RuntimeError("CUDA H2D staging pool slot key is invalid")
        evt = _cache_get(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=f"{oldest_key}_evt",
        )
        if evt is _STAGE_H2D_QUARANTINED:
            # Event publication failed after a copy might have entered the
            # original stream.  Keep the source tensor referenced by its cache
            # key, but retire it from the reusable FIFO forever: without an
            # event there can never be a completion proof.
            slots.pop(oldest_index)
            if slots:
                state["next"] = oldest_index % len(slots)
                continue
            next_slot = int(state.get("next_slot", 0))
            if next_slot <= 0:
                raise RuntimeError(
                    "CUDA H2D staging pool slot generation is invalid"
                )
            newest_key = _stage_ring_slot_key(base_key, next_slot)
            state["next_slot"] = next_slot + 1
            slots.append(newest_key)
            state["next"] = 0
            return newest_key
        # [GUARD-NO-SWALLOW] query 失败意味着无法证明 pinned source 可覆写，
        # 必须原样传播，不能把坏事件当成空闲槽。
        if evt is None or evt.query():
            state["next"] = (oldest_index + 1) % len(slots)
            return oldest_key
        break

    next_slot = int(state.get("next_slot", 0))
    if next_slot <= 0:
        raise RuntimeError("CUDA H2D staging pool slot generation is invalid")
    newest_key = _stage_ring_slot_key(base_key, next_slot)
    state["next_slot"] = next_slot + 1
    # The physical tail is not the temporal tail when the cursor is nonzero.
    # Insert the new submission immediately before the oldest physical index;
    # the old oldest shifts right and remains the cursor.  Logical FIFO order
    # is therefore [oldest, ..., previous-newest, new-slot] after wrap/growth.
    slots.insert(oldest_index, newest_key)
    state["next"] = oldest_index + 1
    return newest_key


def _acquire_stage_h2d_group(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    base_key: str,
    device: torch.device,
) -> tuple[str, object]:
    """Acquire one slot shared by a group of pinned source tensors.

    Callers derive the same slot suffix for every member and record the slot
    event only after the group's final H2D.  This preserves the atomic lifetime
    formerly expressed by one blocking event around multiple source tensors.
    """

    if device.type != "cuda":
        raise ValueError("CUDA H2D staging group requires a CUDA device")
    stream, stream_contract = _current_cuda_stream_contract(device)
    slot_key = _acquire_stage_ring_slot(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        base_key=base_key,
        stream_contract=stream_contract,
    )
    return slot_key, stream


def _record_stage_h2d_evt(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    key: str,
    device: torch.device,
    stream: object | None = None,
) -> None:
    if device.type != "cuda":
        return
    # [GUARD-NO-SWALLOW] record 失败时静默返回=槽位没有完成证明，
    # 下次 acquire 会把在飞 source 当成未使用并无声覆写，必须炸。
    try:
        evt = _cache_get(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=key,
        )
        if evt is _STAGE_H2D_QUARANTINED:
            evt = None
        if evt is not None:
            event_device = getattr(evt, "device", None)
            target_index = (
                int(device.index)
                if device.index is not None
                else int(torch.cuda.current_device())
            )
            if (
                event_device is not None
                and (
                    torch.device(event_device).type != "cuda"
                    or torch.device(event_device).index != target_index
                )
            ):
                # TP worker/device 迁移后旧 event 仍绑定原 CUDA device；CPU source
                # 已由 acquire 证明空闲，但 event 本体不能跨 device 重新 record。
                evt = None
        if evt is None:
            evt = torch.cuda.Event(enable_timing=False)
        evt.record(
            stream if stream is not None else torch.cuda.current_stream(device=device)
        )
        # Publishing the recorded event is part of the same ownership
        # transaction.  If cache publication fails, treating the slot as if
        # it still carried an older/empty event would be a false completion
        # proof just as surely as a failed Event.record().
        _cache_set(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=key,
            value=evt,
        )
    except BaseException as exc:
        # A prior event can already be complete, so leaving it cached after a
        # failed re-record would create a false completion proof.  Replace it
        # with a permanent quarantine marker before propagating the failure.
        try:
            _cache_set(
                stage_cache=stage_cache,
                cache_owner=cache_owner,
                key=key,
                value=_STAGE_H2D_QUARANTINED,
            )
        except BaseException as quarantine_exc:
            _annotate_stage_h2d_error(
                exc,
                "failed to publish the H2D completion event and to quarantine "
                f"its source slot: {quarantine_exc!r}",
            )
        raise


def _protect_failed_stage_h2d(
    *,
    stage_cache: MutableMapping[str, object] | None,
    cache_owner: object | None,
    key: str,
    device: torch.device,
    stream: object,
    error: BaseException,
) -> None:
    """Fence a possibly partial H2D submission without masking its error."""

    try:
        _record_stage_h2d_evt(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=key,
            device=device,
            stream=stream,
        )
    except BaseException as fence_exc:
        # _record_stage_h2d_evt already quarantined the slot.  Preserve the
        # original copy failure while making the failed completion proof
        # visible to diagnostics.
        _annotate_stage_h2d_error(
            error,
            "failed to publish completion proof for a partial H2D submission; "
            f"the source slot was quarantined: {fence_exc!r}",
        )


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
    cpu_base_key = f"_cpu_gpu_stage_{cache_name}_cpu"
    if bool(reuse_unchanged) and out is None:
        cached_values = _cache_get(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=values_key,
        )
        if gpu_stage_reused and cached_values == values_tuple:
            if length > 0:
                _validate_unchanged_stage_stream_owner(
                    stage_cache=stage_cache,
                    cache_owner=cache_owner,
                    base_key=cpu_base_key,
                    device=device,
                )
            return gpu_stage if int(gpu_stage.numel()) == length else gpu_stage[:length]

    # Empty row/slot cohorts are legal at capture-layout boundaries. There is
    # no source lifetime to protect because no H2D is submitted; preserve the
    # existing GPU empty-view/value-cache result without touching CUDA stream
    # or pool state.
    if length == 0:
        if out is None:
            _cache_set(
                stage_cache=stage_cache,
                cache_owner=cache_owner,
                key=values_key,
                value=values_tuple,
            )
        return gpu_stage[:0]

    # The cached value label is the commit record for the stable GPU
    # destination.  Invalidate it before a changed H2D can touch that
    # destination, then publish the new tuple only after both the copy and its
    # completion event have been published.  A partial copy/record failure can
    # therefore never make an older label describe contaminated GPU storage.
    # This write exists only on the changed-copy path; unchanged hits above pay
    # no extra branch or cache mutation.
    _cache_set(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=values_key,
        value=_STAGE_GPU_VALUE_INVALID,
    )

    # The unchanged-value hit above performs only a host-side stream-identity
    # check. Event queries and source-pool acquisition remain H2D-only work.
    current_stream, stream_contract = _current_cuda_stream_contract(device)
    cpu_key = _acquire_stage_ring_slot(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        base_key=cpu_base_key,
        stream_contract=stream_contract,
    )
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

    # [STAGE-FILL-BULK 2026-07-06] 逐元素 Python setitem（本函数是布局/payload
    # 小张量 staging 的公共地板，capture eager 步每层 20-40 次调用）换 C 层一次
    # 构造+整块拷贝：值/位置逐位等价（torch.as_tensor 对 bool/int 的转换与原
    # 逐元素 bool()/int() 同语义），META-STAGE-BULK（d63a188）同款。
    cpu_stage[:length].copy_(torch.as_tensor(values_tuple, dtype=dtype))
    try:
        gpu_stage[:length].copy_(cpu_stage[:length], non_blocking=True)
    except BaseException as exc:
        _protect_failed_stage_h2d(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=f"{cpu_key}_evt",
            device=device,
            stream=current_stream,
            error=exc,
        )
        raise
    _record_stage_h2d_evt(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=f"{cpu_key}_evt",
        device=device,
        stream=current_stream,
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
    cpu_base_name = f"_cpu_gpu_stage_{cache_name}_cpu"
    gpu_name = f"_cpu_gpu_stage_{cache_name}_gpu"
    cached_key = _cache_get(stage_cache=stage_cache, cache_owner=cache_owner, key=key_name)
    gpu_stage = _cache_get(stage_cache=stage_cache, cache_owner=cache_owner, key=gpu_name)
    gpu_stage_reused = not (
        cached_key != key
        or not isinstance(gpu_stage, torch.Tensor)
        or gpu_stage.device != device
        or gpu_stage.dtype != dtype
        or tuple(gpu_stage.shape) != shape
    )
    if not gpu_stage_reused:
        gpu_stage = torch.empty(shape, device=device, dtype=dtype)
        _cache_set(stage_cache=stage_cache, cache_owner=cache_owner, key=key_name, value=key)
        _cache_set(stage_cache=stage_cache, cache_owner=cache_owner, key=gpu_name, value=gpu_stage)
    values_name = f"_cpu_gpu_stage_{cache_name}_values"
    if bool(reuse_unchanged):
        values_tuple = tuple(int(v) for v in tensor.reshape(-1).tolist())
        cached_values = _cache_get(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=values_name,
        )
        # 容量/shape/dtype/device 换代后的 GPU stage 尚未写入，即便 host
        # values 与上代相同也不能复用；否则会把未初始化新 buffer 当命中返回。
        if gpu_stage_reused and cached_values == values_tuple:
            if tensor.numel() > 0:
                _validate_unchanged_stage_stream_owner(
                    stage_cache=stage_cache,
                    cache_owner=cache_owner,
                    base_key=cpu_base_name,
                    device=device,
                )
            return gpu_stage
    else:
        values_tuple = None
    # See cached_sequence_to_device: the value label commits only after the
    # H2D event is safely published.  Even a caller that currently opts out of
    # unchanged reuse must invalidate an older label for the same cache name;
    # otherwise a later opt-in call could mistake partially overwritten GPU
    # storage for that older value.  No tuple materialization is added to the
    # opt-out path.
    _cache_set(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=values_name,
        value=_STAGE_GPU_VALUE_INVALID,
    )
    # The unchanged-value hit above performs only a host-side stream-identity
    # check. Event queries and source-pool acquisition remain H2D-only work.
    current_stream, stream_contract = _current_cuda_stream_contract(device)
    cpu_name = _acquire_stage_ring_slot(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        base_key=cpu_base_name,
        stream_contract=stream_contract,
    )
    cpu_stage = _cache_get(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=cpu_name,
    )
    if (
        not isinstance(cpu_stage, torch.Tensor)
        or cpu_stage.device.type != "cpu"
        or cpu_stage.dtype != dtype
        or tuple(cpu_stage.shape) != shape
    ):
        cpu_stage = _new_cpu_tensor(shape, dtype=dtype, pin_memory=True)
        _cache_set(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=cpu_name,
            value=cpu_stage,
        )
    cpu_stage.copy_(tensor)
    try:
        gpu_stage.copy_(cpu_stage, non_blocking=True)
    except BaseException as exc:
        _protect_failed_stage_h2d(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=f"{cpu_name}_evt",
            device=device,
            stream=current_stream,
            error=exc,
        )
        raise
    _record_stage_h2d_evt(
        stage_cache=stage_cache,
        cache_owner=cache_owner,
        key=f"{cpu_name}_evt",
        device=device,
        stream=current_stream,
    )
    if values_tuple is not None:
        _cache_set(
            stage_cache=stage_cache,
            cache_owner=cache_owner,
            key=values_name,
            value=values_tuple,
        )
    return gpu_stage
