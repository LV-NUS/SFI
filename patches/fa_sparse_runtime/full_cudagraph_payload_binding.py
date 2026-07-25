"""Graph-owned inputs for post-replay sparse refresh payload construction.

The tensors in this registry are raw CUDA Graph inputs.  Their lifetime and
identity belong to the captured graph, not to request-scoped ``LayerState``.
Registration is capture-only; replay lookup runs only on refresh-trigger
steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Dict, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True, slots=True)
class FullCudagraphReplayLayerPayloadBinding:
    cache_key: int
    layer_index: int
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    q: torch.Tensor
    cu_seqlens_q: torch.Tensor
    softmax_scale: float
    softcap: float
    window_size: Optional[Tuple[int, int]]
    alibi_slopes: Optional[torch.Tensor]
    k_descale: Optional[torch.Tensor]
    static_signature: Tuple[object, ...]


@dataclass(frozen=True, slots=True)
class FullCudagraphReplayPayloadGraphBindings:
    capture_generation: int
    graph_batch_size: int
    num_actual_tokens: int
    ordered_layer_keys: Tuple[int, ...]
    ordered_layer_bindings: Tuple[
        FullCudagraphReplayLayerPayloadBinding,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class FullCudagraphReplayPayloadCapture:
    """Opaque ownership token for one actual FULL CUDA Graph capture."""

    graph_key: str
    capture_generation: int


@dataclass(slots=True)
class _CaptureGraphBindings:
    capture: FullCudagraphReplayPayloadCapture
    graph_batch_size: int
    num_actual_tokens: int
    layer_bindings: Dict[int, FullCudagraphReplayLayerPayloadBinding]


def _tensor_signature(tensor: torch.Tensor) -> Tuple[object, ...]:
    return (
        int(tensor.data_ptr()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    rank: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dim() != rank:
        raise RuntimeError(
            "FULL cudagraph replay payload capture requires "
            f"rank-{rank} tensor {name}"
        )
    return value


def _optional_tensor(
    value: object,
    *,
    name: str,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            "FULL cudagraph replay payload capture requires optional tensor "
            f"{name}"
        )
    return value


def _normalize_optional_scalar(value: object, *, name: str) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(
            "FULL cudagraph replay payload capture requires numeric scalar "
            f"{name}"
        )
    return float(value)


class FullCudagraphReplayPayloadRegistry:
    """Own exact graph bindings across request-state resets."""

    __slots__ = (
        "_active_capture",
        "_capture_record",
        "_capture_generation",
        "_records",
    )

    def __init__(self) -> None:
        self._active_capture: Optional[
            FullCudagraphReplayPayloadCapture
        ] = None
        self._capture_record: Optional[_CaptureGraphBindings] = None
        self._capture_generation = 0
        self._records: Dict[
            str,
            FullCudagraphReplayPayloadGraphBindings,
        ] = {}

    @property
    def capture_generation(self) -> int:
        return int(self._capture_generation)

    def begin_capture(
        self,
        *,
        graph_key: str,
    ) -> FullCudagraphReplayPayloadCapture:
        if not isinstance(graph_key, str):
            raise RuntimeError(
                "FULL cudagraph replay payload capture requires string graph "
                "identity"
            )
        graph_key_s = graph_key
        if (
            not graph_key_s.startswith("full:")
            or graph_key_s.startswith("full:num_tokens=")
        ):
            raise RuntimeError(
                "FULL cudagraph replay payload capture requires exact "
                "BatchDescriptor graph identity"
            )
        if self._active_capture is not None:
            raise RuntimeError(
                "FULL cudagraph replay payload capture began before the prior "
                "capture was finalized"
            )
        self._capture_generation += 1
        capture = FullCudagraphReplayPayloadCapture(
            graph_key=graph_key_s,
            capture_generation=int(self._capture_generation),
        )
        self._active_capture = capture
        self._capture_record = None
        return capture

    def clear(self) -> None:
        self._active_capture = None
        self._capture_record = None
        self._records.clear()
        self._capture_generation = 0

    def register_layer(
        self,
        *,
        capture: FullCudagraphReplayPayloadCapture,
        cache_key: int,
        layer_index: int,
        graph_batch_size: int,
        num_actual_tokens: int,
        key_cache: object,
        value_cache: object,
        block_table: object,
        q: object,
        cu_seqlens_q: object,
        softmax_scale: object,
        softcap: object,
        window_size: object,
        alibi_slopes: object,
        k_descale: object,
    ) -> FullCudagraphReplayLayerPayloadBinding:
        if (
            not isinstance(capture, FullCudagraphReplayPayloadCapture)
            or capture is not self._active_capture
        ):
            raise RuntimeError(
                "FULL cudagraph replay payload capture has invalid owner token"
            )
        graph_key_s = str(capture.graph_key)
        capture_generation_i = int(capture.capture_generation)
        cache_key_i = int(cache_key)
        layer_index_i = int(layer_index)
        graph_batch_size_i = int(graph_batch_size)
        num_actual_tokens_i = int(num_actual_tokens)
        if (
            capture_generation_i <= 0
            or capture_generation_i != self._capture_generation
            or cache_key_i <= 0
            or layer_index_i < 0
            or graph_batch_size_i <= 0
            or num_actual_tokens_i <= 0
        ):
            raise RuntimeError(
                "FULL cudagraph replay payload capture has invalid identity"
            )

        key_cache_t = _require_tensor(key_cache, name="key_cache", rank=4)
        value_cache_t = _require_tensor(value_cache, name="value_cache", rank=4)
        block_table_t = _require_tensor(block_table, name="block_table", rank=2)
        q_t = _require_tensor(q, name="q", rank=3)
        cu_seqlens_q_t = _require_tensor(
            cu_seqlens_q,
            name="cu_seqlens_q",
            rank=1,
        )
        alibi_slopes_t = _optional_tensor(
            alibi_slopes,
            name="alibi_slopes",
        )
        k_descale_t = _optional_tensor(k_descale, name="k_descale")

        if int(key_cache_t.data_ptr()) != cache_key_i:
            raise RuntimeError(
                "FULL cudagraph replay payload capture cache identity mismatch"
            )
        if (
            key_cache_t.device != value_cache_t.device
            or key_cache_t.device != block_table_t.device
            or key_cache_t.device != q_t.device
            or key_cache_t.device != cu_seqlens_q_t.device
            or key_cache_t.dtype != value_cache_t.dtype
            or tuple(key_cache_t.shape[:3]) != tuple(value_cache_t.shape[:3])
        ):
            raise RuntimeError(
                "FULL cudagraph replay payload capture tensor device/dtype mismatch"
            )
        if (
            int(cu_seqlens_q_t.numel()) != graph_batch_size_i + 1
            or int(block_table_t.shape[0]) < graph_batch_size_i
            or int(q_t.shape[0]) < num_actual_tokens_i
        ):
            raise RuntimeError(
                "FULL cudagraph replay payload capture batch/shape mismatch"
            )

        if window_size is None:
            frozen_window_size = None
        else:
            if not isinstance(window_size, (tuple, list)):
                raise RuntimeError(
                    "FULL cudagraph replay payload capture requires sequence "
                    "window_size"
                )
            try:
                frozen_window_size = tuple(int(value) for value in window_size)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "FULL cudagraph replay payload capture has invalid window_size"
                ) from exc
            if len(frozen_window_size) != 2:
                raise RuntimeError(
                    "FULL cudagraph replay payload capture requires two-value "
                    "window_size"
                )

        softmax_scale_f = _normalize_optional_scalar(
            softmax_scale,
            name="softmax_scale",
        )
        softcap_f = _normalize_optional_scalar(softcap, name="softcap")
        static_signature = (
            cache_key_i,
            layer_index_i,
            graph_batch_size_i,
            num_actual_tokens_i,
            _tensor_signature(key_cache_t),
            _tensor_signature(value_cache_t),
            _tensor_signature(block_table_t),
            _tensor_signature(q_t),
            _tensor_signature(cu_seqlens_q_t),
            softmax_scale_f,
            softcap_f,
            frozen_window_size,
            (
                _tensor_signature(alibi_slopes_t)
                if alibi_slopes_t is not None
                else None
            ),
            _tensor_signature(k_descale_t) if k_descale_t is not None else None,
        )
        binding = FullCudagraphReplayLayerPayloadBinding(
            cache_key=cache_key_i,
            layer_index=layer_index_i,
            key_cache=key_cache_t,
            value_cache=value_cache_t,
            block_table=block_table_t,
            q=q_t,
            cu_seqlens_q=cu_seqlens_q_t,
            softmax_scale=softmax_scale_f,
            softcap=softcap_f,
            window_size=frozen_window_size,
            alibi_slopes=alibi_slopes_t,
            k_descale=k_descale_t,
            static_signature=static_signature,
        )

        record = self._capture_record
        if record is None:
            record = _CaptureGraphBindings(
                capture=capture,
                graph_batch_size=graph_batch_size_i,
                num_actual_tokens=num_actual_tokens_i,
                layer_bindings={},
            )
            self._capture_record = record
        elif (
            record.capture is not capture
            or str(record.capture.graph_key) != graph_key_s
            or int(record.graph_batch_size) != graph_batch_size_i
            or int(record.num_actual_tokens) != num_actual_tokens_i
        ):
            raise RuntimeError(
                "FULL cudagraph replay payload graph geometry changed within capture"
            )

        previous = record.layer_bindings.get(cache_key_i)
        if (
            previous is not None
            and previous.static_signature != binding.static_signature
        ):
            raise RuntimeError(
                "FULL cudagraph replay payload layer binding changed within capture"
            )
        record.layer_bindings[cache_key_i] = binding
        return binding

    def finalize_capture(
        self,
        *,
        capture: FullCudagraphReplayPayloadCapture,
        layer_keys: Sequence[int],
    ) -> FullCudagraphReplayPayloadGraphBindings:
        """Publish one immutable, ordered graph record at capture completion."""
        if (
            not isinstance(capture, FullCudagraphReplayPayloadCapture)
            or capture is not self._active_capture
        ):
            raise RuntimeError(
                "FULL cudagraph replay payload finalization has invalid owner token"
            )
        graph_key_s = str(capture.graph_key)
        capture_generation_i = int(capture.capture_generation)
        if capture_generation_i != self._capture_generation:
            raise RuntimeError(
                "FULL cudagraph replay payload finalization has stale generation"
            )

        record = self._capture_record
        if record is None or record.capture is not capture:
            raise RuntimeError(
                "FULL cudagraph replay payload finalization is missing capture record"
            )

        ordered_keys = tuple(int(cache_key) for cache_key in layer_keys)
        if not ordered_keys or len(set(ordered_keys)) != len(ordered_keys):
            raise RuntimeError(
                "FULL cudagraph replay payload finalization requires unique "
                "ordered layer keys"
            )
        if set(record.layer_bindings) != set(ordered_keys):
            raise RuntimeError(
                "FULL cudagraph replay payload capture has incomplete or foreign "
                "layer bindings"
            )

        ordered_bindings = tuple(
            record.layer_bindings[cache_key] for cache_key in ordered_keys
        )
        for layer_index, binding in enumerate(ordered_bindings):
            if (
                int(binding.cache_key) != int(ordered_keys[layer_index])
                or int(binding.layer_index) != int(layer_index)
            ):
                raise RuntimeError(
                    "FULL cudagraph replay payload capture layer order mismatch"
                )
        finalized_record = FullCudagraphReplayPayloadGraphBindings(
            capture_generation=capture_generation_i,
            graph_batch_size=int(record.graph_batch_size),
            num_actual_tokens=int(record.num_actual_tokens),
            ordered_layer_keys=ordered_keys,
            ordered_layer_bindings=ordered_bindings,
        )
        self._records[graph_key_s] = finalized_record
        self._capture_record = None
        self._active_capture = None
        return finalized_record

    def require_graph(
        self,
        *,
        graph_key: str,
        layer_keys: Sequence[int],
        live_batch_size: int,
    ) -> FullCudagraphReplayPayloadGraphBindings:
        graph_key_s = str(graph_key)
        if graph_key_s.startswith("full:num_tokens="):
            raise RuntimeError(
                "full cudagraph replay refresh payload does not support "
                "UBatch num_tokens graph identity"
            )
        if self._active_capture is not None:
            raise RuntimeError(
                "full cudagraph replay refresh payload graph was not finalized "
                "at capture completion"
            )
        record = self._records.get(graph_key_s)
        if record is None:
            raise RuntimeError(
                "full cudagraph replay refresh payload binding missing for "
                "exact graph key"
            )
        if len(record.ordered_layer_keys) != len(layer_keys):
            raise RuntimeError(
                "full cudagraph replay refresh payload graph layer count changed"
            )
        for layer_index, cache_key in enumerate(layer_keys):
            if int(cache_key) != int(record.ordered_layer_keys[layer_index]):
                raise RuntimeError(
                    "full cudagraph replay refresh payload graph layer order changed"
                )
        live_batch_size_i = int(live_batch_size)
        if (
            live_batch_size_i <= 0
            or live_batch_size_i > int(record.graph_batch_size)
        ):
            raise RuntimeError(
                "full cudagraph replay refresh payload graph batch mismatch"
            )
        return record
