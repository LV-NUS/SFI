from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(slots=True)
class StepRuntimeState:
    ordered_layer_cursor: int = 0
    ordered_layer_count: int = 0
    ordered_layer_epoch: int = -1
    ordered_batch_size: int = 0
    ordered_cache_key: Optional[Tuple[object, ...]] = None

    @staticmethod
    def _require_int(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {type(value).__name__}")
        return value

    def reset_for_new_step(self, *, epoch: int) -> None:
        epoch_i = self._require_int("epoch", epoch)
        self.reset_ordered_layer_plan(epoch=epoch_i)

    def apply_ordered_layer_plan(
        self,
        *,
        epoch: int,
        layer_count: int,
        batch_size: int,
        cache_key: Optional[Tuple[object, ...]],
    ) -> None:
        epoch_i = self._require_int("epoch", epoch)
        layer_count_i = self._require_int("layer_count", layer_count)
        batch_size_i = self._require_int("batch_size", batch_size)
        if layer_count_i <= 0:
            raise ValueError(f"layer_count must be positive, got {layer_count}")
        if batch_size_i < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        self.ordered_layer_epoch = epoch_i
        self.ordered_layer_cursor = 0
        self.ordered_layer_count = layer_count_i
        self.ordered_batch_size = batch_size_i
        self.ordered_cache_key = cache_key

    def reset_ordered_layer_plan(self, *, epoch: int) -> None:
        epoch_i = self._require_int("epoch", epoch)
        self.ordered_layer_cursor = 0
        self.ordered_layer_count = 0
        self.ordered_layer_epoch = epoch_i
        self.ordered_batch_size = 0
        self.ordered_cache_key = None

    def reset_ordered_layer_cursor_for_step(self, *, epoch: int) -> None:
        epoch_i = self._require_int("epoch", epoch)
        if self.ordered_layer_count > 0:
            self.ordered_layer_epoch = epoch_i
            self.ordered_layer_cursor = 0
            return
        self.reset_ordered_layer_plan(epoch=epoch_i)

    def advance_ordered_layer_cursor(self, *, current_epoch: int) -> int:
        current_epoch_i = self._require_int("current_epoch", current_epoch)
        if self.ordered_layer_epoch != current_epoch_i:
            # Epoch changed without applying a new ordered-layer plan.
            # Invalidate stale fields and fail-fast on access.
            self.reset_ordered_layer_plan(epoch=current_epoch_i)
        cursor = self.ordered_layer_cursor
        if cursor < 0 or cursor >= self.ordered_layer_count:
            raise RuntimeError(
                f"Ordered layer cursor out of range: cursor={cursor} layers={self.ordered_layer_count}"
            )
        self.ordered_layer_cursor = cursor + 1
        return cursor
