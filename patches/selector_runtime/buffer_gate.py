from __future__ import annotations


def should_resize_for_batch(*, capacity: int, batch_size: int) -> bool:
    return int(batch_size) > int(capacity)

