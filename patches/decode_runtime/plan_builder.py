from __future__ import annotations


def build_decode_reuse_order_decision(
    *,
    has_cached_layer_list: bool,
    cached_step_data_identity: bool,
    cached_order_cache_key: tuple[object, ...] | None,
    current_cache_key: tuple[object, ...] | None,
) -> bool:
    return (
        bool(has_cached_layer_list)
        and bool(cached_step_data_identity)
        and cached_order_cache_key == current_cache_key
    )

