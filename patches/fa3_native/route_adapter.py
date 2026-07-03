from __future__ import annotations

from dataclasses import dataclass


_ATTR_HAS_SELECTED_CONSUME = "fa3_has_selected_consume"
_ATTR_HAS_CAPTURE = "fa3_has_capture"
_ATTR_HAS_COMPACT_RECENT = "fa3_has_compact_recent"

_DENSE_NATIVE_ROUTE = "flash_attn_varlen_func"
_MIXED_PAGE_ROUTE = "mixed_page_attn_varlen_func"
_COMPACT_RECENT_ROUTE = "compact_recent_attn_varlen_func"


@dataclass(frozen=True)
class LaunchRouteHints:
    has_selected_consume: bool
    has_capture: bool
    has_compact_recent: bool = False


def route_attention_launch(
    *,
    has_selected_consume: bool,
    has_capture: bool,
    has_compact_recent: bool = False,
) -> str:
    if has_compact_recent:
        raise RuntimeError(
            "compact_recent is no longer a production FA3 route; use "
            "mixed_page_attn_varlen_func with selected_page_table_i32 overlay or an "
            "explicit test-only compact_recent reference"
        )
    if not has_selected_consume and not has_capture:
        return _DENSE_NATIVE_ROUTE
    return _MIXED_PAGE_ROUTE


def bind_launch_route_hints(
    attn_metadata: object,
    *,
    has_selected_consume: bool,
    has_capture: bool,
    has_compact_recent: bool = False,
) -> object:
    setattr(attn_metadata, _ATTR_HAS_SELECTED_CONSUME, bool(has_selected_consume))
    setattr(attn_metadata, _ATTR_HAS_CAPTURE, bool(has_capture))
    setattr(attn_metadata, _ATTR_HAS_COMPACT_RECENT, bool(has_compact_recent))
    return attn_metadata


def get_bound_launch_route_hints(attn_metadata: object) -> LaunchRouteHints:
    return LaunchRouteHints(
        has_selected_consume=bool(
            getattr(attn_metadata, _ATTR_HAS_SELECTED_CONSUME, False)
        ),
        has_capture=bool(getattr(attn_metadata, _ATTR_HAS_CAPTURE, False)),
        has_compact_recent=bool(
            getattr(attn_metadata, _ATTR_HAS_COMPACT_RECENT, False)
        ),
    )


def route_attention_launch_from_attn_metadata(attn_metadata: object) -> str:
    hints = get_bound_launch_route_hints(attn_metadata)
    return route_attention_launch(
        has_selected_consume=hints.has_selected_consume,
        has_capture=hints.has_capture,
        has_compact_recent=hints.has_compact_recent,
    )
