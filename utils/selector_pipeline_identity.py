"""Lightweight single source of truth for the selector extension identity."""

from __future__ import annotations


SELECTOR_PIPELINE_SEMANTIC_VERSION = 2026071303
SELECTOR_PIPELINE_EXTENSION_NAME = (
    f"selector_pipeline_ext_v{SELECTOR_PIPELINE_SEMANTIC_VERSION}"
)
SELECTOR_PIPELINE_CPP_SEMANTIC_TOKEN = (
    "__SFI_SELECTOR_PIPELINE_SEMANTIC_VERSION__"
)


def render_selector_pipeline_cpp_source(source: str) -> str:
    """Inject the Python semantic version into exactly one C++ source slot."""
    token_count = source.count(SELECTOR_PIPELINE_CPP_SEMANTIC_TOKEN)
    if token_count != 1:
        raise ValueError(
            "selector pipeline C++ semantic token must occur exactly once: "
            f"count={token_count}"
        )
    return source.replace(
        SELECTOR_PIPELINE_CPP_SEMANTIC_TOKEN,
        str(SELECTOR_PIPELINE_SEMANTIC_VERSION),
    )
