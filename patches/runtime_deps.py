from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

_MISSING = object()


@dataclass(frozen=True, slots=True)
class _DomainDeps:
    values: Dict[str, Any]

    def require(self, name: str) -> Any:
        value = self.values.get(name, _MISSING)
        if value is _MISSING:
            raise RuntimeError(f"runtime dependency not bound in domain deps: {name}")
        return value


@dataclass(frozen=True, slots=True)
class DecodeDeps(_DomainDeps):
    pass


@dataclass(frozen=True, slots=True)
class RefreshDeps(_DomainDeps):
    pass


@dataclass(frozen=True, slots=True)
class SelectorDeps(_DomainDeps):
    pass


_decode_deps = DecodeDeps(values={})
_refresh_deps = RefreshDeps(values={})
_selector_deps = SelectorDeps(values={})
_runtime_deps: Dict[str, Any] = {}


def _merge_domain_deps() -> None:
    _runtime_deps.clear()
    _runtime_deps.update(_decode_deps.values)
    _runtime_deps.update(_refresh_deps.values)
    _runtime_deps.update(_selector_deps.values)


def bind_decode_deps(mapping: Mapping[str, Any]) -> None:
    global _decode_deps
    _decode_deps = DecodeDeps(values=dict(mapping))
    _merge_domain_deps()


def bind_refresh_deps(mapping: Mapping[str, Any]) -> None:
    global _refresh_deps
    _refresh_deps = RefreshDeps(values=dict(mapping))
    _merge_domain_deps()


def bind_selector_deps(mapping: Mapping[str, Any]) -> None:
    global _selector_deps
    _selector_deps = SelectorDeps(values=dict(mapping))
    _merge_domain_deps()


def clear_runtime_deps() -> None:
    global _decode_deps, _refresh_deps, _selector_deps
    _decode_deps = DecodeDeps(values={})
    _refresh_deps = RefreshDeps(values={})
    _selector_deps = SelectorDeps(values={})
    _runtime_deps.clear()


def bind_runtime_deps(mapping: Mapping[str, Any]) -> None:
    """Backward-compatible API: bind all domains to the same mapping."""
    if not mapping:
        return
    bind_decode_deps(mapping)
    bind_refresh_deps(mapping)
    bind_selector_deps(mapping)


def require_runtime_dep(name: str) -> Any:
    value = _runtime_deps.get(name, _MISSING)
    if value is _MISSING:
        raise RuntimeError(
            f"runtime dependency not bound: {name}; "
            "call _bind_runtime_worker_deps() before importing runtime workers"
        )
    return value
