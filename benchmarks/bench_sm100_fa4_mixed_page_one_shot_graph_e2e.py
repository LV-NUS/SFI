from __future__ import annotations

import sys

from benchmarks import bench_sm80_mixed_page_one_shot_graph_e2e as _legacy_runner

SM100_FA4_ONE_SHOT_CASE = "sm100_fa4_one_shot_graph_e2e"


def _option_value(argv: list[str], option: str) -> str | None:
    prefix = option + "="
    for idx, token in enumerate(argv):
        if token == option:
            if idx + 1 >= len(argv):
                return ""
            return str(argv[idx + 1])
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


def _has_option(argv: list[str], option: str) -> bool:
    prefix = option + "="
    return any(token == option or token.startswith(prefix) for token in argv)


def build_legacy_argv(argv: list[str] | None = None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    backend = _option_value(args, "--backend")
    if backend is None:
        args.extend(["--backend", _legacy_runner.BACKEND_FA4_SM100])
    elif backend != _legacy_runner.BACKEND_FA4_SM100:
        raise SystemExit(
            "bench_sm100_fa4_mixed_page_one_shot_graph_e2e.py requires "
            "--backend fa4-sm100"
        )
    if not _has_option(args, "--case"):
        args.extend(["--case", SM100_FA4_ONE_SHOT_CASE])
    return args


def main(argv: list[str] | None = None) -> int:
    return int(_legacy_runner.main(build_legacy_argv(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
