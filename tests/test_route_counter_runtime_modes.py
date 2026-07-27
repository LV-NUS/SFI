from __future__ import annotations

import ast
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import struct
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _worker_record(rank: int, pid: int, values: list[int]) -> dict[str, object]:
    return {
        "rank": rank,
        "slot_count": 2,
        "field_count": 10,
        "values": values,
        "storage": "worker_local",
        "pid": pid,
    }


def test_worker_local_counter_reset_snapshot_and_layout(monkeypatch) -> None:
    from patches.fa3_native import install

    monkeypatch.setenv("VLLM_SPARSE_FA3_ROUTE_COUNTER_SLOTS", "1")
    install.reset_sparse_fa3_route_counters()

    sentinel = object()

    def core(**_kwargs):
        return sentinel

    counted = install.wrap_mixed_page_route_counter(core)
    assert counted(
        page_resolver_kind=4,
        resolved_page_table_row_ptr_u64=object(),
    ) is sentinel
    install.bump_step_compact_row_liveness(2)

    record = install.snapshot_rank_local_route_counter_slot_after_measurement()
    assert record["storage"] == "worker_local"
    assert record["rank"] == 0
    assert record["values"] == [1, 1, 1, 0, 0, 0, 0, 1, 1, 2]
    assert install.reset_rank_local_route_counter_slot_for_measurement()[
        "values"
    ] == [0] * 10


def test_minimal_worker_extension_exposes_only_named_counter_methods() -> None:
    from patches.fa3_native.route_counter_worker_extension import (
        SparseRouteCounterWorkerExtension,
    )

    public_methods = {
        name
        for name, value in vars(SparseRouteCounterWorkerExtension).items()
        if callable(value) and not name.startswith("_")
    }
    assert public_methods == {
        "sfi_reset_sparse_route_counters_for_measurement",
        "sfi_snapshot_sparse_route_counters_after_measurement",
    }


def test_step_core_contains_no_trace_or_counter_observer() -> None:
    source = (
        REPO_ROOT / "patches" / "decode_runtime" / "step_context_worker.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    prepare = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_step_context_impl"
    )
    referenced_names = {
        node.id for node in ast.walk(prepare) if isinstance(node, ast.Name)
    }
    assert referenced_names.isdisjoint(
        {
            "append_fa3_step_trace",
            "build_fa3_step_trace_event",
            "bump_step_compact_row_liveness",
            "fa3_step_trace_enabled",
        }
    )


def test_observer_free_specialization_is_exact_core_identity(monkeypatch) -> None:
    from patches.decode_runtime.step_context_observer import (
        build_observed_prepare_step_context_impl,
    )
    from patches.fa3_native import install

    def core(self, **_kwargs):
        self.step_authority = SimpleNamespace(
            use_compact_by_row=(True, False, True)
        )
        return "context"

    assert (
        build_observed_prepare_step_context_impl(
            core,
            route_counter_enabled=False,
            step_trace_enabled=False,
        )
        is core
    )

    compact_rows: list[int] = []
    trace_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        install,
        "bump_step_compact_row_liveness",
        compact_rows.append,
    )
    monkeypatch.setattr(
        install,
        "build_fa3_step_trace_event",
        lambda **kwargs: {"source": kwargs["source"]},
    )
    monkeypatch.setattr(install, "append_fa3_step_trace", trace_events.append)

    observed = build_observed_prepare_step_context_impl(
        core,
        route_counter_enabled=True,
        step_trace_enabled=True,
    )
    assert observed(SimpleNamespace()) == "context"
    assert compact_rows == [2]
    assert trace_events == [{"source": "prepare_step_context"}]


def test_timed_and_diagnostic_children_are_statically_separated() -> None:
    from benchmarks.sm80_run_pair import classify_speed_child_env_key

    source = (
        REPO_ROOT
        / "benchmarks"
        / "bench_sm80_mixed_page_one_shot_graph_e2e.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def string_literals(function_name: str) -> set[str]:
        return {
            node.value
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

    proof_flag = "--collect-route-counter-proof"
    assert proof_flag in string_literals("_build_phase2_command")
    assert proof_flag not in string_literals("_build_sparse_speed_command")
    assert (
        classify_speed_child_env_key(
            "VLLM_SPARSE_FA3_ROUTE_COUNTER_ENABLED"
        )
        == "observation"
    )


def test_named_preset_accepts_explicit_capacity_only_gpu_utilization() -> None:
    from benchmarks.bench_sm80_mixed_page_one_shot_graph_e2e import parse_args

    base_args = [
        "--output",
        "unused.json",
        "--mode",
        "sparse",
        "--preset",
        "bs2long-cap128",
    ]
    default = parse_args(base_args)
    explicit = parse_args([*base_args, "--gpu-mem-util", "0.75"])
    assert default.gpu_mem_util == 0.4
    assert explicit.gpu_mem_util == 0.75

    for invalid in ("0", "1.1", "nan", "inf"):
        with pytest.raises(SystemExit):
            parse_args([*base_args, "--gpu-mem-util", invalid])


def test_route_counter_wrapper_is_absent_when_proof_is_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from patches.fa3_native import install

    upstream = (
        tmp_path
        / "third_party_upstreams"
        / "vllm-project-flash-attention"
    )
    package = upstream / "vllm_flash_attn"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "flash_attn_interface.py").write_text(
        "def flash_attn_varlen_func(*args, **kwargs):\n"
        "    return 'dense'\n"
        "def mixed_page_attn_varlen_func(*args, **kwargs):\n"
        "    return 'mixed'\n"
        "def get_scheduler_metadata(*args, **kwargs):\n"
        "    return 'metadata'\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("VLLM_SPARSE_FA3_ROUTE_COUNTER_ENABLED", raising=False)
    bridge = install.load_vendored_flash_attn_bridge(repo_root=tmp_path)
    assert not hasattr(
        bridge.mixed_page_attn_varlen_func,
        "_sfi_sparse_fa3_route_counter",
    )

    proof_root = tmp_path / "proof"
    proof_package = (
        proof_root
        / "third_party_upstreams"
        / "vllm-project-flash-attention"
        / "vllm_flash_attn"
    )
    proof_package.mkdir(parents=True)
    (proof_package / "__init__.py").write_text("", encoding="utf-8")
    (proof_package / "flash_attn_interface.py").write_text(
        (package / "flash_attn_interface.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_SPARSE_FA3_ROUTE_COUNTER_ENABLED", "1")
    observed_bridge = install.load_vendored_flash_attn_bridge(
        repo_root=proof_root
    )
    assert getattr(
        observed_bridge.mixed_page_attn_varlen_func,
        "_sfi_sparse_fa3_route_counter",
        False,
    ) is True
    monkeypatch.delenv("VLLM_SPARSE_FA3_ROUTE_COUNTER_ENABLED")
    restored_bridge = install.load_vendored_flash_attn_bridge(
        repo_root=proof_root
    )
    assert not hasattr(
        restored_bridge.mixed_page_attn_varlen_func,
        "_sfi_sparse_fa3_route_counter",
    )


def test_http_snapshot_schema_and_worker_identity(tmp_path: Path) -> None:
    from scripts import snapshot_sparse_route_counters as snapshot

    values = [3, 2, 2, 1, 0, 0, 0, 2, 4, 6]
    records = snapshot._validated_records(
        [_worker_record(1, 102, values), _worker_record(0, 101, values)],
        tp_size=2,
        require_zero=False,
    )
    assert [record["rank"] for record in records] == [0, 1]
    payload = snapshot._snapshot_payload(records, phase="snapshot")
    assert payload["rank_consistent"] is True
    assert len(str(payload["sha256"])) == 64
    assert payload["binary_size_bytes"] == 160
    assert len(str(payload["binary_sha256"])) == 64

    reset_path = tmp_path / "reset.json"
    reset_records = [
        _worker_record(0, 101, [0] * 10),
        _worker_record(1, 102, [0] * 10),
    ]
    reset_path.write_text(
        json.dumps(snapshot._snapshot_payload(reset_records, phase="reset")),
        encoding="utf-8",
    )
    snapshot._assert_worker_identity(records, str(reset_path), tp_size=2)

    assert snapshot._loopback_collective_rpc_url("http://127.0.0.1:8000") == (
        "http://127.0.0.1:8000/collective_rpc"
    )
    for invalid in (
        "http://0.0.0.0:8000",
        "https://127.0.0.1:8000",
        "http://127.0.0.1:8000/v1",
    ):
        try:
            snapshot._loopback_collective_rpc_url(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"unsafe RPC URL was accepted: {invalid}")


def test_collective_rpc_uses_vllm_http_contract(tmp_path: Path) -> None:
    from scripts import snapshot_sparse_route_counters as snapshot

    values = [0] * 10
    records = [_worker_record(0, 101, values), _worker_record(1, 102, values)]
    observed: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            observed.append(payload)
            assert self.path == "/collective_rpc"
            assert self.headers["Authorization"] == "Bearer secret"
            body = json.dumps({"results": records}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    json_output = tmp_path / "reset.json"
    binary_output = tmp_path / "reset.bin"
    try:
        result = snapshot._collective_rpc(
            base_url=base_url,
            api_key="secret",
            method=snapshot.RESET_METHOD,
            timeout=5.0,
        )
        command = subprocess.run(
            [
                sys.executable,
                "-I",
                str(REPO_ROOT / "scripts" / "snapshot_sparse_route_counters.py"),
                "--base-url",
                base_url,
                "--api-key",
                "secret",
                "--tp-size",
                "2",
                "--phase",
                "reset",
                "--json-output",
                str(json_output),
                "--binary-output",
                str(binary_output),
                "--timeout",
                "5",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    assert result == records
    assert command.returncode == 0, command.stdout + command.stderr
    assert observed == [
        {"method": snapshot.RESET_METHOD, "timeout": 5.0},
        {"method": snapshot.RESET_METHOD, "timeout": 5.0},
    ]
    assert json.loads(json_output.read_text(encoding="utf-8"))["phase"] == "reset"
    assert binary_output.read_bytes() == struct.pack("20q", *([0] * 20))


def test_speed_summary_reads_route_proof_only_from_diagnostic_child(
    tmp_path: Path,
) -> None:
    from benchmarks.run_sparse_only import (
        _fa3_route_counter_rpc_snapshot_artifact,
    )
    from scripts.check_run_speed_summary import _tp_sparse_rank_reasons

    values = [3, 2, 2, 1, 0, 0, 0, 2, 4, 6]
    reset_records = [
        _worker_record(0, 101, [0] * 10),
        _worker_record(1, 102, [0] * 10),
    ]
    snapshot_records = [
        _worker_record(0, 101, values),
        _worker_record(1, 102, values),
    ]
    per_rank = [
        {
            "rank": rank,
            "actual_fwd_mixed_page_count": 3,
            "resolved_row_ptr_fwd_mixed_page_count": 2,
            "has_resolved_row_ptr_count": 2,
            "page_resolver_kind_counts": {
                "0": 1,
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 2,
            },
            "compact_row_steps": 4,
            "compact_rows": 6,
        }
        for rank in range(2)
    ]
    snapshot = {
        "route_counter_slots": 2,
        "route_counter_rank_consistent": True,
        "per_rank_route_counters": per_rank,
    }
    metrics = {
        "route_counter_proof_collected": True,
        "route_counter_scope": "measurement_window",
        "route_counter_available": True,
        "route_counter_rank_slots": 2,
        "route_counter_rank_consistent": True,
        "route_counter_authority": "worker_collective_rpc",
        "route_counter_reset_records": reset_records,
        "route_counter_snapshot_records": snapshot_records,
        "route_counter_rpc_snapshot_artifact": (
            _fa3_route_counter_rpc_snapshot_artifact(
                snapshot_records,
                snapshot,
            )
        ),
        "route_counter_per_rank": per_rank,
        "route_counter_actual_fwd_mixed_page_count": 6,
        "route_counter_resolved_row_ptr_fwd_mixed_page_count": 4,
        "route_counter_has_resolved_row_ptr_count": 4,
        "route_counter_page_resolver_kind0_count": 2,
        "route_counter_page_resolver_kind1_count": 0,
        "route_counter_page_resolver_kind2_count": 0,
        "route_counter_page_resolver_kind3_count": 0,
        "route_counter_page_resolver_kind4_count": 4,
        "route_counter_compact_row_steps": 8,
        "route_counter_compact_rows": 12,
    }
    metrics_path = tmp_path / "sample_diag_decode_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    summary_path = tmp_path / "sample_summary.json"
    reasons = _tp_sparse_rank_reasons(
        {
            "run_provenance": {"tensor_parallel_size": 2},
            "diagnostic_decode_metrics_path": str(metrics_path),
        },
        summary_path=summary_path,
        expected_tier=None,
        selector_cache_root=None,
    )
    assert reasons == []


def test_liveness_rejects_snapshot_without_replay_generation(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "refresh.log"
    profile.write_text(
        "101\tstep\t"
        + json.dumps({"refresh_payloads": 1, "sentence_trigger_intents": 0})
        + "\n",
        encoding="utf-8",
    )
    counter = tmp_path / "route_counter_snapshot.bin"
    counter.write_bytes(struct.pack("10q", 1, 1, 1, 0, 0, 0, 0, 1, 1, 2))

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPO_ROOT / "scripts" / "check_sparse_liveness.py"),
            "--route-counter-snapshot",
            str(counter),
            "--baseline-compact-row-steps",
            "0",
            "--min-compact-row-step-delta",
            "1",
            "--expected-route-counter-ranks",
            "1",
            "--phase",
            "trace-free-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "logical refresh 世代 0 < 1" in result.stdout

    retired_arguments = (
        ("--refresh-profile-log", str(profile)),
        ("--refresh-profile-offset", "0"),
        ("--route-counter-mmap", str(counter)),
    )
    for flag, value in retired_arguments:
        retired = subprocess.run(
            [
                sys.executable,
                "-I",
                str(REPO_ROOT / "scripts" / "check_sparse_liveness.py"),
                flag,
                value,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert retired.returncode != 0
        assert f"unrecognized arguments: {flag}" in retired.stderr


def _replay_enqueue_event(
    *,
    pid: int,
    payload_count: int = 48,
    step_id: int = 306,
) -> dict[str, object]:
    return {
        "event": (
            "mixed_page_full_cudagraph_replay_refresh_payload_enqueue"
        ),
        "pid": pid,
        "step_id": step_id,
        "graph_key": "full:BatchDescriptor(num_tokens=1,num_reqs=1)",
        "refresh_reason": "compact_threshold_crossed",
        "refresh_intent_req_ids": ["request-1"],
        "payload_count": payload_count,
    }


def _replay_route_event(*, pid: int, step_id: int = 309) -> dict[str, object]:
    return {
        "event": "mixed_page_full_cudagraph_replay_hook_check",
        "pid": pid,
        "step_id": step_id,
        "batch_descriptor": "BatchDescriptor(num_tokens=1,num_reqs=1)",
        "captured_route_family": "resolved_row_ptr",
        "current_route_family": "resolved_row_ptr",
        "route_family_mismatch": False,
    }


def _run_replay_liveness(
    tmp_path: Path,
    *,
    route_events: list[dict[str, object]],
    rank_count: int = 2,
) -> subprocess.CompletedProcess[str]:
    route = tmp_path / "route.jsonl"
    route.write_text(
        "".join(json.dumps(event) + "\n" for event in route_events),
        encoding="utf-8",
    )
    counter = tmp_path / "route_counter_snapshot.bin"
    rank_values = [48, 0, 0, 48, 0, 0, 0, 0, 76, 76]
    counter.write_bytes(
        b"".join(struct.pack("10q", *rank_values) for _ in range(rank_count))
    )
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPO_ROOT / "scripts" / "check_sparse_liveness.py"),
            "--route-trace",
            str(route),
            "--route-counter-snapshot",
            str(counter),
            "--baseline-compact-row-steps",
            "0",
            "--min-compact-row-step-delta",
            "1",
            "--expected-route-counter-ranks",
            str(rank_count),
            "--phase",
            "full-replay-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_liveness_accepts_tp_complete_full_replay_without_profile(
    tmp_path: Path,
) -> None:
    result = _run_replay_liveness(
        tmp_path,
        route_events=[
            _replay_enqueue_event(pid=101),
            _replay_enqueue_event(pid=202),
            _replay_route_event(pid=101),
            _replay_route_event(pid=202),
        ],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "generation_count=1" in result.stdout
    assert "source=replay_batched_route" in result.stdout
    assert "FULL replay route authority" in result.stdout


@pytest.mark.parametrize(
    ("enqueue_events", "failure"),
    [
        (
            [],
            "logical refresh 世代 0 < 1",
        ),
        (
            [_replay_enqueue_event(pid=101)],
            "TP-incomplete=1",
        ),
        (
            [
                _replay_enqueue_event(pid=101, payload_count=48),
                _replay_enqueue_event(pid=202, payload_count=47),
            ],
            "payload-mismatch=1",
        ),
    ],
)
def test_liveness_rejects_incomplete_replay_generation_evidence(
    tmp_path: Path,
    enqueue_events: list[dict[str, object]],
    failure: str,
) -> None:
    result = _run_replay_liveness(
        tmp_path,
        route_events=[
            *enqueue_events,
            _replay_route_event(pid=101),
            _replay_route_event(pid=202),
        ],
    )

    assert result.returncode != 0
    assert failure in result.stdout


def test_liveness_rejects_full_replay_route_family_mismatch(
    tmp_path: Path,
) -> None:
    mismatched_route = _replay_route_event(pid=202)
    mismatched_route["current_route_family"] = "native"
    mismatched_route["route_family_mismatch"] = True
    result = _run_replay_liveness(
        tmp_path,
        route_events=[
            _replay_enqueue_event(pid=101),
            _replay_enqueue_event(pid=202),
            _replay_route_event(pid=101),
            mismatched_route,
        ],
    )

    assert result.returncode != 0
    assert "FULL replay route 证据不完整" in result.stdout
    assert "mismatch=1" in result.stdout


def test_liveness_route_judge_uses_only_fresh_offset_window(
    tmp_path: Path,
) -> None:
    stale = json.dumps(
        {"event": "mixed_page_call", "mode": "resolved_row_ptr"}
    ) + "\n"
    fresh = "".join(
        json.dumps(event) + "\n"
        for event in (
            _replay_enqueue_event(pid=101),
            {"event": "mixed_page_call", "mode": "native"},
        )
    )
    route = tmp_path / "route.jsonl"
    route.write_text(stale + fresh, encoding="utf-8")
    counter = tmp_path / "route_counter_snapshot.bin"
    counter.write_bytes(struct.pack("10q", 1, 1, 1, 0, 0, 0, 0, 1, 1, 1))

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPO_ROOT / "scripts" / "check_sparse_liveness.py"),
            "--route-trace",
            str(route),
            "--route-trace-offset",
            str(len(stale.encode("utf-8"))),
            "--route-counter-snapshot",
            str(counter),
            "--expected-route-counter-ranks",
            "1",
            "--phase",
            "fresh-window-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "fresh route window 无 sparse read authority" in result.stdout


def test_full_replay_serve_retires_refresh_profile_authority() -> None:
    serve_source = (
        REPO_ROOT / "scripts" / "serve_sparse.sh"
    ).read_text(encoding="utf-8")
    longbench_source = (
        REPO_ROOT / "scripts" / "run_longbench_v2.sh"
    ).read_text(encoding="utf-8")
    checker_source = (
        REPO_ROOT / "scripts" / "check_sparse_liveness.py"
    ).read_text(encoding="utf-8")
    installer_source = (
        REPO_ROOT / "patches" / "patch_installer.py"
    ).read_text(encoding="utf-8")

    retired_profile_env = (
        "VLLM_SPARSE_REFRESH_PROFILE",
        "VLLM_SPARSE_REFRESH_PROFILE_DETAIL",
        "VLLM_SPARSE_REFRESH_PROFILE_CALL_MIN",
        "VLLM_SPARSE_REFRESH_PROFILE_EVERY",
        "VLLM_SPARSE_REFRESH_PROFILE_LOG",
    )
    assert "REFRESH_PROFILE_LOG=" not in serve_source
    assert "export VLLM_SPARSE_REFRESH_PROFILE=" not in serve_source
    assert '"refresh_profile_log"' not in serve_source
    for name in retired_profile_env:
        assert f"unset {name}" in serve_source
        assert f'"{name}"' in longbench_source
    assert '"schema": 7' in serve_source
    assert '"refresh_profile_log"' not in longbench_source
    assert 'data["schema"] != 7' in longbench_source
    assert "--refresh-profile-log" not in checker_source
    assert "--refresh-profile-offset" not in checker_source
    assert "_refresh_profile_batched_hint_emitted" not in installer_source
