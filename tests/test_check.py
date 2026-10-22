import json
import math
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from copy import deepcopy

import pytest

from torchscan import check_performance, compare_benchmarks
from torchscan.__main__ import main
from torchscan.report import metric_result


def benchmark(latency=1.0):
    return {
        "schema_version": 1,
        "context": {
            "device": "cpu",
            "device_name": "test CPU",
            "processor": "test CPU",
            "machine": "x86_64",
            "platform": "test OS",
            "torch_version": "test",
            "torchscan_version": "test",
            "python_version": "3.11",
            "cuda_version": None,
            "work_units": 1,
            "work_unit": "calls",
            "inputs_source": "caller_supplied",
            "min_repeats": 2,
        },
        "inputs": {"kind": "tensor", "shape": [8], "dtype": "torch.float32", "device": "cpu"},
        "totals": {
            name: metric_result(status="complete", value=value, unit=unit, scope=scope, method=method)
            for name, value, unit, scope, method in (
                ("latency", latency, "seconds", "warmed_block_average", "torch.utils.benchmark.Timer"),
                ("latency_iqr", 0, "seconds", "warmed_block_average", "torch.utils.benchmark.Timer"),
                ("first_call_latency", 1, "seconds", "first_call", "synchronized_perf_counter"),
                ("throughput", 1 / latency, "calls/s", "warmed_blocks", "torch.utils.benchmark.Timer"),
            )
        },
        "measurement": {"number_per_run": 1, "raw_times_seconds": [latency, latency], "measured_calls": 2},
    }


def comparison(latency=1.0, *, passed=True, memory=None):
    before, after = benchmark(), benchmark(latency)
    if memory is not None:
        after["totals"]["peak_memory"] = memory
    return compare_benchmarks(before, after, check=lambda: passed)


def memory_metric(value=100, *, status="complete", scope="pytorch_tensor_bytes", unit="bytes"):
    return metric_result(
        status=status,
        value=value if status == "complete" else None,
        known_value=value if status == "partial" else None,
        unit=unit,
        scope=scope,
        method="torch.profiler.memory_timeline",
    )


@pytest.mark.parametrize(
    ("latency", "limit", "status"),
    [
        (1, 0, "passed"),
        (math.nextafter(1, math.inf), 0, "failed"),
        (1.05, 0.05, "passed"),
        (math.nextafter(1.05, math.inf), 0.05, "failed"),
        (0.9, 0, "passed"),
    ],
)
def test_latency_boundaries(latency, limit, status):
    result = check_performance(comparison(latency), max_latency_regression=limit)
    assert result["status"] == status
    assert result["reasons"]
    assert "fractional regression" in result["reasons"][0]


@pytest.mark.parametrize(("value", "status"), [(0, "passed"), (100, "passed"), (101, "failed")])
def test_memory_boundaries(value, status):
    result = check_performance(
        comparison(memory=memory_metric(value)),
        max_latency_regression=0,
        memory_limit={"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": 100},
    )
    assert result["status"] == status
    assert "peak_memory" in result["reasons"][-1]
    assert "pytorch_tensor_bytes" in result["reasons"][-1]


@pytest.mark.parametrize("status", ["partial", "unavailable"])
def test_incomplete_requested_evidence(status):
    before, after = benchmark(), benchmark()
    after["totals"]["latency"] = metric_result(
        status=status,
        known_value=1 if status == "partial" else None,
        unit="seconds",
        scope="warmed_block_average",
        method="torch.utils.benchmark.Timer",
    )
    result = check_performance(compare_benchmarks(before, after, check=lambda: True), max_latency_regression=0)
    assert result["status"] == "inconclusive"
    assert status in result["reasons"][0]
    result = check_performance(
        comparison(memory=memory_metric(status=status)),
        max_latency_regression=0,
        memory_limit={"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": 100},
    )
    assert result["status"] == "inconclusive"
    assert status in result["reasons"][-1]


@pytest.mark.parametrize(
    ("memory", "metric", "scope", "reason"),
    [
        (None, "peak_memory", "pytorch_tensor_bytes", "missing"),
        (memory_metric(), "process_peak_rss", "child_process_lifetime", "missing"),
        (memory_metric(scope="child_process_lifetime"), "peak_memory", "pytorch_tensor_bytes", "matching byte"),
        (memory_metric(scope="allocator_allocated_peak"), "peak_memory", "allocator_reserved_peak", "matching byte"),
        (memory_metric(unit="MiB"), "peak_memory", "pytorch_tensor_bytes", "matching byte"),
        (memory_metric(-1), "peak_memory", "pytorch_tensor_bytes", "invalid evidence"),
    ],
)
def test_memory_identity_and_unavailability(memory, metric, scope, reason):
    result = check_performance(
        comparison(memory=memory),
        max_latency_regression=0,
        memory_limit={"metric": metric, "scope": scope, "max_bytes": 100},
    )
    assert result["status"] == "inconclusive"
    assert reason in result["reasons"][-1]


@pytest.mark.parametrize("passed", [True, False])
def test_failure_precedes_incomplete_evidence(passed):
    before, after = benchmark(), benchmark()
    after["measurement"] = {}
    after["totals"]["peak_memory"] = memory_metric(101)
    report = compare_benchmarks(before, after, check=lambda: passed)
    result = check_performance(
        report,
        max_latency_regression=0,
        memory_limit={"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": 100},
    )
    assert result["status"] == "failed"
    assert ("ceiling" if passed else "output check failed") in result["reasons"][-1]


def test_exceeded_latency_with_missing_memory_fails():
    result = check_performance(
        comparison(2),
        max_latency_regression=0,
        memory_limit={"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": 100},
    )
    assert result["status"] == "failed"
    assert "missing" in result["reasons"][-1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("number_per_run", 0),
        ("number_per_run", True),
        ("number_per_run", 10**400),
        ("raw_times_seconds", []),
        ("raw_times_seconds", [1]),
        ("raw_times_seconds", [0, 0]),
        ("raw_times_seconds", [True, 1]),
        ("raw_times_seconds", [2, 2]),
        ("raw_times_seconds", None),
        ("measured_calls", 3),
    ],
)
def test_insufficient_timing(field, value):
    before, after = benchmark(), benchmark()
    after["measurement"][field] = value
    result = check_performance(compare_benchmarks(before, after, check=lambda: True), max_latency_regression=0)
    assert result["status"] == "inconclusive"
    assert "after" in result["reasons"][0]


@pytest.mark.parametrize("repeats", [None, 1, 3])
def test_insufficient_repeats(repeats):
    before, after = benchmark(), benchmark()
    after["context"]["min_repeats"] = repeats
    result = check_performance(compare_benchmarks(before, after, check=lambda: True), max_latency_regression=0)
    assert result["status"] == "inconclusive"


@pytest.mark.parametrize("field", ["scope", "method"])
def test_unsupported_or_mismatched_latency(field):
    before, after = benchmark(), benchmark()
    before["totals"]["latency"][field] = "other"
    after["totals"]["latency"][field] = "other"
    result = check_performance(compare_benchmarks(before, after, check=lambda: True), max_latency_regression=0)
    assert result["status"] == "inconclusive"
    report = comparison()
    report["after"]["totals"]["latency"][field] = "other"
    result = check_performance(report, max_latency_regression=0)
    assert result["status"] == "inconclusive"
    assert "incompatible" in result["reasons"][0]


@pytest.mark.parametrize("field", ["before", "after", "totals", "output_check", "latency_change", "context_changes"])
def test_missing_comparison_fields(field):
    report = comparison()
    del report[field]
    assert check_performance(report, max_latency_regression=0)["status"] == "inconclusive"


@pytest.mark.parametrize("report", [None, [], {}, {"schema_version": True}, {"schema_version": 2}])
def test_malformed_reports(report):
    result = check_performance(report, max_latency_regression=0)
    assert result["status"] == "inconclusive"
    assert "comparison" in result["reasons"][0]


def test_reuses_saved_comparison_validation_and_does_not_mutate_or_execute(monkeypatch):
    report = json.loads(json.dumps(comparison(memory=memory_metric(status="partial"))))
    original = deepcopy(report)

    def unexpected():
        pytest.fail("Checking saved evidence must not execute collectors or output checks.")

    monkeypatch.setattr("torchscan.benchmark._run_timing", unexpected)
    monkeypatch.setattr("torchscan.benchmark_compare.compare_benchmarks", unexpected)
    assert check_performance(report, max_latency_regression=0)["status"] == "passed"
    assert report == original
    report["totals"]["latency"]["delta"] = 999
    assert "deltas" in check_performance(report, max_latency_regression=0)["reasons"][0]
    report = deepcopy(original)
    report["after"]["context"]["device_name"] = "different CPU"
    assert "contexts" in check_performance(report, max_latency_regression=0)["reasons"][0]


@pytest.mark.parametrize("limit", [-1, math.inf, math.nan, True, "0.05"])
def test_invalid_latency_limits(limit):
    with pytest.raises(ValueError):
        check_performance(comparison(), max_latency_regression=limit)


@pytest.mark.parametrize(
    "limit",
    [
        {},
        {"metric": "peak_memory", "max_bytes": 100},
        {"metric": "", "scope": "pytorch_tensor_bytes", "max_bytes": 100},
        {"metric": "peak_memory", "scope": "", "max_bytes": 100},
        {"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": -1},
        {"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": 1.5},
        {"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": True},
    ],
)
def test_invalid_memory_limits(limit):
    with pytest.raises(ValueError):
        check_performance(comparison(), max_latency_regression=0, memory_limit=limit)


@pytest.mark.parametrize(("latency", "passed", "code"), [(1, True, 0), (2, True, 1), (1, False, 1), (1, True, 2)])
def test_module_cli_exit_codes(tmp_path, latency, passed, code):
    path = tmp_path / "comparison.json"
    report = comparison(latency, passed=passed)
    if code == 2:
        report["after"]["measurement"] = {}
    path.write_text(json.dumps(report), encoding="utf-8")
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-m", "torchscan", "check", str(path), "--max-latency-regression", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == code
    output = json.loads(result.stdout)
    assert output["status"] == {0: "passed", 1: "failed", 2: "inconclusive"}[code]
    assert output["reasons"]


@pytest.mark.parametrize("payload", ["{", "[]", '{"schema_version":2}', "\udcff", "NaN", '{"a":0,"a":1}'])
def test_cli_invalid_json_and_report(tmp_path, capsys, payload):
    path = tmp_path / "comparison.json"
    path.write_bytes(payload.encode("utf-8", errors="surrogateescape"))
    assert main(["check", str(path), "--max-latency-regression", "0"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "inconclusive"


@pytest.mark.parametrize(
    ("extra", "code"),
    [
        (["--memory-metric", "peak_memory"], 2),
        (["--max-latency-regression", "-1"], 2),
        (["--max-latency-regression", "nan"], 2),
        (["--memory-metric", "peak_memory", "--memory-scope", "pytorch_tensor_bytes", "--max-memory-bytes", "100"], 0),
        (
            ["--memory-metric", "peak_memory", "--memory-scope", "child_process_lifetime", "--max-memory-bytes", "100"],
            2,
        ),
        (
            [
                "--memory-metric",
                "process_peak_rss",
                "--memory-scope",
                "child_process_lifetime",
                "--max-memory-bytes",
                "100",
            ],
            2,
        ),
        (["--memory-metric", "peak_memory", "--memory-scope", "pytorch_tensor_bytes", "--max-memory-bytes", "99"], 1),
    ],
)
def test_cli_limits(tmp_path, capsys, extra, code):
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison(memory=memory_metric())), encoding="utf-8")
    assert main(["check", str(path), "--max-latency-regression", "0", *extra]) == code
    assert json.loads(capsys.readouterr().out)["reasons"]


def test_cli_missing_file_and_required_threshold(tmp_path, capsys):
    assert main(["check", str(tmp_path / "missing.json"), "--max-latency-regression", "0"]) == 2
    assert "Invalid input" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["check", "comparison.json"])
    assert error.value.code == 2


def test_cyclic_api_input_is_inconclusive():
    report = comparison()
    report["extra"] = report
    assert check_performance(report, max_latency_regression=0)["status"] == "inconclusive"


@pytest.mark.parametrize("snapshot", [True, False])
def test_boolean_diff_values_are_invalid(snapshot):
    report = comparison()
    if snapshot:
        report["totals"]["latency"]["before"]["value"] = True
    else:
        report["totals"]["latency"]["delta"] = False
    result = check_performance(report, max_latency_regression=0)
    assert result["status"] == "inconclusive"
    assert "finite number" in result["reasons"][0]


@pytest.mark.parametrize("method", ["", "not_requested", "unavailable"])
def test_unmeasured_memory_method(method):
    memory = memory_metric()
    memory["method"] = method
    result = check_performance(
        comparison(memory=memory),
        max_latency_regression=0,
        memory_limit={"metric": "peak_memory", "scope": "pytorch_tensor_bytes", "max_bytes": 100},
    )
    assert result["status"] == "inconclusive"


@pytest.mark.parametrize("key", ["latency", "latency_iqr"])
def test_incomplete_iqr_and_missing_latency(key):
    report = comparison()
    before, after = report["before"], report["after"]
    if key == "latency":
        del after["totals"][key]
    else:
        after["totals"][key].update(status="partial", value=None)
        report = compare_benchmarks(before, after, check=lambda: True)
    assert check_performance(report, max_latency_regression=0)["status"] == "inconclusive"


@pytest.mark.parametrize(("latency", "passed", "code"), [(1, True, 2), (1, False, 1)])
def test_cli_insufficient_timing(tmp_path, capsys, latency, passed, code):
    report = comparison(latency, passed=passed)
    report["after"]["measurement"] = {}
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert main(["check", str(path), "--max-latency-regression", "0"]) == code
    assert json.loads(capsys.readouterr().out)["reasons"]
