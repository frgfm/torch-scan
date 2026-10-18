import json
from unittest.mock import Mock, call

import pytest
import torch

from torchscan import benchmark_compare as comparison
from torchscan import measure_comparison, metric_result


def timing_report(latency, iqr=0.25):
    return {
        "schema_version": 1,
        "context": {
            "device": "cpu",
            "device_name": "test CPU",
            "processor": "test CPU",
            "machine": "test machine",
            "platform": "test platform",
            "torch_version": "test torch",
            "torchscan_version": "test torchscan",
            "python_version": "test python",
            "cuda_version": None,
            "work_units": 1,
            "work_unit": "calls",
            "inputs_source": "caller_supplied",
        },
        "inputs": {"kind": "tensor", "shape": [8], "dtype": "torch.float32"},
        "totals": {
            name: metric_result(status="complete", value=value, unit=unit, scope=scope, method=method)
            for name, value, unit, scope, method in (
                ("first_call_latency", latency, "seconds", "first_call", "synchronized_perf_counter"),
                ("latency", latency, "seconds", "warmed_block_average", "torch.utils.benchmark.Timer"),
                ("latency_iqr", iqr, "seconds", "warmed_block_average", "torch.utils.benchmark.Timer"),
                ("throughput", 1 / latency, "calls/s", "warmed_blocks", "torch.utils.benchmark.Timer"),
            )
        },
        "measurement": {
            "number_per_run": 1,
            "raw_times_seconds": [latency - iqr, latency, latency + iqr],
            "measured_calls": 3,
        },
    }


def stub_measurements(monkeypatch, pairs):
    ordered = []
    for index, (before, after) in enumerate(pairs):
        ordered.extend((before, after) if index % 2 == 0 else (after, before))
    measure = Mock(side_effect=ordered)
    monkeypatch.setattr(comparison, "measure_latency", measure)
    return measure


def test_alternating_order_shared_settings_and_untimed_checks(monkeypatch):
    pairs = [(timing_report(10 + index), timing_report(5 + index)) for index in range(4)]
    measure = stub_measurements(monkeypatch, pairs)
    events = Mock()
    events.attach_mock(measure, "measure")
    check = Mock(return_value=None)
    events.attach_mock(check, "check")
    baseline, candidate = Mock(), Mock()
    settings = {
        "device": torch.device("cpu"),
        "inputs": {"dataset": "fixed"},
        "work_units": 8,
        "work_unit": "samples",
        "warmup": 2,
        "min_run_time": 0.03,
        "min_repeats": 4,
    }
    result = measure_comparison(baseline, candidate, check=check, min_latency_change=1, trials=4, **settings)
    assert events.mock_calls == [
        event
        for index in range(4)
        for event in (
            call.measure(baseline if index % 2 == 0 else candidate, **settings),
            call.measure(candidate if index % 2 == 0 else baseline, **settings),
            call.check(),
        )
    ]
    assert result["verdict"] == "faster"
    assert result["min_latency_change"] == 1
    assert len(result["trials"]) == 4
    for trial, (before, after) in zip(result["trials"], pairs, strict=True):
        assert trial == comparison.compare_benchmarks(before, after, check=lambda: True)
        assert trial["before"] is not before
        assert trial["after"] is not after
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize(
    ("latencies", "iqr", "verdict"),
    [
        ([7, 8, 6], 0.25, "faster"),
        ([13, 12, 14], 0.25, "slower"),
        ([7, 13, 6], 0.25, "inconclusive"),
        ([9.5, 9.5, 9.5], 0.25, "inconclusive"),
        ([10.5, 10.5, 10.5], 0.25, "inconclusive"),
        ([9, 9, 9], 0.25, "inconclusive"),
        ([10, 10, 10], 0.25, "inconclusive"),
        ([7, 9.5, 6], 0.25, "inconclusive"),
        ([7, 8, 6], 4, "inconclusive"),
    ],
)
def test_consistent_evidence_and_meaningful_change(monkeypatch, latencies, iqr, verdict):
    pairs = [(timing_report(10, iqr), timing_report(latency, iqr)) for latency in latencies]
    stub_measurements(monkeypatch, pairs)
    result = measure_comparison(
        lambda: None, lambda: None, check=lambda: True, min_latency_change=1, device="cpu", inputs={}
    )
    assert result["verdict"] == verdict
    assert len(result["trials"]) == 3
    assert [trial["after"] for trial in result["trials"]] == [pair[1] for pair in pairs]


@pytest.mark.parametrize("assertion", [True, False])
def test_failed_check_overrides_speed_and_keeps_all_trials(monkeypatch, assertion):
    pairs = [(timing_report(10), timing_report(5)) for _ in range(3)]
    measure = stub_measurements(monkeypatch, pairs)
    checks = iter([True, False, None])

    def check():
        outcome = next(checks)
        if outcome is False and assertion:
            raise AssertionError("outputs differ")
        return outcome

    result = measure_comparison(lambda: None, lambda: None, check=check, min_latency_change=1, device="cpu", inputs={})
    assert result["verdict"] == "check_failed"
    assert measure.call_count == 6
    assert [trial["output_check"] for trial in result["trials"]] == ["passed", "failed", "passed"]
    failed = result["trials"][1]
    assert failed["latency_change"] == "unavailable"
    assert all(metric["delta"] is None and metric["status"] == "unavailable" for metric in failed["totals"].values())
    assert failed["before"] == pairs[1][0]
    assert failed["after"] == pairs[1][1]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("incomplete", ["latency", "latency_iqr"])
def test_unavailable_trial_is_inconclusive(monkeypatch, incomplete):
    pairs = [(timing_report(10), timing_report(5)) for _ in range(3)]
    pairs[1][1]["totals"][incomplete].update(status="unavailable", value=None, known_value=None)
    stub_measurements(monkeypatch, pairs)
    result = measure_comparison(
        lambda: None, lambda: None, check=lambda: True, min_latency_change=1, device="cpu", inputs={}
    )
    assert result["verdict"] == "inconclusive"
    assert result["trials"][1]["latency_change"] == "unavailable"


@pytest.mark.parametrize("across_trials", [True, False])
def test_context_mismatch_raises_before_check(monkeypatch, across_trials):
    pairs = [(timing_report(10), timing_report(5)) for _ in range(2)]
    pairs[1 if across_trials else 0][0]["context"]["device_name"] = "other CPU"
    if across_trials:
        pairs[1][1]["context"]["device_name"] = "other CPU"
    stub_measurements(monkeypatch, pairs)
    check = Mock(return_value=True)
    with pytest.raises(ValueError, match="contexts"):
        measure_comparison(
            lambda: None, lambda: None, check=check, min_latency_change=1, trials=2, device="cpu", inputs={}
        )
    assert check.call_count == int(across_trials)


@pytest.mark.parametrize("outcome", ["truthy", 1])
def test_invalid_check_result_raises(monkeypatch, outcome):
    measure = stub_measurements(monkeypatch, [(timing_report(10), timing_report(5))])
    with pytest.raises(ValueError, match="check must return"):
        measure_comparison(
            lambda: None, lambda: None, check=lambda: outcome, min_latency_change=1, device="cpu", inputs={}
        )
    assert measure.call_count == 2


def test_check_exception_propagates(monkeypatch):
    stub_measurements(monkeypatch, [(timing_report(10), timing_report(5))])
    error = RuntimeError("check failed unexpectedly")
    with pytest.raises(RuntimeError) as caught:
        measure_comparison(
            lambda: None, lambda: None, check=Mock(side_effect=error), min_latency_change=1, device="cpu", inputs={}
        )
    assert caught.value is error


@pytest.mark.parametrize("failing", ["baseline", "candidate"])
def test_workload_exception_propagates(failing):
    error = RuntimeError("workload failed")
    workloads = {"baseline": lambda: None, "candidate": lambda: None}
    workloads[failing] = Mock(side_effect=error)
    check = Mock()
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError) as caught:
        measure_comparison(
            **workloads,
            check=check,
            min_latency_change=1,
            device="cpu",
            inputs={},
            warmup=0,
            min_run_time=0.001,
            min_repeats=2,
        )
    assert caught.value is error
    assert torch.get_num_threads() == threads
    check.assert_not_called()


@pytest.mark.parametrize(
    ("options", "error_type"),
    [
        ({"baseline": None}, TypeError),
        ({"candidate": None}, TypeError),
        ({"check": None}, TypeError),
        *[({"trials": value}, ValueError) for value in [0, 1, -2, True, 2.0]],
        *[
            ({"min_latency_change": value}, ValueError)
            for value in [0, -1, True, "1", float("nan"), float("inf"), 10**1000]
        ],
        ({"inputs": None}, ValueError),
        ({"work_units": 0}, ValueError),
        ({"work_unit": " "}, ValueError),
        ({"warmup": -1}, ValueError),
        ({"min_run_time": 0}, ValueError),
        ({"min_repeats": 1}, ValueError),
        ({"device": "meta"}, NotImplementedError),
    ],
)
def test_invalid_settings_do_not_run(options, error_type):
    baseline, candidate, check = Mock(), Mock(), Mock()
    settings = {
        "baseline": baseline,
        "candidate": candidate,
        "check": check,
        "min_latency_change": 1,
        "device": "cpu",
        "inputs": {},
    }
    with pytest.raises(error_type):
        measure_comparison(**(settings | options))
    baseline.assert_not_called()
    candidate.assert_not_called()
    check.assert_not_called()


def test_cpu_integration_without_speed_expectation():
    inputs = torch.arange(8, dtype=torch.float32)

    def baseline():
        return inputs + 1

    def candidate():
        return torch.add(inputs, 1)

    check = Mock(side_effect=lambda: torch.testing.assert_close(baseline(), candidate()))
    threads = torch.get_num_threads()
    result = measure_comparison(
        baseline,
        candidate,
        check=check,
        min_latency_change=1e-6,
        device="cpu",
        inputs=inputs,
        trials=2,
        warmup=0,
        min_run_time=0.001,
        min_repeats=2,
    )
    assert result["verdict"] in {"faster", "slower", "inconclusive"}
    assert check.call_count == len(result["trials"]) == 2
    assert torch.get_num_threads() == threads
    for trial in result["trials"]:
        assert trial["output_check"] == "passed"
        for report in (trial["before"], trial["after"]):
            assert all(metric["status"] == "complete" for metric in report["totals"].values())
            assert len(report["measurement"]["raw_times_seconds"]) >= 2
            assert report["context"]["device"] == "cpu"
            assert report["inputs"]["shape"] == [8]
    assert json.loads(json.dumps(result, allow_nan=False)) == result
