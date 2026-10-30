from copy import deepcopy

import pytest

from torchscan import compare_benchmarks
from torchscan._experiment_decision import decide


def benchmark(latency, memory, *, iqr=0.0001, context_changes=None, memory_metric=None):
    context = {
        "device": "cpu",
        "device_name": "test cpu",
        "processor": "test cpu",
        "machine": "x86_64",
        "platform": "Linux-test",
        "torch_version": "2.14.0",
        "torchscan_version": "0.2.0",
        "python_version": "3.11.0",
        "cuda_version": None,
        "num_threads": 1,
        "num_interop_threads": 1,
        "work_units": 8,
        "work_unit": "samples",
        "inputs_source": "caller_supplied",
        "warmup": 2,
        "min_run_time": 0.01,
        "min_repeats": 5,
    }
    context.update(context_changes or {})

    def metric(value, unit, scope, method):
        return {
            "status": "complete",
            "value": value,
            "known_value": value,
            "unit": unit,
            "scope": scope,
            "method": method,
        }

    memory = memory_metric or metric(memory, "bytes", "pytorch_tensor_bytes", "torch.profiler.memory_timeline")
    return {
        "schema_version": 1,
        "context": context,
        "inputs": {"kind": "tensor", "shape": [8, 4], "dtype": "torch.float32", "device": "cpu"},
        "totals": {
            "first_call_latency": metric(latency, "seconds", "first_call", "synchronized_perf_counter"),
            "latency": metric(latency, "seconds", "warmed_block_average", "torch.utils.benchmark.Timer"),
            "latency_iqr": metric(iqr, "seconds", "warmed_block_average", "torch.utils.benchmark.Timer"),
            "throughput": metric(8 / latency, "samples/s", "warmed_blocks", "torch.utils.benchmark.Timer"),
            "peak_memory": memory,
        },
        "measurement": {
            "number_per_run": 1,
            "raw_times_seconds": [latency - 2 * iqr, latency - iqr / 2, latency, latency + iqr / 2, latency + 2 * iqr],
            "measured_calls": 5,
        },
    }


def comparison(
    before_latency, after_latency, before_memory=20_000, after_memory=20_000, *, output_check=True, iqr=0.0001
):
    before = benchmark(before_latency, before_memory, iqr=iqr)
    after = benchmark(after_latency, after_memory, iqr=iqr)
    return compare_benchmarks(before, after, check=lambda: output_check)


def settings(objective="latency", **overrides):
    values = {
        "objective": objective,
        "min_improvement": 0.1,
        "max_memory_bytes": 30_000,
        "max_latency_seconds": 0.02,
    }
    values.update(overrides)
    return values


def test_keep_latency_only_when_every_trial_beats_threshold_and_memory_ceiling():
    trials = [comparison(0.01, 0.008), comparison(0.01, 0.0085)]

    decision, reasons = decide(trials, settings())

    assert decision == "keep"
    assert "Every trial" in reasons[0]


@pytest.mark.parametrize(
    "trials",
    [
        [comparison(0.01, 0.01), comparison(0.01, 0.01)],
        [comparison(0.01, 0.008), comparison(0.01, 0.012)],
    ],
)
def test_aa_and_mixed_latency_trials_are_inconclusive(trials):
    decision, _ = decide(trials, settings())

    assert decision == "inconclusive"


@pytest.mark.parametrize(("before", "after"), [(1.0, 0.9), (0.01, 0.009), (0.03, 0.027)])
def test_improvement_equal_to_threshold_does_not_pass(before, after):
    decision, reasons = decide([comparison(before, after)] * 2, settings(min_improvement=0.1))

    assert decision == "inconclusive"
    assert "threshold" in reasons[0]


def test_failed_output_check_and_known_memory_overage_discard():
    failed_check = comparison(0.01, 0.008, output_check=False)
    memory_overage = comparison(0.01, 0.008, after_memory=30_001)

    assert decide([failed_check, comparison(0.01, 0.008)], settings())[0] == "discard"
    assert decide([memory_overage, comparison(0.01, 0.008)], settings())[0] == "discard"


def test_memory_objective_keeps_smaller_candidate_within_latency_cap():
    trials = [comparison(0.01, 0.012, before_memory=20_000, after_memory=15_000)] * 2

    decision, reasons = decide(trials, settings("memory", max_latency_seconds=0.02))

    assert decision == "keep"
    assert "memory objective" in reasons[0]


def test_rss_objective_uses_named_scope_and_latency_can_have_no_memory_cap():
    before, after = benchmark(0.01, 20_000), benchmark(0.008, 15_000)
    for report, peak in ((before, 200_000), (after, 100_000)):
        report["totals"]["process_peak_rss"] = {
            **report["totals"]["peak_memory"],
            "value": peak,
            "known_value": peak,
            "scope": "child_process_lifetime",
            "method": "os.wait4.ru_maxrss",
        }
    trial = compare_benchmarks(before, after, check=lambda: True)
    assert decide([trial] * 2, settings("memory", memory_metric="process_peak_rss"))[0] == "keep"
    uncapped = settings()
    uncapped.pop("max_memory_bytes")
    assert decide([trial] * 2, uncapped)[0] == "keep"
    for report in (before, after):
        report["totals"]["process_peak_rss"]["scope"] = "unknown"
    wrong_scope = compare_benchmarks(before, after, check=lambda: True)
    assert decide([wrong_scope] * 2, settings("memory", memory_metric="process_peak_rss"))[0] == "inconclusive"


def test_memory_objective_discards_latency_breach_and_marks_cap_overlap_inconclusive():
    slower = [comparison(0.01, 0.012, before_memory=20_000, after_memory=15_000)] * 2
    overlap = [comparison(0.01, 0.012, before_memory=20_000, after_memory=15_000, iqr=0.0002)] * 2

    assert decide(slower, settings("memory", max_latency_seconds=0.011))[0] == "discard"
    assert decide(overlap, settings("memory", max_latency_seconds=0.0119))[0] == "inconclusive"


def test_missing_supported_memory_and_changed_fixed_context_are_inconclusive():
    unavailable = benchmark(
        0.01,
        None,
        memory_metric={
            "status": "unavailable",
            "value": None,
            "known_value": None,
            "unit": "bytes",
            "scope": "pytorch_tensor_bytes",
            "method": "unavailable",
        },
    )
    unsupported = benchmark(
        0.01,
        20_000,
        memory_metric={
            "status": "complete",
            "value": 20_000,
            "known_value": 20_000,
            "unit": "bytes",
            "scope": "child_process_lifetime",
            "method": "os.wait4.ru_maxrss",
        },
    )
    missing_trial = compare_benchmarks(unavailable, deepcopy(unavailable), check=lambda: True)
    unsupported_trial = compare_benchmarks(unsupported, deepcopy(unsupported), check=lambda: True)
    context_trial = compare_benchmarks(
        benchmark(0.01, 20_000), benchmark(0.008, 20_000, context_changes={"warmup": 3}), check=lambda: True
    )

    assert decide([missing_trial, comparison(0.01, 0.008)], settings())[0] == "inconclusive"
    assert decide([unsupported_trial, comparison(0.01, 0.008)], settings())[0] == "inconclusive"
    assert decide([context_trial, comparison(0.01, 0.008)], settings())[0] == "inconclusive"


def test_input_metadata_must_match_across_trials():
    before, after = benchmark(0.01, 20_000), benchmark(0.008, 20_000)
    before["inputs"]["shape"] = after["inputs"]["shape"] = [4, 8]
    second = compare_benchmarks(before, after, check=lambda: True)

    assert decide([comparison(0.01, 0.008), second], settings())[0] == "inconclusive"


def test_tampered_delta_or_raw_timing_makes_evidence_inconclusive():
    changed_delta = comparison(0.01, 0.008)
    changed_delta["totals"]["latency"]["delta"] = -1
    changed_samples = comparison(0.01, 0.008)
    changed_samples["after"]["measurement"]["raw_times_seconds"] = [0.02] * 5

    assert decide([changed_delta, comparison(0.01, 0.008)], settings())[0] == "inconclusive"
    assert decide([changed_samples, comparison(0.01, 0.008)], settings())[0] == "inconclusive"


def test_known_constraint_failure_wins_over_an_incomplete_trial():
    known_failure = comparison(0.01, 0.008, after_memory=40_000)
    incomplete = deepcopy(comparison(0.01, 0.008))
    incomplete["after"]["measurement"]["raw_times_seconds"] = []

    assert decide([known_failure, incomplete], settings())[0] == "discard"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_memory_bytes", float("nan")),
        ("max_memory_bytes", float("inf")),
        ("max_memory_bytes", -1),
        ("min_improvement", 1),
    ],
)
def test_invalid_settings_are_rejected(name, value):
    with pytest.raises(ValueError):
        decide([comparison(0.01, 0.008)] * 2, settings(**{name: value}))
