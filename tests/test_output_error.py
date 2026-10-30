import pytest
import torch

from torchscan._output_error import compare_outputs


def test_absolute_error_statistics_and_linear_percentiles():
    passed, outputs = compare_outputs(
        torch.tensor([-4.0, -2.0, 0.0, 1.0, 3.0]), torch.zeros(5), {"absolute_error": {"max": 4}}
    )
    assert passed
    assert outputs["/"]["count"] == 5
    for name, expected in {"mean": 2, "median": 2, "p90": 3.6, "p95": 3.8, "p99": 3.96, "max": 4}.items():
        assert outputs["/"][name] == pytest.approx(expected)


def test_percentile_constraint_is_distinct_from_pointwise_maximum():
    actual = torch.zeros(101)
    actual[-1] = 10
    passed, outputs = compare_outputs(actual, torch.zeros(101), {"absolute_error": {"p99": 0}})
    assert passed
    assert outputs["/"]["p99"] == 0
    assert outputs["/"]["max"] == 10
    assert not compare_outputs(actual, torch.zeros(101), {"absolute_error": {"max": 1}})[0]


def test_float_dtype_changes_are_allowed_without_rounding_the_reference():
    reference = torch.tensor([1.0003], dtype=torch.float32)
    candidate = reference.half()
    passed, outputs = compare_outputs(candidate, reference, {"absolute_error": {"mean": 0.0002}})
    assert not passed
    assert outputs["/"]["mean"] > 0.0002
    assert outputs["/"]["candidate_dtype"] == "torch.float16"
    assert compare_outputs(candidate, reference, {"absolute_error": {"mean": 0.001}})[0]
    assert compare_outputs(torch.ones(2).half(), torch.ones(2), {"rtol": 0, "atol": 0})[0]


def test_scalar_float_outputs_use_the_same_error_contract():
    passed, outputs = compare_outputs(1.01, 1.0, {"absolute_error": {"p99": 0.02}})
    assert passed
    assert outputs["/"]["p99"] == pytest.approx(0.01)
    assert compare_outputs(1.01, 1.0, {"rtol": 0, "atol": 0.02})[0]
    assert not compare_outputs(1.0, 1, {"absolute_error": {"p99": 1}})[0]


def test_nested_outputs_are_gated_separately_without_dilution():
    actual = {"large": torch.zeros(1000), "small": [torch.ones(1)]}
    reference = {"large": torch.zeros(1000), "small": [torch.zeros(1)]}
    passed, outputs = compare_outputs(actual, reference, {"absolute_error": {"mean": 0.1}})
    assert not passed
    assert outputs["/large"]["mean"] == 0
    assert outputs["/small/0"]["mean"] == 1


@pytest.mark.parametrize(
    "candidate", [torch.ones(3), torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.ones(1).long()]
)
def test_shape_nonfinite_or_integer_type_mismatch_cannot_pass(candidate):
    passed, _ = compare_outputs(candidate, torch.ones(1), {"absolute_error": {"mean": 1e30}})
    assert not passed


def test_structure_and_empty_outputs_cannot_hide_a_failure():
    for candidate, reference in (({}, {"x": torch.ones(1)}), ([], [None]), (torch.empty(0), torch.empty(0))):
        assert not compare_outputs(candidate, reference, {"absolute_error": {"max": 0}})[0]
