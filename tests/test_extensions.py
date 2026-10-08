import json
import warnings
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import FrozenInstanceError
from itertools import combinations
from math import prod

import pytest
import torch
from torch import nn

from torchscan import (
    IncompleteAnalysisError,
    ModuleCall,
    ModuleEstimates,
    ModuleHandler,
    crawl_module,
    crawler,
    metric_result,
    render_report,
    summary,
)

COMPUTE_METRICS = ("module_flops", "macs", "dmas")
ALL_METRICS = (*COMPUTE_METRICS, "receptive_field", "effective_stride", "effective_padding")
RF_METRICS = ALL_METRICS[3:]
RF_SUBSETS = [subset for size in range(4) for subset in combinations(RF_METRICS, size)]


def _estimates(module_flops=0, macs=0, dmas=0):
    return {
        "module_flops": module_flops,
        "macs": macs,
        "dmas": dmas,
        "receptive_field": 1,
        "effective_stride": 1,
        "effective_padding": 0,
    }


def _layer(report, path, call_index=0):
    return next(layer for layer in report["layers"] if layer["path"] == path and layer["call_index"] == call_index)


def _handler(estimates, owned=()):
    return ModuleHandler(lambda _call: estimates, subtree_metrics=frozenset(owned))


def _analyze(model, handler=None, **kwargs):
    if not {"args", "kwargs", "input_shape"} & kwargs.keys():
        kwargs["args"] = (torch.ones(1, 2),)
    if handler is not None:
        kwargs["custom_modules"] = {type(model): handler}
    return crawl_module(model, **kwargs)


def _assert_metric(result, status, known_value):
    assert result["status"] == status
    assert result["value"] == (known_value if status == "complete" else None)
    assert result["known_value"] == known_value


def _assert_totals(report, **values):
    for name, value in values.items():
        _assert_metric(report["totals"][name], "complete", value)


def _diagnostics(report, **fields):
    return [item for item in report["diagnostics"] if all(item.get(name) == value for name, value in fields.items())]


def _state(model):
    return [
        (child, child.training, len(child._forward_pre_hooks), len(child._forward_hooks)) for child in model.modules()
    ]


def _must_not_run(*_args, **_kwargs):
    pytest.fail("Unexpected estimation execution")


class CustomIdentity(nn.Module):
    def forward(self, input_t):
        return input_t


class CountingIdentity(CustomIdentity):
    executions = 0

    def forward(self, input_t):
        self.executions += 1
        return input_t


class ChildModule(nn.Module):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def forward(self, input_t):
        return self.child(input_t)


class Sine(nn.Module):
    def forward(self, input_t):
        return input_t.sin()


class LinearPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.second = nn.Linear(2, 2)

    def forward(self, input_t):
        return self.second(self.first(input_t))


def test_handler_receives_complete_actual_call_and_nested_output():
    class Nested(nn.Module):
        def forward(self, payload, count, *, options, label="default"):
            del label
            self.output = (payload, {"options": options, "count": count, "metadata": [None, "output"]})
            return self.output

    model = Nested()
    payload = {"tensor": torch.ones(2, 3), "nested": (None, "input", 2 + 3j)}
    options = {"mask": [True, False], "config": object()}
    calls = []

    def estimate(call: ModuleCall) -> ModuleEstimates:
        calls.append(call)
        assert call.module is model
        assert call.args == (payload, 4)
        assert call.args[0] is payload
        assert call.kwargs == {"options": options}
        assert call.kwargs["options"] is options
        assert "label" not in call.kwargs
        assert call.output is model.output
        assert call.output[1]["count"] == 4
        assert call.output[1]["metadata"] == [None, "output"]
        return _estimates(9, 2, 15)

    report = crawl_module(
        model, args=(payload, 4), kwargs={"options": options}, custom_modules={Nested: ModuleHandler(estimate)}
    )

    assert len(calls) == 1
    assert isinstance(calls[0], ModuleCall)
    with pytest.raises(FrozenInstanceError):
        calls[0].output = None
    assert {name: report["totals"][name]["value"] for name in COMPUTE_METRICS} == {
        "module_flops": 9,
        "macs": 2,
        "dmas": 15,
    }
    assert all(_layer(report, "")["metrics"][name]["method"].startswith("custom") for name in ALL_METRICS)
    assert json.loads(json.dumps(report)) == report


def test_custom_handler_does_not_require_tensor_arguments_or_outputs():
    class ScalarModule(nn.Module):
        def forward(self, value, *, multiplier):
            return {"result": value * multiplier, "label": "scalar result"}

    def estimate(call):
        assert call.args == (3,)
        assert call.kwargs == {"multiplier": 4}
        assert call.output == {"result": 12, "label": "scalar result"}
        return _estimates(1)

    report = crawl_module(
        ScalarModule(),
        args=(3,),
        kwargs={"multiplier": 4},
        custom_modules={ScalarModule: ModuleHandler(estimate)},
        strict=True,
    )

    _assert_totals(report, module_flops=1, operator_flops=0)
    assert not report["diagnostics"]


@pytest.mark.parametrize("entrypoint", [crawl_module, summary])
def test_public_entrypoints_forward_custom_modules_and_operator_mapping(entrypoint, capsys):
    def operator_formula(input_shape, *, out_shape):
        assert input_shape == out_shape
        torch.cos(torch.ones(1))
        return 3 * prod(out_shape)

    report = entrypoint(
        Sine(),
        args=(torch.ones(5),),
        custom_modules={Sine: _handler(_estimates(10, 0, 10))},
        custom_mapping={torch.ops.aten.sin: operator_formula},
        strict=True,
    )

    _assert_totals(report, module_flops=10, operator_flops=15)
    assert report["operator_flops"]["by_operator"] == {"aten.sin": 15}
    if entrypoint is summary:
        assert capsys.readouterr().out


def test_caller_handler_overrides_builtins_and_omitted_fields_fall_back():
    model = nn.Linear(4, 2)
    inputs = torch.ones(2, 4)
    baseline = crawl_module(model, args=(inputs,))
    seen = []

    def estimate(call):
        seen.append(call)
        assert call.args == ()
        assert call.kwargs["input"] is inputs
        assert call.output.shape == (2, 2)
        return {"module_flops": 7, "receptive_field": 13}

    report = crawl_module(model, kwargs={"input": inputs}, custom_modules={nn.Linear: ModuleHandler(estimate)})

    assert len(seen) == 1
    metrics = _layer(report, "")["metrics"]
    assert metrics["module_flops"]["value"] == 7
    assert metrics["receptive_field"]["value"] == 13
    for name in ("macs", "dmas", "effective_stride", "effective_padding"):
        assert metrics[name] == _layer(baseline, "")["metrics"][name]
    assert report["operator_flops"] == baseline["operator_flops"]


def test_caller_base_class_handler_overrides_specific_builtin_formula():
    report = _analyze(nn.Linear(2, 2), custom_modules={nn.Module: _handler(_estimates(7, 3, 5))}, strict=True)

    _assert_totals(report, module_flops=7, macs=3, operator_flops=8)


@pytest.mark.parametrize("reverse", [False, True])
def test_subclass_handler_uses_closest_mro_match_independent_of_mapping_order(reverse):
    class Base(CustomIdentity):
        pass

    class Left(Base):
        pass

    class Right(Base):
        pass

    class Diamond(Left, Right):
        pass

    registrations = [
        (nn.Module, _handler(_estimates(1))),
        (Base, _handler(_estimates(2))),
        (Right, _handler(_estimates(3))),
        (Left, _handler(_estimates(4))),
    ]
    if reverse:
        registrations.reverse()
    mapping = dict(registrations)
    report = _analyze(Diamond(), custom_modules=mapping)
    assert report["totals"]["module_flops"]["value"] == 4

    mapping[Diamond] = _handler(_estimates(5))
    report = _analyze(Diamond(), custom_modules=mapping)
    assert report["totals"]["module_flops"]["value"] == 5


def test_missing_unknown_leaf_estimates_remain_explained_and_incomplete():
    report = _analyze(CustomIdentity(), _handler({"module_flops": 4}))

    assert report["totals"]["module_flops"]["value"] == 4
    for name in ("macs", "dmas"):
        result = _layer(report, "")["metrics"][name]
        assert result["status"] in {"partial", "unavailable"}
        assert result["value"] is None
        assert _diagnostics(report, metric=name)


def test_partial_metric_preserves_lower_bound_and_custom_method():
    partial = metric_result(status="partial", known_value=11, unit="FLOPs", scope="workload", method="real_arithmetic")
    report = _analyze(
        nn.Sequential(CustomIdentity(), nn.Linear(2, 2)),
        custom_modules={CustomIdentity: _handler(_estimates(partial, 0, 0))},
    )

    result = _layer(report, "0")["metrics"]["module_flops"]
    _assert_metric(result, "partial", 11)
    assert result["scope"] == "module_call"
    assert "custom" in result["method"]
    assert "real_arithmetic" in result["method"]
    total = report["totals"]["module_flops"]
    _assert_metric(total, "partial", 19)
    assert _diagnostics(report, metric="module_flops", path="0")


@pytest.mark.parametrize(
    "unavailable", [None, metric_result(status="unavailable", unit="MACs", scope="workload", method="not_estimated")]
)
def test_explicit_unavailable_metric_does_not_fall_back_to_builtin(unavailable):
    report = _analyze(nn.Linear(2, 2), _handler({"macs": unavailable}))

    result = _layer(report, "")["metrics"]["macs"]
    _assert_metric(result, "unavailable", None)
    assert result["method"].startswith("custom")
    assert _diagnostics(report, code="custom_metric_unavailable", metric="macs")
    assert report["totals"]["module_flops"]["status"] == "complete"


def test_legitimate_custom_zeros_are_complete_and_pass_strict_mode():
    report = _analyze(CustomIdentity(), _handler(_estimates()), strict=True)

    for name in COMPUTE_METRICS:
        _assert_metric(report["totals"][name], "complete", 0)
    assert not report["diagnostics"]


@pytest.mark.parametrize(
    "invalid",
    [
        -1,
        True,
        float("nan"),
        float("inf"),
        "5",
        {"status": "bogus"},
        {
            **metric_result(status="complete", value=2, unit="FLOPs", scope="module_call", method="estimate"),
            "known_value": 1,
        },
        {
            **metric_result(status="partial", known_value=2, unit="FLOPs", scope="module_call", method="estimate"),
            "value": 2,
        },
        metric_result(status="complete", value=2, unit="MACs", scope="module_call", method="estimate"),
    ],
)
def test_invalid_field_is_unavailable_without_discarding_valid_fields(invalid):
    report = _analyze(CustomIdentity(), _handler(_estimates(invalid, 3, 5)))

    metrics = _layer(report, "")["metrics"]
    _assert_metric(metrics["module_flops"], "unavailable", None)
    assert metrics["macs"]["value"] == 3
    assert metrics["dmas"]["value"] == 5
    assert _diagnostics(report, code="custom_metric_invalid", metric="module_flops")


@pytest.mark.parametrize("invalid_result", [None, 7, [1, 2], {"flops": 10}])
def test_invalid_callback_result_is_diagnostic(invalid_result):
    report = _analyze(nn.Identity(), _handler(invalid_result))

    for name in COMPUTE_METRICS:
        _assert_metric(report["totals"][name], "unavailable", None)
    assert _diagnostics(report, code="custom_handler_invalid")


def test_callback_failure_is_diagnostic_and_does_not_fall_back():
    def fail(_call):
        raise RuntimeError("caller formula failed")

    report = _analyze(nn.Identity(), ModuleHandler(fail))

    for name in ALL_METRICS:
        result = _layer(report, "")["metrics"][name]
        _assert_metric(result, "unavailable", None)
        assert any(
            diagnostic["code"] == "custom_handler_error"
            and diagnostic["metric"] == name
            and "caller formula failed" in diagnostic["message"]
            for diagnostic in report["diagnostics"]
        )


def test_strict_mode_raises_with_custom_failure_report():
    model = nn.Identity().train()
    before = _state(model)

    with pytest.raises(IncompleteAnalysisError) as exc_info:
        _analyze(model, _handler(_estimates(None, 0, 0)), strict=True)

    assert exc_info.value.report["totals"]["module_flops"]["status"] == "unavailable"
    assert _diagnostics(exc_info.value.report, code="custom_metric_unavailable")
    assert _state(model) == before


def test_composite_exclusive_estimate_adds_to_children_and_missing_fields_delegate():
    baseline = _analyze(LinearPair())
    report = _analyze(LinearPair(), _handler({"module_flops": 3}))

    assert report["totals"]["module_flops"]["value"] == baseline["totals"]["module_flops"]["value"] + 3
    for name in ("macs", "dmas"):
        assert report["totals"][name] == baseline["totals"][name]
        assert name not in _layer(report, "")["metrics"]
    assert _layer(report, "")["metrics"]["module_flops"]["scope"] == "module_call"


def test_composite_ownership_is_per_metric_and_preserves_rows_and_parameters():
    baseline = _analyze(LinearPair())
    report = _analyze(LinearPair(), _handler({"module_flops": 100}, {"module_flops"}))

    assert [layer["path"] for layer in report["layers"]] == ["", "first", "second"]
    assert report["totals"]["module_flops"]["value"] == 100
    assert _layer(report, "")["metrics"]["module_flops"]["scope"] == "subtree"
    for path in ("first", "second"):
        assert "module_flops" not in _layer(report, path)["metrics"]
        assert _layer(report, path)["metric_owners"]["module_flops"] == {"path": "", "call_index": 0}
        assert _layer(report, path)["metrics"]["macs"]["status"] == "complete"
    for name in ("parameters", "parameter_bytes", "macs", "dmas"):
        assert report["totals"][name] == baseline["totals"][name]
    assert sum(layer["parameters"]["trainable"] for layer in report["layers"]) == 12


def test_subtree_ownership_suppresses_owned_builtin_formula_execution(monkeypatch):
    monkeypatch.setattr(crawler, "module_flops", _must_not_run)
    report = _analyze(LinearPair(), _handler({"module_flops": 100}, {"module_flops"}), strict=True)
    _assert_totals(report, module_flops=100, macs=8)
    assert all(_layer(report, path)["metrics"]["macs"]["status"] == "complete" for path in ("first", "second"))


def test_full_subtree_ownership_skips_descendant_callbacks_but_keeps_structure():
    handlers = {LinearPair: _handler(_estimates(100, 50, 75), ALL_METRICS), nn.Linear: ModuleHandler(_must_not_run)}
    report = _analyze(LinearPair(), custom_modules=handlers, strict=True)
    assert len(report["layers"]) == 3
    _assert_totals(report, parameters=12, module_flops=100, macs=50, dmas=75)
    assert all(set(layer["metrics"]) == {"calls"} for layer in report["layers"][1:])


def test_nested_owners_keep_outer_ownership_and_delegate_other_metrics():
    model = ChildModule(LinearPair())
    model.register_parameter("marker", nn.Parameter(torch.ones(1)))
    model.register_buffer("metadata", torch.ones(3))
    handlers = {
        ChildModule: _handler({"module_flops": 100}, {"module_flops"}),
        LinearPair: _handler({"module_flops": 999, "macs": 50}, {"module_flops", "macs"}),
    }
    report = _analyze(model, custom_modules=handlers, strict=True)
    _assert_totals(report, module_flops=100, macs=50, parameters=13, buffer_elements=3)
    assert _layer(report, "")["parameters"]["trainable"] == 1
    assert _layer(report, "")["buffers"]["elements"] == 3
    assert "module_flops" not in _layer(report, "child")["metrics"]
    for path in ("child.first", "child.second"):
        owners = _layer(report, path)["metric_owners"]
        assert owners["module_flops"] == {"path": "", "call_index": 0}
        assert owners["macs"] == {"path": "child", "call_index": 0}
        assert _layer(report, path)["metrics"]["dmas"]["status"] == "complete"


@pytest.mark.parametrize(
    "owned_estimate",
    [None, -1, metric_result(status="partial", known_value=7, unit="FLOPs", scope="workload", method="bound")],
)
def test_incomplete_owner_never_falls_back_to_child_counts(owned_estimate):
    report = _analyze(LinearPair(), _handler({"module_flops": owned_estimate}, {"module_flops"}))

    result = report["totals"]["module_flops"]
    _assert_metric(
        result,
        "partial" if isinstance(owned_estimate, dict) else "unavailable",
        7 if isinstance(owned_estimate, dict) else None,
    )
    assert all("module_flops" not in layer["metrics"] for layer in report["layers"][1:])


@pytest.mark.parametrize("failure", ["missing", "exception"])
def test_missing_or_failed_owner_estimates_do_not_delegate_owned_metric(failure):
    def estimate(_call):
        if failure == "exception":
            raise RuntimeError("inclusive estimate failed")
        return {}

    report = _analyze(LinearPair(), ModuleHandler(estimate, subtree_metrics=frozenset({"module_flops"})))

    _assert_metric(report["totals"]["module_flops"], "unavailable", None)
    assert all("module_flops" not in layer["metrics"] for layer in report["layers"][1:])
    assert _diagnostics(report, metric="module_flops", path="")


def test_shared_child_ownership_depends_on_active_invocation():
    shared = nn.Linear(2, 2)
    model = nn.Sequential(ChildModule(shared), shared)
    child_calls = []

    def child_estimate(call):
        child_calls.append(call)
        return _estimates(8, 4, 10)

    handlers = {ChildModule: _handler(_estimates(100, 50, 75), ALL_METRICS), nn.Linear: ModuleHandler(child_estimate)}
    report = _analyze(model, custom_modules=handlers)
    assert len(child_calls) == 1
    _assert_totals(report, module_flops=108, macs=54, parameters=6)
    owned, outside = [_layer(report, "0.child", index) for index in (0, 1)]
    assert "module_flops" not in owned["metrics"]
    assert owned["metric_owners"]["module_flops"] == {"path": "0", "call_index": 0}
    assert outside["metrics"]["module_flops"]["value"] == 8
    assert "module_flops" not in outside.get("metric_owners", {})


def test_repeated_composite_invocations_own_only_their_own_calls():
    pair = LinearPair()
    calls = []

    def estimate(call):
        calls.append(call)
        return _estimates(20, 10, 30)

    handler = ModuleHandler(estimate, subtree_metrics=frozenset(ALL_METRICS))
    report = _analyze(nn.Sequential(pair, pair), custom_modules={LinearPair: handler})
    assert len(calls) == 2
    _assert_totals(report, module_flops=40, parameters=12)
    for call_index in (0, 1):
        for path in ("0.first", "0.second"):
            assert _layer(report, path, call_index)["metric_owners"]["module_flops"] == {
                "path": "0",
                "call_index": call_index,
            }


def test_callbacks_execute_once_and_tensor_bookkeeping_is_not_workload_compute():
    model = CountingIdentity().train()
    calls = []
    before = _state(model)

    def estimate(call):
        calls.append(call)
        assert not call.module.training
        left = torch.ones(2, 3)
        right = torch.ones(3, 4)
        torch.sin(left @ right)
        return _estimates(5)

    report = _analyze(model, ModuleHandler(estimate), strict=True)

    assert model.executions == 1
    assert len(calls) == 1
    assert report["operator_flops"]["by_operator"] == {}
    assert report["totals"]["operator_flops"]["value"] == 0
    assert not report["diagnostics"]
    assert _state(model) == before


def test_training_states_and_hooks_restore_after_callback_and_model_failure():
    model = LinearPair().train()
    model.second.eval()
    before = _state(model)

    def fail_estimate(_call):
        raise RuntimeError("formula failure")

    _analyze(model, ModuleHandler(fail_estimate))

    assert _state(model) == before

    def fail_forward(_input):
        raise RuntimeError("model failure")

    model.second.forward = fail_forward
    with pytest.raises(RuntimeError, match="model failure"):
        _analyze(model, ModuleHandler(fail_estimate))

    assert _state(model) == before


def test_structure_mode_skips_custom_callbacks_and_operator_formulas():
    model = LinearPair().train()
    handler = ModuleHandler(_must_not_run, subtree_metrics=frozenset(ALL_METRICS))
    report = _analyze(
        model, handler, custom_mapping={torch.ops.aten.addmm: _must_not_run}, mode="structure", strict=True
    )
    assert [layer["path"] for layer in report["layers"]] == ["", "first", "second"]
    _assert_totals(report, parameters=12)
    assert all(report["totals"][name]["method"] == "not_requested" for name in (*COMPUTE_METRICS, "operator_flops"))
    assert all(set(layer["metrics"]) == {"calls"} for layer in report["layers"])
    assert model.training


def test_custom_registration_is_scoped_to_one_analysis_and_does_not_mutate_mapping():
    model = nn.Linear(2, 2)
    inputs = torch.ones(1, 2)
    baseline = _analyze(model, args=(inputs,))
    handler = _handler(_estimates(123, 456, 789))
    mapping = {nn.Linear: handler}

    custom = _analyze(model, args=(inputs,), custom_modules=mapping)
    after = _analyze(model, args=(inputs,))
    explicit_empty = _analyze(model, args=(inputs,), custom_modules={})

    assert custom["totals"]["module_flops"]["value"] == 123
    assert after == explicit_empty == baseline
    assert mapping == {nn.Linear: handler}
    with pytest.raises(FrozenInstanceError):
        handler.estimate = lambda _call: {}


def test_operator_overrides_are_scoped_to_one_analysis():
    def formula(input_shape, *, out_shape):
        assert input_shape == out_shape
        return 2 * prod(out_shape)

    model = Sine()
    inputs = torch.ones(3)
    module_mapping = {Sine: _handler(_estimates(3))}
    operator_mapping = {torch.ops.aten.sin: formula}
    custom = _analyze(model, args=(inputs,), custom_modules=module_mapping, custom_mapping=operator_mapping)
    after = _analyze(model, args=(inputs,), custom_modules=module_mapping)

    assert custom["totals"]["operator_flops"]["status"] == "complete"
    assert custom["totals"]["operator_flops"]["value"] == 6
    assert after["totals"]["operator_flops"]["status"] == "complete"
    assert after["operator_flops"]["by_operator"] == {"aten.sin": 3}
    assert not _diagnostics(after, operator="aten.sin")
    assert operator_mapping == {torch.ops.aten.sin: formula}


@pytest.mark.parametrize(
    ("registration", "error_type"),
    [
        ([CustomIdentity], TypeError),
        ({"CustomIdentity": _handler({})}, TypeError),
        ({str: _handler({})}, TypeError),
        ({CustomIdentity: lambda _call: {}}, TypeError),
        ({CustomIdentity: ModuleHandler(None)}, TypeError),
        ({CustomIdentity: ModuleHandler(lambda _call: {}, subtree_metrics={"module_flops"})}, ValueError),
        ({CustomIdentity: ModuleHandler(lambda _call: {}, subtree_metrics=frozenset({"operator_flops"}))}, ValueError),
    ],
)
def test_invalid_registration_fails_before_model_execution(registration, error_type):
    model = CountingIdentity().train()
    with pytest.raises(error_type):
        _analyze(model, custom_modules=registration)

    assert model.executions == 0
    assert model.training
    assert not model._forward_pre_hooks
    assert not model._forward_hooks


def test_explicit_complex_formula_uses_six_real_flops_per_complex_multiply():
    class ComplexScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3, dtype=torch.complex64))

        def forward(self, input_t):
            return input_t * self.weight

    def estimate(call):
        assert call.args[0].is_complex()
        assert call.output.is_complex()
        # (a+ib)(c+id): four real multiplies and two real additions.
        elements = call.output.numel()
        return _estimates(6 * elements, 0, 2 * elements + call.module.weight.numel())

    def complex_multiply(_left_shape, _right_shape, *, out_shape):
        return 6 * prod(out_shape)

    report = crawl_module(
        ComplexScale(),
        args=(torch.ones(2, 3, dtype=torch.complex64),),
        custom_modules={ComplexScale: ModuleHandler(estimate)},
        custom_mapping={torch.ops.aten.mul: complex_multiply},
        strict=True,
    )

    _assert_totals(report, parameters=3, module_flops=36, operator_flops=36)
    assert report["operator_flops"]["by_operator"] == {"aten.mul": 36}
    _assert_totals(report, macs=0, dmas=15)


def test_caught_forward_failure_releases_subtree_ownership_before_fallback():
    class FailingOwner(ChildModule):
        def forward(self, input_t):
            self.child(input_t)
            raise RuntimeError("caught forward failure")

    class Recovery(nn.Sequential):
        def forward(self, input_t):
            with suppress(RuntimeError):
                self[0](input_t)
            return self[1](input_t)

    model = Recovery(FailingOwner(nn.Linear(2, 2)), nn.Linear(2, 2)).train()
    before = _state(model)
    handler = ModuleHandler(_must_not_run, subtree_metrics=frozenset({"module_flops"}))
    report = _analyze(model, custom_modules={FailingOwner: handler})
    assert _layer(report, "0")["output"] == {"kind": "failed"}
    assert _layer(report, "0.child")["metric_owners"]["module_flops"] == {"path": "0", "call_index": 0}
    assert "module_flops" not in _layer(report, "1").get("metric_owners", {})
    assert _layer(report, "1")["metrics"]["module_flops"]["value"] == 8
    _assert_metric(report["totals"]["module_flops"], "partial", 8)
    _assert_totals(report, parameters=12)
    assert _diagnostics(report, code="module_forward_error")
    assert _state(model) == before


def test_successful_none_output_reaches_callback():
    class ReturnsNone(nn.Module):
        def forward(self):
            return None

    outputs = []

    def estimate(call):
        outputs.append(call.output)
        return _estimates()

    report = _analyze(ReturnsNone(), ModuleHandler(estimate), args=(), strict=True)
    assert outputs == [None]
    assert _layer(report, "")["output"] == {"kind": "none"}


def test_recursive_composite_keeps_outer_invocation_ownership():
    class Recursive(nn.Module):
        def forward(self, input_t, remaining):
            return self(input_t, remaining - 1) if remaining else input_t

    calls = []

    def estimate(call):
        calls.append(call.args[1])
        return _estimates(10, 0, 0)

    report = _analyze(
        Recursive(),
        ModuleHandler(estimate, subtree_metrics=frozenset(ALL_METRICS)),
        args=(torch.ones(1), 2),
        strict=True,
    )
    assert calls == [2]
    assert [layer["call_index"] for layer in report["layers"]] == [0, 1, 2]
    assert report["totals"]["module_flops"]["value"] == 10
    assert all(
        layer["metric_owners"]["module_flops"] == {"path": "", "call_index": 0} for layer in report["layers"][1:]
    )


def test_nested_tensor_context_keeps_actual_objects_without_rectangular_metadata():
    inputs = torch.nested.nested_tensor([torch.ones(2, 3), torch.ones(4, 3)])
    calls = []

    def estimate(call):
        calls.append(call)
        assert call.args[0] is inputs
        assert call.output is inputs
        return _estimates(0, 0, 0)

    report = _analyze(CustomIdentity(), ModuleHandler(estimate), args=(inputs,), strict=True)
    assert len(calls) == 1
    assert report["inputs"]["args"][0]["kind"] == "nested_tensor"
    assert _layer(report, "")["output"]["kind"] == "nested_tensor"
    assert "shape" not in _layer(report, "")["output"]
    assert json.loads(json.dumps(report)) == report


def test_registered_atomic_composite_does_not_add_inclusive_fallback_to_children(monkeypatch):
    # Test the legacy atomic fallback separately from registered native handlers.
    monkeypatch.setattr(crawler, "_builtin_module_handlers", dict)
    model = nn.Transformer(d_model=4, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=8, dropout=0)
    report = _analyze(model, _handler({}), args=(torch.ones(3, 1, 4), torch.ones(2, 1, 4)))
    assert len(report["layers"]) > 1
    assert not any(name in _layer(report, "")["metrics"] for name in COMPUTE_METRICS)
    for name in COMPUTE_METRICS:
        known = sum(
            layer["metrics"][name]["known_value"] or 0 for layer in report["layers"] if name in layer["metrics"]
        )
        assert report["totals"][name]["known_value"] == known


@pytest.mark.parametrize("inclusive_custom", [False, True])
def test_builtin_fallback_owns_only_fields_it_supplies(monkeypatch, inclusive_custom):
    builtin = _handler(_estimates(100, 50, 75), ALL_METRICS)
    monkeypatch.setattr(crawler, "_builtin_module_handlers", lambda: {ChildModule: builtin})
    handlers = {
        ChildModule: _handler({"module_flops": 3}, {"module_flops"} if inclusive_custom else ()),
        # Inclusive fallback must prune these failed child fields and their diagnostics.
        CustomIdentity: _handler(_estimates(5, None, -1)),
    }
    report = _analyze(ChildModule(CustomIdentity()), custom_modules=handlers, strict=True)
    _assert_totals(report, module_flops=3 if inclusive_custom else 8, macs=50, dmas=75)
    child = _layer(report, "child")
    assert child["metric_owners"]["macs"] == {"path": "", "call_index": 0}
    assert "macs" not in child["metrics"]
    assert "dmas" not in child["metrics"]
    metrics = _layer(report, "")["metrics"]
    assert metrics["macs"]["method"] == "torchscan_module_formula"
    assert metrics["module_flops"]["method"].startswith("custom")
    assert not report["diagnostics"]


def test_builtin_error_uses_builtin_diagnostic_method_and_keeps_custom_field(monkeypatch):
    def builtin_estimate(_call):
        raise ValueError("builtin estimate failed")

    monkeypatch.setattr(crawler, "_builtin_module_handlers", lambda: {nn.Identity: ModuleHandler(builtin_estimate)})
    report = _analyze(nn.Identity(), _handler({"module_flops": 3}))
    assert report["totals"]["module_flops"]["value"] == 3
    assert report["totals"]["macs"]["status"] == "unavailable"
    assert all(diagnostic["code"] == "module_metric_error" for diagnostic in report["diagnostics"])
    assert all(diagnostic["metric"] != "module_flops" for diagnostic in report["diagnostics"])
    assert _layer(report, "")["metrics"]["macs"]["method"] == "torchscan_module_formula"


def test_omitted_non_tensor_leaf_receptive_metrics_remain_explicitly_unavailable():
    class Scalar(nn.Module):
        def forward(self, scalar):
            return scalar + 1

    with pytest.raises(IncompleteAnalysisError) as exc_info:
        _analyze(Scalar(), _handler({"module_flops": 1, "macs": 0, "dmas": 2}), args=(1,), strict=True)
    metrics = _layer(exc_info.value.report, "")["metrics"]
    for name in ("receptive_field", "effective_stride", "effective_padding"):
        assert metrics[name]["status"] == "unavailable"
        assert _diagnostics(exc_info.value.report, metric=name)


def test_inclusive_extent_fallback_does_not_hide_unowned_stride_failure(monkeypatch):
    builtin = _handler({"receptive_field": 7}, {"receptive_field"})
    monkeypatch.setattr(crawler, "_builtin_module_handlers", lambda: {ChildModule: builtin})
    compute_only = _handler({"module_flops": 0, "macs": 0, "dmas": 0})
    with pytest.raises(IncompleteAnalysisError) as exc_info:
        _analyze(
            ChildModule(CustomIdentity()),
            custom_modules={ChildModule: compute_only, CustomIdentity: compute_only},
            strict=True,
        )
    report = exc_info.value.report
    assert _layer(report, "")["metrics"]["receptive_field"]["value"] == 7
    assert "receptive_field" not in _layer(report, "child")["metrics"]
    _assert_metric(_layer(report, "child")["metrics"]["effective_stride"], "unavailable", None)
    assert _diagnostics(report, metric="effective_stride")


@pytest.fixture(params=["warning", "exception"])
def failed_receptive_formula(monkeypatch, request):
    def fail(*_args):
        if request.param == "exception":
            raise RuntimeError("legacy RF failed")
        warnings.warn("legacy RF failed", UserWarning, stacklevel=1)
        return 1, 1, 0

    monkeypatch.setattr(crawler, "module_rf", fail)
    return "module_metric_error" if request.param == "exception" else "unsupported_module_metric"


@pytest.mark.parametrize("overrides", RF_SUBSETS)
def test_receptive_failure_diagnoses_only_omitted_fields(failed_receptive_formula, overrides):
    handler = _handler(dict.fromkeys(overrides, 7)) if overrides else None
    report = _analyze(nn.Identity(), handler)
    omitted = [name for name in RF_METRICS if name not in overrides]
    metrics = _layer(report, "")["metrics"]
    for name in RF_METRICS:
        _assert_metric(
            metrics[name], "complete" if name in overrides else "unavailable", 7 if name in overrides else None
        )
    expected = [(failed_receptive_formula, omitted[0], "")] if omitted else []
    assert [(item["code"], item["metric"], item["path"]) for item in report["diagnostics"]] == expected


@pytest.mark.parametrize("covered", RF_SUBSETS)
def test_receptive_failure_survives_only_unowned_fields(monkeypatch, failed_receptive_formula, covered):
    builtin = _handler(dict.fromkeys(covered, 9), covered)
    monkeypatch.setattr(crawler, "_builtin_module_handlers", lambda: {ChildModule: builtin})
    handlers = {
        ChildModule: _handler(dict.fromkeys(COMPUTE_METRICS, 0)),
        nn.Identity: _handler({"receptive_field": 7}),
    }
    report = _analyze(ChildModule(nn.Identity()), custom_modules=handlers)
    child = _layer(report, "child")
    remaining = [name for name in RF_METRICS[1:] if name not in covered]
    for name in covered:
        assert name not in child["metrics"]
        assert child["metric_owners"][name] == {"path": "", "call_index": 0}
    for name in remaining:
        _assert_metric(child["metrics"][name], "unavailable", None)
    expected = [(failed_receptive_formula, remaining[0], "child")] if remaining else []
    assert [(item["code"], item["metric"], item["path"]) for item in report["diagnostics"]] == expected


@pytest.mark.usefixtures("failed_receptive_formula")
def test_pruned_receptive_failure_does_not_retarget_unrelated_builtin_field(monkeypatch):
    builtins = {
        ChildModule: _handler({"effective_padding": 0}, {"effective_padding"}),
        nn.Identity: _handler({"effective_stride": None}),
    }
    monkeypatch.setattr(crawler, "_builtin_module_handlers", lambda: builtins)
    handlers = {
        ChildModule: _handler(dict.fromkeys(COMPUTE_METRICS, 0)),
        nn.Identity: _handler({"receptive_field": 7}),
    }
    report = _analyze(ChildModule(nn.Identity()), custom_modules=handlers)
    assert [(item["code"], item["metric"], item["path"]) for item in report["diagnostics"]] == [
        ("unavailable_module_metric", "effective_stride", "child")
    ]


class FailingMapping(Mapping):
    def __init__(self, values, *, iteration=False, key=None, error=RuntimeError):
        self.values, self.iteration, self.key, self.error = values, iteration, key, error

    def __iter__(self):
        if self.iteration:
            raise RuntimeError("cannot list estimate fields")
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        if key == self.key:
            raise self.error("cannot read estimate field")
        return self.values[key]


@pytest.mark.parametrize(
    "failure", ["fields_iteration", "field_value", "field_missing", "metric_iteration", "metric_value"]
)
def test_mapping_access_failures_are_diagnostic_and_restore_state(failure):
    estimates = _estimates(5, 7, 9)
    if failure.startswith("metric"):
        metric = metric_result(status="complete", value=5, unit="FLOPs", scope="module_call", method="mapping")
        estimates["module_flops"] = FailingMapping(metric, iteration=failure == "metric_iteration", key="value")
    else:
        estimates = FailingMapping(
            estimates,
            iteration=failure == "fields_iteration",
            key="module_flops",
            error=KeyError if failure == "field_missing" else RuntimeError,
        )
    calls = []

    def estimate(call):
        calls.append(call)
        return estimates

    model = CustomIdentity().train()
    before = _state(model)
    handler = ModuleHandler(estimate)
    report = _analyze(model, handler)
    _assert_metric(report["totals"]["module_flops"], "unavailable", None)
    code = "custom_handler_error" if failure == "fields_iteration" else "custom_metric_invalid"
    assert _diagnostics(report, code=code, metric="module_flops")
    if failure == "fields_iteration":
        assert all(_layer(report, "")["metrics"][name]["status"] == "unavailable" for name in ALL_METRICS)
    else:
        _assert_totals(report, macs=7, dmas=9)
        assert len(report["diagnostics"]) == 1
    assert len(calls) == 1
    assert _state(model) == before
    with pytest.raises(IncompleteAnalysisError):
        _analyze(model, handler, strict=True)
    assert len(calls) == 2
    assert _state(model) == before


@pytest.mark.parametrize("registered_type", [nn.Linear, nn.Embedding])
def test_atomic_root_descendant_registration_expands_only_observed_calls(registered_type, monkeypatch):
    monkeypatch.setattr(crawler, "_builtin_module_handlers", dict)
    model = nn.Transformer(d_model=4, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=8, dropout=0)
    model.unused = nn.Embedding(3, 4)
    args = (torch.ones(3, 1, 4), torch.ones(2, 1, 4))
    baseline = _analyze(model, args=args)
    calls = []

    def estimate(call):
        calls.append(call)
        return _estimates(2, 1, 1)

    handlers = {registered_type: ModuleHandler(estimate)}
    structure = _analyze(model, args=args, custom_modules=handlers, mode="structure", strict=True)
    assert not calls
    assert not structure["diagnostics"]
    report = _analyze(model, args=args, custom_modules=handlers)
    assert report["totals"]["parameters"] == baseline["totals"]["parameters"]
    if registered_type is nn.Linear:
        assert calls
        assert len(_diagnostics(report, code="expanded_atomic_boundary", path="")) == 3
        for name in COMPUTE_METRICS:
            _assert_metric(_layer(report, "")["metrics"][name], "unavailable", None)
            known = sum(
                layer["metrics"][name]["known_value"] or 0 for layer in report["layers"] if name in layer["metrics"]
            )
            _assert_metric(report["totals"][name], "partial", known)
        with pytest.raises(IncompleteAnalysisError) as exc_info:
            _analyze(model, args=args, custom_modules=handlers, strict=True)
        assert len(_diagnostics(exc_info.value.report, code="expanded_atomic_boundary", path="")) == 3
    else:
        assert not calls
        assert report["totals"] == baseline["totals"]
        assert report["diagnostics"] == baseline["diagnostics"]
        for name in COMPUTE_METRICS:
            assert _layer(report, "")["metrics"][name]["scope"] == "subtree"
            assert all(name not in layer["metrics"] for layer in report["layers"][1:])
    assert render_report(report).startswith("<!doctype html>")
    assert _analyze(model, args=args) == baseline
