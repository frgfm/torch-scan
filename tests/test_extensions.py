import json
from contextlib import suppress
from dataclasses import FrozenInstanceError
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
    summary,
)

COMPUTE_METRICS = ("module_flops", "macs", "dmas")
ALL_METRICS = (*COMPUTE_METRICS, "receptive_field", "effective_stride", "effective_padding")


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


class CustomIdentity(nn.Module):
    def forward(self, input_t):
        return input_t


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

    assert report["totals"]["module_flops"]["value"] == 1
    assert report["totals"]["operator_flops"]["value"] == 0
    assert not report["diagnostics"]


@pytest.mark.parametrize("entrypoint", [crawl_module, summary])
def test_public_entrypoints_forward_custom_modules_and_operator_mapping(entrypoint, capsys):
    class Sine(nn.Module):
        def forward(self, input_t):
            return torch.sin(input_t)

    def operator_formula(input_shape, *, out_shape):
        assert input_shape == out_shape
        torch.cos(torch.ones(1))
        return 3 * prod(out_shape)

    report = entrypoint(
        Sine(),
        args=(torch.ones(5),),
        custom_modules={Sine: ModuleHandler(lambda _call: _estimates(10, 0, 10))},
        custom_mapping={torch.ops.aten.sin: operator_formula},
        strict=True,
    )

    assert report["totals"]["module_flops"]["value"] == 10
    assert report["totals"]["operator_flops"]["value"] == 15
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
    report = crawl_module(
        nn.Linear(2, 2),
        args=(torch.ones(1, 2),),
        custom_modules={nn.Module: ModuleHandler(lambda _call: _estimates(7, 3, 5))},
        strict=True,
    )

    assert report["totals"]["module_flops"]["value"] == 7
    assert report["totals"]["macs"]["value"] == 3
    assert report["totals"]["operator_flops"]["value"] == 8


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
        (nn.Module, ModuleHandler(lambda _call: _estimates(1))),
        (Base, ModuleHandler(lambda _call: _estimates(2))),
        (Right, ModuleHandler(lambda _call: _estimates(3))),
        (Left, ModuleHandler(lambda _call: _estimates(4))),
    ]
    if reverse:
        registrations.reverse()
    mapping = dict(registrations)
    report = crawl_module(Diamond(), args=(torch.ones(1),), custom_modules=mapping)
    assert report["totals"]["module_flops"]["value"] == 4

    mapping[Diamond] = ModuleHandler(lambda _call: _estimates(5))
    report = crawl_module(Diamond(), args=(torch.ones(1),), custom_modules=mapping)
    assert report["totals"]["module_flops"]["value"] == 5


def test_missing_unknown_leaf_estimates_remain_explained_and_incomplete():
    report = crawl_module(
        CustomIdentity(),
        args=(torch.ones(1),),
        custom_modules={CustomIdentity: ModuleHandler(lambda _call: {"module_flops": 4})},
    )

    assert report["totals"]["module_flops"]["value"] == 4
    for name in ("macs", "dmas"):
        result = _layer(report, "")["metrics"][name]
        assert result["status"] in {"partial", "unavailable"}
        assert result["value"] is None
        assert any(diagnostic["metric"] == name for diagnostic in report["diagnostics"])


def test_partial_metric_preserves_lower_bound_and_custom_method():
    partial = metric_result(status="partial", known_value=11, unit="FLOPs", scope="workload", method="real_arithmetic")
    report = crawl_module(
        nn.Sequential(CustomIdentity(), nn.Linear(2, 2)),
        args=(torch.ones(1, 2),),
        custom_modules={CustomIdentity: ModuleHandler(lambda _call: _estimates(partial, 0, 0))},
    )

    result = _layer(report, "0")["metrics"]["module_flops"]
    assert result["status"] == "partial"
    assert result["value"] is None
    assert result["known_value"] == 11
    assert result["scope"] == "module_call"
    assert "custom" in result["method"]
    assert "real_arithmetic" in result["method"]
    total = report["totals"]["module_flops"]
    assert total["status"] == "partial"
    assert total["value"] is None
    assert total["known_value"] == 19
    assert any(
        diagnostic["metric"] == "module_flops" and diagnostic["path"] == "0" for diagnostic in report["diagnostics"]
    )


@pytest.mark.parametrize(
    "unavailable", [None, metric_result(status="unavailable", unit="MACs", scope="workload", method="not_estimated")]
)
def test_explicit_unavailable_metric_does_not_fall_back_to_builtin(unavailable):
    report = crawl_module(
        nn.Linear(2, 2),
        args=(torch.ones(1, 2),),
        custom_modules={nn.Linear: ModuleHandler(lambda _call: {"macs": unavailable})},
    )

    result = _layer(report, "")["metrics"]["macs"]
    assert result["status"] == "unavailable"
    assert result["value"] is None
    assert result["known_value"] is None
    assert result["method"].startswith("custom")
    assert any(
        diagnostic["code"] == "custom_metric_unavailable" and diagnostic["metric"] == "macs"
        for diagnostic in report["diagnostics"]
    )
    assert report["totals"]["module_flops"]["status"] == "complete"


def test_legitimate_custom_zeros_are_complete_and_pass_strict_mode():
    report = crawl_module(
        CustomIdentity(),
        args=(torch.ones(1),),
        custom_modules={CustomIdentity: ModuleHandler(lambda _call: _estimates())},
        strict=True,
    )

    for name in COMPUTE_METRICS:
        assert report["totals"][name]["status"] == "complete"
        assert report["totals"][name]["value"] == 0
        assert report["totals"][name]["known_value"] == 0
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
    report = crawl_module(
        CustomIdentity(),
        args=(torch.ones(1),),
        custom_modules={CustomIdentity: ModuleHandler(lambda _call: _estimates(invalid, 3, 5))},
    )

    metrics = _layer(report, "")["metrics"]
    assert metrics["module_flops"]["status"] == "unavailable"
    assert metrics["module_flops"]["known_value"] is None
    assert metrics["macs"]["value"] == 3
    assert metrics["dmas"]["value"] == 5
    assert any(
        diagnostic["code"] == "custom_metric_invalid" and diagnostic["metric"] == "module_flops"
        for diagnostic in report["diagnostics"]
    )


@pytest.mark.parametrize("invalid_result", [None, 7, [1, 2], {"flops": 10}])
def test_invalid_callback_result_is_diagnostic(invalid_result):
    report = crawl_module(
        nn.Identity(), args=(torch.ones(1),), custom_modules={nn.Identity: ModuleHandler(lambda _call: invalid_result)}
    )

    for name in COMPUTE_METRICS:
        assert report["totals"][name]["status"] == "unavailable"
        assert report["totals"][name]["known_value"] is None
    assert any(diagnostic["code"] == "custom_handler_invalid" for diagnostic in report["diagnostics"])


def test_callback_failure_is_diagnostic_and_does_not_fall_back():
    def fail(_call):
        raise RuntimeError("caller formula failed")

    report = crawl_module(nn.Identity(), args=(torch.ones(1),), custom_modules={nn.Identity: ModuleHandler(fail)})

    for name in ALL_METRICS:
        result = _layer(report, "")["metrics"][name]
        assert result["status"] == "unavailable"
        assert result["known_value"] is None
        assert any(
            diagnostic["code"] == "custom_handler_error"
            and diagnostic["metric"] == name
            and "caller formula failed" in diagnostic["message"]
            for diagnostic in report["diagnostics"]
        )


def test_strict_mode_raises_with_custom_failure_report():
    model = nn.Identity().train()
    hooks_before = (len(model._forward_pre_hooks), len(model._forward_hooks))

    with pytest.raises(IncompleteAnalysisError) as exc_info:
        crawl_module(
            model,
            args=(torch.ones(1),),
            custom_modules={nn.Identity: ModuleHandler(lambda _call: _estimates(None, 0, 0))},
            strict=True,
        )

    assert exc_info.value.report["totals"]["module_flops"]["status"] == "unavailable"
    assert any(diagnostic["code"] == "custom_metric_unavailable" for diagnostic in exc_info.value.report["diagnostics"])
    assert model.training
    assert (len(model._forward_pre_hooks), len(model._forward_hooks)) == hooks_before


def test_composite_exclusive_estimate_adds_to_children_and_missing_fields_delegate():
    baseline = crawl_module(LinearPair(), args=(torch.ones(1, 2),))
    report = crawl_module(
        LinearPair(),
        args=(torch.ones(1, 2),),
        custom_modules={LinearPair: ModuleHandler(lambda _call: {"module_flops": 3})},
    )

    assert report["totals"]["module_flops"]["value"] == baseline["totals"]["module_flops"]["value"] + 3
    for name in ("macs", "dmas"):
        assert report["totals"][name] == baseline["totals"][name]
        assert name not in _layer(report, "")["metrics"]
    assert _layer(report, "")["metrics"]["module_flops"]["scope"] == "module_call"


def test_composite_ownership_is_per_metric_and_preserves_rows_and_parameters():
    baseline = crawl_module(LinearPair(), args=(torch.ones(1, 2),))
    report = crawl_module(
        LinearPair(),
        args=(torch.ones(1, 2),),
        custom_modules={
            LinearPair: ModuleHandler(lambda _call: {"module_flops": 100}, subtree_metrics=frozenset({"module_flops"}))
        },
    )

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
    def unexpected_flops(*_args):
        pytest.fail("Owned child FLOP formulas must not execute")

    monkeypatch.setattr("torchscan.crawler.module_flops", unexpected_flops)
    report = crawl_module(
        LinearPair(),
        args=(torch.ones(1, 2),),
        custom_modules={
            LinearPair: ModuleHandler(lambda _call: {"module_flops": 100}, subtree_metrics=frozenset({"module_flops"}))
        },
        strict=True,
    )

    assert report["totals"]["module_flops"]["value"] == 100
    assert report["totals"]["macs"]["value"] == 8
    assert all(_layer(report, path)["metrics"]["macs"]["status"] == "complete" for path in ("first", "second"))


def test_full_subtree_ownership_skips_descendant_callbacks_but_keeps_structure():
    def unexpected_callback(_call):
        pytest.fail("Descendant formula must not execute for owned metrics")

    report = crawl_module(
        LinearPair(),
        args=(torch.ones(1, 2),),
        custom_modules={
            LinearPair: ModuleHandler(lambda _call: _estimates(100, 50, 75), subtree_metrics=frozenset(ALL_METRICS)),
            nn.Linear: ModuleHandler(unexpected_callback),
        },
        strict=True,
    )

    assert len(report["layers"]) == 3
    assert report["totals"]["parameters"]["value"] == 12
    assert report["totals"]["module_flops"]["value"] == 100
    assert report["totals"]["macs"]["value"] == 50
    assert report["totals"]["dmas"]["value"] == 75
    assert all(set(layer["metrics"]) == {"calls"} for layer in report["layers"][1:])


def test_nested_owners_keep_outer_ownership_and_delegate_other_metrics():
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = LinearPair()
            self.marker = nn.Parameter(torch.ones(1))
            self.register_buffer("metadata", torch.ones(3))

        def forward(self, input_t):
            return self.inner(input_t)

    report = crawl_module(
        Outer(),
        args=(torch.ones(1, 2),),
        custom_modules={
            Outer: ModuleHandler(lambda _call: {"module_flops": 100}, subtree_metrics=frozenset({"module_flops"})),
            LinearPair: ModuleHandler(
                lambda _call: {"module_flops": 999, "macs": 50}, subtree_metrics=frozenset({"module_flops", "macs"})
            ),
        },
        strict=True,
    )

    assert report["totals"]["module_flops"]["value"] == 100
    assert report["totals"]["macs"]["value"] == 50
    assert report["totals"]["parameters"]["value"] == 13
    assert report["totals"]["buffer_elements"]["value"] == 3
    assert _layer(report, "")["parameters"]["trainable"] == 1
    assert _layer(report, "")["buffers"]["elements"] == 3
    assert "module_flops" not in _layer(report, "inner")["metrics"]
    for path in ("inner.first", "inner.second"):
        owners = _layer(report, path)["metric_owners"]
        assert owners["module_flops"] == {"path": "", "call_index": 0}
        assert owners["macs"] == {"path": "inner", "call_index": 0}
        assert _layer(report, path)["metrics"]["dmas"]["status"] == "complete"


@pytest.mark.parametrize(
    "owned_estimate",
    [None, -1, metric_result(status="partial", known_value=7, unit="FLOPs", scope="workload", method="bound")],
)
def test_incomplete_owner_never_falls_back_to_child_counts(owned_estimate):
    report = crawl_module(
        LinearPair(),
        args=(torch.ones(1, 2),),
        custom_modules={
            LinearPair: ModuleHandler(
                lambda _call: {"module_flops": owned_estimate}, subtree_metrics=frozenset({"module_flops"})
            )
        },
    )

    result = report["totals"]["module_flops"]
    assert result["status"] == ("partial" if isinstance(owned_estimate, dict) else "unavailable")
    assert result["value"] is None
    assert result["known_value"] == (7 if isinstance(owned_estimate, dict) else None)
    assert all("module_flops" not in layer["metrics"] for layer in report["layers"][1:])


@pytest.mark.parametrize("failure", ["missing", "exception"])
def test_missing_or_failed_owner_estimates_do_not_delegate_owned_metric(failure):
    def estimate(_call):
        if failure == "exception":
            raise RuntimeError("inclusive estimate failed")
        return {}

    report = crawl_module(
        LinearPair(),
        args=(torch.ones(1, 2),),
        custom_modules={LinearPair: ModuleHandler(estimate, subtree_metrics=frozenset({"module_flops"}))},
    )

    assert report["totals"]["module_flops"]["status"] == "unavailable"
    assert report["totals"]["module_flops"]["known_value"] is None
    assert all("module_flops" not in layer["metrics"] for layer in report["layers"][1:])
    assert any(
        diagnostic["metric"] == "module_flops" and diagnostic["path"] == "" for diagnostic in report["diagnostics"]
    )


def test_shared_child_ownership_depends_on_active_invocation():
    class Owner(nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.shared = shared

        def forward(self, input_t):
            return self.shared(input_t)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = Owner(nn.Linear(2, 2))
            self.shared = self.owner.shared

        def forward(self, input_t):
            return self.shared(self.owner(input_t))

    child_calls = []

    def child_estimate(call):
        child_calls.append(call)
        return _estimates(8, 4, 10)

    report = crawl_module(
        Model(),
        args=(torch.ones(1, 2),),
        custom_modules={
            Owner: ModuleHandler(lambda _call: _estimates(100, 50, 75), subtree_metrics=frozenset(ALL_METRICS)),
            nn.Linear: ModuleHandler(child_estimate),
        },
    )

    assert len(child_calls) == 1
    assert report["totals"]["module_flops"]["value"] == 108
    assert report["totals"]["macs"]["value"] == 54
    assert report["totals"]["parameters"]["value"] == 6
    owned = _layer(report, "owner.shared", 0)
    outside = _layer(report, "owner.shared", 1)
    assert "module_flops" not in owned["metrics"]
    assert owned["metric_owners"]["module_flops"] == {"path": "owner", "call_index": 0}
    assert outside["metrics"]["module_flops"]["value"] == 8
    assert "module_flops" not in outside.get("metric_owners", {})


def test_repeated_composite_invocations_own_only_their_own_calls():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pair = LinearPair()

        def forward(self, input_t):
            return self.pair(self.pair(input_t))

    calls = []

    def estimate(call):
        calls.append(call)
        return _estimates(20, 10, 30)

    report = crawl_module(
        Model(),
        args=(torch.ones(1, 2),),
        custom_modules={LinearPair: ModuleHandler(estimate, subtree_metrics=frozenset(ALL_METRICS))},
    )

    assert len(calls) == 2
    assert report["totals"]["module_flops"]["value"] == 40
    assert report["totals"]["parameters"]["value"] == 12
    for call_index in (0, 1):
        for path in ("pair.first", "pair.second"):
            assert _layer(report, path, call_index)["metric_owners"]["module_flops"] == {
                "path": "pair",
                "call_index": call_index,
            }


def test_callbacks_execute_once_and_tensor_bookkeeping_is_not_workload_compute():
    class Counting(CustomIdentity):
        executions = 0

        def forward(self, input_t):
            self.executions += 1
            return super().forward(input_t)

    model = Counting().train()
    calls = []
    hooks_before = (len(model._forward_pre_hooks), len(model._forward_hooks))

    def estimate(call):
        calls.append(call)
        assert not call.module.training
        left = torch.ones(2, 3)
        right = torch.ones(3, 4)
        torch.sin(left @ right)
        return _estimates(5)

    report = crawl_module(model, args=(torch.ones(1),), custom_modules={Counting: ModuleHandler(estimate)}, strict=True)

    assert model.executions == 1
    assert len(calls) == 1
    assert report["operator_flops"]["by_operator"] == {}
    assert report["totals"]["operator_flops"]["value"] == 0
    assert not report["diagnostics"]
    assert model.training
    assert (len(model._forward_pre_hooks), len(model._forward_hooks)) == hooks_before


def test_training_states_and_hooks_restore_after_callback_and_model_failure():
    model = LinearPair().train()
    model.second.eval()
    original_states = [(child, child.training) for child in model.modules()]
    original_hooks = [(child, len(child._forward_pre_hooks), len(child._forward_hooks)) for child in model.modules()]

    def fail_estimate(_call):
        raise RuntimeError("formula failure")

    crawl_module(model, args=(torch.ones(1, 2),), custom_modules={LinearPair: ModuleHandler(fail_estimate)})

    assert all(child.training == training for child, training in original_states)
    assert all(
        (len(child._forward_pre_hooks), len(child._forward_hooks)) == (pre, post) for child, pre, post in original_hooks
    )

    def fail_forward(_input):
        raise RuntimeError("model failure")

    model.second.forward = fail_forward
    with pytest.raises(RuntimeError, match="model failure"):
        crawl_module(model, args=(torch.ones(1, 2),), custom_modules={LinearPair: ModuleHandler(fail_estimate)})

    assert all(child.training == training for child, training in original_states)
    assert all(
        (len(child._forward_pre_hooks), len(child._forward_hooks)) == (pre, post) for child, pre, post in original_hooks
    )


def test_structure_mode_skips_custom_callbacks_and_operator_formulas():
    def unexpected_callback(_call):
        pytest.fail("Structure mode must not execute callbacks")

    def unexpected_operator(*_args, **_kwargs):
        pytest.fail("Structure mode must not execute operator formulas")

    model = LinearPair().train()
    report = crawl_module(
        model,
        args=(torch.ones(1, 2),),
        custom_modules={LinearPair: ModuleHandler(unexpected_callback, subtree_metrics=frozenset(ALL_METRICS))},
        custom_mapping={torch.ops.aten.addmm: unexpected_operator},
        mode="structure",
        strict=True,
    )

    assert [layer["path"] for layer in report["layers"]] == ["", "first", "second"]
    assert report["totals"]["parameters"]["value"] == 12
    assert all(report["totals"][name]["method"] == "not_requested" for name in (*COMPUTE_METRICS, "operator_flops"))
    assert all(set(layer["metrics"]) == {"calls"} for layer in report["layers"])
    assert model.training


def test_custom_registration_is_scoped_to_one_analysis_and_does_not_mutate_mapping():
    model = nn.Linear(2, 2)
    inputs = torch.ones(1, 2)
    baseline = crawl_module(model, args=(inputs,))
    handler = ModuleHandler(lambda _call: _estimates(123, 456, 789))
    mapping = {nn.Linear: handler}

    custom = crawl_module(model, args=(inputs,), custom_modules=mapping)
    after = crawl_module(model, args=(inputs,))
    explicit_empty = crawl_module(model, args=(inputs,), custom_modules={})

    assert custom["totals"]["module_flops"]["value"] == 123
    assert after == explicit_empty == baseline
    assert mapping == {nn.Linear: handler}
    with pytest.raises(FrozenInstanceError):
        handler.estimate = lambda _call: {}


def test_operator_overrides_are_scoped_to_one_analysis():
    class Sine(nn.Module):
        def forward(self, input_t):
            return input_t.sin()

    def formula(input_shape, *, out_shape):
        assert input_shape == out_shape
        return prod(out_shape)

    model = Sine()
    inputs = torch.ones(3)
    module_mapping = {Sine: ModuleHandler(lambda _call: _estimates(3))}
    operator_mapping = {torch.ops.aten.sin: formula}
    custom = crawl_module(model, args=(inputs,), custom_modules=module_mapping, custom_mapping=operator_mapping)
    after = crawl_module(model, args=(inputs,), custom_modules=module_mapping)

    assert custom["totals"]["operator_flops"]["status"] == "complete"
    assert custom["totals"]["operator_flops"]["value"] == 3
    assert after["totals"]["operator_flops"]["status"] == "partial"
    assert after["operator_flops"]["by_operator"] == {}
    assert any(diagnostic.get("operator") == "aten.sin" for diagnostic in after["diagnostics"])
    assert operator_mapping == {torch.ops.aten.sin: formula}


@pytest.mark.parametrize(
    ("registration", "error_type"),
    [
        ([CustomIdentity], TypeError),
        ({"CustomIdentity": ModuleHandler(lambda _call: {})}, TypeError),
        ({str: ModuleHandler(lambda _call: {})}, TypeError),
        ({CustomIdentity: lambda _call: {}}, TypeError),
        ({CustomIdentity: ModuleHandler(None)}, TypeError),
        ({CustomIdentity: ModuleHandler(lambda _call: {}, subtree_metrics={"module_flops"})}, ValueError),
        ({CustomIdentity: ModuleHandler(lambda _call: {}, subtree_metrics=frozenset({"operator_flops"}))}, ValueError),
    ],
)
def test_invalid_registration_fails_before_model_execution(registration, error_type):
    class Counting(CustomIdentity):
        executions = 0

        def forward(self, input_t):
            self.executions += 1
            return input_t

    model = Counting().train()
    with pytest.raises(error_type):
        crawl_module(model, args=(torch.ones(1),), custom_modules=registration)

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

    assert report["totals"]["parameters"]["value"] == 3
    assert report["totals"]["module_flops"]["value"] == 36
    assert report["totals"]["operator_flops"]["value"] == 36
    assert report["operator_flops"]["by_operator"] == {"aten.mul": 36}
    assert report["totals"]["macs"]["value"] == 0
    assert report["totals"]["dmas"]["value"] == 15


def test_caught_forward_failure_releases_subtree_ownership_before_fallback():
    class FailingOwner(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Linear(2, 2)

        def forward(self, input_t):
            self.child(input_t)
            raise RuntimeError("caught forward failure")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = FailingOwner()
            self.fallback = nn.Linear(2, 2)

        def forward(self, input_t):
            with suppress(RuntimeError):
                self.owner(input_t)
            return self.fallback(input_t)

    calls = []

    def estimate(call):
        calls.append(call)
        assert call.output is None
        return {"module_flops": None}

    model = Model().train()
    hooks_before = [(child, len(child._forward_pre_hooks), len(child._forward_hooks)) for child in model.modules()]
    report = crawl_module(
        model,
        args=(torch.ones(1, 2),),
        custom_modules={
            FailingOwner: ModuleHandler(estimate, subtree_metrics=frozenset({"module_flops"})),
        },
    )

    assert not calls
    assert _layer(report, "owner")["output"] == {"kind": "failed"}
    assert _layer(report, "owner.child")["metric_owners"]["module_flops"] == {"path": "owner", "call_index": 0}
    assert "module_flops" not in _layer(report, "fallback").get("metric_owners", {})
    assert _layer(report, "fallback")["metrics"]["module_flops"]["value"] == 8
    assert report["totals"]["module_flops"]["status"] == "partial"
    assert report["totals"]["module_flops"]["known_value"] == 8
    assert report["totals"]["parameters"]["value"] == 12
    assert any(diagnostic["code"] == "module_forward_error" for diagnostic in report["diagnostics"])
    assert model.training
    assert all(
        (len(child._forward_pre_hooks), len(child._forward_hooks)) == (pre, post) for child, pre, post in hooks_before
    )


def test_successful_none_output_reaches_callback():
    class ReturnsNone(nn.Module):
        def forward(self):
            return None

    outputs = []

    def estimate(call):
        outputs.append(call.output)
        return _estimates()

    report = crawl_module(ReturnsNone(), args=(), custom_modules={ReturnsNone: ModuleHandler(estimate)}, strict=True)
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

    report = crawl_module(
        Recursive(),
        args=(torch.ones(1), 2),
        custom_modules={Recursive: ModuleHandler(estimate, subtree_metrics=frozenset(ALL_METRICS))},
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

    report = crawl_module(
        CustomIdentity(), args=(inputs,), custom_modules={CustomIdentity: ModuleHandler(estimate)}, strict=True
    )
    assert len(calls) == 1
    assert report["inputs"]["args"][0]["kind"] == "nested_tensor"
    assert _layer(report, "")["output"]["kind"] == "nested_tensor"
    assert "shape" not in _layer(report, "")["output"]
    assert json.loads(json.dumps(report)) == report


def test_registered_atomic_composite_does_not_add_inclusive_fallback_to_children():
    model = nn.Transformer(d_model=4, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=8, dropout=0)
    report = crawl_module(
        model,
        args=(torch.ones(3, 1, 4), torch.ones(2, 1, 4)),
        custom_modules={nn.Transformer: ModuleHandler(lambda _call: {})},
    )
    assert len(report["layers"]) > 1
    assert not any(name in _layer(report, "")["metrics"] for name in COMPUTE_METRICS)
    for name in COMPUTE_METRICS:
        known = sum(
            layer["metrics"][name]["known_value"] or 0 for layer in report["layers"] if name in layer["metrics"]
        )
        assert report["totals"][name]["known_value"] == known


@pytest.mark.parametrize("inclusive_custom", [False, True])
def test_builtin_fallback_owns_only_fields_it_supplies(monkeypatch, inclusive_custom):
    class BuiltinComposite(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = CustomIdentity()

        def forward(self, input_t):
            return self.child(input_t)

    def builtin_estimate(_call):
        return _estimates(100, 50, 75)

    builtin = ModuleHandler(builtin_estimate, subtree_metrics=frozenset(ALL_METRICS))
    monkeypatch.setattr(crawler, "_builtin_module_handlers", lambda: {BuiltinComposite: builtin})
    report = crawl_module(
        BuiltinComposite(),
        args=(torch.ones(1),),
        custom_modules={
            BuiltinComposite: ModuleHandler(
                lambda _call: {"module_flops": 3},
                subtree_metrics=frozenset({"module_flops"}) if inclusive_custom else frozenset(),
            ),
            # These failed child fields must be pruned with their diagnostics
            # when an inclusive builtin fallback supplies the parent's fields.
            CustomIdentity: ModuleHandler(lambda _call: _estimates(5, None, -1)),
        },
        strict=True,
    )
    assert report["totals"]["module_flops"]["value"] == (3 if inclusive_custom else 8)
    assert report["totals"]["macs"]["value"] == 50
    assert report["totals"]["dmas"]["value"] == 75
    child = _layer(report, "child")
    assert child["metric_owners"]["macs"] == {"path": "", "call_index": 0}
    assert "macs" not in child["metrics"]
    assert "dmas" not in child["metrics"]
    assert _layer(report, "")["metrics"]["macs"]["method"] == "torchscan_module_formula"
    assert _layer(report, "")["metrics"]["module_flops"]["method"].startswith("custom")
    assert not report["diagnostics"]


def test_builtin_error_uses_builtin_diagnostic_method_and_keeps_custom_field(monkeypatch):
    def builtin_estimate(_call):
        raise ValueError("builtin estimate failed")

    monkeypatch.setattr(crawler, "_builtin_module_handlers", lambda: {nn.Identity: ModuleHandler(builtin_estimate)})
    report = crawl_module(
        nn.Identity(),
        args=(torch.ones(1),),
        custom_modules={nn.Identity: ModuleHandler(lambda _call: {"module_flops": 3})},
    )
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
        crawl_module(
            Scalar(),
            args=(1,),
            custom_modules={Scalar: ModuleHandler(lambda _call: {"module_flops": 1, "macs": 0, "dmas": 2})},
            strict=True,
        )
    metrics = _layer(exc_info.value.report, "")["metrics"]
    for name in ("receptive_field", "effective_stride", "effective_padding"):
        assert metrics[name]["status"] == "unavailable"
        assert any(diagnostic["metric"] == name for diagnostic in exc_info.value.report["diagnostics"])


def test_inclusive_extent_fallback_does_not_hide_unowned_stride_failure(monkeypatch):
    class Composite(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = CustomIdentity()

        def forward(self, input_t):
            return self.child(input_t)

    monkeypatch.setattr(
        crawler,
        "_builtin_module_handlers",
        lambda: {Composite: ModuleHandler(lambda _call: {"receptive_field": 7}, frozenset({"receptive_field"}))},
    )
    with pytest.raises(IncompleteAnalysisError) as exc_info:
        crawl_module(
            Composite(),
            args=(torch.ones(1),),
            custom_modules={
                Composite: ModuleHandler(lambda _call: {"module_flops": 0, "macs": 0, "dmas": 0}),
                CustomIdentity: ModuleHandler(lambda _call: {"module_flops": 0, "macs": 0, "dmas": 0}),
            },
            strict=True,
        )
    report = exc_info.value.report
    assert _layer(report, "")["metrics"]["receptive_field"]["value"] == 7
    assert "receptive_field" not in _layer(report, "child")["metrics"]
    assert _layer(report, "child")["metrics"]["effective_stride"]["status"] == "unavailable"
    assert any(diagnostic["metric"] == "effective_stride" for diagnostic in report["diagnostics"])
