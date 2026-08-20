import json
import warnings

import pytest
import torch
from torch import nn

from torchscan import crawler, modules


def _total(report, name):
    return report["totals"][name]["value"]


def _layer(report, path):
    return next(layer for layer in report["layers"] if layer["path"] == path)


def test_crawl_module_report_and_summary(capsys):
    mod = nn.Conv2d(3, 8, 3)

    report = crawler.crawl_module(mod, (3, 32, 32))
    layer = report["layers"][0]

    assert report["schema_version"] == 1
    assert _total(report, "parameters") == 224
    assert _total(report, "module_flops") == 388_800
    assert _total(report, "macs") == 194_400
    assert _total(report, "dmas") == 201_824
    assert layer["output"]["shape"] == [1, 8, 30, 30]
    assert json.loads(json.dumps(report)) == report

    returned = crawler.summary(mod, (3, 32, 32))
    assert returned == report
    assert "conv2d    Conv2d    (1, 8, 30, 30)    224" in capsys.readouterr().out


def test_internal_metadata_and_formula_boundaries():
    class Inputs(nn.Module):
        def forward(self, first, _second):
            return first

    class Variadic(nn.Module):
        def forward(self, *inputs):
            return inputs[0]

    inputs_signature = crawler.inspect.signature(Inputs().forward)
    assert crawler._ordered_inputs(inputs_signature, (1,), {}) == (1,)
    assert crawler._ordered_inputs(inputs_signature, (), {"unexpected": 2}) == (2,)
    assert crawler._ordered_inputs(crawler.inspect.signature(Variadic().forward), (1, 2), {}) == (1, 2)
    assert crawler._ordered_inputs(None, (1,), {"extra": 2}) == (1, 2)

    diagnostics = []

    def warned_formula():
        warnings.warn("formula warning", UserWarning, stacklevel=1)
        return 3

    result = crawler._measure_module_metric("flops", "FLOPs", "", diagnostics, warned_formula)
    assert result["status"] == "partial"
    assert diagnostics[0]["code"] == "module_metric_warning"
    assert crawler._aggregate_metric([], "flops", "FLOPs")["status"] == "unavailable"


@pytest.mark.parametrize("error_type", [TypeError, ValueError])
def test_uninspectable_metric_leaf_uses_fallback_and_cleans_up(monkeypatch, error_type):
    model = nn.Identity().train()
    original_signature = crawler.inspect.signature
    hook_counts = (len(model._forward_pre_hooks), len(model._forward_hooks))

    def unavailable_signature(callable_):
        if getattr(callable_, "__self__", None) is model:
            raise error_type
        return original_signature(callable_)

    monkeypatch.setattr(crawler.inspect, "signature", unavailable_signature)

    report = crawler.crawl_module(model, args=(torch.ones(1),))

    assert report["layers"][0]["path"] == ""
    assert model.training
    assert (len(model._forward_pre_hooks), len(model._forward_hooks)) == hook_counts


def test_reused_metric_leaf_signature_is_inspected_once_per_crawl(monkeypatch):
    class ReusedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(4, 4)

        def forward(self, input_t):
            return self.shared(self.shared(input_t))

    model = ReusedModel()
    input_t = torch.randn(2, 4)
    original_signature = crawler.inspect.signature
    signature_calls = 0

    def counting_signature(callable_):
        nonlocal signature_calls
        if getattr(callable_, "__self__", None) is model.shared:
            signature_calls += 1
        return original_signature(callable_)

    monkeypatch.setattr(crawler.inspect, "signature", counting_signature)

    report = crawler.crawl_module(model, args=(input_t,))

    assert signature_calls == 1
    assert [(layer["path"], layer["call_index"]) for layer in report["layers"] if layer["path"] == "shared"] == [
        ("shared", 0),
        ("shared", 1),
    ]


def test_package_fallback_and_object_metadata(monkeypatch):
    def missing_package(_):
        raise crawler.PackageNotFoundError

    monkeypatch.setattr(crawler, "version", missing_package)
    assert crawler._package_version() == "unknown"

    class IgnoreObject(nn.Module):
        def forward(self, _value):
            return torch.ones(1)

    report = crawler.crawl_module(IgnoreObject(), args=(object(),))
    assert report["inputs"]["args"][0]["kind"] == "object"


def test_receptive_field_failure_is_diagnostic(monkeypatch):
    def fail_receptive_field(*_args):
        raise RuntimeError("receptive field failed")

    monkeypatch.setattr(crawler, "module_rf", fail_receptive_field)
    report = crawler.crawl_module(nn.Identity(), args=(torch.ones(1),))

    assert any(
        item["metric"] == "receptive_field" and item["code"] == "module_metric_error" for item in report["diagnostics"]
    )


def test_crawl_module_shared_parameters_and_buffers():
    mod = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    mod[1].weight = mod[0].weight
    mod[0].register_buffer("stats", torch.ones(4))
    mod[1].register_buffer("stats", mod[0].stats)

    report = crawler.crawl_module(mod, (4,))
    first, second = _layer(report, "0"), _layer(report, "1")
    num_params = mod[0].weight.numel()
    num_buffers = mod[0].stats.numel()

    assert _total(report, "parameters") == num_params
    assert _total(report, "parameter_bytes") == num_params * mod[0].weight.element_size()
    assert _total(report, "buffer_elements") == num_buffers
    assert _total(report, "buffer_bytes") == num_buffers * mod[0].stats.element_size()
    assert (first["parameters"]["trainable"], second["parameters"]["trainable"]) == (num_params, 0)
    assert (first["buffers"]["elements"], second["buffers"]["elements"]) == (num_buffers, 0)
    assert first["parameters"]["shared"] is False
    assert second["parameters"]["shared"] is True


def test_crawl_module_aggregates_compute_metrics():
    report = crawler.crawl_module(nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2)), (8,))

    assert {name: _total(report, name) for name in ("module_flops", "macs", "dmas")} == {
        "module_flops": 84,
        "macs": 40,
        "dmas": 72,
    }
    for name in ("module_flops", "macs", "dmas"):
        assert _total(report, name) == sum(
            layer["metrics"][name]["value"] for layer in report["layers"] if name in layer["metrics"]
        )


def test_crawl_module_collects_operator_flops_from_the_same_forward():
    report = crawler.crawl_module(nn.Linear(4, 2), (4,))

    assert report["totals"]["operator_flops"] == report["operator_flops"]["total"]
    assert report["operator_flops"]["by_operator"] == {"aten.addmm": 16}


def test_module_formula_tensor_ops_do_not_pollute_operator_report(monkeypatch):
    original = crawler.module_flops

    def formula(*args, **kwargs):
        torch.sin(torch.ones(1))
        return original(*args, **kwargs)

    monkeypatch.setattr(crawler, "module_flops", formula)
    report = crawler.crawl_module(nn.Linear(4, 2), (4,))

    assert "aten.sin" not in report["operator_flops"]["by_operator"]
    assert all(diagnostic.get("operator") != "aten.sin" for diagnostic in report["diagnostics"])


def test_crawl_module_preserves_nested_output_metadata_without_values():
    class NestedOutputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, input_t):
            output = self.linear(input_t)
            return output, [output[:, :2], {"scores": output[:, :1], "label": "secret"}]

    report = crawler.crawl_module(NestedOutputs(), (4,))
    metadata = report["layers"][0]["output"]

    assert metadata["kind"] == "tuple"
    assert metadata["items"][0]["shape"] == [1, 4]
    assert "secret" not in json.dumps(metadata)


def test_crawl_module_maxpool_with_indices():
    mod = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.MaxPool2d(2, return_indices=True))
    report = crawler.crawl_module(mod, (1, 4, 4))
    pool = _layer(report, "1")

    assert pool["output"]["kind"] == "tuple"
    assert [item["shape"] for item in pool["output"]["items"]] == [[1, 1, 2, 2], [1, 1, 2, 2]]


def test_crawl_module_multihead_attention():
    mod = nn.MultiheadAttention(8, 2, batch_first=True)
    query = torch.rand((1, 4, 8))

    report = crawler.crawl_module(mod, args=(query, query, query))
    layer = report["layers"][0]

    assert len(report["layers"]) == 1
    assert layer["output"]["kind"] == "tuple"
    assert layer["metrics"]["module_flops"]["value"] == modules.module_flops(
        mod, (query, query, query), mod(query, query, query)
    )


def _tiny_transformer(batch_first):
    return nn.Transformer(
        d_model=4,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=8,
        dropout=0,
        batch_first=batch_first,
    )


@pytest.mark.parametrize("batch_first", [False, True])
def test_crawl_module_transformer_formula(batch_first):
    mod = _tiny_transformer(batch_first)
    if batch_first:
        src, tgt = torch.rand((1, 3, 4)), torch.rand((1, 2, 4))
    else:
        src, tgt = torch.rand((3, 1, 4)), torch.rand((2, 1, 4))

    report = crawler.crawl_module(mod, args=(src, tgt))

    assert len(report["layers"]) == 1
    assert report["layers"][0]["metrics"]["module_flops"]["value"] == 2635


def test_metric_failure_is_diagnostic_and_hooks_are_removed():
    mod = nn.MultiheadAttention(4, 2, batch_first=True, add_zero_attn=True)
    query = torch.rand((1, 3, 4))
    expected_hook_counts = len(mod._forward_pre_hooks), len(mod._forward_hooks)

    report = crawler.crawl_module(mod, args=(query, query, query))

    assert report["totals"]["module_flops"]["status"] == "unavailable"
    assert any(item["code"] == "module_metric_error" for item in report["diagnostics"])
    assert (len(mod._forward_pre_hooks), len(mod._forward_hooks)) == expected_hook_counts


def test_non_tensor_output_is_reported_without_exposing_value():
    class NoTensorOutput(nn.Module):
        def forward(self, _):
            return {"label": "secret"}

    report = crawler.crawl_module(NoTensorOutput(), (4,))

    assert report["totals"]["module_flops"]["status"] == "unavailable"
    assert "secret" not in json.dumps(report)


def test_generated_integer_input():
    class CaptureEmbedding(nn.Embedding):
        def forward(self, tensor):
            self.received = tensor
            return super().forward(tensor)

    mod = CaptureEmbedding(8, 4)
    crawler.crawl_module(mod, (3,), dtype=torch.long)

    assert mod.received.shape == (1, 3)
    assert mod.received.dtype == torch.long


def test_summary_depth_and_trainable_column(capsys):
    mod = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    mod[0].requires_grad_(False)

    crawler.summary(mod, (4,), max_depth=1)
    output = capsys.readouterr().out

    assert "Trainable" in output
    assert "Linear" in output
    assert "False" in output
