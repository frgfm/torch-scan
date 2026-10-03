import json
from math import prod

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

from scripts.benchmark import metric_cell
from torchscan import IncompleteAnalysisError, crawl_module
from torchscan.flops import measure_flops
from torchscan.modules import module_flops, module_macs
from torchscan.report import metric_result


def test_measure_flops_matmul_and_module_hierarchy():
    left = torch.ones(2, 3)
    right = torch.ones(3, 4)
    linear = nn.Linear(4, 2)

    report = measure_flops(lambda: linear(left @ right), modules=linear)

    assert report["total"] == {
        "status": "complete",
        "value": 80,
        "known_value": 80,
        "unit": "FLOPs",
        "scope": "workload",
        "method": "torch.utils.flop_counter.FlopCounterMode",
    }
    assert report["by_operator"] == {"aten.addmm": 32, "aten.mm": 48}
    assert report["by_module"]["Linear"] == 32
    assert report["diagnostics"] == []
    assert json.loads(json.dumps(report)) == report


def test_measure_flops_custom_mapping():
    inputs = torch.ones(5)

    def sin_flops(input_shape, *, out_shape):
        assert input_shape == out_shape
        return prod(out_shape)

    report = measure_flops(lambda: torch.sin(inputs), custom_mapping={torch.ops.aten.sin: sin_flops})

    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == 5
    assert report["by_operator"] == {"aten.sin": 5}
    assert report["diagnostics"] == []


def test_measure_flops_uses_explicit_modules_on_legacy_counter(monkeypatch):
    calls = []

    class LegacyCounter:
        def __init__(self, mods=None, **_kwargs):
            calls.append(mods)
            self.flop_mapping = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_flop_counts(self):
            return {}

    monkeypatch.setattr("torchscan.flops.FlopCounterMode", LegacyCounter)
    module = nn.Identity()

    report = measure_flops(lambda: None, modules=module)

    assert calls == [None, module]
    assert report["total"]["value"] == 0


def test_measure_flops_rejects_overload_custom_mapping_keys():
    calls = 0

    def workload():
        nonlocal calls
        calls += 1

    with pytest.raises(TypeError, match="operator packets"):
        measure_flops(workload, custom_mapping={torch.ops.aten.sin.default: lambda *_args, **_kwargs: 1})

    assert calls == 0


def test_measure_flops_requires_an_inspectable_native_mapping(monkeypatch):
    class OpaqueCounter:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr("torchscan.flops.FlopCounterMode", OpaqueCounter)
    with pytest.raises(NotImplementedError, match="does not expose its formula mapping"):
        measure_flops(lambda: pytest.fail("Workload must not run without an inspectable mapping"))


def test_measure_flops_custom_zero_is_complete():
    inputs = torch.ones(5)

    def zero_flops(input_shape, *, out_shape):
        del input_shape, out_shape
        return 0

    mapping = {torch.ops.aten.sin: zero_flops}

    report = measure_flops(lambda: torch.sin(inputs), custom_mapping=mapping)

    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == 0
    assert report["by_operator"] == {"aten.sin": 0}
    assert mapping.keys() == {torch.ops.aten.sin}


def test_custom_formula_tensor_ops_are_not_recorded_as_workload_ops():
    inputs = torch.ones(5)

    def sin_flops(input_shape, *, out_shape):
        assert input_shape == out_shape
        torch.cos(torch.ones(1))
        return prod(out_shape)

    report = measure_flops(lambda: torch.sin(inputs), custom_mapping={torch.ops.aten.sin: sin_flops})

    assert report["by_operator"] == {"aten.sin": 5}
    assert all(diagnostic.get("operator") != "aten.cos" for diagnostic in report["diagnostics"])


def test_measure_flops_reports_uncounted_operator_as_partial():
    left = torch.ones(2, 3)
    right = torch.ones(3, 4)

    report = measure_flops(lambda: torch.sin(left @ right))

    assert report["total"]["status"] == "partial"
    assert report["total"]["value"] is None
    assert report["total"]["known_value"] == 48
    assert report["by_operator"] == {"aten.mm": 48}
    assert report["diagnostics"] == [
        {
            "code": "uncounted_operator",
            "severity": "warning",
            "metric": "flops",
            "operator": "aten.sin",
            "message": "aten.sin was observed 1 time(s), but no FLOP formula is registered.",
        }
    ]


def test_measure_flops_explicitly_ignores_zero_flop_operator():
    inputs = torch.ones(2, 3)

    report = measure_flops(lambda: inputs.view(3, 2))

    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == 0
    assert report["ignored_operators"] == {"aten.view": {"calls": 1, "reason": "Metadata-only tensor view."}}


def test_measure_flops_invokes_workload_once():
    calls = 0

    def workload():
        nonlocal calls
        calls += 1

    report = measure_flops(workload)

    assert calls == 1
    assert report["total"]["value"] == 0


def test_measure_flops_preserves_workload_exception():
    error = RuntimeError("workload failed")
    calls = 0

    def workload():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(RuntimeError) as exc_info:
        measure_flops(workload)

    assert exc_info.value is error
    assert calls == 1


# Independent arithmetic examples. Expected counts below are derived by hand.
def _complete_count(workload):
    total = measure_flops(workload)["total"]
    assert total["status"] == "complete"
    return total["value"]


@pytest.mark.parametrize(
    ("module", "shape", "output_shape", "expected"),
    [
        # 60 outputs, each with 2 channels x 3 taps: 360 MACs, 660 exact ops.
        (nn.Conv1d(4, 6, 3, groups=2, bias=False), (2, 4, 7), (2, 6, 5), (660, 720, 360)),
        # Depthwise multiplier 2: 150 outputs x 9 taps. Bias restores one op/output.
        (nn.Conv2d(3, 6, 3, groups=3, padding=1), (1, 3, 5, 5), (1, 6, 5, 5), (2700, 2700, 1350)),
        # 16 outputs x 8 taps, despite dilation spacing the taps apart.
        (nn.Conv3d(2, 2, 2, groups=2, dilation=2), (1, 2, 4, 4, 4), (1, 2, 2, 2, 2), (256, 256, 128)),
        # 108 outputs x (2 channels x 6 taps), asymmetric stride/padding/dilation.
        (
            nn.Conv2d(4, 6, (2, 3), groups=2, stride=(2, 1), padding=(1, 0), dilation=(2, 1), bias=False),
            (2, 4, 6, 5),
            (2, 6, 3, 3),
            (2484, 2592, 1296),
        ),
        # Unbatched grouped convolution has the same channel arithmetic.
        (nn.Conv1d(4, 6, 3, groups=2, bias=False), (4, 7), (6, 5), (330, 360, 180)),
        # Transpose: 24 input values x (3 destination channels x 3 taps) = 216 MACs.
        # 72 output biases; padding/output_padding determine shape, not dense MACs.
        (
            nn.ConvTranspose1d(4, 6, 3, groups=2, stride=2, padding=1, output_padding=1),
            (2, 4, 3),
            (2, 6, 6),
            (504, 432, 216),
        ),
        # 12 inputs x 2 destination channels x 6 taps = 144 MACs; 100 output biases.
        (
            nn.ConvTranspose2d(2, 4, (2, 3), groups=2, stride=2, padding=(0, 1), output_padding=(1, 0)),
            (1, 2, 2, 3),
            (1, 4, 5, 5),
            (388, 288, 144),
        ),
        # Depthwise transpose: 16 inputs x 8 taps, dilation leaves holes in output.
        (
            nn.ConvTranspose3d(2, 2, 2, groups=2, stride=2, dilation=2, bias=False),
            (1, 2, 2, 2, 2),
            (1, 2, 5, 5, 5),
            (256, 256, 128),
        ),
    ],
)
def test_convolution_arithmetic(module, shape, output_shape, expected):
    inputs = torch.ones(shape)
    module.eval()
    with torch.no_grad():
        output = module(inputs)
        count = _complete_count(lambda: module(inputs))
    assert output.shape == output_shape
    assert (
        module_flops(module, (inputs,), output),
        count,
        module_macs(module, inputs, output),
    ) == expected


@pytest.mark.parametrize("bias", [False, True])
def test_linear_module_and_functional_arithmetic(bias):
    inputs = torch.ones(2, 3, 4)
    module = nn.Linear(4, 5, bias=bias)
    # Six vectors, five outputs: 30 dots with four multiplies and three adds.
    # Module: 210 plus 30 biases. Native: 120 MACs x 2, fused bias excluded.
    output = module(inputs)
    assert module_flops(module, (inputs,), output) == (240 if bias else 210)
    assert _complete_count(lambda: F.linear(inputs, module.weight, module.bias)) == 240


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("tracking", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_batch_norm_statistics_arithmetic(training, tracking, affine):
    inputs = torch.ones(2, 3, 4)
    module = nn.BatchNorm1d(3, affine=affine, track_running_stats=tracking).train(training)
    # Each channel has 8 samples: mean (7 adds + divide) = 8; variance
    # (8 subtracts + 8 squares + 7 adds + divide) = 24; total statistics 96.
    # Normalize 24 values: 48; eps/sqrt per channel: 6; affine: 48.
    # Running variance unbias + two weighted averages: 8 ops/channel = 24.
    expected = 54 + (48 if affine else 0) + (96 if training or not tracking else 0)
    if training and tracking:
        expected += 24
    output = module(inputs)
    assert module_flops(module, (inputs,), output) == expected
    assert _complete_count(lambda: module(inputs)) == expected


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        # Four rows of six: stats 4 x (6+18), eps/sqrt 8, normalize 48, affine 48.
        (nn.LayerNorm((2, 3)), 200),
        (nn.LayerNorm((2, 3), elementwise_affine=False), 152),
        # Six groups of four: stats 6 x (4+12), eps/sqrt 12, normalize/affine 96.
        (nn.GroupNorm(3, 6), 204),
        (nn.GroupNorm(3, 6, affine=False), 156),
    ],
)
def test_normalization_arithmetic(module, expected):
    inputs = torch.ones(2, 6, 2) if isinstance(module, nn.GroupNorm) else torch.ones(2, 2, 2, 3)
    output = module(inputs)
    assert module_flops(module, (inputs,), output) == expected
    assert _complete_count(lambda: module(inputs)) == expected


@pytest.mark.parametrize(
    ("training", "probability", "expected"), [(False, 0.5, 0), (True, 0, 0), (True, 0.5, 12), (True, 1, 6)]
)
def test_dropout_execution_state(training, probability, expected):
    module = nn.Dropout(probability).train(training)
    inputs = torch.ones(2, 3)
    assert module_flops(module, (inputs,), module(inputs)) == expected


def test_functional_residual_softmax_and_reductions():
    inputs = torch.ones(2, 3)
    other = torch.ones(3)

    def workload():
        # Broadcast add 6; scaled add 12; relu 6; multiply 6; softmax
        # two rows: each max2 + subtract3 + exp3 + sum2 + divide3 = 13.
        values = F.relu(torch.add(inputs + other, inputs, alpha=2)) * 2
        probabilities = F.softmax(values, dim=-1)
        return probabilities.sum() + probabilities.mean()

    # sum5 + mean(5+1) + scalar add1 = 12; total = 68.
    assert _complete_count(workload) == 68


@pytest.mark.parametrize("source_length", [3, 5])
@pytest.mark.parametrize("mask_kind", [None, "float", "bool", "causal"])
def test_functional_cpu_attention_arithmetic(source_length, mask_kind):
    q = torch.ones(2, 2, 3, 2)
    k = torch.ones(2, 2, source_length, 2)
    mask = torch.zeros(3, source_length, dtype=torch.bool if mask_kind == "bool" else torch.float32)
    # Four heads/batches, three query rows. For S=3: 36 scores and 24
    # outputs, each with 3 value terms. Products: 72+72 = 144 => 288 ops.
    # Scale36; softmax 12 x (max2+sub3+exp3+sum2+div3) =156 =>480.
    # S=5: products120+120 =240 =>480; scale60; softmax276 =>816.
    expected = 480 if source_length == 3 else 816
    if mask_kind is not None:
        expected += 36 if source_length == 3 else 60
    if mask_kind == "bool":
        # Boolean-to-additive mask conversion selects once per stored mask entry.
        expected += 3 * source_length
    report = measure_flops(
        lambda: F.scaled_dot_product_attention(
            q, k, k, attn_mask=mask if mask_kind in {"float", "bool"} else None, is_causal=mask_kind == "causal"
        )
    )
    if "aten._scaled_dot_product_flash_attention_for_cpu" in report["by_operator"]:
        assert report["total"]["status"] == "complete"
        assert report["total"]["value"] == expected
    elif "aten._scaled_dot_product_flash_attention" in report["by_operator"]:
        # PyTorch 2.1 uses its core-only native formula for CPU too.
        assert report["total"]["status"] == "partial"
        assert report["total"]["value"] is None
        assert report["by_operator"]["aten._scaled_dot_product_flash_attention"] == (288 if source_length == 3 else 480)
    else:
        # Older masked attention decomposes instead; the separate graph test
        # validates its arithmetic independently of its choice of scaling path.
        assert report["total"]["known_value"] >= (288 if source_length == 3 else 480)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("need_weights", [False, True])
@pytest.mark.parametrize("separate_dimensions", [False, True])
def test_cross_attention_projection_arithmetic(batch_first, need_weights, separate_dimensions):
    module = nn.MultiheadAttention(
        4,
        2,
        batch_first=batch_first,
        dropout=0,
        kdim=6 if separate_dimensions else 4,
        vdim=2 if separate_dimensions else 4,
    )
    shape = lambda length, width: (2, length, width) if batch_first else (length, 2, width)
    q = torch.ones(shape(3, 4))
    k = torch.ones(shape(5, 6 if separate_dimensions else 4))
    v = torch.ones(shape(5, 2 if separate_dimensions else 4))
    # Projection dots: Q 24 x 8 =192; K+V 40 x 8 each =640,
    # or K 40 x 12 + V 40 x 4 =640. Output 24 x 8 =192.
    # Q scale24; scores60 x (2 multiplies + 1 add)=180;
    # softmax 12 rows x23=276; weighted values24 x9=216. Total1720.
    output = module(q, k, v, need_weights=need_weights)
    assert module_flops(module, (q, k, v), output) == 1720 + (60 if need_weights else 0)
    # Native projection MACs: Q96, K/V320, output96 =>1024 FLOPs.
    # Batch-first inputs transpose into noncontiguous projections, adding
    # 24+40+40 explicit biases. Sequence-first projections fuse/omit bias.
    # With weights: scale24 + products480 + softmax276 + average60 =840.
    # Without weights the CPU fused ancillary count is816. PyTorch 2.1's
    # native fused count includes only the480 matrix FLOPs and stays partial.
    report = measure_flops(lambda: module(q, k, v, need_weights=need_weights))
    expected = 1024 + (104 if batch_first else 0)
    if need_weights:
        expected += 840
        assert report["total"]["status"] == "complete"
    elif "aten._scaled_dot_product_flash_attention_for_cpu" in report["by_operator"]:
        expected += 816
        assert report["total"]["status"] == "complete"
    else:
        expected += 480
        assert report["total"]["status"] == "partial"
        assert report["total"]["value"] is None
        assert report["diagnostics"]
    assert report["total"]["known_value"] == expected


def test_explicit_attention_graph_derivation():
    q, k, v = torch.ones(1, 2, 3, 2), torch.ones(1, 2, 5, 2), torch.ones(1, 2, 5, 3)
    mask = torch.zeros(3, 5)
    # QK: 30 dots x 2 MACs x2 =120; scale30, mask30, softmax6x23=138;
    # AV: 18 dots x 5 MACs x2 =180. Different value width, total498.
    assert _complete_count(lambda: F.softmax(q @ k.transpose(-1, -2) * 0.5 + mask, -1) @ v) == 498


def test_math_attention_and_safe_softmax_arithmetic():
    q, k, v = torch.ones(1, 2, 3, 2), torch.ones(1, 2, 5, 2), torch.ones(1, 2, 5, 3)
    report = measure_flops(lambda: F.scaled_dot_product_attention(q, k, v))
    # The math path scales 12 Q and 20 K values, counts 300 matrix FLOPs,
    # and normalizes six rows of five (138 FLOPs). Safe softmax additionally
    # compares 30 scores and selects 30 probabilities: 530 overall.
    expected = 530 if "aten._safe_softmax" in report["by_operator"] else 470
    assert report["total"]["status"] == "complete"
    assert report["total"]["value"] == expected


def test_scalar_softmax_and_integer_control_arithmetic():
    scalar = torch.ones(())
    integers = torch.ones(3, dtype=torch.long)
    assert _complete_count(lambda: scalar.softmax(0)) == 3
    assert _complete_count(lambda: integers + 1) == 0
    assert _complete_count(lambda: integers + 0.5) == 3
    assert _complete_count(lambda: integers / 2) == 3
    assert _complete_count(lambda: torch.exp(integers)) == 3
    assert _complete_count(lambda: torch.sqrt(integers)) == 3
    assert _complete_count(lambda: torch.rsqrt(integers)) == 3


def test_empty_normalized_rows_remain_visibly_unsupported():
    inputs = torch.ones(2, 0)
    module = nn.LayerNorm(0)
    report = crawl_module(module, args=(inputs,))
    assert report["totals"]["module_flops"]["status"] == "unavailable"
    assert report["totals"]["operator_flops"]["status"] == "partial"
    assert report["totals"]["operator_flops"]["known_value"] == 0
    assert any(item["code"] == "unsupported_operator_formula" for item in report["diagnostics"])


def test_normalization_backward_remains_partial():
    inputs = torch.ones(2, 3, requires_grad=True)
    report = measure_flops(lambda: F.layer_norm(inputs, (3,)).sum().backward())
    assert report["total"]["status"] == "partial"
    assert report["total"]["value"] is None
    # Forward: stats24 + normalize12 + eps/sqrt4 =40; loss reduction5.
    assert report["total"]["known_value"] == 45
    assert any(item.get("operator") == "aten.native_layer_norm_backward" for item in report["diagnostics"])


@pytest.mark.parametrize(("overload", "expected"), [("no_stats", 150), ("default", 174), ("eval", 54)])
def test_native_batch_norm_overload_arithmetic(overload, expected):
    inputs = torch.ones(2, 3, 4)
    mean, variance = torch.zeros(3), torch.ones(3)

    def workload():
        if overload == "no_stats":
            return torch.ops.aten._native_batch_norm_legit.no_stats(inputs, None, None, True, 0.1, 1e-5)
        if overload == "eval":
            return torch.ops.aten._native_batch_norm_legit_no_training(inputs, None, None, mean, variance, 0.1, 1e-5)
        return torch.ops.aten._native_batch_norm_legit(inputs, None, None, mean, variance, True, 0.1, 1e-5)

    # Normalize48 + eps/sqrt6. Batch statistics add96; tracked updates add24.
    assert _complete_count(workload) == expected


def test_empty_attention_source_is_visibly_incomplete():
    module = nn.MultiheadAttention(4, 2, batch_first=True)
    query, memory = torch.ones(2, 3, 4), torch.ones(2, 0, 4)
    report = crawl_module(module, args=(query, memory, memory), kwargs={"need_weights": False})
    assert report["totals"]["module_flops"]["status"] == "unavailable"
    assert any("nonempty query and source" in item["message"] for item in report["diagnostics"])


def test_forward_and_backward_matrix_work():
    left = torch.ones(2, 3, requires_grad=True)
    right = torch.ones(3, 4, requires_grad=True)
    forward = measure_flops(lambda: left @ right)
    # Forward: 2x3x4 =24 MACs. dLeft and dRight each also require24
    # MACs. Scalar loss sum over8 outputs adds7. Total144+7=151.
    training = measure_flops(lambda: (left @ right).sum().backward())
    assert forward["total"]["value"] == 48
    assert training["total"]["status"] == "complete"
    assert training["total"]["value"] == 151


def test_scoped_rules_preserve_native_registry_and_caller_override():
    before = FlopCounterMode(display=False)
    registry = dict(getattr(before, "flop_registry", getattr(before, "flop_mapping", {})))
    inputs = torch.ones(2, 3)
    custom = {torch.ops.aten.add: lambda *_args, **_kwargs: 99}
    assert measure_flops(lambda: inputs + inputs, custom_mapping=custom)["total"]["value"] == 99
    assert measure_flops(lambda: inputs + inputs)["total"]["value"] == 6
    after = FlopCounterMode(display=False)
    assert dict(getattr(after, "flop_registry", getattr(after, "flop_mapping", {}))) == registry
    assert len(custom) == 1


def test_mixed_supported_and_unsupported_division_remains_partial():
    inputs = torch.ones(2, 3)
    report = measure_flops(lambda: (inputs / 2, torch.div(inputs, 2, rounding_mode="floor")))
    assert report["total"]["status"] == "partial"
    assert report["total"]["value"] is None
    assert report["total"]["known_value"] == 6
    assert any(item["code"] == "unsupported_operator_formula" for item in report["diagnostics"])


def test_unknown_functional_work_and_strict_mode():
    class Unsupported(nn.Module):
        def forward(self, inputs):
            return torch.sin(inputs @ inputs)

    model = Unsupported()
    report = crawl_module(model, args=(torch.ones(2, 2),))
    assert report["totals"]["operator_flops"]["status"] == "partial"
    assert report["totals"]["operator_flops"]["known_value"] == 16
    assert any(item.get("operator") == "aten.sin" for item in report["diagnostics"])
    with pytest.raises(IncompleteAnalysisError):
        crawl_module(model, args=(torch.ones(2, 2),), strict=True)


def test_complex_arithmetic_remains_visibly_incomplete():
    left, right = torch.ones(1, 2, dtype=torch.complex64), torch.ones(2, 1, dtype=torch.complex64)
    report = measure_flops(lambda: (left @ right, left + left))
    assert report["total"]["status"] == "partial"
    assert report["total"]["value"] is None
    # Native keeps its shape count: two complex MACs x2 =4, which is a lower
    # bound on real arithmetic. The supplemental add is unsupported, contributing0.
    assert report["total"]["known_value"] == 4
    assert {item["code"] for item in report["diagnostics"]} == {
        "incomplete_operator_formula",
        "unsupported_operator_formula",
    }
    module = nn.Linear(2, 1, dtype=torch.complex64)
    assert crawl_module(module, args=(left,))["totals"]["module_flops"]["status"] == "unavailable"


def test_native_fused_attention_core_is_visibly_incomplete():
    # Exercise the upstream GPU-shaped operator on meta tensors; this is formula
    # validation only and makes no claim about CUDA execution.
    q = torch.empty(1, 2, 3, 4, device="meta")
    k = torch.empty(1, 2, 5, 4, device="meta")
    report = measure_flops(lambda: torch.ops.aten._scaled_dot_product_flash_attention(q, k, k))
    # Score products:30x4, value products:24x5 =>240 MACs /480 FLOPs.
    assert report["total"]["status"] == "partial"
    assert report["total"]["value"] is None
    assert report["total"]["known_value"] == 480
    assert any(item["code"] == "incomplete_operator_formula" for item in report["diagnostics"])


@pytest.mark.parametrize(
    ("status", "value", "expected"),
    [
        ("complete", 0, "0 [complete]"),
        ("partial", 0, ">=0 [partial]"),
        ("unavailable", None, "n/a [unavailable]"),
    ],
)
def test_benchmark_preserves_metric_states(status, value, expected):
    metric = metric_result(status=status, value=value, known_value=value, unit="FLOPs", scope="forward", method="test")
    assert metric_cell(metric) == expected
