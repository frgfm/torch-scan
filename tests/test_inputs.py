import inspect

import pytest
import torch
from torch import nn

from torchscan import crawl_module


def test_real_args_and_kwargs_are_forwarded_unchanged():
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, input_t, scale, optional, *, payload, attention_mask):
            self.received = (input_t, scale, optional, payload, attention_mask)
            return input_t.masked_fill(~attention_mask, 0) * scale

    model = Recorder()
    input_t = torch.randn(2, 4)
    mask = torch.tensor([[True, False, True, True], [False, True, True, False]])
    payload = {"nested": [input_t, None, {"enabled": True}]}
    scale = 2.0

    crawl_module(
        model,
        args=(input_t, scale, None),
        kwargs={"payload": payload, "attention_mask": mask},
    )

    assert model.received is not None
    assert model.received[0] is input_t
    assert model.received[1] is scale
    assert model.received[2] is None
    assert model.received[3] is payload
    assert model.received[4] is mask


def test_kwargs_only_is_a_real_input_source():
    class KeywordOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, *, input_t):
            self.received = input_t
            return input_t

    model = KeywordOnly()
    input_t = torch.randn(2, 4)

    crawl_module(model, kwargs={"input_t": input_t})

    assert model.received is input_t


def test_exactly_one_generated_or_real_input_source_is_required():
    model = nn.Linear(4, 2)
    input_t = torch.randn(2, 4)

    with pytest.raises(ValueError, match="Exactly one"):
        crawl_module(model)
    with pytest.raises(ValueError, match="Exactly one"):
        crawl_module(model, (4,), args=(input_t,))
    with pytest.raises(ValueError, match="Exactly one"):
        crawl_module(model, (4,), kwargs={"input": input_t})


def test_input_data_is_not_part_of_the_public_contract():
    assert "input_data" not in inspect.signature(crawl_module).parameters


def test_generated_input_shape_and_dtype_lengths_must_match():
    class Pair(nn.Module):
        def forward(self, left, right):
            return left + right

    with pytest.raises(ValueError, match=r"length|same number"):
        crawl_module(Pair(), [(4,), (4,)], [torch.float32])


def test_generated_input_infers_parameter_device_and_dtype():
    class RecordingLinear(nn.Linear):
        def __init__(self):
            super().__init__(4, 2, dtype=torch.float64)
            self.received = None

        def forward(self, input_t):
            self.received = input_t
            return super().forward(input_t)

    model = RecordingLinear()

    crawl_module(model, (4,))

    assert model.received is not None
    assert model.received.device == model.weight.device
    assert model.received.dtype == model.weight.dtype


def test_generated_input_infers_buffer_device_and_dtype():
    class BufferOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("template", torch.ones(1, dtype=torch.float64))
            self.received = None

        def forward(self, input_t):
            self.received = input_t
            return input_t

    model = BufferOnly()

    crawl_module(model, (4,))

    assert model.received is not None
    assert model.received.device == model.template.device
    assert model.received.dtype == model.template.dtype


def test_generated_input_supports_parameterless_modules_and_explicit_device():
    class Parameterless(nn.Module):
        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, input_t):
            self.received = input_t
            return input_t

    model = Parameterless()

    report = crawl_module(model, (4,), dtype=torch.float64, device="cpu")

    assert report["layers"]
    assert model.received is not None
    assert model.received.device.type == "cpu"
    assert model.received.dtype == torch.float64


def test_training_flags_are_restored_and_batchnorm_state_is_unchanged():
    class MixedMode(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_norm = nn.BatchNorm1d(4)
            self.dropout = nn.Dropout()

        def forward(self, input_t):
            return self.dropout(self.batch_norm(input_t))

    model = MixedMode().train()
    model.dropout.eval()
    training_flags = {name: module.training for name, module in model.named_modules()}
    running_mean = model.batch_norm.running_mean.clone()
    running_var = model.batch_norm.running_var.clone()
    batches = model.batch_norm.num_batches_tracked.clone()

    crawl_module(model, args=(torch.randn(8, 4),))

    assert {name: module.training for name, module in model.named_modules()} == training_flags
    assert torch.equal(model.batch_norm.running_mean, running_mean)
    assert torch.equal(model.batch_norm.running_var, running_var)
    assert torch.equal(model.batch_norm.num_batches_tracked, batches)


def test_hooks_and_training_flags_are_restored_after_forward_error():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, input_t):
            self.linear(input_t)
            raise RuntimeError("expected failure")

    model = Broken().train()
    model.linear.eval()
    modules = list(model.modules())
    hook_counts = [(len(module._forward_pre_hooks), len(module._forward_hooks)) for module in modules]
    training_flags = [module.training for module in modules]

    with pytest.raises(RuntimeError, match="expected failure"):
        crawl_module(model, args=(torch.randn(2, 4),))

    assert [(len(module._forward_pre_hooks), len(module._forward_hooks)) for module in modules] == hook_counts
    assert [module.training for module in modules] == training_flags
