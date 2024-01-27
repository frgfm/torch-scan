import pytest
import torch
from torch import nn

from torchscan import modules


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()


def test_module_flops_warning():
    with pytest.warns(UserWarning):
        modules.module_flops(MyModule(), None, None)


@pytest.mark.parametrize(
    ("mod", "input_shape", "output_shape", "expected_val"),
    [
        # Check for unknown module that it returns 0 and throws a warning
        (MyModule(), (1,), (1,), 0),
        # Fully-connected
        (nn.Linear(8, 4), (1, 8), (1, 4), 4 * (2 * 8 - 1) + 4),
        (nn.Linear(8, 4, bias=False), (1, 8), (1, 4), 4 * (2 * 8 - 1)),
        (nn.Linear(8, 4), (1, 2, 8), (1, 2, 4), 2 * (4 * (2 * 8 - 1) + 4)),
        # Activations
        (nn.Identity(), (1, 8), (1, 8), 0),
        (nn.Flatten(), (1, 8), (1, 8), 0),
        (nn.ReLU(), (1, 8), (1, 8), 8),
        (nn.ELU(), (1, 8), (1, 8), 48),
        (nn.LeakyReLU(), (1, 8), (1, 8), 32),
        (nn.ReLU6(), (1, 8), (1, 8), 16),
        (nn.Tanh(), (1, 8), (1, 8), 48),
        (nn.Sigmoid(), (1, 8), (1, 8), 32),
        # BN
        (nn.BatchNorm1d(8), (1, 8, 4), (1, 8, 4), 144 + 32 + 32 * 3 + 48),
        # Pooling
        (nn.MaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AvgPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        (nn.AdaptiveMaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AdaptiveMaxPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AdaptiveAvgPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        (nn.AdaptiveAvgPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        # Dropout
        (nn.Dropout(), (1, 8), (1, 8), 8),
        (nn.Dropout(p=0), (1, 8), (1, 8), 0),
        # Conv
        (nn.Conv2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 30, 30), 388800),
        (nn.ConvTranspose2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 34, 34), 499408),
    ],
)
def test_module_flops(mod, input_shape, output_shape, expected_val):
    assert modules.module_flops(mod, (torch.zeros(input_shape),), torch.zeros(output_shape)) == expected_val


def test_transformer_flops():
    mod = nn.Transformer(d_model=64, nhead=4, num_encoder_layers=3)
    src = torch.rand((10, 16, 64))
    tgt = torch.rand((20, 16, 64))
    assert modules.module_flops(mod, (src, tgt), mod(src, tgt)) == 774952841


def test_module_macs_warning():
    with pytest.warns(UserWarning):
        modules.module_macs(MyModule(), None, None)


@pytest.mark.parametrize(
    ("mod", "input_shape", "output_shape", "expected_val"),
    [
        # Check for unknown module that it returns 0 and throws a warning
        (MyModule(), (1,), (1,), 0),
        # Fully-connected
        (nn.Linear(8, 4), (1, 8), (1, 4), 8 * 4),
        (nn.Linear(8, 4), (1, 2, 8), (1, 2, 4), 8 * 4 * 2),
        # Activations
        (nn.ReLU(), (1, 8), (1, 8), 0),
        # BN
        (nn.BatchNorm1d(8), (1, 8, 4), (1, 8, 4), 64 + 24 + 56 + 32),
        # Pooling
        (nn.MaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AvgPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        (nn.AdaptiveMaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AdaptiveMaxPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 3 * 32),
        (nn.AdaptiveAvgPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        (nn.AdaptiveAvgPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 5 * 32),
        # Dropout
        (nn.Dropout(), (1, 8), (1, 8), 0),
        # Conv
        (nn.Conv2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 30, 30), 194400),
        (nn.ConvTranspose2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 34, 34), 249704),
    ],
)
def test_module_macs(mod, input_shape, output_shape, expected_val):
    assert modules.module_macs(mod, torch.zeros(input_shape), torch.zeros(output_shape)) == expected_val


def test_module_dmas_warning():
    with pytest.warns(UserWarning):
        modules.module_dmas(MyModule(), None, None)


@pytest.mark.parametrize(
    ("mod", "input_shape", "output_shape", "expected_val"),
    [
        # Check for unknown module that it returns 0 and throws a warning
        (MyModule(), (1,), (1,), 0),
        # Fully-connected
        (nn.Linear(8, 4), (1, 8), (1, 4), 4 * (8 + 1) + 8 + 4),
        (nn.Linear(8, 4), (1, 2, 8), (1, 2, 4), 4 * (8 + 1) + 2 * (8 + 4)),
        # Activations
        (nn.Identity(), (1, 8), (1, 8), 8),
        (nn.Flatten(), (1, 8), (1, 8), 16),
        (nn.ReLU(), (1, 8), (1, 8), 8 * 2),
        (nn.ReLU(inplace=True), (1, 8), (1, 8), 8),
        (nn.ELU(), (1, 8), (1, 8), 17),
        (nn.Tanh(), (1, 8), (1, 8), 24),
        (nn.Sigmoid(), (1, 8), (1, 8), 16),
        # BN
        (nn.BatchNorm1d(8), (1, 8, 4), (1, 8, 4), 32 + 17 + 16 + 1 + 17 + 32),
        # Pooling
        (nn.MaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32 + 32),
        (nn.MaxPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32 + 32),
        (nn.AdaptiveMaxPool2d((2, 2)), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32 + 32),
        (nn.AdaptiveMaxPool2d(2), (1, 8, 4, 4), (1, 8, 2, 2), 4 * 32 + 32),
        # Dropout
        (nn.Dropout(), (1, 8), (1, 8), 17),
        # Conv
        (nn.Conv2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 30, 30), 201824),
        (nn.ConvTranspose2d(3, 8, 3), (1, 3, 32, 32), (1, 8, 34, 34), 259178),
    ],
)
def test_module_dmas(mod, input_shape, output_shape, expected_val):
    assert modules.module_dmas(mod, torch.zeros(input_shape), torch.zeros(output_shape)) == expected_val


# @torch.no_grad()
# def test_module_rf(self):

#     # Check for unknown module that it returns 0 and throws a warning
#     self.assertEqual(modules.module_rf(MyModule(), None, None), (1, 1, 0))
#     self.assertWarns(UserWarning, modules.module_rf, MyModule(), None, None)

#     # Common unit tests
#     # Linear
#     self.assertEqual(modules.module_rf(nn.Linear(8, 4), torch.zeros((1, 8)), torch.zeros((1, 4))),
#                      (1, 1, 0))
#     # Activation
#     self.assertEqual(modules.module_rf(nn.Identity(), torch.zeros((1, 8)), torch.zeros((1, 8))), (1, 1, 0))
#     self.assertEqual(modules.module_rf(nn.Flatten(), torch.zeros((1, 8)), torch.zeros((1, 8))), (1, 1, 0))
#     self.assertEqual(modules.module_rf(nn.ReLU(), torch.zeros((1, 8)), torch.zeros((1, 8))), (1, 1, 0))
#     self.assertEqual(modules.module_rf(nn.ELU(), torch.zeros((1, 8)), torch.zeros((1, 8))), (1, 1, 0))
#     self.assertEqual(modules.module_rf(nn.Sigmoid(), torch.zeros((1, 8)), torch.zeros((1, 8))), (1, 1, 0))
#     self.assertEqual(modules.module_rf(nn.Tanh(), torch.zeros((1, 8)), torch.zeros((1, 8))), (1, 1, 0))
#     # Conv
#     input_t = torch.rand((1, 3, 32, 32))
#     mod = nn.Conv2d(3, 8, 3)
#     self.assertEqual(modules.module_rf(mod, input_t, mod(input_t)), (3, 1, 0))
#     # Check for dilation support
#     mod = nn.Conv2d(3, 8, 3, dilation=2)
#     self.assertEqual(modules.module_rf(mod, input_t, mod(input_t)), (5, 1, 0))
#     # ConvTranspose
#     mod = nn.ConvTranspose2d(3, 8, 3)
#     self.assertEqual(modules.module_rf(mod, input_t, mod(input_t)), (-3, 1, 0))
#     # BN
#     self.assertEqual(modules.module_rf(nn.BatchNorm1d(8), torch.zeros((1, 8, 4)), torch.zeros((1, 8, 4))),
#                      (1, 1, 0))

#     # Pooling
#     self.assertEqual(modules.module_rf(nn.MaxPool2d((2, 2)),
#                                        torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
#                      (2, 2, 0))
#     self.assertEqual(modules.module_rf(nn.AdaptiveMaxPool2d((2, 2)),
#                                        torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
#                      (2, 2, 0))

#     # Dropout
#     self.assertEqual(modules.module_rf(nn.Dropout(), torch.zeros((1, 8)), torch.zeros((1, 8))), (1, 1, 0))
