import unittest
import torch
from torch import nn
from torchscan import modules


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()


class Tester(unittest.TestCase):

    def test_module_flops(self):

        # Check for unknown module that it returns 0 and throws a warning
        self.assertEqual(modules.module_flops(MyModule(), None, None), 0)
        self.assertWarns(UserWarning, modules.module_flops, MyModule(), None, None)

        # Common unit tests
        self.assertEqual(modules.module_flops(nn.Linear(8, 4), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         4 * (2 * 8 - 1) + 4)
        self.assertEqual(modules.module_flops(nn.Linear(8, 4, bias=False), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         4 * (2 * 8 - 1))
        # Activations
        self.assertEqual(modules.module_flops(nn.Identity(), torch.zeros((1, 8)), torch.zeros((1, 8))), 0)
        self.assertEqual(modules.module_flops(nn.ReLU(), torch.zeros((1, 8)), torch.zeros((1, 8))), 8)
        self.assertEqual(modules.module_flops(nn.ELU(), torch.zeros((1, 8)), torch.zeros((1, 8))), 48)
        self.assertEqual(modules.module_flops(nn.LeakyReLU(), torch.zeros((1, 8)), torch.zeros((1, 8))), 32)
        self.assertEqual(modules.module_flops(nn.ReLU6(), torch.zeros((1, 8)), torch.zeros((1, 8))), 16)
        self.assertEqual(modules.module_flops(nn.Tanh(), torch.zeros((1, 8)), torch.zeros((1, 8))), 48)
        self.assertEqual(modules.module_flops(nn.Sigmoid(), torch.zeros((1, 8)), torch.zeros((1, 8))), 32)

        # BN
        self.assertEqual(modules.module_flops(nn.BatchNorm1d(8), torch.zeros((1, 8, 4)), torch.zeros((1, 8, 4))),
                         144 + 32 + 32 * 3 + 48)

        # Pooling
        self.assertEqual(modules.module_flops(nn.MaxPool2d((2, 2)),
                                              torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         3 * 32)
        self.assertEqual(modules.module_flops(nn.AvgPool2d((2, 2)),
                                              torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         5 * 32)
        self.assertEqual(modules.module_flops(nn.AdaptiveMaxPool2d((2, 2)),
                                              torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         3 * 32)
        self.assertEqual(modules.module_flops(nn.AdaptiveAvgPool2d((2, 2)),
                                              torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         5 * 32)

        # Dropout
        self.assertEqual(modules.module_flops(nn.Dropout(), torch.zeros((1, 8)), torch.zeros((1, 8))), 8)
        self.assertEqual(modules.module_flops(nn.Dropout(p=0), torch.zeros((1, 8)), torch.zeros((1, 8))), 0)

        # Conv
        input_t = torch.rand((1, 3, 32, 32))
        mod = nn.Conv2d(3, 8, 3)
        self.assertEqual(modules.module_flops(mod, input_t, mod(input_t)), 388800)
        # ConvTranspose
        mod = nn.ConvTranspose2d(3, 8, 3)
        self.assertEqual(modules.module_flops(mod, input_t, mod(input_t)), 499408)

    def test_module_macs(self):

        # Check for unknown module that it returns 0 and throws a warning
        self.assertEqual(modules.module_macs(MyModule(), None, None), 0)
        self.assertWarns(UserWarning, modules.module_macs, MyModule(), None, None)

        # Linear
        self.assertEqual(modules.module_macs(nn.Linear(8, 4), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         8 * 4)
        # Activations
        self.assertEqual(modules.module_macs(nn.ReLU(), None, None), 0)
        # Conv
        input_t = torch.rand((1, 3, 32, 32))
        mod = nn.Conv2d(3, 8, 3)
        self.assertEqual(modules.module_macs(mod, input_t, mod(input_t)), 194400)
        # ConvTranspose
        mod = nn.ConvTranspose2d(3, 8, 3)
        self.assertEqual(modules.module_macs(mod, input_t, mod(input_t)), 249704)
        # BN
        self.assertEqual(modules.module_macs(nn.BatchNorm1d(8), torch.zeros((1, 8, 4)), torch.zeros((1, 8, 4))),
                         64 + 24 + 56 + 32)

        # Pooling
        self.assertEqual(modules.module_macs(nn.MaxPool2d((2, 2)),
                                             torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         3 * 32)
        self.assertEqual(modules.module_macs(nn.AvgPool2d((2, 2)),
                                             torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         5 * 32)
        self.assertEqual(modules.module_macs(nn.AdaptiveMaxPool2d((2, 2)),
                                             torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         3 * 32)
        self.assertEqual(modules.module_macs(nn.AdaptiveAvgPool2d((2, 2)),
                                             torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         5 * 32)

        # Dropout
        self.assertEqual(modules.module_macs(nn.Dropout(), torch.zeros((1, 8)), torch.zeros((1, 8))), 0)

    def test_module_dmas(self):

        # Check for unknown module that it returns 0 and throws a warning
        self.assertEqual(modules.module_dmas(MyModule(), None, None), 0)
        self.assertWarns(UserWarning, modules.module_dmas, MyModule(), None, None)

        # Common unit tests
        # Linear
        self.assertEqual(modules.module_dmas(nn.Linear(8, 4), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         4 * (8 + 1) + 8 + 4)
        # Activation
        self.assertEqual(modules.module_dmas(nn.Identity(), torch.zeros((1, 8)), torch.zeros((1, 8))), 8)
        self.assertEqual(modules.module_dmas(nn.ReLU(), torch.zeros((1, 8)), torch.zeros((1, 8))), 8 * 2)
        self.assertEqual(modules.module_dmas(nn.ReLU(inplace=True), torch.zeros((1, 8)), None), 8)
        self.assertEqual(modules.module_dmas(nn.ELU(), torch.zeros((1, 8)), torch.zeros((1, 8))), 17)
        self.assertEqual(modules.module_dmas(nn.Sigmoid(), torch.zeros((1, 8)), torch.zeros((1, 8))), 16)
        self.assertEqual(modules.module_dmas(nn.Tanh(), torch.zeros((1, 8)), torch.zeros((1, 8))), 24)
        # Conv
        input_t = torch.rand((1, 3, 32, 32))
        mod = nn.Conv2d(3, 8, 3)
        self.assertEqual(modules.module_dmas(mod, input_t, mod(input_t)), 201824)
        # ConvTranspose
        mod = nn.ConvTranspose2d(3, 8, 3)
        self.assertEqual(modules.module_dmas(mod, input_t, mod(input_t)), 259178)
        # BN
        self.assertEqual(modules.module_dmas(nn.BatchNorm1d(8), torch.zeros((1, 8, 4)), torch.zeros((1, 8, 4))),
                         32 + 17 + 1 + 16 + 17 + 32)

        # Pooling
        self.assertEqual(modules.module_dmas(nn.MaxPool2d((2, 2)),
                                             torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         4 * 32 + 32)
        self.assertEqual(modules.module_dmas(nn.AdaptiveMaxPool2d((2, 2)),
                                             torch.zeros((1, 8, 4, 4)), torch.zeros((1, 8, 2, 2))),
                         4 * 32 + 32)

        # Dropout
        self.assertEqual(modules.module_dmas(nn.Dropout(), torch.zeros((1, 8)), torch.zeros((1, 8))), 17)


if __name__ == '__main__':
    unittest.main()
