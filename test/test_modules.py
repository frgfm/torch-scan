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
        input_t = torch.rand((1, 3, 32, 32))
        self.assertEqual(modules.module_flops(nn.Linear(8, 4), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         4 * (2 * 8 - 1) + 4)
        self.assertEqual(modules.module_flops(nn.Linear(8, 4, bias=False), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         4 * (2 * 8 - 1))
        self.assertEqual(modules.module_flops(nn.ReLU(), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         8)
        mod = nn.Conv2d(3, 8, 3)
        self.assertEqual(modules.module_flops(mod, input_t, mod(input_t)), 388800)

    def test_module_macs(self):

        # Check for unknown module that it returns 0 and throws a warning
        self.assertEqual(modules.module_macs(MyModule(), None, None), 0)
        self.assertWarns(UserWarning, modules.module_macs, MyModule(), None, None)

        # Common unit tests
        input_t = torch.rand((1, 3, 32, 32))
        self.assertEqual(modules.module_macs(nn.Linear(8, 4), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         8 * 4)
        self.assertEqual(modules.module_macs(nn.ReLU(), None, None), 0)
        mod = nn.Conv2d(3, 8, 3)
        self.assertEqual(modules.module_macs(mod, input_t, mod(input_t)), 194400)

    def test_module_dmas(self):

        # Check for unknown module that it returns 0 and throws a warning
        self.assertEqual(modules.module_dmas(MyModule(), None, None), 0)
        self.assertWarns(UserWarning, modules.module_dmas, MyModule(), None, None)

        # Common unit tests
        input_t = torch.rand((1, 3, 32, 32))
        self.assertEqual(modules.module_dmas(nn.Linear(8, 4), torch.zeros((1, 8)), torch.zeros((1, 4))),
                         4 * (8 + 1) + 8 + 4)
        self.assertEqual(modules.module_dmas(nn.ReLU(), torch.zeros((1, 8)), torch.zeros((1, 8))), 8 * 2)
        self.assertEqual(modules.module_dmas(nn.ReLU(inplace=True), torch.zeros((1, 8)), None), 8)
        mod = nn.Conv2d(3, 8, 3)
        self.assertEqual(modules.module_dmas(mod, input_t, mod(input_t)), 201824)


if __name__ == '__main__':
    unittest.main()
