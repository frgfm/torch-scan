# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import unittest

import torch

from torchscan import process


class Tester(unittest.TestCase):
    def test_get_process_gpu_ram(self):

        if torch.cuda.is_initialized:
            self.assertGreaterEqual(process.get_process_gpu_ram(os.getpid()), 0)
        else:
            self.assertEqual(process.get_process_gpu_ram(os.getpid()), 0)


if __name__ == '__main__':
    unittest.main()
