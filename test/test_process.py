import unittest
import os
from torchscan import process
import torch


class Tester(unittest.TestCase):

    def test_get_process_gpu_ram(self):

        if torch.cuda.is_initialized:
            self.assertGreaterEqual(process.get_process_gpu_ram(os.getpid()), 0)
        else:
            self.assertEqual(process.get_process_gpu_ram(os.getpid()), 0)


if __name__ == '__main__':
    unittest.main()
