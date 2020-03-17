import unittest
import os
from torchscan import process


class Tester(unittest.TestCase):

    def test_get_process_gpu_ram(self):
        self.assertEqual(process.get_process_gpu_ram(os.getpid()), 0)


if __name__ == '__main__':
    unittest.main()
