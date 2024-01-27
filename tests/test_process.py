import os

import torch

from torchscan import process


def test_get_process_gpu_ram():
    if torch.cuda.is_initialized:
        assert process.get_process_gpu_ram(os.getpid()) >= 0
    else:
        assert process.get_process_gpu_ram(os.getpid()) == 0
