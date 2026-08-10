from unittest.mock import Mock

import pytest
import torch

from torchscan import process


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("", 0.0),
        ("GPU:0", 0.0),
        ("GPU:0\nprocess", 0.0),
        ("GPU:0\nprocess 123 uses 456.000 MB GPU memory", 456.0),
    ],
)
def test_get_process_gpu_ram_fallback(monkeypatch, output, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "list_gpu_processes", lambda: output)
    monkeypatch.setattr("torchscan.process.memory.subprocess.run", Mock(side_effect=FileNotFoundError("nvidia-smi")))

    with pytest.warns(UserWarning, match="Parsing NVIDIA-SMI failed"):
        assert process.get_process_gpu_ram(123) == expected


def test_get_process_gpu_ram_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.warns(UserWarning, match=r"CUDA is unavailable to PyTorch\."):
        assert process.get_process_gpu_ram(123) == 0
