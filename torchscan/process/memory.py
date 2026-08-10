# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] S404
import warnings

import torch

__all__ = ["get_process_gpu_ram"]


def get_process_gpu_ram(pid: int) -> float:
    """Gets the amount of RAM used by a given process on GPU devices

    Args:
        pid: process ID
    Returns:
        RAM usage in Megabytes
    """
    # PyTorch is not responsible for GPU usage
    if not torch.cuda.is_available():
        warnings.warn("CUDA is unavailable to PyTorch.", stacklevel=1)
        return 0.0

    # Query the running processes on GPUs
    try:
        res = subprocess.run(["nvidia-smi", "-q", "-d", "PIDS"], capture_output=True).stdout.decode()
        # Try to locate the process
        pids = re.findall(r"Process ID\s+:\s([^\D]*)", res)
        for idx, pid_ in enumerate(pids):
            if int(pid_) == pid:
                return float(re.findall(r"Used GPU Memory\s+:\s([^\D]*)", res)[idx])

        # Query total memory used by nvidia
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"], capture_output=True
        ).stdout.decode()
        return float(res.split("\n")[1].split()[0])
    except FileNotFoundError as e:
        warnings.warn(f"raised: {e}. Parsing NVIDIA-SMI failed.", stacklevel=1)

    # Default to overall RAM usage for this process on the GPU
    ram_str = torch.cuda.list_gpu_processes().splitlines()
    # Take the first process running on the GPU
    if len(ram_str) > 1:
        process_info = ram_str[1].split()
        if len(process_info) > 3 and process_info[0] == "process":
            return float(process_info[3])

    # Otherwise assume the process is running exclusively on CPU
    return 0.0
