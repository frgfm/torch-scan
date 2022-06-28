# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import re
import subprocess
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
        warnings.warn("CUDA is unavailable to PyTorch.")
        return 0.0

    # Query the running processes on GPUs
    try:
        res = subprocess.run(["nvidia-smi", "-q", "-d", "PIDS"], capture_output=True).stdout.decode()
        # Try to locate the process
        pids = re.findall(r"Process ID\s+:\s([^\D]*)", res)
        for idx, _pid in enumerate(pids):
            if int(_pid) == pid:
                return float(re.findall(r"Used GPU Memory\s+:\s([^\D]*)", res)[idx])

        # Query total memory used by nvidia
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"], capture_output=True
        ).stdout.decode()
        return float(res.split("\n")[1].split()[0])
    except Exception as e:
        warnings.warn(f"raised: {e}. Parsing NVIDIA-SMI failed.")

    # Default to overall RAM usage for this process on the GPU
    ram_str = torch.cuda.list_gpu_processes().split("\n")
    # Take the first process running on the GPU
    if ram_str[1].startswith("process"):
        return float(ram_str[1].split()[3])

    # Otherwise assume the process is running exclusively on CPU
    return 0.0
