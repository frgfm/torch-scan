# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import re
import subprocess
import warnings

__all__ = ["get_process_gpu_ram"]


def get_process_gpu_ram(pid: int) -> float:
    """Gets the amount of RAM used by a given process on GPU devices

    Args:
        pid: process ID
    Returns:
        RAM usage in Megabytes
    """

    # Query the running processes on GPUs
    try:
        res = subprocess.run(["nvidia-smi", "-q", "-d", "PIDS"], capture_output=True).stdout.decode()
        # Try to locate the process
        pids = re.findall(r"Process ID\s+:\s([^\D]*)", res)
        for idx, _pid in enumerate(pids):
            if int(_pid) == pid:
                return float(re.findall(r"Used GPU Memory\s+:\s([^\D]*)", res)[idx])
    except Exception as e:
        warnings.warn(f"raised: {e}. Assuming no GPU is available.")

    # Otherwise assume the process is running exclusively on CPU
    return 0.0
