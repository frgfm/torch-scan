#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Process memory
"""

import re
import subprocess
import warnings

__all__ = ['get_process_gpu_ram']


def get_process_gpu_ram(pid):
    """Gets the amount of RAM used by a given process on GPU devices

    Args:
        pid (int): process ID
    Returns:
        float: RAM usage in Megabytes
    """

    # Query the running processes on GPUs
    try:
        res = subprocess.run(["nvidia-smi", "-q", "-d", "PIDS"], capture_output=True).stdout.decode()
        # Try to locate the process
        pids = re.findall("Process ID\s+:\s([^\D]*)", res)
        for idx, _pid in enumerate(pids):
            if int(_pid) == pid:
                return float(re.findall("Used GPU Memory\s+:\s([^\D]*)", res)[idx])
    except Exception as e:
        warnings.warn(f"raised: {e}. Assuming no GPU is available.")

    # Otherwise assume the process is running exclusively on CPU
    return 0.
