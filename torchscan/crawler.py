#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Module crawler
"""

import os
import torch

from .modules import module_flops, module_macs, module_dmas
from .process import get_process_gpu_ram
from .utils import aggregate_info, format_info

__all__ = ['crawl_module', 'summary']


def apply(module, fn, depth=0, name=None):
    """Modified version of `torch.nn.Module.apply` method

    Args:
        module (torch.nn.Module): target module
        fn (callable): function to apply to each module
        depth (int, optional): current depth of `module`
        name (str, optional): name of the current module
    """

    if name is None:
        name = module.__class__.__name__.lower()
    fn(module, depth, name)
    for n, m in module.named_children():
        apply(m, fn, depth + 1, n)


def crawl_module(module, input_shape, dtype=None, max_depth=None):
    """Retrieves module information for an expected input tensor shape

    Example::
        >>> import torch.nn as nn
        >>> from torchscan import summary
        >>> mod = nn.Conv2d(3, 8, 3)
        >>> module_info = crawl_module(mod, (3, 224, 224))

    Args:
        module (torch.nn.Module): module to inspect
        input_shape (tuple<int>): expected input shapes
        dtype (type): data type of each input argument to the module
        max_depth (int, optional): maximum depth of layer information
    Returns:
        dict: layer and overhead information
    """

    # Get device and data types from model
    p = next(module.parameters())
    device = p.device

    cuda_overhead, framework_overhead = 0, 0
    if torch.cuda.is_available():
        # Process RAM - allocator RAM
        cuda_overhead = get_process_gpu_ram(os.getpid()) - (torch.cuda.memory_reserved() / 1024 ** 2)
        # Allocator RAM - Used RAM
        framework_overhead = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024 ** 2

    # input
    if isinstance(input_shape[0], int):
        input_shape = [input_shape]
    if dtype is None:
        dtype = p.data.dtype
    if isinstance(dtype, torch.dtype):
        dtype = [dtype] * len(input_shape)
    # Tensor arguments
    input_ts = [torch.rand(1, *in_shape).to(dtype=_dtype, device=device)
                for in_shape, _dtype in zip(input_shape, dtype)]

    # Hook definition
    def _hook_info(module, depth, name):

        def _pre_hook(module, input):
            """Pre-forward hook"""

            # Params
            grad_params, nograd_params, param_size = 0, 0, 0
            num_buffers, buffer_size = 0, 0
            is_shared = False
            if not any(module.children()):
                # Parameters
                for p in module.parameters():
                    if id(p) not in param_ids:
                        if p.requires_grad:
                            grad_params += p.data.numel()
                        else:
                            nograd_params += p.data.numel()
                        param_size += p.data.numel() * p.data.element_size()
                        param_ids.append(id(p))
                    else:
                        is_shared = True
                # Buffers
                for b in module.buffers():
                    if id(b) not in param_ids:
                        num_buffers += b.numel()
                        buffer_size += b.numel() * b.element_size()
                        param_ids.append(id(b))
                    else:
                        is_shared = True

            call_idxs[id(module)] = len(info)

            info.append(dict(name=name,
                             depth=depth,
                             type=module.__class__.__name__,
                             input_shape=(-1, *input[0][0].shape[1:]),
                             output_shape=None,
                             grad_params=grad_params,
                             nograd_params=nograd_params,
                             param_size=param_size,
                             num_buffers=num_buffers,
                             buffer_size=buffer_size,
                             flops=0,
                             macs=0,
                             dmas=0,
                             is_shared=is_shared,
                             is_leaf=not any(module.children())))

            # Remove the hook by using its handle
            pre_fw_handle.remove()

        def _fwd_hook(module, input, output):
            """Post-forward hook"""

            # Retrieve forward index
            fw_idx = call_idxs[id(module)]

            if any(module.children()):
                tot_flops, tot_macs, tot_dmas = 0, 0, 0
            else:
                # Compute stats for standalone layers
                tot_flops = module_flops(module, input[0], output)
                tot_macs = module_macs(module, input[0], output)
                tot_dmas = module_dmas(module, input[0], output)

            # Update layer information
            info[fw_idx]['output_shape'] = (-1, *output.shape[1:])
            # Add them, since some modules can be used several times
            info[fw_idx]['flops'] = tot_flops
            info[fw_idx]['macs'] = tot_macs
            info[fw_idx]['dmas'] = tot_dmas

            # Remove the hook by using its handle
            post_fw_handle.remove()

        # Hook only leaf children
        pre_fw_handle = module.register_forward_pre_hook(_pre_hook)
        post_fw_handle = module.register_forward_hook(_fwd_hook)

    # Hook model
    info = []
    param_ids = []
    call_idxs = {}
    apply(module, _hook_info)

    # Forward
    with torch.no_grad():
        module(*input_ts)

    reserved_ram, diff_ram = 0, 0
    if torch.cuda.is_available():
        reserved_ram = torch.cuda.memory_reserved() / 1024 ** 2
        diff_ram = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024 ** 2
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    grad_params, nograd_params, param_size = 0, 0, 0
    num_buffers, buffer_size = 0, 0
    for p in module.parameters():
        if p.requires_grad:
            grad_params += p.data.numel()
        else:
            nograd_params += p.data.numel()
        param_size += p.data.numel() * p.data.element_size()
    for b in module.buffers():
        num_buffers += b.numel()
        buffer_size += b.numel() * b.element_size()

    return dict(overheads=dict(cuda=dict(pre=cuda_overhead, fwd=get_process_gpu_ram(os.getpid()) - reserved_ram),
                               framework=dict(pre=framework_overhead, fwd=diff_ram)),
                layers=info,
                overall=dict(grad_params=grad_params, nograd_params=nograd_params, param_size=param_size,
                             num_buffers=num_buffers, buffer_size=buffer_size))


def summary(module, input_shape, wrap_mode='mid', max_depth=None):
    """Print module summary for an expected input tensor shape

    Example::
        >>> import torch.nn as nn
        >>> from torchscan import summary
        >>> mod = nn.Conv2d(3, 8, 3)
        >>> summary(mod, (3, 224, 224))

    Args:
        module (torch.nn.Module): module to inspect
        input_shape (tuple<int>): expected input shapes
        wrap_mode (str, optional): if a value is too long, where the wrapping should be performed
        max_depth (int, optional): maximum depth of layer information
    """

    # Get the summary dict
    module_info = crawl_module(module, input_shape)
    # Aggregate until max_depth
    if isinstance(max_depth, int):
        module_info = aggregate_info(module_info, max_depth)
    # Format it and print it
    print(format_info(module_info, wrap_mode))
