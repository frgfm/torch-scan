#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Utils
"""


def format_name(name, depth=0):
    """Format a string for nested data printing

    Args:
        name (str): input string
        depth (int, optional): depth of the nested information
    Returns:
        str: formatted string
    """

    if depth == 0:
        return name
    elif depth == 1:
        return f"├─{name}"
    else:
        return f"{'|    ' * (depth - 1)}└─{name}"


def wrap_string(s, max_len, delimiter='.', wrap='[...]', mode='end'):
    """Wrap a string into a given length

    Args:
        s (str): input string
        max_len (int): maximum string length
        delimiter (str, optional): character used for delimiting information categories
        wrap (str, optional): wrapping sequence used
        mode (str, optional): wrapping mode
    Returns:
        str: wrapped string
    """

    if len(s) <= max_len or mode is None:
        return s

    if mode == 'end':
        return s[:max_len - len(wrap)] + wrap
    elif mode == 'mid':
        final_part = s.rpartition(delimiter)[-1]
        wrapped_end = f"{wrap}.{final_part}"
        return s[:max_len - len(wrapped_end)] + wrapped_end
    else:
        raise ValueError("received an unexpected value of argument `mode`")


def unit_scale(val):
    """Rescale value using scale units

    Args:
        val (float): input value
    Returns:
        float: rescaled value
        str: unit
    """

    if val // 1e12 > 0:
        return val / 1e12, 'T'
    elif val // 1e9 > 0:
        return val / 1e9, 'G'
    elif val // 1e6 > 0:
        return val / 1e6, 'M'
    elif val // 1e3 > 0:
        return val / 1e3, 'k'
    else:
        return val, ''


def format_info(module_info, wrap_mode='mid'):
    """Print module summary for an expected input tensor shape

    Args:
        module_info (dict): dictionary output of `crawl_module`
        wrap_mode (str, optional): wrapping mode
    """

    # Define separating lines
    line_length = 90
    thin_line = ('_' * line_length) + '\n'
    thick_line = ('=' * line_length) + '\n'
    dot_line = ('-' * line_length) + '\n'

    # Header
    info_str = thin_line
    info_str += f"{'Layer':<27}  {'Type':<20}  {'Output Shape':<25} {'Param #':<15}\n"
    info_str += thick_line

    # Layer information
    for layer in module_info['layers']:
        # name, type, output_shape, nb_params
        info_str += (f"{wrap_string(format_name(layer['name'], layer['depth']), 30, mode=wrap_mode):<27.25}  "
                     f"{layer['type']:<20}  {str(layer['output_shape']):<25} "
                     f"{layer['grad_params'] + layer['nograd_params'] + layer['num_buffers']:<15,}\n")

    # Parameter information
    info_str += thick_line

    info_str += f"Trainable params: {module_info['overall']['grad_params']:,}\n"
    info_str += f"Non-trainable params: {module_info['overall']['nograd_params']:,}\n"
    info_str += f"Total params: {module_info['overall']['grad_params'] + module_info['overall']['nograd_params']:,}\n"

    # Static RAM usage
    info_str += dot_line

    # Convert to Megabytes
    param_size = (module_info['overall']['param_size'] + module_info['overall']['buffer_size']) / 1024 ** 2
    overhead = module_info['overheads']['framework']['fwd'] + module_info['overheads']['cuda']['fwd']

    info_str += f"Model size (params + buffers): {param_size:.2f} Mb\n"
    info_str += f"Framework & CUDA overhead: {overhead:.2f} Mb\n"
    info_str += f"Total RAM usage: {param_size + overhead:.2f} Mb\n"

    # FLOPS information
    info_str += dot_line

    flops, flops_units = unit_scale(sum(layer['flops'] for layer in module_info['layers']))
    macs, macs_units = unit_scale(sum(layer['macs'] for layer in module_info['layers']))
    dmas, dmas_units = unit_scale(sum(layer['dmas'] for layer in module_info['layers']))

    info_str += f"Floating Point Operations on forward: {flops:.2f} {flops_units}FLOPs\n"
    info_str += f"Multiply-Accumulations on forward: {macs:.2f} {macs_units}MACs\n"
    info_str += f"Direct memory accesses on forward: {dmas:.2f} {dmas_units}DMAs\n"

    info_str += thin_line

    return info_str


def aggregate_info(info, max_depth):
    """Aggregate module information to a maximum depth

    Args:
        info (dict): dictionary output of `crawl_module`
        max_depth (int, optional): depth at which parent node aggregates children information
    Returns:
        dict: edited dictionary information
    """

    if not any(layer['depth'] == max_depth for layer in info['layers']):
        raise ValueError(f"The `max_depth` argument cannot be higher than module depth.")

    for fw_idx, layer in enumerate(info['layers']):
        # Need to aggregate information
        if not layer['is_leaf'] and layer['depth'] == max_depth:
            grad_p, nograd_p, p_size, num_buffers, b_size = 0, 0, 0, 0, 0
            flops, macs, dmas = 0, 0, 0
            for l in info['layers'][fw_idx + 1:]:
                # Children have superior depth and were hooked after parent
                if l['depth'] <= max_depth:
                    break
                # Aggregate all information (flops, macc, ram)
                flops += l['flops']
                macs += l['macs']
                dmas += l['dmas']
                grad_p += l['grad_params']
                nograd_p += l['nograd_params']
                p_size += l['param_size']
                num_buffers += l['num_buffers']
                b_size += l['buffer_size']

            # Update info
            info['layers'][fw_idx]['flops'] = flops
            info['layers'][fw_idx]['macs'] = macs
            info['layers'][fw_idx]['dmas'] = dmas
            info['layers'][fw_idx]['grad_params'] = grad_p
            info['layers'][fw_idx]['nograd_params'] = nograd_p
            info['layers'][fw_idx]['param_size'] = p_size
            info['layers'][fw_idx]['num_buffers'] = num_buffers
            info['layers'][fw_idx]['buffer_size'] = b_size

    # Filter out further depth information
    info['layers'] = [layer for layer in info['layers'] if layer['depth'] <= max_depth]

    return info
