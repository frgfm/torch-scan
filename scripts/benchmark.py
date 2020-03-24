#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Torchvision benchmark
"""

import torch
from torchvision import models

from torchscan import crawl_module

TORCHVISION_MODELS = [
    'alexnet',
    'googlenet',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'inception_v3',
    'squeezenet1_0', 'squeezenet1_1',
    'wide_resnet50_2', 'wide_resnet101_2',
    'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'resnext50_32x4d', 'resnext101_32x8d',
    'mobilenet_v2',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
    'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'
]


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"{'Model':<20} | {'Params (M)':<10} | {'FLOPs (G)':<10} | {'MACs (G)':<10} | {'DMAs (G)':<10}")
    print('-' * 71)
    for name in TORCHVISION_MODELS:
        model = models.__dict__[name]().eval().to(device)
        dsize = (3, 224, 224)
        if "inception" in name:
            dsize = (3, 299, 299)
        model_info = crawl_module(model, dsize)

        tot_params = sum(layer['grad_params'] + layer['nograd_params'] for layer in model_info['layers'])
        tot_flops = sum(layer['flops'] for layer in model_info['layers'])
        tot_macs = sum(layer['macs'] for layer in model_info['layers'])
        tot_dmas = sum(layer['dmas'] for layer in model_info['layers'])
        print(f"{name:<20} | {tot_params / 1e6:<10.2f} | {tot_flops / 1e9:<10.2f} | {tot_macs / 1e9:<10.2f} | "
              f"{tot_dmas / 1e9:<10.2f}")


if __name__ == "__main__":

    main()
