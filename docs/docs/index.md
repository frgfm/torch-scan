# TorchScan: inspect your PyTorch models

The `torchscan` package provides tools for analyzing your PyTorch modules and models. In addition to performance benchmarks, a comprehensive architecture comparison requires insights into model complexity and its use of computational and memory resources.

This project is meant for:

- :zap: **exploration**: easily assess the influence of your architecture on resource consumption
- :woman_scientist: **research**: quickly implement your own ideas to mitigate latency

## Supported layers

Here is the list of supported layers for FLOPs, MACs, DMAs and receptive field computation:

### Non-linear activations

- [`torch.nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
- [`torch.nn.ELU`](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html)
- [`torch.nn.LeakyReLU`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
- [`torch.nn.ReLU6`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html)
- [`torch.nn.Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
- [`torch.nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)

### Linear layers

- [`torch.nn.Identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html)
- [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

### Convolutions

- [`torch.nn.Conv1d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [`torch.nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [`torch.nn.Conv3d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
- [`torch.nn.ConvTranspose1d`](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html)
- [`torch.nn.ConvTranspose2d`](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
- [`torch.nn.ConvTranspose3d`](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html)

### Pooling

- [`torch.nn.MaxPool1d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html)
- [`torch.nn.MaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
- [`torch.nn.MaxPool3d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html)
- [`torch.nn.AvgPool1d`](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html)
- [`torch.nn.AvgPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
- [`torch.nn.AvgPool3d`](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool3d.html)
- [`torch.nn.AdaptiveMaxPool1d`](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html)
- [`torch.nn.AdaptiveMaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html)
- [`torch.nn.AdaptiveMaxPool3d`](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool3d.html)
- [`torch.nn.AdaptiveAvgPool1d`](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html)
- [`torch.nn.AdaptiveAvgPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html)
- [`torch.nn.AdaptiveAvgPool3d`](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html)

### Normalization

- [`torch.nn.BatchNorm1d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [`torch.nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [`torch.nn.BatchNorm3d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html)

### Other

- [`torch.nn.Flatten`](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
- [`torch.nn.Dropout`](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

*Please note that the functional API of PyTorch is not supported.*
