# TorchScan: inspect your PyTorch models

TorchScan prints the structure, parameter count, memory use, computation estimates, and receptive field of a PyTorch model.

## 60-second quickstart

Install the stable release:

```shell
pip install torchscan
```

Then inspect a model on CPU:

```python
import torch.nn as nn
from torchscan import summary

model = nn.Sequential(
    nn.Conv2d(3, 8, 3),
    nn.ReLU(),
).eval()

summary(model, (3, 32, 32), max_depth=1)
```

Representative output (abridged):

```text
Layer                        Type                  Output Shape              Param #
==========================================================================================
sequential                   Sequential            (-1, 8, 30, 30)           0
├─0                          Conv2d                (-1, 8, 30, 30)           224
├─1                          ReLU                  (-1, 8, 30, 30)           0
==========================================================================================
Trainable params: 224
Model size (params + buffers): 0.00 Mb
Floating Point Operations on forward: 396.00 kFLOPs
Multiply-Accumulations on forward: 194.40 kMACs
Direct memory accesses on forward: 216.22 kDMAs
```

`input_shape` excludes the batch dimension. TorchScan creates a synthetic batch of one on the model's current device, using its current mode; calling `eval()` avoids training-state updates during inspection.

## Next steps

- Choose between the stable and development releases in [Installation](installing.md).
- Check supported inputs, outputs, and modules in [Model and input support](model-support.md).
- Learn how to interpret estimates in [Understanding results](metrics.md).
- See every public function in the [`torchscan` API reference](torchscan.md).
