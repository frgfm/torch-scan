**************************************
TorchScan: inspect your PyTorch models
**************************************

The :mod:`torchscan` package provides tools for analyzing your PyTorch modules and models. Additionally to performance benchmarks, a comprehensive architecture comparison require some insights in the model complexity, its usage of computational and memory resources.


This project is meant for:

* |:zap:| **exploration**: easily assess the influence of your architecture on resource consumption
* |:woman_scientist:| **research**: quickly implement your own ideas to mitigate latency


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installing


.. toctree::
   :maxdepth: 1
   :caption: Package Reference
   :hidden:

   torchscan
   modules
   process
   utils

.. toctree::
   :maxdepth: 2
   :caption: Notes
   :hidden:

   changelog


Supported layers
^^^^^^^^^^^^^^^^

Here is the list of supported layers for FLOPS, MACs, DMAs and receptive field computation:

Non-linear activations
""""""""""""""""""""""

* `torch.nn.ReLU <https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html>`_
* `torch.nn.ELU <https://pytorch.org/docs/stable/generated/torch.nn.ELU.html>`_
* `torch.nn.LeakyReLU <https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html>`_
* `torch.nn.ReLU6 <https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html>`_
* `torch.nn.Tanh <https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html>`_
* `torch.nn.Sigmoid <https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html>`_

Linear layers
"""""""""""""

* `torch.nn.Identity <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_
* `torch.nn.Linear <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_

Convolutions
""""""""""""

* `torch.nn.Conv1d <https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html>`_
* `torch.nn.Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_
* `torch.nn.Conv3d <https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html>`_
* `torch.nn.ConvTranspose1d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html>`_
* `torch.nn.ConvTranspose2d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html>`_
* `torch.nn.ConvTranspose3d <https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html>`_

Pooling
"""""""

* `torch.nn.MaxPool1d <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html>`_
* `torch.nn.MaxPool2d <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html>`_
* `torch.nn.MaxPool3d <https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html>`_
* `torch.nn.AvgPool1d <https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html>`_
* `torch.nn.AvgPool2d <https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html>`_
* `torch.nn.AvgPool3d <https://pytorch.org/docs/stable/generated/torch.nn.AvgPool3d.html>`_
* `torch.nn.AdaptiveMaxPool1d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html>`_
* `torch.nn.AdaptiveMaxPool2d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html>`_
* `torch.nn.AdaptiveMaxPool3d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool3d.html>`_
* `torch.nn.AdaptiveAvgPool1d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html>`_
* `torch.nn.AdaptiveAvgPool2d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html>`_
* `torch.nn.AdaptiveAvgPool3d <https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html>`_

Normalization
"""""""""""""

* `torch.nn.BatchNorm1d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>`_
* `torch.nn.BatchNorm2d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_
* `torch.nn.BatchNorm3d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html>`_

Other
"""""

* `torch.nn.Flatten <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_
* `torch.nn.Dropout <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html>`_


*Please note that the functional API of PyTorch is not supported.*