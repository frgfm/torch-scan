torchscan.modules
=================

The modules subpackage contains tools for inspection of modules.

.. currentmodule:: torchscan.modules


FLOPs
-----
Related to the number of floating point operations performed during model inference.

.. autofunction:: module_flops


MACs
-----
Related to the number of multiply-accumulate operations performed during model inference

.. autofunction:: module_macs


DMAs
----
Related to the number of direct memory accesses during model inference

.. autofunction:: module_dmas


Receptive field
---------------
Related to the effective receptive field of a layer

.. autofunction:: module_rf
