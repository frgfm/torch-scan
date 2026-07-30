# `torchscan.modules`

The modules subpackage contains tools for inspecting modules. See [Understanding results](metrics.md) for counting conventions and [Model and input support](model-support.md) for the capability matrix.

## FLOPs

Related to the number of floating-point operations performed during model inference.

::: torchscan.modules.module_flops

## MACs

Related to the number of multiply-accumulate operations performed during model inference.

::: torchscan.modules.module_macs

## DMAs

Related to the number of direct memory accesses during model inference.

::: torchscan.modules.module_dmas

## Receptive field

Related to the effective receptive field of a layer.

::: torchscan.modules.module_rf
