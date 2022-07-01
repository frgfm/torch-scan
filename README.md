<p align="center">
  <img src="https://github.com/frgfm/torch-scan/releases/download/v0.1.1/logo_text.png" width="40%">
</p>

<p align="center">
  <a href="https://github.com/frgfm/torch-scan/actions?query=workflow%3Abuilds">
    <img alt="CI Status" src="https://img.shields.io/github/workflow/status/frgfm/torch-scan/builds?label=CI&logo=github&style=flat-square">
  </a>
  <a href="https://frgfm.github.io/torch-scan">
    <img src="https://img.shields.io/github/workflow/status/frgfm/torch-scan/docs?label=docs&logo=read-the-docs&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/frgfm/torch-scan">
    <img src="https://img.shields.io/codecov/c/github/frgfm/torch-scan.svg?logo=codecov&style=flat-square" alt="Test coverage percentage">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://www.codacy.com/gh/frgfm/torch-scan/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/torch-scan&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/9dc68e8bfce34d9dbc8b44a350e9adc7"/></a>
</p>
<p align="center">
  <a href="https://pypi.org/project/torchscan/">
    <img src="https://img.shields.io/pypi/v/torchscan.svg?logo=python&amp;logoColor=fff&amp;style=flat-square" alt="PyPi Status">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/torchscan.svg?style=flat-square" alt="pyversions">
  <img src="https://img.shields.io/pypi/l/torchscan.svg?style=flat-square" alt="license">
</p>

The very useful [summary](https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary) method of `tf.keras.Model` but for PyTorch, with more useful information.


## Quick Tour

### Inspecting your PyTorch architecture

Similarly to the `torchsummary` implementation, `torchscan` brings useful module information into readable format. For nested complex architectures, you can use a maximum depth of display as follows:

```python
from torchvision.models import densenet121
from torchscan import summary

model = densenet121().eval().cuda()
summary(model, (3, 224, 224), max_depth=2)
```

which would yield

```shell
__________________________________________________________________________________________
Layer                        Type                  Output Shape              Param #        
==========================================================================================
densenet                     DenseNet              (-1, 1000)                0              
├─features                   Sequential            (-1, 1024, 7, 7)          0              
|    └─conv0                 Conv2d                (-1, 64, 112, 112)        9,408          
|    └─norm0                 BatchNorm2d           (-1, 64, 112, 112)        257            
|    └─relu0                 ReLU                  (-1, 64, 112, 112)        0              
|    └─pool0                 MaxPool2d             (-1, 64, 56, 56)          0              
|    └─denseblock1           _DenseBlock           (-1, 256, 56, 56)         338,316        
|    └─transition1           _Transition           (-1, 128, 28, 28)         33,793         
|    └─denseblock2           _DenseBlock           (-1, 512, 28, 28)         930,072        
|    └─transition2           _Transition           (-1, 256, 14, 14)         133,121        
|    └─denseblock3           _DenseBlock           (-1, 1024, 14, 14)        2,873,904      
|    └─transition3           _Transition           (-1, 512, 7, 7)           528,385        
|    └─denseblock4           _DenseBlock           (-1, 1024, 7, 7)          2,186,272      
|    └─norm5                 BatchNorm2d           (-1, 1024, 7, 7)          4,097          
├─classifier                 Linear                (-1, 1000)                1,025,000      
==========================================================================================
Trainable params: 7,978,856
Non-trainable params: 0
Total params: 7,978,856
------------------------------------------------------------------------------------------
Model size (params + buffers): 30.76 Mb
Framework & CUDA overhead: 423.57 Mb
Total RAM usage: 454.32 Mb
------------------------------------------------------------------------------------------
Floating Point Operations on forward: 5.74 GFLOPs
Multiply-Accumulations on forward: 2.87 GMACs
Direct memory accesses on forward: 2.90 GDMAs
__________________________________________________________________________________________
```

Results are aggregated to the selected depth for improved readability.

For reference, here are explanations of a few acronyms:

- **FLOPs**: floating-point operations (not to be confused with FLOPS which is FLOPs per second)
- **MACs**: mutiply-accumulate operations (cf. [wikipedia](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation))
- **DMAs**: direct memory accesses (many argue that it is more relevant than FLOPs or MACs to compare model inference speeds cf. [wikipedia](https://en.wikipedia.org/wiki/Direct_memory_access))



Additionally, for highway nets (models without multiple branches / skip connections), `torchscan` supports receptive field estimation.

```python
from torchvision.models import vgg16
from torchscan import summary

model = vgg16().eval().cuda()
summary(model, (3, 224, 224), receptive_field=True, max_depth=0)
```

which will add the layer's receptive field (relatively to the last convolutional layer) to the summary.


## Setup

Python 3.6 (or higher) and [pip](https://pip.pypa.io/en/stable/)/[conda](https://docs.conda.io/en/latest/miniconda.html) are required to install Torchscan.

### Stable release

You can install the last stable release of the package using [pypi](https://pypi.org/project/torch-scan/) as follows:

```shell
pip install torchscan
```

or using [conda](https://anaconda.org/frgfm/torchscan):

```shell
conda install -c frgfm torchscan
```

### Developer installation

Alternatively, if you wish to use the latest features of the project that haven't made their way to a release yet, you can install the package from source:

```shell
git clone https://github.com/frgfm/torch-scan.git
pip install -e torch-scan/.
```


## Benchmark

Below are the results for classification models supported by `torchvision` for a single image with 3 color channels of size `224x224` (apart from  `inception_v3`   which uses `299x299`).

| Model              | Params (M) | FLOPs (G) | MACs (G) | DMAs (G) | RF   |
| ------------------ | ---------- | --------- | -------- | -------- | ---- |
| alexnet            | 61.1       | 1.43      | 0.71     | 0.72     | 195  |
| googlenet          | 6.62       | 3.01      | 1.51     | 1.53     | --   |
| vgg11              | 132.86     | 15.23     | 7.61     | 7.64     | 150  |
| vgg11_bn           | 132.87     | 15.26     | 7.63     | 7.66     | 150  |
| vgg13              | 133.05     | 22.63     | 11.31    | 11.35    | 156  |
| vgg13_bn           | 133.05     | 22.68     | 11.33    | 11.37    | 156  |
| vgg16              | 138.36     | 30.96     | 15.47    | 15.52    | 212  |
| vgg16_bn           | 138.37     | 31.01     | 15.5     | 15.55    | 212  |
| vgg19              | 143.67     | 39.28     | 19.63    | 19.69    | 268  |
| vgg19_bn           | 143.68     | 39.34     | 19.66    | 19.72    | 268  |
| resnet18           | 11.69      | 3.64      | 1.82     | 1.84     | --   |
| resnet34           | 21.8       | 7.34      | 3.67     | 3.7      | --   |
| resnet50           | 25.56      | 8.21      | 4.11     | 4.15     | --   |
| resnet101          | 44.55      | 15.66     | 7.83     | 7.9      | --   |
| resnet152          | 60.19      | 23.1      | 11.56    | 11.65    | --   |
| inception_v3       | 27.16      | 11.45     | 5.73     | 5.76     | --   |
| squeezenet1_0      | 1.25       | 1.64      | 0.82     | 0.83     | --   |
| squeezenet1_1      | 1.24       | 0.7       | 0.35     | 0.36     | --   |
| wide_resnet50_2    | 68.88      | 22.84     | 11.43    | 11.51    | --   |
| wide_resnet101_2   | 126.89     | 45.58     | 22.8     | 22.95    | --   |
| densenet121        | 7.98       | 5.74      | 2.87     | 2.9      | --   |
| densenet161        | 28.68      | 15.59     | 7.79     | 7.86     | --   |
| densenet169        | 14.15      | 6.81      | 3.4      | 3.44     | --   |
| densenet201        | 20.01      | 8.7       | 4.34     | 4.39     | --   |
| resnext50_32x4d    | 25.03      | 8.51      | 4.26     | 4.3      | --   |
| resnext101_32x8d   | 88.79      | 32.93     | 16.48    | 16.61    | --   |
| mobilenet_v2       | 3.5        | 0.63      | 0.31     | 0.32     | --   |
| shufflenet_v2_x0_5 | 1.37       | 0.09      | 0.04     | 0.05     | --   |
| shufflenet_v2_x1_0 | 2.28       | 0.3       | 0.15     | 0.15     | --   |
| shufflenet_v2_x1_5 | 3.5        | 0.6       | 0.3      | 0.31     | --   |
| shufflenet_v2_x2_0 | 7.39       | 1.18      | 0.59     | 0.6      | --   |
| mnasnet0_5         | 2.22       | 0.22      | 0.11     | 0.12     | --   |
| mnasnet0_75        | 3.17       | 0.45      | 0.23     | 0.24     | --   |
| mnasnet1_0         | 4.38       | 0.65      | 0.33     | 0.34     | --   |
| mnasnet1_3         | 6.28       | 1.08      | 0.54     | 0.56     | --   |

The above results were produced using the `scripts/benchmark.py` script.

*Note: receptive field computation is currently only valid for highway nets.*



## What else

### Documentation

The full package documentation is available [here](https://frgfm.github.io/torch-scan/) for detailed specifications.


### Example script

An example script is provided for you to benchmark torchvision models using the library:

```shell
python scripts/benchmark.py
```


## Credits

This project is developed and maintained by the repo owner, but the implementation was inspired or helped by the following contributions:

- [Pytorch summary](https://github.com/sksq96/pytorch-summary): existing PyTorch porting of `tf.keras.Model.summary`
- [Torchstat](https://github.com/Swall0w/torchstat): another module inspection tool
- [Flops counter Pytorch](https://github.com/sovrasov/flops-counter.pytorch): operation counter tool
- [THOP](https://github.com/Lyken17/pytorch-OpCounter): PyTorch Op counter
- Number of operations and memory estimation articles by [Matthijs Hollemans](https://machinethink.net/blog/how-fast-is-my-model/), and [Sicara](https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks)
- [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)


## Citation

If you wish to cite this project, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@misc{torchscan2020,
    title={Torchscan: meaningful module insights},
    author={François-Guillaume Fernandez},
    year={2020},
    month={March},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/frgfm/torch-scan}}
}
```


## Contributing

Any sort of contribution is greatly appreciated!

You can find a short guide in [`CONTRIBUTING`](CONTRIBUTING) to help grow this project!



## License

Distributed under the Apache 2.0 License. See [`LICENSE`](LICENSE) for more information.
