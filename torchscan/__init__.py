from importlib.metadata import version

from torchscan import modules, process, utils
from torchscan.benchmark import *
from torchscan.benchmark_compare import *
from torchscan.compare import *
from torchscan.crawler import *
from torchscan.extensions import *
from torchscan.flops import *
from torchscan.profiler import *
from torchscan.render import *
from torchscan.report import *
from torchscan.workload import *

__version__ = version("torchscan")
