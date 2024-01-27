from contextlib import suppress
from torchscan import modules, process, utils
from torchscan.crawler import *

with suppress(ImportError):
    from .version import __version__
