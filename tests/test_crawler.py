import io
import sys
from collections import OrderedDict

import pytest
import torch.nn as nn

from torchscan import crawler


def test_apply():
    multi_convs = nn.Sequential(nn.Conv2d(16, 32, 3), nn.Conv2d(32, 64, 3))
    mod = nn.Sequential(nn.Conv2d(3, 16, 3), multi_convs)

    # Tag module attributes
    def tag_name(mod, name):
        mod.__depth__ = len(name.split(".")) - 1
        mod.__name__ = name.rpartition(".")[-1]

    crawler.apply(mod, tag_name)

    assert mod[1][1].__depth__ == 2
    assert mod[1][1].__name__ == "1"


def test_crawl_module():

    mod = nn.Conv2d(3, 8, 3)

    res = crawler.crawl_module(mod, (3, 32, 32))
    assert isinstance(res, dict)
    assert res["overall"]["grad_params"] == 224
    assert res["layers"][0]["output_shape"] == (-1, 8, 30, 30)


def test_summary():

    mod = nn.Conv2d(3, 8, 3)

    # Redirect stdout with StringIO object
    captured_output = io.StringIO()
    sys.stdout = captured_output
    crawler.summary(mod, (3, 32, 32))
    # Reset redirect.
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue().split("\n")[7] == "Total params: 224"

    # Check receptive field
    captured_output = io.StringIO()
    sys.stdout = captured_output
    crawler.summary(mod, (3, 32, 32), receptive_field=True)
    # Reset redirect.
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue().split("\n")[1].rpartition("  ")[-1] == "Receptive field"
    assert captured_output.getvalue().split("\n")[3].split()[-1] == "3"
    # Check effective stats
    captured_output = io.StringIO()
    sys.stdout = captured_output
    crawler.summary(mod, (3, 32, 32), receptive_field=True, effective_rf_stats=True)
    # Reset redirect.
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue().split("\n")[1].rpartition("  ")[-1] == "Effective padding"
    assert captured_output.getvalue().split("\n")[3].split()[-1] == "0"

    # Max depth > model hierarchy
    with pytest.raises(ValueError):
        crawler.summary(mod, (3, 32, 32), max_depth=1)

    mod = nn.Sequential(
        OrderedDict(
            [
                ("features", nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(inplace=True))),
                ("pool", nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))),
                ("classifier", nn.Linear(8, 1)),
            ]
        )
    )

    crawler.summary(mod, (3, 32, 32), max_depth=1)
    # Reset redirect.
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue().split("\n")[4].startswith("├─features ")
