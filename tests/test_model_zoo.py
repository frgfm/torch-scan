import json

import pytest
import torch

from torchscan import crawl_module


def _assert_smoke_report(report):
    assert report["schema_version"] == 1
    assert report["layers"]
    assert report["totals"]["parameters"]["value"] > 0
    assert report["totals"]["operator_flops"]["status"] in {"complete", "partial"}
    json.dumps(report)


def test_torchvision_resnet18_without_downloads():
    models = pytest.importorskip("torchvision.models")
    model = models.resnet18(weights=None)

    _assert_smoke_report(crawl_module(model, args=(torch.randn(1, 3, 32, 32),)))


def test_timm_resnet18_without_downloads():
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet18", pretrained=False, num_classes=10)

    _assert_smoke_report(crawl_module(model, args=(torch.randn(1, 3, 32, 32),)))


def test_transformers_bert_kwargs_without_downloads():
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = transformers.BertModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    attention_mask = torch.ones_like(input_ids)

    _assert_smoke_report(crawl_module(model, args=(input_ids,), kwargs={"attention_mask": attention_mask}))
