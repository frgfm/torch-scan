import json

import pytest
import torch

from torchscan import crawl_module


def _assert_smoke_report(report):
    assert report["schema_version"] == 1
    assert report["layers"]
    assert report["totals"]["parameters"]["value"] > 0
    assert report["totals"]["operator_flops"]["status"] in {"complete", "partial"}
    if report["totals"]["operator_flops"]["status"] == "partial":
        assert report["totals"]["operator_flops"]["value"] is None
        assert report["operator_flops"]["diagnostics"]
    json.dumps(report)


@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v2", "resnext50_32x4d"])
def test_torchvision_cnn_without_downloads(name):
    models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = getattr(models, name)(weights=None)

    _assert_smoke_report(crawl_module(model, args=(torch.ones(1, 3, 32, 32),)))


@pytest.mark.parametrize("name", ["resnet18", "vit_tiny_patch16_224"])
def test_timm_cnn_and_vit_without_downloads(name):
    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    model = timm.create_model(name, pretrained=False, num_classes=10, **({"img_size": 32} if "vit" in name else {}))

    _assert_smoke_report(crawl_module(model, args=(torch.ones(1, 3, 32, 32),)))


def test_transformers_bert_kwargs_without_downloads():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = transformers.BertModel(config)
    input_ids = torch.arange(8).reshape(1, 8)
    attention_mask = torch.ones_like(input_ids)

    _assert_smoke_report(crawl_module(model, args=(input_ids,), kwargs={"attention_mask": attention_mask}))


def test_transformers_t5_cross_attention_without_downloads():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.T5Config(
        vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=0
    )
    model = transformers.T5Model(config)
    _assert_smoke_report(
        crawl_module(
            model,
            kwargs={
                "input_ids": torch.arange(5).reshape(1, 5),
                "decoder_input_ids": torch.arange(3).reshape(1, 3),
            },
        )
    )
