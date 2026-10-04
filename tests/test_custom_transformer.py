import json

import pytest
import torch

from scripts.custom_transformer import HEAD_HANDLER, CustomTransformer, SineHead, main
from torchscan import IncompleteAnalysisError, crawl_module


def test_custom_transformer_example(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["custom_transformer.py", "--json", str(path)])
    main()
    report = json.loads(path.read_text())
    # Independent counts from the runnable guide, including the head's projection once.
    for name, value in (("module_flops", 1278), ("macs", 528), ("dmas", 962), ("parameters", 180)):
        assert report["totals"][name]["status"] == "complete"
        assert report["totals"][name]["value"] == value
    assert report["totals"]["operator_flops"]["status"] == "partial"


def test_custom_transformer_mask_and_ownership():
    with pytest.raises(IncompleteAnalysisError) as error:
        crawl_module(
            CustomTransformer(),
            kwargs={"source": torch.ones(1, 3, 4), "src_mask": torch.ones(3, 3, dtype=torch.bool).triu(1)},
            custom_modules={SineHead: HEAD_HANDLER},
            strict=True,
        )
    report = error.value.report
    for name, value in (("module_flops", 1296), ("macs", 528), ("dmas", 1007)):
        assert report["totals"][name]["status"] == "complete"
        assert report["totals"][name]["value"] == value
        assert {row["path"] for row in report["layers"] if name in row["metrics"]} == {"encoder", "head"}
    rows = {row["path"]: row for row in report["layers"]}
    assert rows["encoder"]["token_dependencies"]["sources"][0]["relation"] == {"kind": "prefix"}
    assert set(rows["head.projection"]["metrics"]) == {"calls"}
    assert rows["head.projection"]["metric_owners"]["macs"] == {"path": "head", "call_index": 0}
    operators = report["totals"]["operator_flops"]
    assert operators["status"] == "partial"
    assert operators["value"] is None
    assert operators["known_value"] is not None
    assert any(item.get("operator") == "aten.sin" for item in report["diagnostics"])
