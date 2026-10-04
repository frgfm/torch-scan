import json

import pytest
import torch

from scripts.custom_transformer import HEAD_HANDLER, CustomTransformer, SineHead, estimate_head, main
from torchscan import IncompleteAnalysisError, ModuleHandler, crawl_module

_COMPUTE = ("module_flops", "macs", "dmas")


@pytest.mark.parametrize("keyword_source", [False, True])
@pytest.mark.parametrize("causal_mask", [False, True])
def test_custom_transformer_counts_and_call_context(keyword_source, causal_mask):
    model = CustomTransformer()
    # Mixed flags must survive the scan, including on a child covered by a handler.
    model.encoder.eval()
    model.head.projection.eval()
    training_flags = [module.training for module in model.modules()]
    source = torch.ones(1, 3, 4)
    mask = torch.ones(3, 3, dtype=torch.bool).triu(1) if causal_mask else None
    kwargs = {"scale": 0.25, "tag": "private-example-tag", "src_mask": mask}
    received = []
    calls = []

    def record_source(_module, args, options):
        received.append((args[0], options["src_mask"]))

    def estimate(call):
        calls.append((call.kwargs["scale"], call.kwargs["tag"], call.output["tag"]))
        return estimate_head(call)

    handle = model.encoder.register_forward_pre_hook(record_source, with_kwargs=True)
    try:
        report = crawl_module(
            model,
            args=() if keyword_source else (source,),
            kwargs={"source": source, **kwargs} if keyword_source else kwargs,
            custom_modules={SineHead: ModuleHandler(estimate, subtree_metrics=HEAD_HANDLER.subtree_metrics)},
        )
    finally:
        handle.remove()

    assert len(received) == 1
    assert received[0][0] is source
    assert received[0][1] is mask
    assert calls == [(0.25, "private-example-tag", "private-example-tag")]
    assert [module.training for module in model.modules()] == training_flags
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())

    # Independent tiny-case derivations are documented in custom-transformer.md.
    expected = {"module_flops": 1296 if causal_mask else 1278, "macs": 528, "dmas": 1007 if causal_mask else 962}
    for name, value in expected.items():
        assert report["totals"][name]["status"] == "complete", report["diagnostics"]
        assert report["totals"][name]["value"] == value
    assert report["totals"]["parameters"]["value"] == 180
    assert sum(row["parameters"]["trainable"] + row["parameters"]["frozen"] for row in report["layers"]) == 180

    rows = {row["path"]: row for row in report["layers"]}
    for name in _COMPUTE:
        assert {row["path"] for row in report["layers"] if name in row["metrics"]} == {"encoder", "head"}
        assert rows["head"]["metrics"][name]["scope"] == "subtree"
        assert rows["head"]["metrics"][name]["method"].startswith("custom_module_handler:")
    projection = rows["head.projection"]
    assert projection["parameters"]["trainable"] == 8
    assert projection["metrics"]["calls"]["value"] == 1
    for name in HEAD_HANDLER.subtree_metrics:
        assert name not in projection["metrics"]
        assert projection["metric_owners"][name] == {"path": "head", "call_index": 0}
    dependencies = rows["encoder"]["token_dependencies"]
    assert dependencies["status"] == "complete"
    assert dependencies["sources"] == [
        {
            "arguments": ["src"],
            "sequence_axis": 1,
            "length": 3,
            "relation": {"kind": "prefix" if causal_mask else "all"},
        }
    ]
    output = rows["head"]["output"]["items"][0]["value"]
    assert output["kind"] == "tuple"
    assert output["items"][0]["shape"] == [1, 3, 2]
    assert "private-example-tag" not in json.dumps(report)


def test_custom_transformer_strict_preserves_the_operator_gap():
    model = CustomTransformer()
    with pytest.raises(IncompleteAnalysisError) as error:
        crawl_module(model, args=(torch.ones(1, 3, 4),), custom_modules={SineHead: HEAD_HANDLER}, strict=True)

    report = error.value.report
    for name, value in (("module_flops", 1278), ("macs", 528), ("dmas", 962)):
        assert report["totals"][name]["status"] == "complete"
        assert report["totals"][name]["value"] == value
    operators = report["totals"]["operator_flops"]
    assert operators["status"] == "partial"
    assert operators["value"] is None
    assert operators["known_value"] is not None
    assert any(
        item.get("operator") == "aten.sin" and item["code"] == "uncounted_operator" for item in report["diagnostics"]
    )
    assert all(module.training for module in model.modules())
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())


def test_custom_transformer_structure_does_not_run_estimates():
    def unexpected(_call):
        pytest.fail("Structure mode must not call the custom estimator.")

    report = crawl_module(
        CustomTransformer(),
        args=(torch.ones(1, 3, 4),),
        custom_modules={SineHead: ModuleHandler(unexpected, subtree_metrics=HEAD_HANDLER.subtree_metrics)},
        mode="structure",
        strict=True,
    )
    assert report["totals"]["parameters"]["value"] == 180
    assert all(set(row["metrics"]) == {"calls"} and "token_dependencies" not in row for row in report["layers"])
    for name in (*_COMPUTE, "operator_flops"):
        assert report["totals"][name]["status"] == "unavailable"
        assert report["totals"][name]["method"] == "not_requested"


def test_custom_transformer_command_saves_a_report(tmp_path, monkeypatch, capsys):
    path = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["custom_transformer.py", "--json", str(path)])
    main()
    report = json.loads(path.read_text())
    assert report["totals"]["macs"]["value"] == 528
    assert report["totals"]["dmas"]["value"] == 962
    assert report["totals"]["operator_flops"]["status"] == "partial"
    assert "Operator FLOPs: partial" in capsys.readouterr().out
