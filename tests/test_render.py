import base64
import copy
import hashlib
import json
import re
from html.parser import HTMLParser

# Only parse XML emitted by the renderer, never caller-supplied XML.
from xml.etree import ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]

import pytest
from torch import nn

from torchscan import ModuleHandler, crawl_module, metric_result, render_report
from torchscan._render_map import build_maps


class Node:
    def __init__(self, tag, attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []

    def text(self):
        return "".join(child if isinstance(child, str) else child.text() for child in self.children)

    def find(self, tag=None, **attributes):
        nodes = []
        if (tag is None or self.tag == tag) and all(self.attrs.get(key) == value for key, value in attributes.items()):
            nodes.append(self)
        for child in self.children:
            if isinstance(child, Node):
                nodes.extend(child.find(tag, **attributes))
        return nodes


class Document(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]
        self.feed(source)
        self.close()

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {"meta", "input", "br", "hr", "link", "img"}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        assert self.stack[-1].tag == tag
        self.stack.pop()

    def handle_data(self, data):
        self.stack[-1].children.append(data)


@pytest.fixture
def report_model():
    return nn.Sequential(nn.Linear(4, 8, bias=False), nn.Identity(), nn.Linear(8, 2, bias=False))


@pytest.fixture
def report(report_model):
    return crawl_module(report_model, (4,))


@pytest.fixture
def covered_report(report_model, request):
    estimate = metric_result(
        status=getattr(request, "param", "complete"),
        value=100,
        known_value=20,
        unit="FLOPs",
        scope="subtree",
        method="inclusive_cost",
    )
    handler = ModuleHandler(lambda _call: {"module_flops": estimate}, frozenset({"module_flops"}))
    return crawl_module(report_model, (4,), custom_modules={nn.Sequential: handler})


def _document(report, **kwargs):
    return Document(render_report(report, **kwargs)).root


def _panel(root, name):
    return root.find("div", **{"data-view": name})[0]


def _embedded(root):
    return json.loads(root.find("script", id="torchscan-data")[0].text())


@pytest.mark.parametrize("covered_report", ["complete", "partial", "unavailable"], indirect=True)
def test_inclusive_handler_coverage_is_not_rendered_as_missing_child_work(covered_report):
    estimate = covered_report["layers"][0]["metrics"]["module_flops"]
    root = _document(covered_report)
    groups = _embedded(root)["maps"]["module_flops"]["groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["kind"] == "recorded"
    assert group["known_total"] == estimate["known_value"]
    for node in group["nodes"][1:]:
        assert node["rail"] == []
        assert node["direct"] is None
        assert node["direct_kind"] == node["display_status"] == "covered"
        call = node["calls"][0]
        assert call["display"] == "covered · included in (root) · call #0 inclusive estimate"
        assert call["display_status"] == "covered"
        assert call["owner"] == {"path": "", "call_index": 0}

    detail = root.find("details", id="call-1")[0]
    assert "module_flops: included in (root) · call #0 inclusive estimate" in detail.text()
    assert "not_recorded" not in detail.text()
    panel = _panel(root, "module_flops")
    assert "Covered by inclusive estimates (3)" in panel.text()
    count = 0 if estimate["status"] == "complete" else 1
    assert f"Unranked incomplete or absent measurements ({count})" in panel.text()
    svg = ET.fromstring(render_report(covered_report, format="svg"))  # ruff: ignore[suspicious-xml-element-tree-usage]
    assert not any(
        "rail-" in element.attrib.get("id", "") and "node-1" in element.attrib["id"] for element in svg.iter()
    )


@pytest.mark.parametrize("invalid", ["scope", "owner_shape", "owner_index", "missing_owner", "duplicate_estimate"])
def test_invalid_ownership_evidence_is_rejected(covered_report, invalid):
    parent, child = covered_report["layers"][:2]
    if invalid == "scope":
        parent["metric_ownership"]["module_flops"] = "unknown"
    elif invalid == "owner_shape":
        child["metric_owners"]["module_flops"] = None
    elif invalid == "owner_index":
        child["metric_owners"]["module_flops"]["call_index"] = True
    elif invalid == "missing_owner":
        child["metric_owners"]["module_flops"]["path"] = "missing"
    else:
        child["metrics"]["module_flops"] = metric_result(
            status="complete", value=1, unit="FLOPs", scope="module_call", method="duplicate"
        )

    with pytest.raises(ValueError):
        render_report(covered_report)


@pytest.fixture
def shared_outside_report():
    class Owner(nn.Sequential):
        pass

    class SharedOutside(nn.Module):
        def __init__(self, include):
            super().__init__()
            self.owner = Owner(nn.Sequential(nn.Identity()))
            self.outside = self.owner[0][0]
            self.zzother = nn.Identity()
            self.include = include

        def forward(self, inputs):
            owned = self.owner(inputs)
            return self.zzother(self.outside(owned) if self.include else owned)

    def collect(status, count, *, include=True, inclusive=100):
        estimate = metric_result(
            status=status, value=count, known_value=count, unit="FLOPs", scope="module_call", method="outside"
        )

        def a_outside(_call):
            return {"module_flops": estimate}

        def zz_owner(_call):
            return {"module_flops": inclusive}

        return crawl_module(
            SharedOutside(include),
            (4,),
            custom_modules={
                Owner: ModuleHandler(zz_owner, frozenset({"module_flops"})),
                nn.Identity: ModuleHandler(a_outside),
            },
        )

    return collect


@pytest.mark.parametrize(("status", "count"), [("complete", 0), ("complete", 5), ("partial", 5), ("unavailable", None)])
def test_covered_container_keeps_shared_descendant_contributions_visible(shared_outside_report, status, count):
    report = shared_outside_report(status, count)
    html = render_report(report, before=report)
    group = next(
        group
        for group in _embedded(Document(html).root)["maps"]["module_flops"]["groups"]
        if group["method"].endswith(":outside")
    )
    node = _node(group, "owner.0")
    assert node["direct_kind"] == "covered"
    assert node["coverage"] is not None
    assert node["has_contributions"]
    assert node["before_has_contributions"]
    assert node["display_status"] == node["subtotal"]["status"] == status
    assert node["known"] == count
    assert node["display"].startswith(status)
    assert node["before_display"].startswith(status)
    for source in (html, render_report(report, format="svg")):
        description = _tile_anchor(_maps(source)[group["id"]], node).attrib["aria-label"]
        assert f"Derived subtree subtotal: {status}" in description
        assert "Own calls covered: included in owner" in description


def test_svg_comparison_keeps_coverage_independent_between_snapshots(shared_outside_report):
    contributed = shared_outside_report("complete", 0, inclusive=0)
    covered = shared_outside_report("complete", 0, include=False, inclusive=0)
    for before, after in ((contributed, covered), (covered, contributed)):
        svg = ET.fromstring(render_report(after, before=before, format="svg"))  # ruff: ignore[suspicious-xml-element-tree-usage]
        inspector = next(element for element in svg.iter() if element.attrib.get("id", "").startswith("inspector-"))
        text = " ".join(" ".join(inspector.itertext()).split())
        complete = "complete · 0 FLOPs"
        included = "covered · included in owner"
        assert f"Before subtree: {complete if before is contributed else included}" in text
        assert f"After subtree: {complete if after is contributed else included}" in text
        assert "unavailable" not in text
        label = "Derived descendant subtotal" if after is contributed else "Covered by inclusive ancestor estimate"
        assert label in text


def test_real_report_html_hierarchy_ranking_methods_and_offline_controls(report):
    root = _document(report)
    assert "not a computational graph" in root.text()
    assert "not measured peak memory" in root.text()
    assert "FLOPs do not establish latency" in root.text()
    assert "torchscan_module_formula" in root.text()
    assert "Output shapes and metadata" in root.text()
    assert '"shape": [' in root.text()
    assert len(root.find("input", type="radio", name="view")) == 5
    assert len(root.find("details", id="call-1")) == 1
    panel = _panel(root, "module_flops")
    rows = panel.find("tbody")[0].find("tr")
    assert "0 · call #0" in rows[0].text()  # Larger Linear precedes the smaller one.
    assert "(root)" in panel.find("ul")[0].text()
    assert "unavailable" in panel.find("ul")[0].text()
    links = root.find("a")
    identifiers = [node.attrs["id"] for node in root.find() if "id" in node.attrs]
    assert len(identifiers) == len(set(identifiers))
    assert all(link.attrs["href"].startswith("#") and link.attrs["href"][1:] in identifiers for link in links)
    assert not root.find("script", src="")
    assert all("src" not in node.attrs for node in root.find())
    assert _embedded(root)["report"] == report


def test_inclusive_operator_counts_never_join_layer_ranking(report):
    report["operator_flops"]["by_module"] = {"Parent": 10000, "Parent.Child": 5000}
    root = _document(report)
    assert "10000" not in _panel(root, "module_flops").text()
    operator_section = root.find("section", id="operators")[0]
    assert "10,000 known inclusive FLOPs" in operator_section.text()
    assert "Do not sum these rows" in operator_section.text()


def test_partial_zero_unavailable_and_complete_zero_are_distinct(report):
    lower = metric_result(status="partial", known_value=0, unit="FLOPs", scope="module_call", method="test")
    report["layers"][1]["metrics"]["module_flops"] = lower
    report["layers"][2]["metrics"]["module_flops"] = metric_result(
        status="unavailable", unit="FLOPs", scope="module_call", method="test"
    )
    report["layers"][3]["metrics"]["module_flops"] = metric_result(
        status="complete", value=0, unit="FLOPs", scope="module_call", method="test"
    )
    root = _document(report)
    panel = _panel(root, "module_flops")
    assert len(panel.find("progress")) == 1
    assert "complete 0 FLOPs" in panel.find("tbody")[0].text()
    assert "partial at least 0 FLOPs (lower bound; full value unknown)" in panel.find("ul")[0].text()
    assert "unavailable unknown" in panel.find("ul")[0].text()
    svg = ET.fromstring(render_report(report, format="svg"))  # ruff: ignore[suspicious-xml-element-tree-usage]
    assert "partial · ≥ 0 FLOPs · full value unknown" in "".join(svg.itertext())


def test_comparison_complete_deltas_and_context(report):
    before = copy.deepcopy(report)
    after = copy.deepcopy(report)
    after["totals"]["module_flops"]["value"] += 12
    after["totals"]["module_flops"]["known_value"] += 12
    after["layers"][1]["metrics"]["module_flops"]["value"] += 12
    after["layers"][1]["metrics"]["module_flops"]["known_value"] += 12
    root = _document(after, before=before)
    assert "complete · +12 FLOPs" in root.find("section", id="comparison")[0].text()
    assert "Changed calls (1)" in root.text()
    assert "Before input and measurement context" in root.text()
    assert "After input and measurement context" in root.text()
    assert _embedded(root)["comparison"]["totals"]["module_flops"]["delta"] == 12


@pytest.mark.parametrize(
    "change",
    [
        "inputs",
        "torch_version",
        "torchscan_version",
        "python_version",
        "execution_mode",
        "devices",
        "dtypes",
        "missing",
    ],
)
def test_comparison_context_mismatch_withholds_compute_but_allows_storage(report, change):
    before = copy.deepcopy(report)
    if change == "inputs":
        before["inputs"]["args"][0]["shape"] = [3, 4]
    elif change == "missing":
        del before["context"]["torch_version"]
    else:
        before["context"][change] = ["different"] if change in ("devices", "dtypes") else "different"
    root = _document(report, before=before)
    diff = _embedded(root)["comparison"]
    assert diff["totals"]["module_flops"]["delta"] is None
    assert diff["totals"]["module_flops"]["status"] == "unavailable"
    assert diff["totals"]["operator_flops"]["delta"] is None
    assert diff["totals"]["parameters"]["delta"] == 0
    assert "not comparable" in root.find("section", id="comparison")[0].text()


def test_comparison_partial_no_numeric_delta_missing_added_removed(report):
    before = copy.deepcopy(report)
    before["totals"]["module_flops"] = metric_result(
        status="partial", known_value=7, unit="FLOPs", scope="forward", method="torchscan_module_formula"
    )
    del before["totals"]["parameters"]
    before["layers"][1]["path"] = "old"
    root = _document(report, before=before)
    comparison = root.find("section", id="comparison")[0]
    assert "partial · delta unknown" in comparison.text()
    assert "unavailable · delta unknown" in comparison.text()
    assert "Added calls (1)" in comparison.text()
    assert "Removed calls (1)" in comparison.text()
    diff = _embedded(root)["comparison"]
    assert diff["totals"]["module_flops"]["delta"] is None
    assert diff["totals"]["parameters"]["delta"] is None


def test_comparison_uses_compare_reports_method_compatibility(report):
    before = copy.deepcopy(report)
    before["totals"]["module_flops"]["method"] = "other method"
    with pytest.raises(ValueError, match="incompatible method"):
        render_report(report, before=before)


def test_rankings_do_not_compare_different_methods_or_units(report):
    report["layers"][1]["metrics"]["module_flops"]["method"] = "custom_formula"
    root = _document(report)
    rows = _panel(root, "module_flops").find("tbody")[0].find("tr")
    assert [row.find("td")[0].text() for row in rows] == ["1", "1", "2"]
    assert float(rows[0].find("progress")[0].attrs["value"]) == pytest.approx(1)
    assert float(rows[1].find("progress")[0].attrs["value"]) == pytest.approx(1)
    assert "Ranks restart for each group" in _panel(root, "module_flops").text()


def test_missing_metrics_are_explicit_in_details_and_svg_unranked(report):
    structure = crawl_module(nn.Linear(4, 2), (4,), mode="structure")
    root = _document(structure)
    assert "unavailable unknown" in root.find("details", id="call-0")[0].text()
    assert "not_requested" in root.find("details", id="call-0")[0].text()
    assert not _panel(root, "module_flops").find("progress")
    assert _panel(root, "parameters").find("progress")
    svg = ET.fromstring(render_report(structure, format="svg"))  # ruff: ignore[suspicious-xml-element-tree-usage]
    text = "".join(svg.itertext())
    assert "structural · unscaled" in text
    assert "unavailable · unknown" in text
    before = copy.deepcopy(report)
    before["layers"][1]["path"] = "old"
    before["layers"][2]["metrics"]["calls"]["value"] = 2
    before["layers"][2]["metrics"]["calls"]["known_value"] = 2
    svg = ET.fromstring(render_report(report, before=before, format="svg"))  # ruff: ignore[suspicious-xml-element-tree-usage]
    text = "".join(svg.itertext())
    assert "Module added in after report." in text
    assert "Module removed in after report." in text
    # Exact changes remain in compare_reports evidence; the map represents the
    # selected cost metric, without treating call-count changes as FLOP deltas.
    root = _document(report, before=before)
    comparison = root.find("section", id="comparison")[0]
    assert "Changed calls (1)" in comparison.text()
    assert "complete · -1 calls" in comparison.text()


def test_invalid_layer_operator_and_diagnostic_shapes_rejected(report):
    cases = [
        (lambda r: r["layers"][0].update(path="bad..path"), "empty module component"),
        (lambda r: r["layers"][0]["parameters"].update(shared=1), "shared must be a boolean"),
        (lambda r: r["operator_flops"].update(schema_version=2), "operator schema_version"),
        (lambda r: r["operator_flops"]["by_module"].update(bad=-1), "nonnegative integer"),
        (
            lambda r: r["operator_flops"]["ignored_operators"].update(bad={"calls": 1, "reason": 0}),
            "reason must be a string",
        ),
        (
            lambda r: r["diagnostics"].append({"code": "bad", "severity": "info", "metric": "flops", "message": "bad"}),
            "invalid severity",
        ),
    ]
    for mutate, message in cases:
        invalid = copy.deepcopy(report)
        mutate(invalid)
        with pytest.raises(ValueError, match=message):
            render_report(invalid)


def test_suggestions_link_facts_and_separate_experiments(report):
    root = _document(report)
    suggestions = root.find("li", **{"class": "suggestion"})
    assert len(suggestions) >= 2
    for suggestion in suggestions:
        assert "Recorded fact:" in suggestion.text()
        assert "Experiment to try:" in suggestion.text()
        assert suggestion.find("a")[0].attrs["href"].startswith("#call-")
    assert "benchmark latency" in suggestions[0].text()
    assert "measure peak memory" in suggestions[1].text()


def test_optional_shape_diagnostics_do_not_imply_missing_compute(report):
    report["diagnostics"].append({
        "code": "module_metric_error",
        "severity": "warning",
        "metric": "receptive_field",
        "message": "unsupported receptive-field metadata",
        "path": "0",
    })
    root = _document(report)
    suggestions = root.find("li", **{"class": "suggestion"})
    assert len(suggestions) == 2
    assert "compute diagnostic" not in "".join(suggestion.text() for suggestion in suggestions)
    assert root.find("details", id="call-1")[0].find("a", href="#diagnostic-0")


def test_unexecuted_parameters_stay_in_totals_and_do_not_create_calls():
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(4, 4, bias=False)
            self.unused = nn.Linear(4, 16, bias=False)

        def forward(self, inputs):
            return self.used(inputs)

    report = crawl_module(Branch(), (4,))
    assert report["totals"]["parameters"]["value"] == 80
    assert sum(layer["parameters"]["trainable"] for layer in report["layers"]) == 16
    root = _document(report)
    assert "unused" not in root.find("section", **{"class": "tree"})[0].text()
    assert "Authoritative model total: complete 80 elements" in _panel(root, "parameters").text()


def test_untrusted_content_escaped_and_embedded_json_round_trips(report):
    payload = '</script><script>window.pwned=1</script><img src=x onerror="alert(1)">&\u2028\u2029'
    report["context"]["model_type"] = payload
    report["inputs"][payload] = {"nested": payload}
    report["layers"][1]["path"] = payload
    report["layers"][1]["type"] = payload
    report["layers"][1]["metrics"]["calls"]["method"] = payload
    report["layers"][1]["metrics"][payload] = report["layers"][1]["metrics"]["calls"]
    report["diagnostics"].append({"code": payload, "severity": "warning", "metric": payload, "message": payload})
    report["operator_flops"]["by_operator"][payload] = 9
    report["operator_flops"]["by_module"][payload] = 9
    root = _document(report, title=payload, before=copy.deepcopy(report))
    assert len(root.find("script")) == 2
    assert not root.find("img")
    assert all(not attr.startswith("on") for node in root.find() for attr in node.attrs)
    assert _embedded(root)["report"] == report
    assert _embedded(root)["before_diagnostics"][-1]["message"] == payload
    assert payload in root.find("li", id=f"before-diagnostic-{len(report['diagnostics']) - 1}")[0].text()
    raw = root.find("script", id="torchscan-data")[0].text()
    assert "<" not in raw
    assert "&" not in raw
    assert "\u2028" not in raw
    assert "\u2029" not in raw
    assert payload in root.find("title")[0].text()
    script = root.find("script")[-1].text()
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    csp = root.find("meta", **{"http-equiv": "Content-Security-Policy"})[0].attrs["content"]
    assert f"'sha256-{digest}'" in csp
    assert "default-src 'none'" in csp
    svg = ET.fromstring(render_report(report, format="svg", title=payload, before=report))  # ruff: ignore[suspicious-xml-element-tree-usage]
    assert svg.find("{*}title").text == payload
    assert svg.attrib["role"] == "img"
    assert all(link.attrib["tabindex"] == "0" for link in svg.findall(".//{*}a"))
    assert all(element.tag.split("}")[-1] not in {"script", "image", "foreignObject"} for element in svg.iter())


@pytest.mark.parametrize(
    ("field", "repeated", "view"),
    [
        ("method", False, "module_flops"),
        ("scope", True, "module_flops"),
        ("unit", False, "module_flops"),
        ("method", True, "parameters"),
    ],
)
def test_svg_grouped_measurements_keep_independent_scales_and_compatible_evidence(field, repeated, view):
    class Repeated(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Linear(4, 4, bias=False)

        def forward(self, inputs):
            return self.block(self.block(inputs))

    model = Repeated() if repeated else nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False))
    report = crawl_module(model, (4,))
    report["layers"][2]["metrics"]["module_flops"][field] = "aaa_custom"
    groups = _embedded(_document(report))["maps"][view]["groups"]
    source = render_report(report, format="svg", metric=view)
    svg = ET.fromstring(source)  # ruff: ignore[suspicious-xml-element-tree-usage]
    identifiers = {element.attrib["id"]: element for element in svg.iter() if "id" in element.attrib}
    assert len(identifiers) == sum("id" in element.attrib for element in svg.iter())
    if view == "module_flops":
        diagrams = _maps(source)
        for group in groups:
            known = next(node for node in group["nodes"] if node["direct_known"])
            diagram = diagrams[group["id"]]
            assert _rect_geometry(_tile(diagram, known))[0::2] == _rect_geometry(_tile(diagram, _node(group, "")))[0::2]
            assert group["method"] in "".join(diagram.itertext())
    parents = {child: parent for parent in svg.iter() for child in parent}
    links = svg.findall(".//{*}a")
    assert all(link.attrib["href"][1:] in identifiers for link in links)
    experiments = [
        link
        for link in links
        if "Recorded fact:" in "".join(link.itertext()) and "largest complete" in "".join(link.itertext())
    ]
    assert len(experiments) == 2
    for link in experiments:
        # Both compute and parameter experiments refer to the first call. A
        # repeated module can contain calls from two different method groups.
        group = next(
            group
            for group in groups
            if any(call["index"] == 1 and call["in_group"] for node in group["nodes"] for call in node["calls"])
        )
        node = next(node for node in group["nodes"] if any(call["index"] == 1 for call in node["calls"]))
        target = identifiers[link.attrib["href"][1:]]
        if target.attrib["id"] == "call-1":
            while target in parents and target.attrib.get("id") != f"inspector-{group['id']}-{node['id']}":
                target = parents[target]
            assert target.attrib.get("id") == f"inspector-{group['id']}-{node['id']}"
        else:
            assert target.attrib["id"] in {f"static-{group['id']}-{node['id']}", f"rail-{group['id']}-{node['id']}"}


def test_svg_missing_metric_experiment_links_use_the_synthetic_group():
    report = crawl_module(nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False)), (4,))
    del report["layers"][1]["metrics"]["module_flops"]
    # A real user method can have the same name as the missing-measurement group.
    report["layers"][2]["metrics"]["module_flops"]["method"] = "no_recorded_measurement"
    groups = _embedded(_document(report))["maps"]["module_flops"]["groups"]
    missing = next(group for group in groups if group["kind"] == "unrecorded")
    recorded = next(group for group in groups if group["kind"] == "recorded")
    assert recorded["method"] == missing["method"]
    assert _node(recorded, "1")["calls"][0]["display_status"] == "complete"
    assert _node(missing, "1")["calls"][0]["in_group"] is False
    node = next(node for node in missing["nodes"] if node["path"] == "0")
    svg = ET.fromstring(render_report(report, format="svg"))  # ruff: ignore[suspicious-xml-element-tree-usage]
    identifiers = {element.attrib["id"] for element in svg.iter() if "id" in element.attrib}
    links = svg.findall(".//{*}a")
    assert all(link.attrib["href"][1:] in identifiers for link in links)
    storage_experiments = [
        link for link in links if "largest complete attributed parameters" in "".join(link.itertext())
    ]
    assert len(storage_experiments) == 1
    assert storage_experiments[0].attrib["href"] == f"#static-{missing['id']}-{node['id']}"


@pytest.mark.parametrize("removed", [False, True])
@pytest.mark.parametrize("output_format", ["html", "svg"])
def test_comparison_visibly_preserves_baseline_diagnostics(removed, output_format):
    class Sine(nn.Module):
        def forward(self, inputs):
            return inputs.sin()

    before = crawl_module(nn.Sequential(nn.Linear(4, 4, bias=False), Sine()), (4,))
    after = crawl_module(nn.Sequential() if removed else nn.Sequential(nn.Identity(), nn.Identity()), (4,))
    diagnostics = before["diagnostics"] + [
        item for item in before["operator_flops"]["diagnostics"] if item not in before["diagnostics"]
    ]
    source = render_report(after, before=before, format=output_format)
    if output_format == "html":
        root = Document(source).root
        rows = [root.find("li", id=f"before-diagnostic-{index}")[0] for index in range(len(diagnostics))]
        assert all(item["message"] in row.text() for item, row in zip(diagnostics, rows, strict=True))
        assert _embedded(root)["before_diagnostics"] == [
            {**item, "index": index} for index, item in enumerate(diagnostics)
        ]
        identifiers = {node.attrs["id"] for node in root.find() if "id" in node.attrs}
        assert all(link.attrs["href"][1:] in identifiers for link in root.find("a"))
        assert root.find("details", id="before-call-2")[0].find("a", href="#before-diagnostic-0")
        return
    svg = ET.fromstring(source)  # ruff: ignore[suspicious-xml-element-tree-usage]
    visible = " ".join(element.text or "" for element in svg.iter() if element.tag.endswith("}text"))
    assert "Before diagnostics" in visible
    assert "Module type not supported: Sine" in visible
    assert "aten.sin was observed" in visible
    assert all(diagnostic["message"] in visible for diagnostic in diagnostics)
    if removed:
        assert "Before measurement diagnostics" in visible
    identifiers = {element.attrib["id"]: element for element in svg.iter() if "id" in element.attrib}
    assert all(f"before-diagnostic-{index}" in identifiers for index in range(len(diagnostics)))
    context = json.loads(identifiers["svg-evidence"].find("{*}title").text)
    assert context["before"]["diagnostics"] == diagnostics
    assert context["before"]["inputs"] == before["inputs"]
    assert all(link.attrib["href"][1:] in identifiers for link in svg.findall(".//{*}a"))


def test_svg_accessible_names_exclude_raw_call_json_but_tooltips_keep_exact_evidence(report):
    svg = ET.fromstring(render_report(report, format="svg"))  # ruff: ignore[suspicious-xml-element-tree-usage]
    anchors = [element for element in svg.iter() if "data-map-node" in element.attrib]
    for anchor in anchors:
        assert "Call evidence:" not in anchor.attrib["aria-label"]
        assert '"metrics"' not in anchor.attrib["aria-label"]
        tooltip = anchor.find("{*}title").text
        evidence = json.loads(tooltip.split(" Call evidence: ", 1)[1])
        path = evidence[0]["path"]
        assert evidence == [layer for layer in report["layers"] if layer["path"] == path]


@pytest.mark.parametrize("output_format", ["html", "svg"])
def test_deterministic_rendering_and_input_immutability(report, output_format):
    saved = copy.deepcopy(report)
    first = render_report(report, before=report, format=output_format)
    assert first == render_report(report, before=report, format=output_format)
    assert report == saved


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("schema_version", True, "schema_version"),
        ("layers", {}, "layers"),
        ("context", [], "context"),
        ("diagnostics", {}, "diagnostics"),
        ("inputs", {"bad": float("nan")}, "finite number"),
        ("inputs", {"bad": object()}, "finite number"),
        ("inputs", {9: "bad key"}, "string"),
        ("inputs", {"bad": "\x00"}, "invalid text"),
    ],
)
def test_invalid_schema_and_json_rejected(report, field, value, message):
    report[field] = value
    with pytest.raises(ValueError, match=message):
        render_report(report)


@pytest.mark.parametrize(
    ("status", "value", "known", "message"),
    [
        ("bogus", 0, 0, "invalid status"),
        ("complete", None, None, "finite number"),
        ("complete", 1, 2, "invariant"),
        ("partial", 1, 1, "invariant"),
        ("partial", None, None, "finite number"),
        ("unavailable", 0, None, "invariant"),
        ("complete", True, True, "finite number"),
        ("complete", float("inf"), float("inf"), "finite number"),
    ],
)
def test_invalid_status_invariants_rejected(report, status, value, known, message):
    result = report["totals"]["module_flops"]
    result.update(status=status, value=value, known_value=known)
    with pytest.raises(ValueError, match=message):
        render_report(report)


def test_duplicate_invalid_calls_and_inconsistent_operator_total(report):
    original = copy.deepcopy(report)
    report["layers"].append(copy.deepcopy(report["layers"][0]))
    with pytest.raises(ValueError, match="duplicate layer call"):
        render_report(report)
    report = copy.deepcopy(original)
    report["layers"][0]["call_index"] = -1
    with pytest.raises(ValueError, match="nonnegative integer"):
        render_report(report)
    report = copy.deepcopy(original)
    report["totals"]["operator_flops"] = copy.deepcopy(report["operator_flops"]["total"])
    report["totals"]["operator_flops"]["method"] = "inconsistent"
    with pytest.raises(ValueError, match="must match"):
        render_report(report)


def test_invalid_options_and_baseline_rejected(report):
    for output_format in ("pdf", None, []):
        with pytest.raises(ValueError, match="format"):
            render_report(report, format=output_format)
    with pytest.raises(ValueError, match="metric"):
        render_report(report, metric="operator_flops")
    with pytest.raises(ValueError, match="title"):
        render_report(report, title=None)
    with pytest.raises(ValueError, match="schema_version"):
        render_report(report, before={"schema_version": 5})
    report["inputs"]["cycle"] = report
    with pytest.raises(ValueError, match="acyclic"):
        render_report(report)


@pytest.mark.parametrize(("field", "value"), [("devices", 1), ("devices", [1]), ("dtypes", "cpu")])
def test_invalid_display_context_rejected(report, field, value):
    report["context"][field] = value
    with pytest.raises(ValueError, match=f"context.{field} must be a list of strings"):
        render_report(report)


@pytest.mark.parametrize("output_format", ["html", "svg"])
def test_open_ended_shape_metadata_remains_safe_to_display(report, output_format):
    # Schema v1 deliberately permits arbitrary JSON metadata rather than a
    # mandatory tensor shape record. Both formats must tolerate that contract.
    report["inputs"] = {"args": None, "kwargs": ["metadata"]}
    report["layers"][1]["input"] = {"args": 1, "kwargs": ["<img src=x>"]}
    report["layers"][1]["output"] = {"kind": "tensor", "shape": 1}
    output = render_report(report, format=output_format)
    assert "&lt;img src=x&gt;" in output
    assert "<img src=x>" not in output


def _maps(source):
    fragments = re.findall(r'<svg\b[^>]*class="module-map"[^>]*>.*?</svg>', source, flags=re.DOTALL)
    return {
        node.attrib["data-group-id"]: node
        for node in (ET.fromstring(fragment) for fragment in fragments)  # ruff: ignore[suspicious-xml-element-tree-usage]
    }


def _node(group, path):
    return next(node for node in group["nodes"] if node["path"] == path)


def _tile_anchor(diagram, node):
    return next((element for element in diagram.iter() if element.attrib.get("data-node-id") == node["id"]), None)


def _tile(diagram, node):
    anchor = _tile_anchor(diagram, node)
    assert anchor is not None
    return next(element for element in anchor if "data-tile" in element.attrib)


def _rect_geometry(rect):
    return tuple(float(rect.attrib[name]) for name in ("x", "y", "width", "height"))


def _diagram_geometry(diagram):
    return {
        element.attrib["data-node-id"]: _rect_geometry(next(child for child in element if "data-tile" in child.attrib))
        for element in diagram.iter()
        if "data-map-node" in element.attrib
    }


def test_actual_layer_costs_partition_nested_rectangles_at_module_depth():
    model = nn.Sequential(nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU()), nn.Linear(8, 2, bias=False))
    nested_report = crawl_module(model, (4,))
    html = render_report(nested_report)
    svg = render_report(nested_report, format="svg")
    group = build_maps(nested_report)["module_flops"]["groups"][0]
    html_map, svg_map = _maps(html)[group["id"]], _maps(svg)[group["id"]]
    assert _diagram_geometry(html_map) == _diagram_geometry(svg_map)
    root, branch, linear, relu, last = [_node(group, path) for path in ("", "0", "0.0", "0.1", "1")]
    assert [linear["direct_known"], relu["direct_known"], last["direct_known"]] == [56, 8, 30]
    root_rect, branch_rect, linear_rect, relu_rect, last_rect = [
        _tile(html_map, node) for node in (root, branch, linear, relu, last)
    ]
    x_root, y_root, width_root, _ = _rect_geometry(root_rect)
    x_branch, y_branch, width_branch, _ = _rect_geometry(branch_rect)
    x_linear, y_linear, width_linear, _ = _rect_geometry(linear_rect)
    x_relu, y_relu, width_relu, _ = _rect_geometry(relu_rect)
    x_last, y_last, width_last, _ = _rect_geometry(last_rect)
    gap = x_last - (x_branch + width_branch)
    assert gap > 0
    assert (width_branch + gap) / (width_last + gap) == pytest.approx((56 + 8) / 30, abs=0.00002)
    assert (width_linear + gap) / (width_relu + gap) == pytest.approx(56 / 8, abs=0.0001)
    assert x_root == x_branch == x_linear
    assert x_relu > x_linear
    assert x_last + width_last == pytest.approx(x_root + width_root, abs=0.002)
    assert y_branch == y_last
    assert y_linear == y_relu
    assert y_linear - y_branch == y_branch - y_root > 0
    for diagram in (html_map, svg_map):
        assert all(
            anchor.attrib["tabindex"] == "0" and anchor.attrib.get("aria-label")
            for anchor in diagram.iter()
            if "data-map-node" in anchor.attrib
        )


def test_reused_module_is_one_tile_with_both_calls_and_parameters_counted_once():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Linear(4, 4, bias=False))
            self.tied = nn.Linear(4, 4)
            self.tied.weight = self.block[0].weight

        def forward(self, inputs):
            return self.tied(self.block(self.block(inputs)))

    report = crawl_module(Shared(), (4,))
    html = render_report(report)
    data = _embedded(Document(html).root)
    group = data["maps"]["module_flops"]["groups"][0]
    node = _node(group, "block.0")
    diagram = _maps(html)[group["id"]]
    anchor = _tile_anchor(diagram, node)
    assert anchor is not None
    assert "\u00d72 calls" in "".join(anchor.itertext())
    assert node["direct"]["value"] == 56
    assert sum(element.attrib.get("data-node-id") == node["id"] for element in diagram.iter()) == 1
    assert [call["result"]["value"] for call in node["calls"]] == [28, 28]
    root = Document(html).root
    inspector = root.find("div", **{"data-view": "module_flops"})[0].find("aside")[0]
    rows = inspector.find("div", **{"class": "call-card"})
    assert len(rows) == 2
    assert [row.find("a")[0].attrs["href"] for row in rows] == [f"#call-{call['index']}" for call in node["calls"]]
    assert "16 newly attributed parameters" in rows[0].text()
    assert "Shared tensors · no new parameter attribution" in rows[1].text()
    assert all("28 FLOPs" in row.text() and "[1, 4]" in row.text() for row in rows)
    parameter_group = data["maps"]["parameters"]["groups"][0]
    assert parameter_group["known_total"] == report["totals"]["parameters"]["value"] == 20
    assert _node(parameter_group, "block.0")["direct"]["value"] == 16
    assert _node(parameter_group, "tied")["direct"]["value"] == 4
    assert _node(parameter_group, "tied")["calls"][0]["shared"] is True
    parameter_svg = render_report(report, format="svg", metric="parameters")
    parameter_map = _maps(parameter_svg)[parameter_group["id"]]
    assert _diagram_geometry(parameter_map) == _diagram_geometry(_maps(html)[parameter_group["id"]])
    svg_root = ET.fromstring(parameter_svg)  # ruff: ignore[suspicious-xml-element-tree-usage]
    svg_text = "".join(svg_root.itertext())
    assert "16 newly attributed parameters" in svg_text
    assert "Shared tensors · no new parameter attribution" in svg_text
    assert all(
        any(element.attrib.get("id") == f"call-{call['index']}" for element in svg_root.iter())
        for call in node["calls"]
    )


def test_partial_positive_tile_is_hatched_and_partial_zero_keeps_unscaled_visible_card():
    class Sine(nn.Module):
        def forward(self, inputs):
            return inputs.sin()

    report = crawl_module(nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False), Sine()), (4,))
    # Preserve a positive recorded lower bound while declaring its formula
    # incomplete. The unsupported Sine gives a real partial-zero measurement.
    report["layers"][1]["metrics"]["module_flops"] = metric_result(
        status="partial", known_value=7, unit="FLOPs", scope="module_call", method="torchscan_module_formula"
    )
    html, svg = render_report(report), render_report(report, format="svg")
    data = _embedded(Document(html).root)
    group = data["maps"]["module_flops"]["groups"][0]
    positive, unknown = [_node(group, path) for path in ("0", "2")]
    assert positive["known"] == 7
    assert positive["width"] == pytest.approx(7 / (7 + 30))
    assert unknown["known"] == 0
    assert unknown["status"] == "partial"
    root = Document(html).root
    operators = root.find("section", id="operators")[0].text()
    assert "partial" in operators
    assert any(
        "aten.sin" in item.text() for item in root.find("li") if item.attrs.get("id", "").startswith("diagnostic-")
    )
    for diagram in (_maps(html)[group["id"]], _maps(svg)[group["id"]]):
        positive_anchor = _tile_anchor(diagram, positive)
        assert positive_anchor is not None
        assert any(child.attrib.get("fill", "").startswith("url(#") for child in positive_anchor)
        assert "partial" in positive_anchor.attrib["aria-label"]
        assert "full value unknown" in positive_anchor.attrib["aria-label"]
        assert _tile_anchor(diagram, unknown) is None  # It has no numeric width, but retains a visible card.
    panel = root.find("div", **{"data-view": "module_flops"})[0]
    unknown_card = panel.find("a", **{"class": "rail-card unknown", "data-node-id": unknown["id"]})[0]
    assert "at least 0 FLOPs" in unknown_card.text()
    assert "full value unknown" in unknown_card.text()
    assert "deliberately unscaled" in unknown_card.text()
    svg_root = ET.fromstring(svg)  # ruff: ignore[suspicious-xml-element-tree-usage]
    cards = {
        node["id"]: next(
            element for element in svg_root.iter() if element.attrib.get("id") == f"rail-{group['id']}-{node['id']}"
        )
        for node in (positive, unknown)
    }
    assert "not sized by cost" in "".join(cards[unknown["id"]].itertext())
    assert "≥ 0 FLOPs · partial" in "".join(cards[unknown["id"]].itertext())
    positive_width = float(cards[positive["id"]][1].attrib["width"])
    unknown_width = float(cards[unknown["id"]][1].attrib["width"])
    assert positive_width == pytest.approx(unknown_width)
    assert unknown_width > 0


def test_before_after_reversal_keeps_union_geometry_and_paired_measurement_bars():
    before = crawl_module(nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False)), (4,))
    after = crawl_module(nn.Sequential(nn.Linear(4, 1, bias=False), nn.Linear(1, 32, bias=False)), (4,))
    forward, reverse = render_report(after, before=before), render_report(before, before=after)
    data = _embedded(Document(forward).root)
    group = data["maps"]["module_flops"]["groups"][0]
    first, second = [_node(group, path) for path in ("0", "1")]
    assert [first["before_known"], second["before_known"]] == [56, 30]
    assert [first["known"], second["known"]] == [7, 32]
    assert first["width"] == pytest.approx(56 / (56 + 32))
    assert second["width"] == pytest.approx(32 / (56 + 32))
    forward_map = _maps(forward)[group["id"]]
    assert _diagram_geometry(forward_map) == _diagram_geometry(_maps(reverse)[group["id"]])
    assert _diagram_geometry(forward_map) == _diagram_geometry(
        _maps(render_report(after, before=before, format="svg"))[group["id"]]
    )
    for node, before_greater in ((first, True), (second, False)):
        anchor = _tile_anchor(forward_map, node)
        bars = [
            float(element.attrib["width"])
            for element in anchor.iter()
            if element.tag.endswith("rect") and element.attrib.get("height") == "7.000"
        ]
        assert len(bars) == 2
        assert (bars[0] > bars[1]) is before_greater
        assert "Complete comparable delta" in anchor.attrib["aria-label"]
    mismatched = copy.deepcopy(before)
    mismatched["context"]["devices"] = ["different-device"]
    withheld = render_report(after, before=mismatched)
    withheld_group = _embedded(Document(withheld).root)["maps"]["module_flops"]["groups"][0]
    assert all(node["delta"] is None for node in withheld_group["nodes"])
    withheld_map = _maps(withheld)[group["id"]]
    assert _diagram_geometry(forward_map) == _diagram_geometry(withheld_map)
    for element in withheld_map.iter():
        if "data-map-node" in element.attrib:
            assert "Delta unknown: devices differs" in element.attrib["aria-label"]
            assert "Δ " not in "".join(element.itertext())
