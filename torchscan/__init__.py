# ruff: file-ignore[non-empty-init-module]
from importlib import import_module as _import_module
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

__version__ = _version("torchscan")

_MODULE_EXPORTS = {
    "benchmark": ("BenchmarkReport", "measure_latency"),
    "benchmark_compare": ("BenchmarkComparison", "compare_benchmarks"),
    "compare": ("ReportDiff", "compare_reports"),
    "crawler": ("crawl_module", "summary"),
    "extensions": ("ModuleCall", "ModuleEstimates", "ModuleHandler"),
    "flops": ("FlopReport", "measure_flops"),
    "profiler": ("ProfileReport", "profile_workload"),
    "render": ("render_report",),
    "report": (
        "AnalysisReport",
        "Diagnostic",
        "IncompleteAnalysisError",
        "LayerReport",
        "MetricResult",
        "TokenDependency",
        "metric_result",
    ),
    "workload": ("measure_workload",),
}
_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}
__all__ = ["__version__", "modules", "process", "utils", *_MODULE_EXPORTS, *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS or name in {"modules", "process", "utils"}:
        value = _import_module(f".{name}", __name__)
    elif name in _EXPORTS:
        value = getattr(_import_module(f".{_EXPORTS[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
