# ruff: file-ignore[non-empty-init-module]
from importlib import import_module as _import_module
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchscan import modules, process, utils
    from torchscan.benchmark import *
    from torchscan.benchmark_compare import *
    from torchscan.capture import *
    from torchscan.compare import *
    from torchscan.crawler import *
    from torchscan.extensions import *
    from torchscan.flops import *
    from torchscan.profiler import *
    from torchscan.render import *
    from torchscan.replay import *
    from torchscan.report import *
    from torchscan.workload import *

__version__ = _version("torchscan")

# Keep saved-data analysis usable without importing the workload runtime.
_EXPORTS = {
    "BenchmarkReport": "benchmark",
    "measure_latency": "benchmark",
    "BenchmarkComparison": "benchmark_compare",
    "compare_benchmarks": "benchmark_compare",
    "ReportDiff": "compare",
    "compare_reports": "compare",
    "crawl_module": "crawler",
    "summary": "crawler",
    "ModuleCall": "extensions",
    "ModuleEstimates": "extensions",
    "ModuleHandler": "extensions",
    "FlopReport": "report",
    "measure_flops": "flops",
    "ProfileReport": "profiler",
    "profile_workload": "profiler",
    "render_report": "render",
    "AnalysisReport": "report",
    "Diagnostic": "report",
    "IncompleteAnalysisError": "report",
    "LayerReport": "report",
    "MetricResult": "report",
    "TokenDependency": "report",
    "metric_result": "report",
    "measure_workload": "workload",
    "OperatorCapture": "report",
    "analyze_capture": "replay",
    "capture_workload": "capture",
}
__all__ = ["__version__", "modules", "process", "utils", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in {"modules", "process", "utils"}:
        value = _import_module(f".{name}", __name__)
    elif name in _EXPORTS:
        value = getattr(_import_module(f".{_EXPORTS[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
