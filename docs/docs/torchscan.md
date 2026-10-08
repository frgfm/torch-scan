# `torchscan`

This reference covers development APIs, including the #176 preview; [installation](installing.md) distinguishes it from
stable 0.2.0. Start with the [runnable quickstart](index.md) to inspect cost, measure a workload, check a change, and
consume the report. Read [Model and input support](model-support.md) for non-trivial calls.

`summary` and `crawl_module` return an `AnalysisReport` for an evaluation forward. `measure_workload` and `measure_latency`
return `BenchmarkReport` evidence while preserving callable state. Timing is clean; FLOPs, scoped memory, and profiler
evidence use separate passes. Check statuses, methods, and scopes before comparing or consuming numbers.

## Model analysis

::: torchscan.crawl_module

::: torchscan.summary

## Custom module extensions

See [the extension tutorial](extensions.md) for complete examples and subtree ownership.

::: torchscan.ModuleCall

::: torchscan.ModuleEstimates

::: torchscan.ModuleHandler

## Operator FLOPs

::: torchscan.measure_flops

## Model cost comparison

::: torchscan.compare_reports

## Workload timing

`measure_workload` assembles selected FLOPs, timing/throughput, and scoped memory into one report with a readable
terminal summary. `profile=True` adds a separate instrumented pass; RSS requires an explicit `rss_command`.
Use `metrics=("latency", "throughput")` for timing only and `print_summary=False` for automation.
These APIs require the unreleased development version. See [Latency and throughput](metrics.md#latency-and-throughput)
for work-unit throughput, repeated-call behavior, and measurement boundaries. Configure model state, device placement,
precision, and threads yourself. The `inputs` argument records metadata; it is not forwarded to the workload.

::: torchscan.measure_workload

::: torchscan.measure_latency

::: torchscan.BenchmarkReport

See [checked performance experiments](benchmark-comparison.md) for output checks and offline benchmark reports.

::: torchscan.compare_benchmarks

::: torchscan.BenchmarkComparison

## Workload diagnostics

See [workload diagnostics](workload-diagnostics.md) for separate process RAM and profiler passes.

::: torchscan.profile_workload

::: torchscan.ProfileReport

## Consume saved reports

Serialize returned mappings with `json.dumps`. `render_report` renders model or workload HTML without remeasurement;
model reports also support SVG. A saved comparison displays its recorded output check without rerunning it.

::: torchscan.render_report

## Public report types and errors

::: torchscan.AnalysisReport

::: torchscan.LayerReport

::: torchscan.MetricResult

::: torchscan.Diagnostic

::: torchscan.FlopReport

::: torchscan.ReportDiff

::: torchscan.IncompleteAnalysisError
