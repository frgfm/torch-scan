# `torchscan`

This reference follows the development version on `main`. Read [Model and input support](model-support.md) before
using non-trivial calls and [Understanding results](metrics.md) before comparing measurements.

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

## Report comparison

::: torchscan.compare_reports

## Workload timing

This API requires the unreleased development version. See [Latency and throughput](metrics.md#latency-and-throughput)
for a runnable example and measurement boundaries.

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

## Offline visual reports

::: torchscan.render_report

## Public report types and errors

::: torchscan.AnalysisReport

::: torchscan.LayerReport

::: torchscan.MetricResult

::: torchscan.Diagnostic

::: torchscan.FlopReport

::: torchscan.ReportDiff

::: torchscan.IncompleteAnalysisError
