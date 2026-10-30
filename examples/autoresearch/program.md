# Resource autoresearch with an existing coding agent

The owner supplies one model/input contract: `workload.json`. The coding agent changes candidate model code.
TorchScan owns model execution, timing, memory/RSS measurements, output checks and improvement decisions.

## Establish the experiment

1. Install TorchScan and the model's dependencies. The example uses `uv pip install -e ".[model-test]"`.
2. Review `workload.json` with the owner: model/weights, shapes or real input batch, inference, target device,
   objective and constraints. Agree on a maximum number of candidate attempts before starting.
3. Keep that contract, data/weights and original baseline fixed in an isolated experiment branch.
   For the example, preserve `models.py:baseline` and edit only `candidate()` or candidate-specific code.
4. Measure the original model once:

   ```shell
   python -m torchscan.autoresearch examples/autoresearch/workload.json --output /tmp/autoresearch/baseline
   ```

5. Compare baseline with itself using `--candidate-model models.py:baseline`. Never interpret identical-code
   controls as evidence of an optimization. If one promotes, investigate noise before continuing.

## Bounded edit/evaluate loop

Propose one change, commit it, then evaluate its factory against the current incumbent:

```shell
python -m torchscan.autoresearch examples/autoresearch/workload.json \
  --candidate-model models.py:candidate --output /tmp/autoresearch/attempt-001
```

Read the returned JSON and its saved evidence:

Keep FP32 precision fixed while changing architecture or implementation. Do not add internal casts, autocast or
quantization to a candidate. Before searching, the owner can set fixed `check.absolute_error` ceilings on mean,
median, p90, p95, p99 or maximum absolute output error. Read all statistics; p99 can leave large outliers.
Never relax these limits after a failed candidate; the original reference remains fixed.
Without `check`, errors are still measured but do not gate acceptance. Record `output_error_policy`; a resource
winner with `report_only` does not establish numerical equivalence or task quality.

- `keep`: record the confirmed objective gain, output errors and resource tradeoff. Preserve the accepted factory/revision and
  use `--baseline-model` to select it for the next iteration.
- `discard`: record the violated check/constraint or regression, then restore the previous candidate implementation.
- `inconclusive`: preserve logs and do not promote. Inspect the reason; spend another attempt only within the budget.

Every output directory must be new. Keep a ledger with revision, hypothesis, decision and report path.
The candidate-attempt budget is separate from paired `settings.trials` within each evaluation. Stop at the agreed
attempt limit. Never weaken output checks, process fewer inputs, or change reference data to improve the score.

Rerun the final candidate against the unchanged original baseline in a new directory. Return the accepted code
revision, confirmed measurements, tradeoffs and unresolved limits. Synthetic checks on locally initialized weights
establish runtime behavior and output tolerance; they do not establish task accuracy or another device's performance.
