# TorchScan resource autoresearch

Use the bounded edit/evaluate/keep loop below with the user's existing coding agent. The owner supplies the fixed
experiment, reference implementation, inputs, correctness check, device and resource limit. The agent edits the
candidate implementation.

## Before the loop

1. Select CPU or a CUDA device, a latency objective and memory ceiling, and a maximum number of candidate attempts.
   This attempt limit is separate from `SETTINGS["trials"]`, which repeats each measurement. The committed example
   uses CPU. Use CUDA only on a CUDA-enabled host.
2. Install TorchScan and the example dependencies:

   ```shell
   uv pip install -e ".[model-test]"
   ```

3. Review `experiment.py`, `reference.py`, and `baseline.py` with the owner. Configure the device and objective before
   the campaign. Keep the model, seed, input batch, reference, correctness tolerance, and measurement settings fixed.
   The defaults use four 32 × 32 RGB inputs, FP32, eval/inference mode and one CPU thread.
4. Compare the baseline with itself as a noise control. The checks should pass; the improvement decision should be
   `inconclusive`. This does not establish statistical significance.
5. Create a clean experiment branch and commit the baseline, candidate starter and owner contract before the loop.
   Save an unchanged copy of the starting baseline as `baseline-initial.py` for the final comparison.
6. Run the commands below from `examples/autoresearch/`, or the directory containing your own experiment files.

## Experiment loop

Edit only `candidate.py`. Commit each candidate, then compare it with the current accepted `baseline.py`:

```shell
python -m torchscan.autoresearch experiment.py \
  --baseline baseline.py --candidate candidate.py --output /tmp/torchscan-autoresearch/run-001
```

Each output directory must be new. The example writes output to `/tmp` so reports stay outside the source tree.

- **`keep`**: checks pass, measurements are complete, resource limits hold, and the repeated improvement exceeds the
  configured threshold. Promote the candidate to `baseline.py` and commit it before the next experiment.
- **`discard`**: a reference check fails, a complete measurement crosses a resource limit, or the objective consistently worsens. Record why
  and restore the candidate from its previous commit.
- **`inconclusive`**: evidence is missing or noisy, a worker fails or times out, or the candidate does not clear the
  improvement threshold. Do not advance the baseline. Inspect its saved logs and reports.

Use one change per experiment, preserve reports and stop at the owner's candidate-attempt limit. Rerun the final
accepted candidate against `baseline-initial.py` in a new output directory. Never present an inconclusive measurement
as a gain. Keep the unchanged reference even when promoting a new incumbent.

## Evidence boundaries

This example checks output agreement with a locally initialized ResNet18. It does not measure task accuracy. Latency
is warmed block-average call time, not request p95. Memory is PyTorch-tracked CPU tensors or the CUDA reserved
allocator; it is not total device memory or process RSS. CPU runs provide no CUDA speed evidence. Use
`candidate_wrong.py` and `candidate_memory.py` to see check and resource failures.
