# Local model reports

From the repository root, run `python scripts/visual_report.py`. Open `explorer.html` in a browser or open
`explorer.svg` for a static view. Each model also has a source JSON report. Generated files are ignored by Git;
the generator uses small CPU models with locally initialized weights and downloads nothing.

| Files (`.html` / `.svg`) | Example |
| --- | --- |
| `explorer` | Nested convolution model on input `[1, 3, 16, 16]`. A convolution runs twice. A custom `Sine` probe has a zero lower bound and unknown full cost. |
| `explorer-complete` / `explorer-slim` | Complete module estimates with channel widths 8 / 4. |
| `explorer-comparison` | Before/after marks use fixed module positions and a common scale. |
| `shared` | A block runs twice; another layer shares its weight. Compute repeats; parameters count once. |
| `structure` | Shapes and parameters are available; compute is explicitly unavailable. |

Operator support can vary with PyTorch. The convolution examples may have partial operator counts for `AvgPool2d`,
even when module estimates are complete. The reports record software/input context and establish neither latency
nor measured peak memory. See [the guide](../../docs/docs/visual-report.md) for the API, keyboard controls, and status rules.
