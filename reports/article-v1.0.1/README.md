# Validation of article-v1.0.1 public preparation

The checks below ran on the exported public components on 23 October 2026.
[validation.json](validation.json) records source snapshots, import provenance,
software versions and per-suite outcomes. The final documentation commit only
adds this evidence and README/manifest updates to the tested import commits.

| Check | Result | Evidence |
| --- | --- | --- |
| Full evaluator suite with CUDA | 883 passed, no skips; 1115.87 s (18 min 35 s) | [log](parser-gpu-full.log), [JUnit XML](parser-gpu-full.xml) |
| Full evaluator suite without CUDA access | 736 passed, 147 skipped | [log](parser-cpu.log), [JUnit XML](parser-cpu.xml) |
| Article MUX protocol with CUDA | 32 passed, no skips | [per-case log](mux-gpu.log), [JUnit XML](mux-gpu.xml) |
| Server suite with the public evaluator | 110 passed, 1 skipped | [log](server-default.log), [JUnit XML](server-default.xml) |
| Configured server CUDA lifecycle | 1 passed | [log](server-cuda.log), [JUnit XML](server-cuda.xml) |
| Clean UI dependency installation | Successful | [npm ci log](ui-install.log) |
| UI suite | 129 passed across 5 files | [log](ui-test.log) |
| UI TypeScript and Vite build | Successful | [log](ui-build.log) |
| Joint Python package installation | Successful | [log](install.log) |
| HTTP without a loaded model | Health, SIZE evaluation and LaTeX rendering passed | [responses](http-smoke.json) |

The evaluator's 147 skips are CUDA integration tests in the restricted CPU
execution. The separate CUDA MUX run executes all 16 configurations, each with
an independent reference-set check and a count/decision/witness integration
check. Each integration check exercises both total and restricted enumeration.

The default server run skips its optional CUDA lifecycle test because no catalog
was configured. That test then passed separately with the bundled MNIST 3/8
PCs, class 3, and a temporary catalog pointing to this checkout. No new model
artifacts or local server evidence directories were exported.

The full evaluator suite was rerun with CUDA on public commit `8e51ead`,
with a 2400-second limit. It completed successfully in 1115.87 seconds:
883 passed, zero skipped, zero failures and zero errors. Its
[full log](parser-gpu-full.log) and [JUnit XML](parser-gpu-full.xml) are archived.
The earlier attempt had been interrupted by a 240-second limit (exit 124),
shorter than the suite's normal runtime. Its
[partial output](parser-gpu-full-incomplete.log) is retained as historical evidence;
the completed rerun supersedes that incomplete validation status.

## Reproduction

Activate an environment with the optional runtime and server test dependencies.
Run each Python suite from its own component directory using the same interpreter:

```bash
python -m pytest tests/ -rs
```

For the server, ensure it imports the public runtime, for example from
`qfoil-server/` with `PYTHONPATH=../qfoil-parser-evaluator:src`.
The article-specific command is run from `qfoil-parser-evaluator/`:

```bash
python -m pytest tests/mux/test_article_validation.py -v -ra --junitxml=junit.xml
```

For the optional server CUDA lifecycle test, configure a catalog as explained
in the server README and set `QFOIL_MODEL_CATALOG`, `QFOIL_CUDA_MODEL_ID`, and
`QFOIL_CUDA_CLASS_ID` (3 for the MNIST 3/8 example), then run
`python -m pytest tests/test_cuda_lifecycle.py -v -ra` from `qfoil-server/`.

From `qfoil-ui/`, run `npm ci`, `npm test`, and `npm run build`.
The joint installation check used an isolated temporary Python environment
with the installed backend dependencies available, installed both exported
Python packages without downloading dependencies, and verified the loaded
module paths before making the three HTTP requests.

The original article MUX run is also archived under
[`qfoil-parser-evaluator/reports/mux-20261023-115138`](../../qfoil-parser-evaluator/reports/mux-20261023-115138/README.md).
