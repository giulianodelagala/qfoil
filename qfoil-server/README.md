# qfoil-server

Thin FastAPI adapter for the QFOIL Python API. The service owns HTTP, JSON,
request limits and process lifecycle; parsing, evaluation, witness search and
LaTeX generation remain in `qfoil`.

This is the HTTP-adapter component of the proof-of-concept prototype for
*Semirings Meet Interpretability: A Quantitative Logic for Model
Explanations* (Pinto & Arenas). The language runtime (`qfoil-parser-evaluator`)
and web client (`qfoil-ui`) are separate repositories.

See the [QFOIL project overview](https://github.com/giulianodelagala/qfoil)
for the component architecture, joint setup and the coordinated
`article-v1.0.0` source snapshot.

This proof of concept deliberately excludes the proposed declarative query
language, remote function registration, authentication and persistent jobs.
The current endpoints execute QFOIL inline; production scheduling and
isolation of CPU/GPU-bound work remain deployment concerns.

Evaluation defaults to heuristic mode, which never enumerates the domain
and returns `inconclusive` when it cannot certify a result. Per-request
`options.mode: "exact"` enables exact enumeration with restricted domains
(the runtime's `restricted` mode), using mandatory FULL, COMP, subsumption
and SIZE factors to prune candidates. Total-domain `exact` remains
available in the Python API. HTTP evaluation includes nested
queries, under a shared limit of 20,000,000 evaluator node visits per
operation. See `API_CONTRACT.md` for random seeds and local search budgets.

## Development setup

Server evaluation budgets, witness-search defaults and their maximum HTTP
limits are centralized in [config.py](src/qfoil_server/config.py). Changes
require restarting the server. Clients can override witness-search defaults
through request `options`, within those limits; the global evaluation budget
and SIGMA evaluation search budget have server defaults. Evaluation requests
can set `options.evaluation_budget` and `options.sigma_search_budget` from 0
to 20,000,000. The total budget applies to both modes and all nested queries;
request overrides are restored afterwards.

Install QFOIL from the sibling repository first, then install this service:
The runtime repository is now `qfoil-parser-evaluator` (formerly
`SigmaParser`); its Python package remains `qfoil`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../qfoil-parser-evaluator
pip install -e ".[test]"
```

To load real PyJuice models via `/v1/models/{model_id}/load`, also install
the `pyjuice` extra (pulls in Torch and PyJuice):

```bash
pip install -e ".[pyjuice]"
```

A plain `pip install -e ".[test]"` never installs Torch or PyJuice, and the
server never imports either until a client actually requests a load.

Run the server:

```bash
qfoil-server
```

or:

```bash
uvicorn qfoil_server.main:app --reload
```

Run the tests:

```bash
pytest
```

For a per-file pass/fail/skip summary instead of raw pytest output, run
`python scripts/run_validation_report.py`.

The interactive OpenAPI documentation is available at `/docs`. See
[`API_CONTRACT.md`](API_CONTRACT.md) for the transport contract and limits.

## Current endpoints

```text
GET    /health
POST   /v1/expressions/evaluate
POST   /v1/expressions/latex
POST   /v1/witnesses/search
GET    /v1/models
GET    /v1/models/current
POST   /v1/models/{model_id}/load
DELETE /v1/models/current
```

The server always starts with no model loaded; `SIZE`, `FULL`
and LaTeX rendering work either way. `VAL`, `COND`, `EP` and `CEP` need a
model, loaded on demand through the endpoints above -- never at startup and
never by passing a pre-built probabilistic `QFOIL` into `create_app()`.

Which models can be loaded is configured with one environment variable
pointing at a catalog file:

```bash
export QFOIL_MODEL_CATALOG=/path/to/models.json
```

```json
{
  "mux": {
    "backend": "pyjuice",
    "path": "/srv/qfoil/models/mux",
    "device": "cuda:0",
    "min_batch_size": 4
  }
}
```

```bash
curl -X POST http://localhost:8000/v1/models/mux/load
curl http://localhost:8000/v1/models/current
curl -X POST http://localhost:8000/v1/expressions/evaluate \
  -H 'content-type: application/json' \
  -d '{"expression": "EP(x, c)", "bindings": {"x": [null, null], "c": 0}}'
curl -X DELETE http://localhost:8000/v1/models/current
```

See [`API_CONTRACT.md`](API_CONTRACT.md#model-lifecycle) for the full
lifecycle, states and error codes. These HTTP endpoints are the whole
lifecycle surface for this proof of concept; a future declarative `LOAD
MODEL` / `UNLOAD MODEL` statement will call into the same `ModelManager`
rather than replace it.

Tests never need Torch, PyJuice or CUDA: inject a `ModelManager` built with
an explicit catalog and a fake loader instead of relying on
`QFOIL_MODEL_CATALOG` or the real PyJuice backend:

```python
from qfoil_server import create_app
from qfoil_server.model_manager import ModelManager

manager = ModelManager(catalog=my_test_catalog, loader=my_fake_loader)
app = create_app(model_manager=manager)
```

To customize only the backend-independent runtime (used before any model is
loaded and restored after every unload), pass `qfoil=` instead:

```python
from qfoil import QFOIL
from qfoil_server import create_app

app = create_app(qfoil=QFOIL(instance_len=8))
```

## Random Forest oracle

Install the optional backend in both editable projects (`qfoil[random_forest]`
and `qfoil-server[random_forest]`). Parser and PC use do not import sklearn.
The default loader dispatches `backend: "pyjuice"` or `"random_forest"` through
one active model slot and the existing endpoints.

An RF directory contains `model.joblib` (a trusted local fitted sklearn
classifier), `metadata.json` with `backend`, `num_features`, `num_classes`, and
`class_ids`, plus optional paired `test_samples.npy` and `test_targets.npy`.
RF runs on CPU. Its `VAL(x)` returns an integer label for complete binary
instances; valid partial instances return numeric `0` without prediction.
Use `FULL(x) * (VAL(x) == 3)` for classification. `VAL(x,c)`, `EP`, `CEP`, and
`COND` return HTTP 501 / `unsupported_operation` for RF.

The local catalog includes `mux_random_forest` in place of MNIST RF.
Its eight inputs are `S0, S1, D0, D1, D2, D3, N0, N1`, with label
`D[2*S1 + S0]` in {0, 1}; the last two inputs are irrelevant noise bits.
Use `FULL(x) * (VAL(x) == 1)` to query class 1. Rebuild from the parser
repository on CPU:

```bash
conda run -n pc python -m examples.models.mux_random_forest.train
```

The artifact contains the fitted forest, metadata, and all 256 binary inputs
with their labels. The forest trains on this complete truth table and is
verified against every entry, including invariance to the noise bits. This
is an exhaustive rule check, not a held-out accuracy measurement. MNIST RF
artifacts and earlier reproduction evidence remain available outside the
active catalog.

## Implementation

The implementation was developed with assistance from generative AI models.
