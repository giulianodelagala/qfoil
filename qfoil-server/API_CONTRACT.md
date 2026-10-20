# HTTP contract for qfoil-server

## Scope

This contract exposes the approved QFOIL Python API over HTTP. It does not
define or expose the proposed declarative query language.

The service accepts and returns JSON. QFOIL tuples are represented as JSON
arrays and unobserved values as `null`:

```json
[1, null, 0]
```

Incoming arrays are converted recursively to Python tuples before invoking
QFOIL. Tuple results are serialized back to JSON arrays by FastAPI.

## Versioning

Functional endpoints use the `/v1` prefix. `/health`, `/docs` and
`/openapi.json` are operational endpoints and are not versioned.

## Endpoints

### `GET /health`

Response `200`, no model active:

```json
{
  "status": "ok",
  "service": "qfoil-server",
  "model_status": "unloaded"
}
```

Response `200`, model active:

```json
{
  "status": "ok",
  "service": "qfoil-server",
  "model_status": "ready",
  "model_id": "mux"
}
```

`model_id` is present only once a model has been requested. No probabilistic
model being loaded does not mean the service is degraded: `SIZE`, `FULL` and
LaTeX rendering always work.

### `GET /v1/config`

Read-only service configuration for evaluation and search options. Available with or
without a loaded model. Response `200`:

```json
{
  "evaluation": {
    "defaults": {"mode": "heuristic", "sigma_search_budget": 64, "evaluation_budget": null, "rng_seed": null, "seed": null},
    "modes": ["heuristic", "exact"],
    "max_sigma_search_budget": 20000000,
    "max_evaluation_budget": 20000000
  },
  "search": {
    "defaults": {"budget": 1000, "inner_budget": 8, "pool_size": 30},
    "max": {"budget": 100000, "inner_budget": 1000, "pool_size": 1000}
  },
  "evaluation_budget": 20000000,
  "sigma_search_budget": 64
}
```

These values come from `config.py`. Top-level `evaluation_budget` and
`sigma_search_budget` are informational service defaults. Evaluation request
`options.evaluation_budget` overrides the global work limit for that request
only, from 0 to `evaluation.max_evaluation_budget` (20,000,000). Omitting it
or passing null retains the runtime default. `options.sigma_search_budget`
accepts 0 to `evaluation.max_sigma_search_budget` (20,000,000). This endpoint does not maintain session configuration or change the
shared runtime. There is no HTTP configuration update endpoint.

### `POST /v1/expressions/evaluate`

Evaluation accepts an optional `options` object:

```json
{
  "expression": "(SIGMA z. FULL(z)) >= 1",
  "bindings": {},
  "options": {
    "mode": "heuristic",
    "sigma_search_budget": 1000,
    "rng_seed": 38,
    "seed": null
  }
}
```

`mode` is `heuristic` (default) or `exact` (exhaustive). HTTP `exact`
uses the runtime's `restricted` procedure: mandatory product factors
`FULL`, `COMP`, subsumption and `SIZE` bounds restrict the generated
domain. It still computes an exact result; unrecognized restrictions
fall back to the full domain. Conditions within negation or sums do
not narrow the domain. Python `sigma_mode="exact"` retains total-domain
enumeration; it is not exposed over HTTP. All nested queries
inherit the mode. `sigma_search_budget` defaults to 64 and accepts integers
from 0 to 20000000; it applies per heuristic SIGMA attempt. Zero does not select
exact mode. `rng_seed` is an optional integer for reproducible random starts,
moves and restarts. `seed` is a separate optional initial tuple for a direct
SIGMA comparison; its entries must be 0, 1 or null and its length must match
the active tuple domain. Unknown options and invalid values return 422.

Options apply only to this request and are restored after success or error.
Exact mode uses the request `evaluation_budget` override or the service default and returns
`inconclusive` if it exhausts that limit. `/v1/config` exposes `evaluation`
with effective `defaults`, supported `modes`, `max_evaluation_budget` and
`max_sigma_search_budget`.
The existing `search` configuration and witness-search endpoint are retained.


Request:

```json
{
  "expression": "SIZE(x) <= limit",
  "bindings": {
    "x": [1, null, 0],
    "limit": 2
  }
}
```

Response `200`:

```json
{
  "status": "decided",
  "value": 1,
  "last_witness": [null, 1, 0, null],
  "reason": null
}
```

`value` is the result of the expression. `last_witness` is the last verified
witness produced by a SIGMA comparison during this evaluation, found by the
heuristic search (the default procedure; exact mode can also produce witnesses). It is
`null` when this evaluation produced no witness:

```json
{
  "status": "decided",
  "value": 0,
  "last_witness": null,
  "reason": null
}
```

This field does not trigger a new witness search and does not prove that no
witness exists when it is `null`. It describes only the current request: the
server clears any witness left by an earlier evaluation before evaluating the
expression. Use `POST /v1/witnesses/search` for an independent witness search.

An uncertified evaluation returns HTTP `200`, not a false decision:

```json
{
  "status": "inconclusive",
  "value": null,
  "last_witness": null,
  "reason": "bounded_search_incomplete"
}
```

`null` here is an unknown result, never logical false, zero or a verified
witness. The runtime propagates unknown subresults conservatively through
every operation -- negation, comparisons, sums, products, function
arguments, tuples and nested SIGMA alike -- so this never silently becomes a
decided answer. `decided` also covers conclusive numeric expression values,
not just Boolean decisions.

Nested SIGMA execution is supported: this service runs qfoil-parser-evaluator's
*heuristic* execution mode by default (or exhaustive *exact* mode when
requested through evaluation options). Heuristic mode never enumerates -- in evaluation,
nested subqueries, logic-defined functions, witness-search scoring and
witness verification alike -- so a nested decision it cannot certify
through a verified witness returns `inconclusive` with
`reason: "bounded_search_incomplete"` instead of either enumerating or
guessing. A second, independent reason covers a different limit:

```json
{
  "status": "inconclusive",
  "value": null,
  "last_witness": null,
  "reason": "evaluation_budget_exhausted"
}
```

This means the shared global evaluation budget for the request ran out
before a decision could be certified -- it bounds the *entire* operation
(flat evaluation, every nested subquery, logic function, heuristic scoring
step and witness verification), not any one of them independently, and
there is no separate sub-budget per subquery. It is configured once for the
service, with a per-request `options.evaluation_budget` override up to
20,000,000, and is independent from `sigma_search_budget` (the heuristic's own
per-attempt budget, 64 by default) and from the witness-search `budget`
below. Neither reason is a promise about exactly how much work was spent or
that no witness exists -- both just mean this request could not certify one.

The SIGMA tuple domain is configured by the active runtime, never inferred
from request bindings. With a loaded model its dimension is
`model.num_features`; without a loaded model, the backend-independent runtime
uses its configured `instance_len`. Bindings contain only free variables, so a
variable bound by SIGMA must not be included. Any tuple array that is present
in `bindings` must have exactly `num_features` entries when a model is loaded.
Clients must not send a separate `domain_size` field.

### `POST /v1/expressions/latex`

Request:

```json
{
  "expression": "SIGMA z. FULL(z) * (z <<= x)"
}
```

Response `200`:

```json
{
  "latex": "\\sum_{z} \\left( ... \\right)"
}
```

The string contains neither math delimiters nor HTML.

### `POST /v1/witnesses/search`

Request:

```json
{
  "expression": "FULL(z) * (z <<= x)",
  "variable": "z",
  "bindings": {
    "x": [1, 0, 1]
  },
  "options": {
    "budget": 1000,
    "pool_size": 30,
    "inner_budget": 8,
    "seed": null
  }
}
```

Response `200`:

```json
{
  "status": "found",
  "variable": "z",
  "witness": [1, 0, 1],
  "verified": true,
  "seed_used": null,
  "search": {
    "budget": 1000,
    "pool_size": 30,
    "inner_budget": 8
  }
}
```

`inconclusive` replaces the former `not_found` state. It means that the
heuristic returned without a
verified witness. It does not prove that a witness does not exist.
The response has `witness: null`, `verified: false`, and
`reason: "bounded_search_incomplete"`; successful searches have `reason: null`.
The reason covers budget limits and unsupported heuristic scoring; it does
not claim a measured number of evaluations or demonstrated absence. The
expression may itself contain SIGMA: the body is searched over the same
heuristic mode as evaluation, never by enumeration, so a nested subquery it
cannot certify also reports `inconclusive` rather than guessing.
`reason: "evaluation_budget_exhausted"` is possible here too, with the same
meaning as in `POST /v1/expressions/evaluate`, covering this search's own
verification step as well; it is independent from the `budget`/`pool_size`/
`inner_budget` options above.

`options` is optional; omitting it or sending `{}` uses the defaults exposed
by `GET /v1/config`. Partial options use defaults for omitted fields. Valid
ranges are `0 <= budget <= 100000`, `0 <= inner_budget <= 1000`, and
`1 <= pool_size <= 1000`. Unknown fields (including `evaluation_budget` and
`sigma_search_budget`) and out-of-range values return the standard FastAPI
`422` validation response. Options apply only to that request and do not
change service defaults or the shared runtime. In both `found` and
`inconclusive` responses, the existing `search` field reports the effective
`budget`, `inner_budget`, and `pool_size` passed to the search, including
defaults for omitted options; these are limits, not measured work counts.

## Model lifecycle

The server always starts with no model loaded. `SIZE`, `FULL`
and other backend-independent operations, and LaTeX rendering, always work,
with or without a model. `VAL`, `COND`, `EP` and `CEP` require a model to be
loaded first.

At most one model is active per process. These endpoints are a proof-of-concept
HTTP lifecycle; the declarative language's future `LOAD MODEL` / `UNLOAD MODEL`
statements will be another interface over the same lifecycle manager, not a
replacement for it.

### Model catalog

A static catalog of loadable models is configured through one environment
variable:

```
QFOIL_MODEL_CATALOG=/path/to/models.json
```

```json
{
  "mux": {
    "backend": "pyjuice",
    "path": "/srv/qfoil/models/mux",
    "device": "cuda:0",
    "min_batch_size": 4
  },
  "spam": {
    "backend": "pyjuice",
    "path": "/srv/qfoil/models/spam",
    "device": "cuda:0",
    "min_batch_size": 4
  }
}
```

The catalog is parsed and structurally validated when the server starts.
Nothing is loaded at startup. `path` never leaves the process: it is not
accepted from a client and never appears in an HTTP response. Clients only
ever refer to a model by its catalog key (`model_id`).

If `QFOIL_MODEL_CATALOG` is unset, the catalog is empty and `GET /v1/models`
returns no entries; loading is still fully rejected with `unknown_model`, not
skipped.

### States

```
unloaded -> loading -> ready -> unloading -> unloaded
              \-> unloaded (on load failure)
```

`error` is a defensive state reserved for an internal inconsistency; it is
not reachable through the documented request/response flow above.

### `GET /v1/models`

Lists the catalog without loading anything.

Response `200`:

```json
{
  "models": [
    { "id": "mux", "backend": "pyjuice" },
    { "id": "spam", "backend": "pyjuice" }
  ]
}
```

### `GET /v1/models/current`

Response `200`, no model active:

```json
{ "model": { "status": "unloaded" } }
```

Response `200`, model active:

```json
{
  "model": {
    "id": "mux",
    "status": "ready",
    "backend": "pyjuice",
    "num_features": 8,
    "num_classes": 2,
    "device": "cuda:0",
    "loaded_at": "2026-09-24T17:32:00Z"
  }
}
```

Fields that don't apply to the current status (e.g. `id` while `unloaded`)
are omitted rather than sent as `null`.

### `POST /v1/models/{model_id}/load`

`model_id` must be a catalog key; it is never a filesystem path. Loading
blocks the HTTP request until the model is ready or the load fails --
consult `GET /v1/models/current` to poll `loading`/`unloading` status from a
concurrent request.

Response `200` (same shape as `GET /v1/models/current` with a model active).

### Test dataset samples

These read-only endpoints operate only on the test split associated with the
currently loaded model:

```
GET /v1/models/current/dataset/test/samples/{index}
GET /v1/models/current/dataset/test/samples/random
GET /v1/models/current/dataset/test/samples/random?seed=123
```

`index` is a zero-based, stable position in the active test dataset. Its valid
range is `0 <= index < dataset_size`; it is not a database identifier. The
random endpoint chooses one index from the same range. `seed` is an optional
base-10 integer. Selection uses a new local pseudo-random generator for every
request, so the same integer seed selects the same index while the model and
ordered dataset are unchanged. Omitting `seed` uses system randomness. Neither
form maintains a cursor or changes model, dataset, inference, or navigation
state.

Both endpoints return `200` with exactly the same schema:

```json
{
  "model_id": "mnist_binary_small",
  "split": "test",
  "index": 42,
  "dataset_size": 12665,
  "sample": [1, null, 0],
  "expected": 1
}
```

`split` is always `"test"`. `sample` is the already transformed representation
stored for inference (for the current PyJuice artifacts, the binarized feature
tensor), not the original source image or row. It is therefore valid as a
QFOIL tuple binding. A one-dimensional tuple/tensor becomes a JSON array and
unobserved values remain `null`. Nested or multi-component samples preserve
their list/object structure; the server does not flatten them. `expected` is
the aligned test target. Backend tensors, NumPy-like arrays and scalar objects
are recursively converted to JSON arrays, objects, numbers, booleans, strings,
or `null`. Non-finite numbers, objects without such a representation, and
objects with non-string map keys are rejected rather than encoded implicitly.

Dataset access is backend-neutral. A loaded backend may supply an indexed
`(sample, expected)` provider. The PyJuice adapter recognizes a single
`*_test_data*.pt` tensor paired with `test_targets.pt` beside the catalog model;
models without that complete pair load normally but report
`test_dataset_unavailable`. The tensor loader requests memory mapping when the
installed Torch/file format supports it, and each HTTP request indexes only one
item rather than serializing the complete split. Filesystem paths are never
accepted by or returned to clients.

Examples:

```bash
curl http://localhost:8000/v1/models/current/dataset/test/samples/0
curl 'http://localhost:8000/v1/models/current/dataset/test/samples/random?seed=123'
```

### `DELETE /v1/models/current`

Unloads the active model and restores the backend-independent runtime.
Idempotent: calling it with no model active still returns `200`.

Response `200`:

```json
{ "model": { "status": "unloaded" } }
```

### Example: load, query, unload

```bash
curl -X POST http://localhost:8000/v1/models/mux/load
# {"model": {"id": "mux", "status": "ready", "backend": "pyjuice", ...}}

curl -X POST http://localhost:8000/v1/expressions/evaluate \
  -H 'content-type: application/json' \
  -d '{"expression": "EP(x, c)", "bindings": {"x": [null, null, null, null, null, null, null, null], "c": 0}}'
# {"value": 0.42, "last_witness": null}

curl -X DELETE http://localhost:8000/v1/models/current
# {"model": {"status": "unloaded"}}
```

## Errors

Errors generated by QFOIL and the model lifecycle use the same envelope:

```json
{
  "error": {
    "code": "qfoil_syntax_error",
    "message": "..."
  }
}
```

Initial mapping:

| HTTP status | Code | Meaning |
| --- | --- | --- |
| `400` | `qfoil_syntax_error` | Invalid QFOIL expression |
| `422` | `qfoil_validation_error` | Invalid binding or search option |
| `501` | `unsupported_operation` | Valid but unsupported operation |
| `500` | `qfoil_error` | Other QFOIL failure |
| `404` | `unknown_model` | `model_id` is not in the catalog |
| `409` | `model_already_loaded` | A model is already active; unload it first |
| `409` | `model_busy` | Another load or unload is already in flight |
| `501` | `backend_unavailable` | Unknown backend, or the selected backend’s optional dependencies are missing |
| `500` | `model_load_error` | The backend rejected the model, or failed to build a runtime |
| `409` | `model_not_loaded` | No model is currently active |
| `404` | `test_dataset_unavailable` | The active model has no complete indexed test dataset |
| `409` | `test_dataset_empty` | The active test dataset exists but contains no samples |
| `404` | `test_dataset_index_out_of_range` | The non-negative index is not present in the test dataset |
| `500` | `test_dataset_serialization_error` | The selected sample or target cannot be represented as strict JSON |

For these sample routes, a malformed integer or a negative `index`, and a
malformed `seed`, return `422 qfoil_validation_error`. A syntactically valid
non-negative index outside the dataset uses `404` because that indexed sample
does not exist. Missing model and empty-dataset conditions use `409` because
the current server state cannot fulfill an otherwise valid request. A model
without an exposed test resource and a missing indexed resource use `404`.
Serialization failure is an internal model/artifact contract violation and
therefore uses `500`.

Error messages never contain a filesystem path or a stack trace. A load
failure is logged with full detail on the server side and reported to the
client only as `model_load_error` with a generic message.

FastAPI request-schema failures retain its standard `422` response.

## Proof-of-concept limits

Execution no longer limits SIGMA nesting depth: both evaluation and witness
search accept nested SIGMA, including nesting introduced by QFOIL-defined
functions. This is safe because the service defaults to qfoil-parser-evaluator's heuristic
execution mode, and heuristic mode never enumerates the domain
-- in evaluation, nested subqueries, logic functions, witness-search scoring
or witness verification -- it reports `inconclusive` instead whenever it
cannot certify a decision through a verified witness (see
`POST /v1/expressions/evaluate` above). Exact mode (attempts to decide by
enumeration) is available as a reference for small domains, selected
via `sigma_mode="exact"` in Python or `options.mode="exact"` in HTTP evaluation.
A verified witness establishes validity, not minimality. LaTeX rendering
accepts nested formulas, as before.

The one fragment restriction that remains is unrelated to nesting:
recursive logic-function definitions (directly or indirectly, through
another function's body) are rejected with `501 unsupported_operation`
before execution, because qfoil-parser-evaluator does not bound recursive
function-call depth at evaluation time.

A shared global evaluation budget bounds the total work of one request --
flat evaluation, every nested subquery, logic function, heuristic-search
step and witness verification together, not any one of them independently.
Its default is configured for the service; evaluation requests can override
it through `options.evaluation_budget` up to 20,000,000. It is
independent from `sigma_search_budget` (the heuristic's own per-attempt
budget, 64 by default), the witness-search `budget` below, and
`inner_budget`.

- Expression length: 10,000 characters.
- Tuple/array length at any nesting level: 10,000 elements.
- Search budget: at most 100,000 evaluations.
- Pool size: at most 1,000.
- Inner budget: at most 1,000.
- No endpoint exposes `define_function()`.
- No authentication, persistence or asynchronous job management is included.
- Exactly one model may be active per process; no GPU worker
  pool, queue, or multiple resident models.
- `torch.cuda.empty_cache()` is never called automatically on unload.

Deployment must add request-body limits, concurrency limits and timeouts at the
ASGI server or reverse proxy boundary.


## Random Forest backend contract

The existing catalog, load/unload, evaluate, witness and dataset endpoints also
accept catalog entries with `backend: "random_forest"`. The server uses the
same single active model lifecycle. Metadata reports the fitted feature count,
the number of integer labels (not their largest value plus one), and
`device: "cpu"`. The local MNIST 3/8 RF has 196 features and 2 classes.

For this backend:

- `VAL(x)` returns the Python integer classifier label when `x` is FULL,
  has exactly `num_features` positions, and all values are numeric 0 or 1.
  Boolean values are rejected. Length/domain violations produce HTTP 422.
- A valid partial tuple (one or more `null` values) returns numeric integer
  `0`, without invoking the classifier. Length and domain validation still
  apply to partial tuples. This value is not a Boolean decision.
- Classification predicates explicitly use `FULL(x) * (VAL(x) == 3)`.
- `VAL(x,c)`, `COND`, `EP` and `CEP` raise `UnsupportedOperationError` with
  `Consulta no soportada por Random Forest`, mapped by the existing handler to
  HTTP 501 with code `unsupported_operation`. This applies also in SIGMA
  bodies and defined-function wrappers that heuristic evaluation might skip.
- Backend-independent operations and PC probability semantics are unchanged.

The RF test split uses paired NumPy arrays (`allow_pickle=False`, memory-mapped
for indexed access), returned through the existing `TestDataset` interface.
No client-side query capability validation or new endpoints are introduced.
HTTP evaluation retains its 64-candidate heuristic SIGMA budget and does not
accept an evaluation seed. A full 196-feature classification witness should
be requested through the existing seeded witness search endpoint; default
unseeded SIGMA evaluation may return `inconclusive`. Python SIGMA evaluation
can receive a seed through the existing `QFOIL.evaluate(..., seed=...)` API.
