# qfoil-ui

Minimal web interface for the QFOIL proof of concept. The project consumes the
HTTP API exposed by `qfoil-server` and uses KaTeX only to render LaTeX generated
by QFOIL.

This is the web-client component of the proof-of-concept prototype for
*Semirings Meet Interpretability: A Quantitative Logic for Model
Explanations* (Pinto & Arenas). The language runtime
[`qfoil-parser-evaluator`](../qfoil-parser-evaluator/README.md) (Python package
`qfoil`) and HTTP adapter [`qfoil-server`](../qfoil-server/README.md) are sibling
directories in this repository. The transport contract is documented in the
server's [`API_CONTRACT.md`](../qfoil-server/API_CONTRACT.md).

The UI checks server health, renders formulas, evaluates expressions and
inspects witnesses produced during evaluation. It also manages one active
model and inspects its test samples.

See the [QFOIL project overview](../README.md)
for the component architecture, joint setup and the coordinated
`article-v1.0.1` source snapshot.

Evaluation options live in the **Evaluation settings** tab: Heuristic or
Exhaustive mode, the total evaluation budget, the per-SIGMA search budget,
an optional integer random seed and an optional initial tuple.
Settings apply to the next evaluation; the
Playground summarizes the active mode. Both modes respect the server's global
work limit. Both budgets allow up to 20,000,000, as advertised by `/v1/config`.
The total budget is editable in both modes; the per-SIGMA budget applies to
heuristic mode. An inconclusive result is displayed with its reason. Mode, budgets
and random seed persist for the browser tab session; initial tuples do not.
Exhaustive HTTP evaluation uses exact restricted-domain enumeration.

Run the npm commands below from `qfoil-ui/`. The project overview documents
the shared Python environment and full system setup.

## Installation

```bash
npm ci
```

## Running qfoil-ui together with qfoil-server

The UI is a pure client for `qfoil-server`'s HTTP API and does nothing useful
on its own. First follow the [joint setup](../README.md#run-the-system),
installing the runtime, server and any required backend extras. With that
shared environment, start the server from `qfoil-ui/`:

```bash
cd ..
source .venv/bin/activate
qfoil-server            # or: uvicorn qfoil_server.main:app --reload
```

By default this listens on `http://127.0.0.1:8000`. In a second terminal,
from the repository root:

```bash
cd qfoil-ui
npm run dev
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000/*` (see
`vite.config.ts`), so the browser only ever talks to the dev server's own
origin. Open the printed local URL; the header should show `Connected` once
`GET /health` succeeds.

### Configuring a model catalog: `QFOIL_MODEL_CATALOG`

qfoil-server always starts with no model loaded. To let the UI
offer models to load, point `QFOIL_MODEL_CATALOG` (set before starting
`qfoil-server`) at a JSON file describing the catalog:

```bash
export QFOIL_MODEL_CATALOG=/path/to/models.json
```

```json
{
  "mux": { "backend": "pyjuice", "path": "/srv/qfoil/models/mux", "device": "cuda:0", "min_batch_size": 4 },
  "spam": { "backend": "pyjuice", "path": "/srv/qfoil/models/spam", "device": "cuda:0", "min_batch_size": 4 }
}
```

The UI never sees `path` or `min_batch_size`; it only ever gets each entry's
`id` (the JSON key) and `backend` from `GET /v1/models`. If the variable is
unset, the catalog is empty and the model panel shows "No models are
configured on qfoil-server." instead of a selector.

### Pointing at a different server: `VITE_QFOIL_API_URL`

If `qfoil-server` runs somewhere other than the dev proxy target (a different
port, a remote host, or a production build served without the Vite proxy), set
`VITE_QFOIL_API_URL` to its base URL:

```bash
VITE_QFOIL_API_URL=http://127.0.0.1:9000 npm run dev
```

A trailing slash is stripped automatically. When unset, the client uses `/api`
as a relative base, which only works through the Vite dev proxy or an
equivalent reverse-proxy rule in production.

## Commands

```bash
npm run dev       # start the Vite dev server with the /api proxy
npm test          # run the Vitest unit/integration suite (jsdom)
npm run build     # type-check (tsc --noEmit) and produce a production build
npm run preview   # preview the production build locally
```

## Using the playground

### Render a formula and evaluate it

With the default `Expression` (`SIZE(x) <= 2`) and `Bindings`
(`{ "x": [1, null, 0] }`):

- **Render formula** calls `POST /v1/expressions/latex` and renders the
  returned LaTeX with KaTeX; the raw LaTeX string is also shown as text.
- **Evaluate** calls `POST /v1/expressions/evaluate` with the same expression
  and the parsed bindings, and shows `Value: 1`.

### Inspect a witness

An evaluation that produces `last_witness` displays it as JSON; click the
value, or use Enter/Space, to copy it. Heuristic results that cannot be
certified are shown as **Inconclusive**, which does not prove that no witness
exists. Exhaustive evaluation can also be inconclusive if its total work
budget is exhausted.

Use **Evaluation settings** to choose the mode and budgets. An initial tuple
must be a JSON array of `0`, `1` or `null` matching the loaded model's feature
count. Exhaustive mode omits heuristic options. If configuration cannot load,
evaluation uses server defaults; **Reload settings** retries the configuration.

### Loading and unloading a model

`SIZE`, `FULL` and formula rendering work with no model loaded. `EP`, `CEP`,
`VAL` and `COND` need one active.

The header has a plain (non-button) status readout — next to the server
status, not part of it — that always shows the current model state:
`○ No model loaded`, `◌ Loading mux…`, `● mux · PyJuice · cuda:0`,
`◌ Unloading mux…`, or `⚠ Model error`. It's informational only; it stays
visible across all tabs (`Playground`, `Model`, `Evaluation settings` and
`Visualize`) and isn't itself a control.

The **Model** tab is where you manage the lifecycle. With no model loaded,
it offers a `<select>` populated from `GET /v1/models` (see "Configuring a
model catalog" above) and a `Load` button. If the catalog is empty, the tab
says so instead of showing an empty selector. Clicking **Load** calls
`POST /v1/models/{model_id}/load`; the request stays open until the server
finishes loading, during which the selector, `Load` and `Evaluate` are
disabled (`Render formula` stays available).

Once a model is `ready`, the tab shows its `id`, `status`, `backend`,
`num_features`, `num_classes`, `device` and `loaded_at` (as a localized
`<time>`), plus an `Unload` button. **At most one model can be active**, and
the selector stays hidden while one is loaded: loading a second model
directly is not possible. To switch models, click **Unload** (this asks for
confirmation and calls `DELETE /v1/models/current`; repeating that request
with no model loaded still returns `200`)
and then load a different one from the catalog — there is no automatic
"replace" that chains the two calls for you.

The UI reconstructs this state from `GET /v1/models/current` on every page
load (the server is always the source of truth; nothing about the model is
kept in `localStorage`), and refreshes it again after `Load`/`Unload`, when
the window regains focus, when the browser tab becomes visible, or via the
`Refresh` button on the `Model` tab.

### Test samples

For a ready model with test data, the **Model** tab can fetch an indexed or
random sample. An optional integer seed makes random sampling reproducible.
The panel shows the index, dataset size, sample and expected target, with
separate controls to copy the sample or the full response. These controls use
the server's `/v1/models/current/dataset/test/samples` endpoints.

### Random Forest

The selector and lifecycle support `backend: "random_forest"`, displayed as
**Random Forest**. `VAL(x)` returns an integer label for complete binary tuples
and numeric `0` for valid partial tuples. Use `FULL(x)` to distinguish a
complete classification from a partial tuple. `VAL(x,c)`, `EP`, `CEP` and
`COND` are unsupported; the UI displays the server's HTTP 501 error.

The runtime and server READMEs document the eight-feature MUX RF model, with
feature order `S0, S1, D0, D1, D2, D3, N0, N1` and labels 0/1. For class 1,
use `FULL(x) * (VAL(x) == 1)`. Backend extras, artifact generation and catalog
configuration are documented in the runtime and server READMEs.

### Tuple visualization

The experimental **Visualize** tab compares original and explanation tuples
of equal length as colored grids. It accepts JSON arrays of `0`, `1` and
`null`, with controls for layers, columns, cell size and colors. Visualization
runs locally and does not evaluate expressions or query the model.

## Input history

Expression and Bindings each have an independent History dropdown. Selecting
an entry fills only that field; Copy copies the selected text, and Clear removes
that field's history. Each list stores up to 20 unique strings in localStorage.

## Implementation

The implementation was developed with assistance from generative AI models.
