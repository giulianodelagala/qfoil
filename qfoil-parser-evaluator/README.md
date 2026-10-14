# qfoil-parser-evaluator

QFOIL is a logic language and execution engine for reasoning over partial
tuples and ML models such as probabilistic circuits and Boolean classifiers.

This is the language-runtime component of the proof-of-concept prototype for
*Semirings Meet Interpretability: A Quantitative Logic for Model
Explanations* (Pinto & Arenas). It parses and evaluates formulas; the
companion [HTTP adapter](../qfoil-server/README.md) and
[web client](../qfoil-ui/README.md) are sibling directories in this repository.

See the [QFOIL project overview](../README.md)
for the component architecture, joint setup and the coordinated
`article-v1.0.0` source snapshot.

## Quick reference

Tuples use `0`, `1`, and `None` (undefined); `x`, `y`, `z`... are tuple
variables, `a`, `b` numeric terms, `c` a class id.

| Syntax | Meaning |
| --- | --- |
| `(1, 0, None)` | tuple literal |
| `a + b`, `a * b`, `!a` / `NOT a` | sum, product (≡ conjunction on 0/1), zero test |
| `a == b`, `a <= b`, `a >= b` | equality and numeric comparisons |
| `y <<< x` / `y << x` | non-strict / strict subsumption (`y ⊆ x`) |
| `SIGMA y. (phi)` / `SIGMA y (phi)` | counting quantifier; `sigma` and `∑` are aliases, the dot is optional |
| `FULL(x)` | all positions defined |
| `L0(x)`, `L1(x)` / `FEAT(x)` | zero / exactly one position defined |
| `SIZE(x)` | number of defined positions |
| `FREE(x, y)` | defined positions of `x` not retained in `y` |
| `COMP(x, y)` | `y` is a full completion of `x` |
| `SUF(x, y)` | same defined-position set |
| `OPPOSITE(x, y)` | opposite single-feature tuples |
| `PARTITION(x, y)` | disjoint defined-position sets |
| `MEET(x, y, z)`, `JOIN(x, y, z)` | lattice meet / compatible join |
| `BOTTOM()` / `BOTTOM(n)` | fully undefined tuple |
| `VAL(x)`, `VAL(x, c)` | Pr(x) / Pr(x, Y=c) (PC backend); classifier label (RF backend) |
| `COND(x, y, c)` | Pr(x, Y=c \| y) (PC backend only) |
| `EP(x, c)`, `CEP(x, c)` | Pr(Y=c \| x) and its complement (PC backend only) |

`EP`/`CEP`/`COND`/`VAL(x, c)` raise `UnsupportedOperationError` under an RF
`oracle_backend`. See sections below for backend setup, witness search,
execution modes, and defining new functions with `QFOIL.define_function`.

Run the commands below from `qfoil-parser-evaluator/`. For the shared
environment and full system setup, see the project overview.

## Installation

Install only the backend-independent language runtime:

```bash
pip install .
```

Optional integrations are installed explicitly:

```bash
pip install ".[pyjuice]"
pip install ".[z3]"
pip install ".[random_forest]"
```

For development, including both integrations and pytest:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from qfoil import QFOIL

qfoil = QFOIL()
result = qfoil.evaluate("SIZE(x) == 2", x=(1, 0, None))
```

Parsing is independent from the execution runtime:

```python
from qfoil import QFOILParser

ast = QFOILParser().parse("SIGMA y. (y <<< x)")
```

The dot after the bound variable is optional: `SIGMA y (y <<< x)` and
`SIGMA y. (y <<< x)` are equivalent. This also applies to the aliases `sigma`
and `∑`; existing expressions with a dot remain valid. Use parentheses around
the condition to keep its scope explicit.

Probabilistic operations are provided through a backend contract, keeping the
QFOIL evaluator independent from PyJuice and Torch:

```python
from qfoil.backends.pyjuice import PyJuiceModel, PyJuiceProbabilityBackend

model = PyJuiceModel.load("path/to/model")
backend = PyJuiceProbabilityBackend(model)
qfoil = QFOIL(probabilistic_backend=backend)
probability = qfoil.evaluate("EP(x, c)", x=instance, c=class_id)
```

Heuristic SIGMA evaluation supports reproducible randomness and a local
candidate budget for one operation:

```python
value = qfoil.evaluate(
    "(SIGMA y. (y <<< x) * (EP(y, c) >= 0.9)) >= 1",
    bindings={"x": instance, "c": class_id},
    rng_seed=38,
    sigma_search_budget=1000,
)
witness = qfoil.evaluator.sigma.last_witness
```

`rng_seed` initializes the random stream shared by searches and nested
queries in this operation. `seed` is a separate, optional initial tuple
for a direct SIGMA comparison. `sigma_search_budget` limits candidates per
SIGMA attempt; it does not reset or replace the global evaluation budget.
`sigma_mode="exact"` selects exhaustive execution for this operation,
including nested queries, under the same global evaluation budget.
`sigma_mode="restricted"` is a second exact procedure: mandatory product
factors `FULL`, `COMP`, subsumption and `SIZE` bounds restrict the generated
domain. Unknown constraints fall back to the full domain; the original body
is still evaluated for each candidate. Conditions inside negation or sums
do not restrict the domain. Both exact modes share the global work budget.
Overrides are restored after the call. A failed heuristic attempt returns
`Inconclusive`, not a proof that no witness exists. A flat evaluation clears
any witness from a previous facade call. The HTTP server exposes these options through the evaluation request's
`options` object (`mode` selects `heuristic` or `exact`).
HTTP `mode="exact"` uses the restricted exact procedure. Total enumeration
remains available through the Python API with `sigma_mode="exact"`.

Witness search is exposed by the same facade:

```python
witness = qfoil.grow_witness(
    "FULL(y) * (y <<< x)",
    var="y",
    context={"x": instance},
)
```

For a verified result with search metadata, use the additive structured API:

```python
result = qfoil.search_witness(
    "FULL(y) * (y <<< x)",
    var="y",
    context={"x": instance},
    budget=1000,
    pool_size=30,
)

if result.found:
    witness = result.witness
```

`not_found` means that the heuristic search did not find a verified witness
within its budget. It does not prove that no witness exists.

QFOIL expressions can also be rendered as KaTeX-compatible LaTeX without
adding a web or JavaScript dependency:

```python
latex = qfoil.to_latex("SIGMA y. FULL(y) * (y <<< x)")
```

The returned string contains no math delimiters or HTML. Applications may pass
it to KaTeX, MathJax, a notebook display helper, or another renderer.


## Random Forest label oracle

The additive `oracle_backend=` argument accepts the minimal `ValueOracle`
contract (`num_features` and `value(instance) -> int | float`). It is mutually
exclusive with `probabilistic_backend=`.
Labels never pass through `ProbabilisticBackend.probability()`.

```python
from qfoil import QFOIL
from qfoil.backends.random_forest import RandomForestOracle

# forest is a fitted sklearn classifier with integer class labels.
oracle = RandomForestOracle(forest)
qfoil = QFOIL(oracle_backend=oracle, instance_len=oracle.num_features)
label = qfoil.evaluate("VAL(x)", x=instance)
is_three = qfoil.evaluate("FULL(x) * (VAL(x) == 3)", x=instance)
```

RF checks tuple length and the `{0,1,None}` domain (Booleans excluded) before
prediction. FULL binary inputs produce a Python integer label, including when
sklearn returns a NumPy scalar. Valid partial inputs produce integer `0`
without calling `predict`. `VAL(x,c)`, `COND`, `EP`, and `CEP` raise
`UnsupportedOperationError("Consulta no soportada por Random Forest")`.
The same rejection occurs in skipped heuristic SIGMA bodies and wrappers.
The parser and AST are unchanged. Importing the parser, core runtime, or RF
adapter does not import sklearn, NumPy, Torch, or PyJuice. sklearn is required
only for fitting/loading a real forest; use the `random_forest` extra.

The MUX RF artifact is stored in `examples/models/mux_random_forest/` and
replaces MNIST RF in the server's local model catalog. It uses the same
`S0, S1, D0, D1, D2, D3, N0, N1` feature order as the PC MUX and returns
`D[2*S1 + S0]`, with class labels 0 and 1. The noise bits do not affect the
label. It includes `model.joblib`, `metadata.json`, and paired NumPy arrays
covering all 256 inputs. These inputs also train the forest; the exhaustive
check verifies the truth table, rather than estimating held-out accuracy.
Rebuild on CPU with:

```bash
conda run -n pc python -m examples.models.mux_random_forest.train
```

Seeded verification can reuse `QFOIL.evaluate(sigma_expr, bindings={...},
seed=full_instance)` and `search_witness(..., seed=full_instance)`.
Unseeded heuristic searches remain bounded and may be inconclusive.

## Examples

- `examples/intro_basics.ipynb`: parsing, tuple operations, and PyJuice
  queries (`VAL`/`EP`/`CEP`/`COND`) on an MNIST model.
- `examples/example_mux.ipynb`: explanation queries (PSR, PCE, DFS, and
  their minimality/ranking variants) validated on a synthetic 8-feature MUX
  against an independent ground-truth oracle.
- `examples/mnist_sandbox.ipynb`: trains the MNIST PyJuice models stored
  under `examples/models/`.

Model artifacts live under `examples/models/<name>/`, one directory per
model, loaded by path with `PyJuiceModel.load(...)` or
`joblib.load(.../model.joblib)` for the Random Forest variants.

## Implementation

The implementation was developed with assistance from generative AI models.
