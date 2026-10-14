# QFOIL development guide

QFOIL is a pure-Python declarative logic runtime for partial tuples and
probabilistic-circuit models. `SIGMA` is the language's counting quantifier.

## Public API

- `qfoil.QFOIL`: composed parsing and evaluation API.
- `qfoil.QFOILParser`: backend-independent parser.
- `qfoil.QFOILEvaluator`: AST evaluator.
- `qfoil.backends.ProbabilisticBackend`: probability contract.
- `qfoil.backends.pyjuice.PyJuiceProbabilityBackend`: PyJuice adapter.
- `qfoil.backends.pyjuice.PyJuiceInference`: low-level PyJuice queries.
- `qfoil.backends.z3.Z3CegisSolver`: optional Z3-backed search.

There is no compatibility package and no implicit `model=` conversion. Attach
a PyJuice model explicitly:

```python
from qfoil import QFOIL
from qfoil.backends.pyjuice import PyJuiceProbabilityBackend

qfoil = QFOIL(
    probabilistic_backend=PyJuiceProbabilityBackend(model),
)
```

Use `QFOIL.grow_witness(...)` for heuristic witness search. The facade owns
this operation; its SIGMA implementation lives in `qfoil/quantifiers/`.

## Architecture

- `qfoil/language/`: AST and parser.
- `qfoil/semantics/`: tuple lattice, operators, function registry, and the
  declarative standard library.
- `qfoil/runtime/`: AST evaluator, function runtime, and probability runtime.
- `qfoil/quantifiers/`: exact SIGMA evaluation and witness search.
- `qfoil/backends/`: probabilistic protocol, PyJuice, and Z3 integrations.

Tuples use values from `{0, 1, None}`; `None` means unobserved. Probabilistic
functions keep the class as a separate argument: `VAL(x)`, `VAL(x, c)`,
`COND(x, y, c)`, `EP(x, c)`, and `CEP(x, c)`.

## Tests

The test layout mirrors the package boundaries:

- `test_parser.py`: lexer, grammar, and AST construction.
- `test_language_runtime.py`: backend-independent expression evaluation.
- `test_standard_library.py`: native/declarative function integration.
- `test_sigma_runtime.py`: exact quantification and witness search.
- `test_*_runtime.py` and `test_*_semantics.py`: focused component tests.
- `test_z3_backend.py` and PyJuice tests: optional backend integration.

Use the Conda environment `pc`:

```bash
conda run -n pc pytest tests/
```

Packaging metadata and pytest configuration live in `pyproject.toml`.
Use the `pyjuice`, `z3`, `test`, or `dev` extras instead of duplicating
dependency declarations elsewhere.

PyJuice circuit compilation requires CUDA in the installed version. Parser,
semantics, runtime, and Z3 tests can run without the PyJuice integration
fixtures.
