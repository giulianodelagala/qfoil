"""QFOIL functions defined in the language itself.

Keeping these definitions outside the evaluator makes the standard library
data-driven: runtimes only need to register and evaluate ``DefinedFunction``
objects, without embedding the language catalogue in their constructors.
"""

from .functions import DefinedFunction


def standard_definitions() -> tuple[DefinedFunction, ...]:
    """Return the built-in QFOIL source definitions in registration order."""
    cond1 = "((u <<= x) * (u <<= y))"
    cond2 = "((v <<= x) * (v <<= y))"
    cond3 = "((u <<= x) * (v <<= y))"
    cond4 = "((u <<= y) * (v <<= x))"
    cond5 = "(!(u <<= x) * !(u <<= y) * !(v <<= x) * !(v <<= y))"
    count_opposite = "SIGMA u. SIGMA v. (OPPOSITE(u, v))"

    return (
        DefinedFunction("L_FULL", ("x",), "SIGMA y. (x << y) == 0"),
        DefinedFunction("L_COMP", ("x", "y"), "(x <<= y) * L_FULL(y)"),
        DefinedFunction("L_FEAT", ("x",), "SIGMA y. (y << x) == 1"),
        DefinedFunction(
            "L_SIZE", ("x",), "SIGMA y. (L_FEAT(y) * (y <<= x))"
        ),
        DefinedFunction(
            "L_FREE",
            ("x", "y"),
            "SIGMA u. (L_FEAT(u) * (u <<= x) * !(u <<= y))",
        ),
        DefinedFunction(
            "L_MEET",
            ("x", "y", "z"),
            "(z <<= x) * (z <<= y) * (SIGMA w. (w <<= x) * "
            "(w <<= y) * (w <<= z) >= SIGMA w. (w <<= x) * (w <<= y))",
        ),
        DefinedFunction(
            "L_JOIN",
            ("x", "y", "z"),
            "(x <<= z) * (y <<= z) * (SIGMA w. (x <<= w) * "
            "(y <<= w) * (z <<= w) >= SIGMA w. (x <<= w) * (y <<= w))",
        ),
        DefinedFunction("L_L0", ("x",), "SIGMA y. (y <<= x) == 1"),
        DefinedFunction(
            "L_L1",
            ("x",),
            "SIGMA y. (L_L0(y) * ((y <<= x) * !(y == x)) * "
            "(SIGMA z.((y <<= z) * !(y == z) * "
            "(z <<= x) * !(z == x)) == 0 ))",
        ),
        DefinedFunction(
            "L_OPPOSITE",
            ("x", "y"),
            "L1(x) * L1(y) * (SIGMA z.(x <<= z) * (y <<= z) == 0)",
        ),
        DefinedFunction(
            "L_PARTITION",
            ("x", "y"),
            "( SIGMA z. (L_FULL(z) * (x <<= z) * (y <<= z)) >= 1) * "
            "SIGMA z. (L_L0(z) * L_MEET(x, y, z))",
        ),
        DefinedFunction(
            "L_SUF",
            ("x", "y"),
            f"SIGMA u. SIGMA v. (OPPOSITE(u, v) * "
            f"({cond1} + {cond2} + {cond3} + {cond4} + {cond5})) "
            f"== {count_opposite}",
        ),
        DefinedFunction("L_COUNT_SUF", ("x",), "SIGMA y. (SUF(x, y))"),
    )


STANDARD_DEFINITIONS = standard_definitions()

__all__ = ["STANDARD_DEFINITIONS", "standard_definitions"]
