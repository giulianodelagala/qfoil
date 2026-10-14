"""Global work budget shared by one top-level ``QFOILEvaluator.evaluate()``.

This is deliberately separate from the pre-existing local budgets
(`SigmaEvaluator.search_budget`, the independent witness-search `budget`,
and `inner_budget`): those bound one heuristic search call each, while this
one bounds the *entire* operation that invoked `evaluate()`, including every
subquery, logic-defined function body, witness-search scoring step, and
witness verification reached along the way -- because all of them are
ultimately expressed as a call to `QFOILEvaluator.evaluate()`.

Unit of work: one evaluation unit is charged for every call to
`QFOILEvaluator.evaluate()` -- i.e. once per AST node visited, at any
nesting depth. This is the single point every execution path (flat
evaluation, nested SIGMA, logic functions, heuristic scoring, and witness
re-verification) routes through, so metering it there bounds the whole
operation without any separate per-subquery accounting.
"""

from __future__ import annotations

from typing import Optional


class EvaluationBudget:
    """Count evaluation units spent by one top-level `evaluate()` call.

    ``limit=None`` means unlimited: the historical, unbounded behavior.
    The counter is rearmed to ``limit`` every time a *new* top-level
    operation begins (see `QFOILEvaluator.evaluate`), so a budget attached
    to a long-lived evaluator never leaks state between requests.
    """

    def __init__(self, limit: Optional[int] = None):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("evaluation_budget must be None or a non-negative integer")
        self.limit = limit
        self.remaining = limit

    def reset(self) -> None:
        """Rearm the counter to its configured limit for a new operation."""
        self.remaining = self.limit

    def consume(self) -> bool:
        """Charge one unit; return False once the budget is exhausted."""
        if self.remaining is None:
            return True
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


__all__ = ["EvaluationBudget"]
