"""Structured result types returned by the public QFOIL facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional


BOUNDED_SEARCH_INCOMPLETE = "bounded_search_incomplete"
"""Heuristic mode could not certify a decision: no verified witness, or the
decision shape (operator/threshold) has no witness-search strategy at all."""

EVALUATION_BUDGET_EXHAUSTED = "evaluation_budget_exhausted"
"""The shared global work budget for this operation ran out before a
decision could be certified. See `qfoil.runtime.budget.EvaluationBudget`."""


@dataclass(frozen=True)
class Inconclusive:
    """Unknown result, deliberately distinct from false, zero and None."""

    reason: str = BOUNDED_SEARCH_INCOMPLETE

    def __bool__(self):
        raise TypeError("An inconclusive result has no truth value")


INCONCLUSIVE = Inconclusive()


@dataclass(frozen=True)
class WitnessSearchResult:
    """Outcome of a verified heuristic witness search."""

    status: Literal["found", "inconclusive"]
    variable: str
    witness: Optional[tuple]
    verified: bool
    seed_used: Optional[tuple]
    search: Mapping[str, int]
    reason: Optional[str] = None

    @property
    def found(self) -> bool:
        """Whether a verified witness was found."""
        return self.status == "found"

    def to_dict(self) -> dict[str, Any]:
        """Return a transport-neutral description built from plain values."""
        return {
            "status": self.status,
            "variable": self.variable,
            "witness": self.witness,
            "verified": self.verified,
            "seed_used": self.seed_used,
            "search": dict(self.search),
            "reason": self.reason,
        }


__all__ = [
    "WitnessSearchResult",
    "Inconclusive",
    "INCONCLUSIVE",
    "BOUNDED_SEARCH_INCOMPLETE",
    "EVALUATION_BUDGET_EXHAUSTED",
]
