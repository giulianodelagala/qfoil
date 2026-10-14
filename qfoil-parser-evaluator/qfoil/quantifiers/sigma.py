"""Exact evaluation and orchestration for the SIGMA quantifier.

Execution policy is split into two explicit, orthogonal axes:

- ``mode`` ("heuristic", "exact" or "restricted") selects the procedure.
  ``restricted`` performs exact enumeration of a domain narrowed by
  mandatory tuple and size factors; ``exact`` retains the total domain.
  Heuristic mode
  never enumerates, in evaluation, subqueries, logic functions, scoring or
  verification; when it cannot certify an answer it returns
  :data:`~qfoil.results.INCONCLUSIVE`. Exact mode always decides by
  enumeration and is the ground-truth reference available for small
  domains (e.g. the Mux oracle comparison).
- the local *budgets* (``search_budget``, the independent witness-search
  ``budget``, and ``inner_budget``) bound how much work a given attempt may
  spend; they do not select the procedure. In particular
  ``sigma_search_budget=0`` no longer means "use exact mode" -- pass
  ``mode="exact"`` explicitly instead.
"""

from itertools import product
from typing import Any, Dict, Optional

from ..language.ast import BinaryOp, Sigma
from .witness_search import SigmaSearch
from .domains import restricted_candidates
from ..results import EVALUATION_BUDGET_EXHAUSTED, Inconclusive, INCONCLUSIVE


class SigmaEvaluator:
    """Own SIGMA state and combine heuristic search with exact evaluation."""

    COMPARISON_OPERATORS = ("==", "<=", ">=")
    MIRRORED_OPERATOR = {"==": "==", "<=": ">=", ">=": "<="}
    MODES = ("heuristic", "exact", "restricted")

    def __init__(
        self,
        evaluator: Any,
        search_budget: int = 64,
        mode: str = "heuristic",
    ):
        if (
            isinstance(search_budget, bool)
            or not isinstance(search_budget, int)
            or search_budget < 0
        ):
            raise ValueError("sigma_search_budget must be a non-negative integer")
        if mode not in self.MODES:
            raise ValueError(f"sigma_mode must be one of {self.MODES}, got {mode!r}")
        self.evaluator = evaluator
        self.search_budget = search_budget
        self.mode = mode
        self.approximate_depth = 0
        self.last_witness: Optional[tuple] = None
        self.search = SigmaSearch(evaluator, self)

    def _give_up(self) -> Inconclusive:
        """Report Inconclusive, attributing it to the real root cause.

        A local search/enumeration giving up usually means it is genuinely
        uncertain (`bounded_search_incomplete`). But if the shared global
        evaluation budget ran out along the way, intermediate layers (e.g.
        heuristic scoring collapsing Inconclusive to `None`) may have lost
        that specific reason -- recover it here by checking budget state
        directly, since it is the more accurate explanation.
        """
        self.last_witness = None
        budget = getattr(self.evaluator, "budget", None)
        if budget is not None and budget.limit is not None and budget.remaining <= 0:
            return Inconclusive(EVALUATION_BUDGET_EXHAUSTED)
        return INCONCLUSIVE

    def generate_all_combinations(self) -> list[tuple]:
        """Materialize the ternary domain with the configured tuple length."""
        return list(product((0, 1, None), repeat=self.evaluator.instance_len))

    def domain_template(self) -> tuple:
        """Return a template for the runtime's configured SIGMA domain."""
        return (None,) * self.evaluator.instance_len

    def candidates(self, ast: Sigma, context: Dict[str, Any]):
        """Keep total enumeration available alongside restricted exact mode."""
        if self.mode == "restricted":
            return restricted_candidates(ast, context, self.evaluator.instance_len)
        return product((0, 1, None), repeat=self.evaluator.instance_len)

    def evaluate_exact(
        self, ast: Sigma, context: Dict[str, Any]
    ) -> int | float | Inconclusive:
        """Sum over the total or safely restricted ternary domain.

        A bare SIGMA count (not wrapped in a comparison `decide()` can
        certify via a witness) has no way to be answered without summing
        every candidate. Heuristic mode therefore never attempts this: it
        reports Inconclusive unconditionally. Exact mode performs the sum.
        """
        if self.mode == "heuristic":
            return self._give_up()
        count = 0
        for candidate in self.candidates(ast, context):
            local_context = context.copy()
            local_context[ast.bound_var] = candidate
            result = self.evaluator.evaluate(ast.formula, local_context)
            if isinstance(result, Inconclusive):
                self.last_witness = None
                return result
            count += result if isinstance(result, (int, float)) else int(bool(result))
        return count

    def decide(
        self,
        sigma_ast: Sigma,
        operator: str,
        threshold: Any,
        context: Dict[str, Any],
        seed: Optional[tuple] = None,
    ) -> int | Inconclusive:
        """Decide when certified; return Inconclusive otherwise.

        Exact mode (``self.mode == "exact"``) always decides by exhaustive
        enumeration, regardless of operator/threshold shape or budget: it
        is the ground-truth reference, selected explicitly, never a
        fallback.
        Restricted mode uses the same exact decisions with a pruned domain.

        Heuristic mode never enumerates. It only has a certification
        strategy for existential (``>= 1``) and universal (``<= 0`` /
        ``== 0``) shapes, via a verified witness/counterexample search; any
        other shape, an exhausted local search budget, or a failed/
        unverified search all return Inconclusive directly -- never a
        silent fall-through to the exact scan.
        """
        if isinstance(threshold, Inconclusive):
            self.last_witness = None
            return threshold
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Invalid operation: SIGMA {operator} {threshold}")

        if self.mode in ("exact", "restricted"):
            return self._decide_by_enumeration(sigma_ast, operator, threshold, context)

        reference = self.domain_template()
        existential = operator == ">=" and threshold <= 1
        universal = operator in ("<=", "==") and threshold == 0

        if not (existential or universal) or self.search_budget <= 0:
            return self._give_up()

        witness = self.search.local_search(
            sigma_ast,
            context,
            reference,
            self.search_budget,
            seed=seed,
        )
        if witness is not None and self._verify_witness(sigma_ast, context, witness):
            self.last_witness = witness
            return 1 if existential else 0

        return self._give_up()

    def _verify_witness(
        self, sigma_ast: Sigma, context: Dict[str, Any], witness: tuple
    ) -> bool:
        local_context = context.copy()
        local_context[sigma_ast.bound_var] = witness
        result = self.evaluator.evaluate(sigma_ast.formula, local_context)
        if isinstance(result, Inconclusive):
            return False
        return result > 0 if isinstance(result, (int, float)) else bool(result)

    def _decide_by_enumeration(
        self,
        sigma_ast: Sigma,
        operator: str,
        threshold: float,
        context: Dict[str, Any],
    ) -> int:
        count = 0
        for candidate in self.candidates(sigma_ast, context):
            local_context = context.copy()
            local_context[sigma_ast.bound_var] = candidate
            result = self.evaluator.evaluate(sigma_ast.formula, local_context)
            if isinstance(result, Inconclusive):
                self.last_witness = None
                return result
            contribution = (
                result if isinstance(result, (int, float)) else int(bool(result))
            )
            if contribution < 0:
                raise ValueError(
                    "SIGMA early-exit requires non-negative summands, "
                    f"got {contribution} for {sigma_ast.bound_var}={candidate}"
                )
            count += contribution

            verdict = self._early_verdict(operator, count, threshold)
            if verdict is not None:
                self.last_witness = candidate
                return verdict

        self.last_witness = None
        return self._final_verdict(operator, count, threshold)

    @staticmethod
    def _early_verdict(
        operator: str, count: float, threshold: float
    ) -> Optional[int]:
        if operator == ">=" and count >= threshold:
            return 1
        if operator == "<=" and count > threshold:
            return 0
        if operator == "==" and count > threshold:
            return 0
        return None

    @staticmethod
    def _final_verdict(operator: str, count: float, threshold: float) -> int:
        if operator == ">=":
            return int(count >= threshold)
        if operator == "<=":
            return int(count <= threshold)
        return int(count == threshold)

    def evaluate_comparison(
        self,
        ast: BinaryOp,
        context: Dict[str, Any],
        seed: Optional[tuple] = None,
    ) -> int:
        """Evaluate a comparison with a SIGMA expression on either side."""
        left_is_sigma = isinstance(ast.left, Sigma)
        right_is_sigma = isinstance(ast.right, Sigma)

        if left_is_sigma and not right_is_sigma:
            threshold = self.evaluator.evaluate(ast.right, context)
            return self.decide(ast.left, ast.operator, threshold, context, seed)
        if right_is_sigma and not left_is_sigma:
            threshold = self.evaluator.evaluate(ast.left, context)
            operator = self.MIRRORED_OPERATOR[ast.operator]
            return self.decide(ast.right, operator, threshold, context, seed)

        threshold = self.evaluator.evaluate(ast.right, context)
        return self.decide(ast.left, ast.operator, threshold, context, seed)

    def grow_witness(
        self,
        expr: str,
        var: str,
        context: Dict[str, Any],
        budget: int = 1000,
        pool_size: int = 30,
        seed: Optional[tuple] = None,
        inner_budget: int = 8,
    ) -> Optional[tuple]:
        parser = self.evaluator.parser
        if parser is None:
            raise RuntimeError("Parser not set. Call set_parser() first.")
        sigma_ast = Sigma(var, parser.parse(expr))
        reference = self.domain_template()
        return self.search.greedy_ascent(
            sigma_ast,
            context,
            reference,
            budget,
            seed=seed,
            pool_size=pool_size,
            inner_budget=inner_budget,
        )
