"""High-level composition API for parsing and evaluating QFOIL."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from .errors import QFOILSyntaxError, QFOILValidationError
from .language import ASTNode, QFOILParser
from .language.latex import QFOILLatexRenderer
from .results import WitnessSearchResult, Inconclusive
from .runtime import QFOILEvaluator

if TYPE_CHECKING:
    from .backends import ProbabilisticBackend
    from .backends.oracle import ValueOracle


class QFOIL:
    """Compose the QFOIL language frontend with its execution runtime."""

    def __init__(
        self,
        *,
        probabilistic_backend: Optional[ProbabilisticBackend] = None,
        oracle_backend: Optional[ValueOracle] = None,
        instance_len: int = 3,
        sigma_search_budget: int = 64,
        sigma_mode: str = "heuristic",
        evaluation_budget: Optional[int] = None,
        prob_cache_size: int = 10000,
    ) -> None:
        self.parser = QFOILParser(instance_len=instance_len)
        self.evaluator = QFOILEvaluator(
            probabilistic_backend=probabilistic_backend,
            oracle_backend=oracle_backend,
            instance_len=instance_len,
            sigma_search_budget=sigma_search_budget,
            sigma_mode=sigma_mode,
            evaluation_budget=evaluation_budget,
            prob_cache_size=prob_cache_size,
        )
        self.evaluator.set_parser(self.parser)

    def parse(self, expression: str) -> ASTNode:
        """Parse a QFOIL expression into its abstract syntax tree."""
        try:
            return self.parser.parse(expression)
        except ValueError as exc:
            raise QFOILSyntaxError(str(exc)) from exc

    def to_latex(self, expression: str | ASTNode) -> str:
        """Render a QFOIL expression as KaTeX-compatible LaTeX."""
        ast = self.parse(expression) if isinstance(expression, str) else expression
        return QFOILLatexRenderer().render(ast)

    def evaluate(
        self,
        expression: str | ASTNode,
        bindings: Optional[Mapping[str, Any]] = None,
        *,
        seed: Optional[tuple] = None,
        rng_seed: Optional[int] = None,
        sigma_search_budget: Optional[int] = None,
        sigma_mode: Optional[str] = None,
        evaluation_budget: Optional[int] = None,
        **variables: Any,
    ) -> Any:
        """Evaluate with optional per-operation work/search budgets and RNG seed.

        ``seed`` is an initial tuple; ``rng_seed`` is an integer that makes
        random initialization, moves and restarts reproducible.
        ``evaluation_budget`` overrides total node visits for this operation,
        including nested queries; the configured limit is restored afterwards.
        """
        if bindings is not None and variables:
            raise ValueError("use either bindings or keyword variables, not both")

        context = dict(bindings or variables)
        ast = self.parse(expression) if isinstance(expression, str) else expression
        with self.evaluator.operation(
            rng_seed=rng_seed, sigma_search_budget=sigma_search_budget,
            sigma_mode=sigma_mode, evaluation_budget=evaluation_budget,
        ):
            return self.evaluator.evaluate(ast, context, seed=seed)

    def define_function(
        self, name: str, parameters: Sequence[str], body: str
    ) -> None:
        """Define or replace a QFOIL function for this runtime."""
        self.evaluator.define_logic_function(name, parameters, body)

    def grow_witness(
        self,
        expr: str,
        var: str,
        context: Mapping[str, Any],
        *,
        budget: int = 1000,
        pool_size: int = 30,
        seed: Optional[tuple] = None,
        inner_budget: int = 8,
    ) -> Optional[tuple]:
        """Search heuristically for a tuple satisfying ``expr``."""
        if self.evaluator.oracle_backend is not None:
            self.evaluator.validate_oracle_operations(self.parse(expr))
        with self.evaluator.operation():
            return self.evaluator.sigma.grow_witness(
                expr,
                var,
                dict(context),
                budget,
                pool_size,
                seed,
                inner_budget,
            )

    def search_witness(
        self,
        expr: str,
        var: str,
        context: Mapping[str, Any],
        *,
        budget: int = 1000,
        pool_size: int = 30,
        seed: Optional[tuple] = None,
        inner_budget: int = 8,
    ) -> WitnessSearchResult:
        """Search for and independently verify one heuristic witness."""
        self._validate_search_request(
            expr=expr,
            var=var,
            context=context,
            budget=budget,
            pool_size=pool_size,
            seed=seed,
            inner_budget=inner_budget,
        )
        with self.evaluator.operation():
            context_dict = dict(context)
            witness = self.grow_witness(
                expr,
                var,
                context_dict,
                budget=budget,
                pool_size=pool_size,
                seed=seed,
                inner_budget=inner_budget,
            )
            verified = False
            if witness is not None:
                verification_context = dict(context_dict)
                verification_context[var] = witness
                value = self.evaluate(expr, bindings=verification_context)
                verified = False if isinstance(value, Inconclusive) else (value > 0 if isinstance(value, (int, float)) else bool(value))

            found = witness is not None and verified
            return WitnessSearchResult(
                status="found" if found else "inconclusive",
                variable=var,
                witness=witness if found else None,
                verified=verified,
                reason=None if found else (
                    "evaluation_budget_exhausted"
                    if self.evaluator.budget.remaining == 0
                    else "bounded_search_incomplete"
                ),
                seed_used=seed,
                search=MappingProxyType(
                    {
                        "budget": budget,
                        "pool_size": pool_size,
                        "inner_budget": inner_budget,
                    }
                ),
            )

    def _validate_search_request(
        self,
        *,
        expr: str,
        var: str,
        context: Mapping[str, Any],
        budget: int,
        pool_size: int,
        seed: Optional[tuple],
        inner_budget: int,
    ) -> None:
        self.parse(expr)
        if (
            not isinstance(var, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", var) is None
        ):
            raise QFOILValidationError("var must be a valid QFOIL identifier")
        self._require_integer_option("budget", budget, minimum=0)
        self._require_integer_option("pool_size", pool_size, minimum=1)
        self._require_integer_option("inner_budget", inner_budget, minimum=0)
        if seed is not None:
            if not isinstance(seed, tuple):
                raise QFOILValidationError("seed must be a tuple or None")
            if len(seed) != self.evaluator.instance_len:
                raise QFOILValidationError(
                    "seed must have the same length as instance_len"
                )

    @staticmethod
    def _require_integer_option(name: str, value: Any, *, minimum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise QFOILValidationError(
                f"{name} must be an integer greater than or equal to {minimum}"
            )
