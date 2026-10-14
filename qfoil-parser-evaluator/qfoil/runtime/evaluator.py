"""Interpret QFOIL ASTs and adapt probabilistic calls to optional backends.

SIGMA evaluation and witness search live in :mod:`qfoil.quantifiers`. Pure
logic works without a model; only VAL, COND, EP, and CEP require a
probabilistic backend.
"""

from __future__ import annotations

from contextlib import contextmanager
import random

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)

from ..errors import UnsupportedOperationError

from ..language.ast import (
    ASTNode,
    Variable,
    NumberLiteral,
    BinaryOp,
    UnaryOp,
    Sigma,
    FunctionCall,
    TupleLiteral,
)
from ..quantifiers import SigmaEvaluator
from ..results import EVALUATION_BUDGET_EXHAUSTED, Inconclusive
from ..semantics import (
    OperatorSemantics,
    STANDARD_DEFINITIONS,
    TupleSemantics,
)
from .budget import EvaluationBudget
from .functions import FunctionRuntime
from .probability import ProbabilityRuntime

if TYPE_CHECKING:
    from ..backends.probabilistic import ProbabilisticBackend
    from ..backends.oracle import ValueOracle
    from ..language.parser import QFOILParser


class QFOILEvaluator:
    """Interpret QFOIL expressions and delegate probabilistic operations.

    The evaluator is backend-neutral. Optional integrations are supplied
    through the probabilistic backend protocol.
    """

    def __init__(
        self,
        probabilistic_backend: Optional[ProbabilisticBackend] = None,
        instance_len: int = 3,
        parser: Optional[QFOILParser] = None,
        sigma_search_budget: int = 64,
        sigma_mode: str = "heuristic",
        evaluation_budget: Optional[int] = None,
        prob_cache_size: int = 10000,
        *,
        oracle_backend: Optional[ValueOracle] = None,
    ):
        """Initialize the QFOIL evaluator.

        Args:
            probabilistic_backend: Backend implementing QFOIL's probability
                contract. Pure logic works without a backend.
            oracle_backend: Numeric VAL(x) oracle (e.g. Random Forest).
                Mutually exclusive with probabilistic_backend.
            instance_len: Length of tuples for domain (default: 3)
            parser: Optional logic parser for function definitions
            sigma_search_budget: Max evaluations the grow-shrink heuristic
                (`SigmaSearch.local_search`) spends looking for a witness to
                `SIGMA x. formula >= 1`, in `sigma_mode="heuristic"`. Does
                not select the mode: `0` simply means the heuristic search
                is skipped (never a silent switch to exact enumeration --
                pass `sigma_mode="exact"` for that instead).
            sigma_mode: `"heuristic"` (default) never enumerates -- in
                evaluation, subqueries, logic functions, scoring and
                witness verification alike -- and returns Inconclusive
                whenever it cannot certify a witness-backed decision.
                `"exact"` always decides by exhaustive enumeration and is
                the ground-truth reference for small domains (e.g. the Mux
                oracle comparison); select it explicitly, it is never
                triggered implicitly by a budget value.
                `"restricted"` also decides exactly, pruning the domain
                using mandatory FULL, COMP, subsumption and SIZE factors.
            evaluation_budget: Global work-unit limit shared by one
                top-level `evaluate()` call, covering subqueries, logic
                functions, heuristic scoring and witness verification (see
                `qfoil.runtime.budget.EvaluationBudget`). `None` (default)
                is unlimited. Exhausting it returns Inconclusive with
                reason `"evaluation_budget_exhausted"`, and is independent
                from `sigma_search_budget`, the witness-search `budget`,
                and `inner_budget`.
            prob_cache_size: Max entries in the VAL/COND/EP/CEP memoization
                cache (`ProbabilityRuntime.cached`), LRU beyond that; `0` disables
                it. Safe because these are pure functions of their
                arguments for a fixed backend - a fresh evaluator starts
                with an empty cache.
        """
        if probabilistic_backend is not None and oracle_backend is not None:
            raise ValueError("Supply only one active backend")
        self.oracle_backend = oracle_backend
        self.domain = {0, 1, None}
        self.instance_len = instance_len
        self.tuple_semantics = TupleSemantics(instance_len)
        self.operator_semantics = OperatorSemantics(self.tuple_semantics)
        self.parser = parser
        self.sigma = SigmaEvaluator(self, sigma_search_budget, mode=sigma_mode)
        self.budget = EvaluationBudget(evaluation_budget)
        self._evaluation_depth = 0
        self._operation_depth = 0

        self.probabilistic_backend = probabilistic_backend
        self.probability_runtime = ProbabilityRuntime(
            probabilistic_backend,
            cache_size=prob_cache_size,
            oracle_backend=oracle_backend,
        )

        native_handlers: Dict[
            str,
            Optional[Callable[[List[Any]], Any]],
        ] = {
            **self.tuple_semantics.handlers,
            **self.probability_runtime.handlers,
        }
        self.function_runtime = FunctionRuntime(
            native_handlers,
            STANDARD_DEFINITIONS,
            parser=parser,
        )

    def set_parser(self, parser: QFOILParser) -> None:
        """Set the parser reference for parsing logic functions."""
        self.parser = parser

    @property
    def parser(self) -> Optional[QFOILParser]:
        return self._parser

    @parser.setter
    def parser(self, parser: Optional[QFOILParser]) -> None:
        self._parser = parser
        if hasattr(self, "function_runtime"):
            self.function_runtime.set_parser(parser)

    def define_logic_function(
        self, name: str, params: Sequence[str], body: str
    ) -> None:
        self.function_runtime.define(name, params, body)

    def evaluate_function(
        self, func_call: FunctionCall, context: Dict[str, Any]
    ) -> Any:
        """Evaluate function call - native or logic-defined."""
        func_name = func_call.function_name.upper()

        # Evaluate arguments
        args = [self.evaluate(arg, context) for arg in func_call.arguments]
        inconclusive_arg = next(
            (arg for arg in args if isinstance(arg, Inconclusive)), None
        )
        if inconclusive_arg is not None:
            return inconclusive_arg

        # Check native functions first
        if self.function_runtime.registry.has_native(func_name):
            return self.function_runtime.evaluate_native(func_name, args)

        # Check logic-defined functions
        if self.function_runtime.registry.has_defined(func_name):
            return self.function_runtime.evaluate_defined(
                func_name,
                args,
                context,
                self.evaluate,
            )

        raise ValueError(f"Unknown function: {func_name}")

    @contextmanager
    def operation(
        self,
        *,
        rng_seed: Optional[int] = None,
        sigma_search_budget: Optional[int] = None,
        sigma_mode: Optional[str] = None,
        evaluation_budget: Optional[int] = None,
    ):
        """Share a budget and random stream across one public operation.

        Overrides belong to the outermost operation and are restored on
        exit, including exceptions. Nested queries inherit both settings.
        """
        if evaluation_budget is not None:
            EvaluationBudget(evaluation_budget)
        if rng_seed is not None and (
            isinstance(rng_seed, bool) or not isinstance(rng_seed, int)
        ):
            raise ValueError("rng_seed must be None or an integer")
        if sigma_search_budget is not None and (
            isinstance(sigma_search_budget, bool)
            or not isinstance(sigma_search_budget, int)
            or sigma_search_budget < 0
        ):
            raise ValueError("sigma_search_budget must be a non-negative integer")
        if sigma_mode is not None and sigma_mode not in self.sigma.MODES:
            raise ValueError(f"sigma_mode must be one of {self.sigma.MODES}")
        outermost = self._operation_depth == 0 and self._evaluation_depth == 0
        if not outermost and any(
            option is not None for option in (rng_seed, sigma_search_budget, sigma_mode, evaluation_budget)
        ):
            raise ValueError("search overrides must be set on the outermost operation")
        original_rng = self.sigma.search.rng
        original_search_budget = self.sigma.search_budget
        original_mode = self.sigma.mode
        original_evaluation_budget = self.budget.limit
        if outermost:
            if evaluation_budget is not None:
                self.budget.limit = evaluation_budget
            self.budget.reset()
            self.sigma.last_witness = None
            if rng_seed is not None:
                self.sigma.search.rng = random.Random(rng_seed)
            if sigma_search_budget is not None:
                self.sigma.search_budget = sigma_search_budget
            if sigma_mode is not None:
                self.sigma.mode = sigma_mode
        self._operation_depth += 1
        try:
            yield
        finally:
            self._operation_depth -= 1
            if outermost:
                self.sigma.search.rng = original_rng
                self.sigma.search_budget = original_search_budget
                self.sigma.mode = original_mode
                self.budget.limit = original_evaluation_budget

    def evaluate(
        self,
        ast: ASTNode,
        context: Dict[str, Any],
        seed: Optional[tuple] = None,
        *,
        rng_seed: Optional[int] = None,
        sigma_search_budget: Optional[int] = None,
        sigma_mode: Optional[str] = None,
    ) -> Any:
        """Evaluate an AST, optionally configuring this operation's search.

        ``seed`` is an initial tuple for a direct SIGMA comparison only.
        ``rng_seed`` seeds the shared random stream for all searches in the
        operation, including nested queries. ``sigma_search_budget`` limits
        candidates per SIGMA attempt, independently of the global budget.
        """
        if any(option is not None for option in (rng_seed, sigma_search_budget, sigma_mode)):
            with self.operation(
                rng_seed=rng_seed, sigma_search_budget=sigma_search_budget,
                sigma_mode=sigma_mode,
            ):
                return self.evaluate(ast, context, seed=seed)
        is_root_evaluation = self._evaluation_depth == 0
        if is_root_evaluation:
            self.validate_oracle_operations(ast)
            bound_bindings = self._bound_variables(ast).intersection(context)
            if bound_bindings:
                names = ", ".join(sorted(bound_bindings))
                raise ValueError(
                    "bindings must not include variables bound by SIGMA: "
                    f"{names}"
                )
            # Rearm the shared global budget for this new top-level
            # operation so no state leaks in from a previous call.
            if self._operation_depth == 0:
                self.budget.reset()

        if not self.budget.consume():
            return Inconclusive(EVALUATION_BUDGET_EXHAUSTED)

        self._evaluation_depth += 1
        try:
            return self._evaluate_node(ast, context, seed)
        finally:
            self._evaluation_depth -= 1

    def _evaluate_node(
        self, ast: ASTNode, context: Dict[str, Any], seed: Optional[tuple]
    ) -> Any:
        """Evaluate one node after root-level binding validation."""
        if isinstance(ast, Variable):
            if ast.name not in context:
                raise ValueError(f"Undefined variable: {ast.name}")
            return context[ast.name]

        if isinstance(ast, NumberLiteral):
            return ast.value

        if isinstance(ast, TupleLiteral):
            values = tuple(self.evaluate(elem, context) for elem in ast.elements)
            inconclusive = next(
                (v for v in values if isinstance(v, Inconclusive)), None
            )
            return inconclusive if inconclusive is not None else values

        if isinstance(ast, Sigma):
            # SIGMA EVALUATION VARIABLE SEMANTICS:
            # In "SIGMA z. ((k <<< z) * FULL(z) * (EP(z, c) >= delta))":
            # - z is QUANTIFIED (bound by SIGMA) - varies over all candidates
            # - k, c, delta are FREE (from context) - constant during evaluation
            return self.sigma.evaluate_exact(ast, context)

        if isinstance(ast, FunctionCall):
            return self.evaluate_function(ast, context)

        if isinstance(ast, UnaryOp) and ast.operator.upper() in ("NOT", "!"):
            value = self.evaluate(ast.operand, context)
            return value if isinstance(value, Inconclusive) else int(value == 0)

        if isinstance(ast, BinaryOp):
            return self._evaluate_binary_operation(ast, context, seed)

        raise ValueError(f"Unknown AST node: {ast}")

    def validate_oracle_operations(self, ast: ASTNode, stack=()) -> None:
        """Reject RF probability calls even when a heuristic skips the body.

        This is a fixed first-version RF contract, not a capabilities registry.
        Pure PC evaluation is unchanged. Expand defined functions so a skipped
        SIGMA body cannot hide an unsupported operation behind a wrapper.
        """
        if self.oracle_backend is None:
            return
        if isinstance(ast, FunctionCall):
            name = ast.function_name.upper()
            if name in ("EP", "CEP", "COND") or (name == "VAL" and len(ast.arguments) == 2):
                raise UnsupportedOperationError("Consulta no soportada por Random Forest")
            for argument in ast.arguments:
                self.validate_oracle_operations(argument, stack)
            if self.function_runtime.registry.has_defined(name) and name not in stack:
                definition = self.function_runtime.registry.defined_function(name)
                if self.parser is not None:
                    self.validate_oracle_operations(self.parser.parse(definition.body), (*stack, name))
        elif isinstance(ast, Sigma):
            self.validate_oracle_operations(ast.formula, stack)
        elif isinstance(ast, BinaryOp):
            self.validate_oracle_operations(ast.left, stack)
            self.validate_oracle_operations(ast.right, stack)
        elif isinstance(ast, UnaryOp):
            self.validate_oracle_operations(ast.operand, stack)
        elif isinstance(ast, TupleLiteral):
            for element in ast.elements:
                self.validate_oracle_operations(element, stack)

    @classmethod
    def _bound_variables(cls, ast: ASTNode) -> set[str]:
        """Collect every variable name introduced by a SIGMA node."""
        if isinstance(ast, Sigma):
            return {ast.bound_var} | cls._bound_variables(ast.formula)
        if isinstance(ast, BinaryOp):
            return cls._bound_variables(ast.left) | cls._bound_variables(ast.right)
        if isinstance(ast, UnaryOp):
            return cls._bound_variables(ast.operand)
        if isinstance(ast, FunctionCall):
            result: set[str] = set()
            for argument in ast.arguments:
                result.update(cls._bound_variables(argument))
            return result
        if isinstance(ast, TupleLiteral):
            result = set()
            for element in ast.elements:
                result.update(cls._bound_variables(element))
            return result
        return set()

    def _evaluate_binary_operation(
        self,
        ast: BinaryOp,
        context: Dict[str, Any],
        seed: Optional[tuple],
    ) -> Any:
        """Evaluate one binary node, preserving Sigma's early-exit path."""
        operator = ast.operator
        if operator in self.sigma.COMPARISON_OPERATORS and (
            isinstance(ast.left, Sigma) or isinstance(ast.right, Sigma)
        ):
            return self.sigma.evaluate_comparison(ast, context, seed=seed)

        left = self.evaluate(ast.left, context)
        right = self.evaluate(ast.right, context)
        if isinstance(left, Inconclusive) or isinstance(right, Inconclusive):
            # Zero annihilates an unknown scalar product; otherwise propagate
            # whichever operand actually carries the Inconclusive reason.
            if operator == "*" and any(
                isinstance(v, (int, float)) and v == 0 for v in (left, right)
            ):
                return 0
            return left if isinstance(left, Inconclusive) else right
        return self.operator_semantics.evaluate(operator, left, right)
