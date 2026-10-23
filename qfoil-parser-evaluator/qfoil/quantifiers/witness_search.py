"""Heuristic witness search for SIGMA expressions.

This module only proposes candidates.  ``SigmaEvaluator`` remains responsible
for verifying every witness and selecting heuristic or exact execution.
"""

import random
from contextlib import contextmanager, nullcontext
from math import inf, log, sqrt
from typing import Any, Dict, List, Optional

from ..results import Inconclusive
from ..semantics.tuples import TupleSemantics
from .domains import TupleDomain, plan_domain

from ..language.ast import (
    ASTNode,
    BinaryOp,
    FunctionCall,
    NumberLiteral,
    Sigma,
    TupleLiteral,
    UnaryOp,
    Variable,
)


class SigmaSearch:
    """Score formulas and search the ternary tuple domain for witnesses."""

    VIOLATION_FUNCTIONS = {
        "FULL",
        "L0",
        "L1",
        "FEAT",
        "PARTITION",
        "MEET",
        "JOIN",
        "OPPOSITE",
        "SUF",
        "COMP",
    }

    def __init__(self, evaluator: Any, sigma_evaluator: Any, seed: int = 1234):
        self.evaluator = evaluator
        self.sigma_evaluator = sigma_evaluator
        self.rng = random.Random(seed)

    @staticmethod
    def violation_subseteq(left: Any, right: Any) -> float:
        """Return the number of fixed positions in ``left`` that disagree."""
        # Invalid operands are query errors, not unsatisfied constraints.
        TupleSemantics.subseteq(left, right)
        return float(
            sum(
                left_value is not None and left_value != right_value
                for left_value, right_value in zip(left, right)
            )
        )

    def violation_for_native(self, func_name: str, args: List[Any]) -> float:
        """Return zero when a supported native relation is satisfied."""
        if func_name in self.VIOLATION_FUNCTIONS:
            # Share the exact evaluator's arity, type and length validation.
            self.evaluator.function_runtime.evaluate_native(func_name, args)
        try:
            if func_name == "FULL":
                (x,) = args
                return float(sum(value is None for value in x))
            if func_name == "L0":
                (x,) = args
                return float(sum(value is not None for value in x))
            if func_name in ("L1", "FEAT"):
                (x,) = args
                return float(abs(sum(value is not None for value in x) - 1))
            if func_name == "PARTITION":
                x, y = args
                if len(x) != len(y):
                    return inf
                return float(
                    sum(
                        x_value is not None and y_value is not None
                        for x_value, y_value in zip(x, y)
                    )
                )
            if func_name == "MEET":
                x, y, z = args
                if not (len(x) == len(y) == len(z)):
                    return inf
                return float(
                    sum(
                        z_value is not None
                        if x_value != y_value
                        else z_value != x_value
                        for x_value, y_value, z_value in zip(x, y, z)
                    )
                )
            if func_name == "JOIN":
                x, y, z = args
                if not (len(x) == len(y) == len(z)):
                    return inf
                violation = 0
                for x_value, y_value, z_value in zip(x, y, z):
                    if x_value is not None and y_value is not None:
                        violation += x_value != y_value or z_value != x_value
                    elif x_value is not None:
                        violation += z_value != x_value
                    elif y_value is not None:
                        violation += z_value != y_value
                return float(violation)
            if func_name == "OPPOSITE":
                x, y = args
                if len(x) != len(y):
                    return inf
                shared = 0
                differing = False
                violation = 0
                for x_value, y_value in zip(x, y):
                    if x_value is not None and y_value is not None:
                        shared += 1
                        differing |= x_value != y_value
                    elif (x_value is None) != (y_value is None):
                        violation += 1
                violation += abs(shared - 1)
                if shared >= 1 and not differing:
                    violation += 1
                return float(violation)
            if func_name == "SUF":
                x, y = args
                if len(x) != len(y):
                    return inf
                return float(
                    sum(
                        (x_value is None) != (y_value is None)
                        for x_value, y_value in zip(x, y)
                    )
                )
            if func_name == "COMP":
                x, y = args
                return self.violation_subseteq(x, y) + float(
                    sum(value is None for value in y)
                )
        except (TypeError, ValueError, IndexError):
            return inf
        return inf

    def contains_sigma(self, ast: ASTNode, stack=()) -> bool:
        if isinstance(ast, Sigma):
            return True
        if isinstance(ast, BinaryOp):
            return self.contains_sigma(ast.left, stack) or self.contains_sigma(ast.right, stack)
        if isinstance(ast, UnaryOp):
            return self.contains_sigma(ast.operand, stack)
        if isinstance(ast, FunctionCall):
            if any(self.contains_sigma(argument, stack) for argument in ast.arguments):
                return True
            name = ast.function_name.upper()
            functions = self.evaluator.function_runtime
            if functions.registry.has_defined(name):
                if name in stack:
                    return True
                body = self.evaluator.parser.parse(functions.registry.defined_function(name).body)
                return self.contains_sigma(body, (*stack, name))
            return False
        if isinstance(ast, TupleLiteral):
            return any(self.contains_sigma(element, stack) for element in ast.elements)
        return False

    @contextmanager
    def temporarily_limit_budget(self, inner_budget: int):
        original_budget = self.sigma_evaluator.search_budget
        self.sigma_evaluator.search_budget = inner_budget
        self.sigma_evaluator.approximate_depth += 1
        try:
            yield
        finally:
            self.sigma_evaluator.search_budget = original_budget
            self.sigma_evaluator.approximate_depth -= 1

    def approximate_mode(self, formula: ASTNode, inner_budget: int):
        if not self.contains_sigma(formula):
            return nullcontext()
        return self.temporarily_limit_budget(inner_budget)

    def comparison_violation(
        self, comparison: BinaryOp, context: Dict[str, Any], negate: bool
    ) -> Optional[float]:
        if self.contains_sigma(comparison.left) or self.contains_sigma(
            comparison.right
        ):
            result = self.evaluator.evaluate(comparison, context)
            if isinstance(result, Inconclusive):
                return None
            satisfied = result > 0 if isinstance(result, (int, float)) else bool(result)
            return 0.0 if satisfied != negate else 1.0

        left = self.evaluator.evaluate(comparison.left, context)
        right = self.evaluator.evaluate(comparison.right, context)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None

        # RF labels are categories, not coordinates on a numeric scale.
        if comparison.operator == "==" and self.evaluator.oracle_backend is not None:
            def is_label(node):
                return isinstance(node, FunctionCall) and node.function_name.upper() == "VAL"

            if is_label(comparison.left) or is_label(comparison.right):
                satisfied = left == right
                return 0.0 if satisfied != negate else 1.0

        difference = float(left) - float(right)
        if not negate:
            if comparison.operator == ">=":
                return max(0.0, -difference)
            if comparison.operator == "<=":
                return max(0.0, difference)
            return abs(difference)

        epsilon = 1e-9
        if comparison.operator == ">=":
            return max(0.0, difference + epsilon)
        if comparison.operator == "<=":
            return max(0.0, epsilon - difference)
        return max(0.0, epsilon - abs(difference))

    def soft_violation(
        self, ast: ASTNode, context: Dict[str, Any]
    ) -> Optional[float]:
        if isinstance(ast, BinaryOp):
            operator = ast.operator.upper()
            if operator in ("SUBSETEQ", "<<=", "SUBSET", "<<"):
                left = self.evaluator.evaluate(ast.left, context)
                right = self.evaluator.evaluate(ast.right, context)
                if isinstance(left, Inconclusive) or isinstance(right, Inconclusive):
                    return None
                violation = self.violation_subseteq(left, right)
                if operator in ("SUBSET", "<<"):
                    violation += float(left == right)
                return violation
            if ast.operator in ("==", "<=", ">="):
                return self.comparison_violation(ast, context, negate=False)
            if ast.operator == "*":
                left = self.soft_violation(ast.left, context)
                right = self.soft_violation(ast.right, context)
                return None if left is None or right is None else left + right
            return None

        if isinstance(ast, FunctionCall):
            func_name = ast.function_name.upper()
            if func_name not in self.VIOLATION_FUNCTIONS:
                return None
            args = [
                self.evaluator.evaluate(argument, context)
                for argument in ast.arguments
            ]
            if any(isinstance(argument, Inconclusive) for argument in args):
                return None
            return self.violation_for_native(func_name, args)

        if isinstance(ast, UnaryOp) and ast.operator.upper() in ("NOT", "!"):
            inner = ast.operand
            if isinstance(inner, BinaryOp) and inner.operator in ("==", "<=", ">="):
                return self.comparison_violation(inner, context, negate=True)
            if isinstance(inner, UnaryOp) and inner.operator.upper() in ("NOT", "!"):
                return self.soft_violation(inner.operand, context)
            violation = self.soft_violation(inner, context)
            if violation is None:
                return None
            return 0.0 if violation > 0 else 1.0
        return None

    def _random_candidate(
        self,
        sigma_ast: Sigma,
        context: Dict[str, Any],
        size: int,
        *,
        initial: bool = False,
        domain: Optional[TupleDomain] = None,
    ) -> Optional[List[Any]]:
        """Sample within the same necessary constraints as restricted exact mode."""
        if domain is None:
            domain = plan_domain(sigma_ast, context, size)
        return domain.sample(
            self.rng,
            initial=initial,
            prefer_full=self.evaluator.oracle_backend is not None,
        )

    def local_search(
        self,
        sigma_ast: Sigma,
        context: Dict[str, Any],
        reference_tuple: tuple,
        budget: int,
        seed: Optional[tuple] = None,
        inner_budget: int = 8,
    ) -> Optional[tuple]:
        """Stochastic grow-shrink search with random starts and restarts.

        A feasible explicit seed is tried first; others are projected onto
        the necessary constraints. All initial scores, moves and
        restarts share the candidate budget. UCB1 chooses coordinates;
        sideways moves and occasional noisy moves escape local plateaus.
        A zero score proposes a witness, which the caller must verify.
        """
        size = len(reference_tuple)
        if size == 0 or budget <= 0:
            return None
        bound_var = sigma_ast.bound_var
        domain = plan_domain(sigma_ast, context, size)
        mutable = [i for i, values in enumerate(domain.choices) if len(values) > 1]

        with self.approximate_mode(sigma_ast.formula, inner_budget):
            evaluations = 0

            def score(values):
                nonlocal evaluations
                if evaluations >= budget:
                    return None
                evaluations += 1
                return self.soft_violation(
                    sigma_ast.formula, {**context, bound_var: tuple(values)}
                )

            candidate = (
                domain.sample(self.rng, seed=seed)
                if seed is not None and len(seed) == size
                else self._random_candidate(
                    sigma_ast, context, size, initial=True, domain=domain
                )
            )
            if candidate is None:
                # Preserve runtime errors even when necessary constraints
                # conflict. This diagnostic score never proposes a witness.
                score(reference_tuple)
                return None
            best_score = score(candidate)
            if best_score is None:
                # Nested predicates may be uncertifiable at a random start.
                if not self.contains_sigma(sigma_ast.formula):
                    return None
                candidate = self._random_candidate(
                    sigma_ast, context, size, domain=domain
                )
                if candidate is None:
                    return None
                best_score = score(candidate)
                if best_score is None:
                    return None
            tries = [0] * size
            gains = [0.0] * size
            stagnant = 0
            attempt_start = 0
            attempt_budget = max(8, 4 * size)

            while True:
                if best_score == 0:
                    return tuple(candidate)
                if evaluations >= budget or self.evaluator.budget.remaining == 0:
                    return None
                if not mutable:
                    return None
                if (
                    stagnant >= len(mutable)
                    or evaluations - attempt_start >= attempt_budget
                ):
                    candidate = self._random_candidate(
                        sigma_ast, context, size, domain=domain
                    )
                    if candidate is None:
                        return None
                    best_score = score(candidate)
                    if best_score is None:
                        return None
                    tries = [0] * size
                    gains = [0.0] * size
                    stagnant = 0
                    attempt_start = evaluations
                    continue

                untried = [i for i in mutable if tries[i] == 0]
                total = sum(tries)
                index = (
                    self.rng.choice(untried)
                    if untried
                    else max(
                        mutable,
                        key=lambda i: gains[i] / tries[i]
                        + sqrt(2 * log(total) / tries[i]),
                    )
                )
                current = candidate[index]
                trials = []
                for value in domain.choices[index]:
                    if value == current or evaluations >= budget:
                        continue
                    proposal = domain.move(candidate, index, value, self.rng)
                    if proposal is None:
                        continue
                    violation = score(proposal)
                    if violation is not None:
                        if violation == 0:
                            return tuple(proposal)
                        trials.append((proposal, violation))
                if not trials:
                    tries[index] += 1
                    stagnant += 1
                    continue
                local_best = min(v for _, v in trials)
                choices = [(value, v) for value, v in trials if v == local_best]
                chosen, trial_score = self.rng.choice(choices)
                improvement = max(best_score - trial_score, 0.0)
                tries[index] += 1
                gains[index] += improvement
                if trial_score <= best_score or self.rng.random() < 0.1:
                    candidate = chosen
                    best_score = trial_score
                stagnant = 0 if improvement > 0 else stagnant + 1

    def greedy_ascent(
        self,
        sigma_ast: Sigma,
        context: Dict[str, Any],
        reference_tuple: tuple,
        budget: int,
        seed: Optional[tuple] = None,
        pool_size: int = 30,
        inner_budget: int = 8,
    ) -> Optional[tuple]:
        """Improve a feasible start, then grow its witness monotonically."""
        size = len(reference_tuple)
        if size == 0:
            return None
        bound_var = sigma_ast.bound_var
        domain = plan_domain(sigma_ast, context, size)

        with self.approximate_mode(sigma_ast.formula, inner_budget):

            def score(candidate_values: List[Any]) -> Optional[float]:
                local_context = context.copy()
                local_context[bound_var] = tuple(candidate_values)
                return self.soft_violation(sigma_ast.formula, local_context)

            if budget <= 0:
                # A zero move budget may check the supplied seed (or BOTTOM),
                # but must not construct a new witness through planning.
                candidate = (
                    list(seed) if seed is not None and len(seed) == size
                    else list(reference_tuple)
                )
                violation = score(candidate)
                return (
                    tuple(candidate)
                    if violation == 0 and domain.accepts(candidate) else None
                )

            candidate = domain.sample(
                self.rng,
                initial=True,
                seed=seed if seed is not None and len(seed) == size else None,
            )
            if candidate is None:
                score(list(reference_tuple))
                return None
            violation = score(candidate)
            if violation is None:
                return None

            evaluations = 0
            while evaluations < budget and self.evaluator.budget.remaining != 0:
                already_witness = violation == 0
                remaining = [
                    i for i, values in enumerate(domain.choices)
                    if len(values) > 1 and (not already_witness or candidate[i] is None)
                ]
                if not remaining:
                    break
                pool = self.rng.sample(remaining, min(pool_size, len(remaining)))
                best_candidate = None
                best_violation = violation

                for position in pool:
                    for value in domain.choices[position]:
                        if evaluations >= budget:
                            break
                        proposal = domain.move(candidate, position, value, self.rng)
                        if proposal is None:
                            continue
                        if already_witness and any(
                            old is not None and new != old
                            for old, new in zip(candidate, proposal)
                        ):
                            continue
                        trial = score(proposal)
                        evaluations += 1
                        if trial is None:
                            continue
                        helps = trial < best_violation or (
                            already_witness and trial == 0
                        )
                        if helps and (
                            best_candidate is None or trial < best_violation
                        ):
                            best_violation = trial
                            best_candidate = proposal
                    if evaluations >= budget:
                        break

                if best_candidate is None:
                    break
                candidate = best_candidate
                violation = best_violation
            return tuple(candidate) if violation == 0 else None
