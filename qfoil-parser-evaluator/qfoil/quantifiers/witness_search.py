"""Heuristic witness search for SIGMA expressions.

This module only proposes candidates.  ``SigmaEvaluator`` remains responsible
for verifying every witness and selecting heuristic or exact execution.
"""

import random
from contextlib import contextmanager, nullcontext
from math import floor, inf, log, sqrt
from typing import Any, Dict, List, Optional

from ..results import Inconclusive

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
        if (
            not isinstance(left, tuple)
            or not isinstance(right, tuple)
            or len(left) != len(right)
        ):
            return inf
        return float(
            sum(
                left_value is not None and left_value != right_value
                for left_value, right_value in zip(left, right)
            )
        )

    def violation_for_native(self, func_name: str, args: List[Any]) -> float:
        """Return zero when a supported native relation is satisfied."""
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
            if operator in ("SUBSETEQ", "<<="):
                return self.violation_subseteq(
                    self.evaluator.evaluate(ast.left, context),
                    self.evaluator.evaluate(ast.right, context),
                )
            if operator in ("SUBSET", "<<"):
                left = self.evaluator.evaluate(ast.left, context)
                right = self.evaluator.evaluate(ast.right, context)
                return self.violation_subseteq(left, right) + float(left == right)
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
        self, sigma_ast: Sigma, context: Dict[str, Any], size: int, *, initial: bool = False
    ) -> List[Any]:
        """Sample full/partial tuples respecting simple conjunctive bounds.

        Only positive top-level conjunctions are inspected. Bounds hidden in
        negation or other operators must not constrain the search domain.
        ``initial`` requests the most conservative candidate consistent with
        the bounds (BOTTOM plus any forced lower-bound values) instead of a
        density-sampled one, so the very first attempt -- the only one a
        tight budget may afford on a large domain -- starts feasible with
        respect to any SIZE cap. Restarts still sample densely for diversity.
        """
        lower = [None] * size
        upper = None
        max_size = None
        prefer_full = self.evaluator.oracle_backend is not None

        def is_bound_size(node):
            return (
                isinstance(node, FunctionCall)
                and node.function_name.upper() == "SIZE"
                and len(node.arguments) == 1
                and isinstance(node.arguments[0], Variable)
                and node.arguments[0].name == sigma_ast.bound_var
            )

        def resolve_threshold(node):
            if isinstance(node, NumberLiteral):
                return node.value
            if isinstance(node, Variable):
                return context.get(node.name)
            return None

        def inspect(node):
            nonlocal upper, prefer_full, max_size
            if isinstance(node, FunctionCall) and node.function_name.upper() == "FULL":
                if len(node.arguments) == 1 and isinstance(node.arguments[0], Variable):
                    prefer_full |= node.arguments[0].name == sigma_ast.bound_var
            if not isinstance(node, BinaryOp):
                return
            if node.operator == "*":
                inspect(node.left)
                inspect(node.right)
            elif node.operator.upper() in ("<<=", "SUBSETEQ", "<<", "SUBSET"):
                if isinstance(node.left, Variable) and node.left.name == sigma_ast.bound_var:
                    other = node.right
                    direction = "upper"
                elif isinstance(node.right, Variable) and node.right.name == sigma_ast.bound_var:
                    other = node.left
                    direction = "lower"
                else:
                    return
                # Resolve only free variables; do not evaluate arbitrary ASTs
                # outside the candidate-evaluation budget.
                if not isinstance(other, Variable) or other.name == sigma_ast.bound_var:
                    return
                values = context.get(other.name)
                if not isinstance(values, tuple) or len(values) != size:
                    return
                if direction == "upper":
                    upper = list(values) if upper is None else [
                        a if a == b else None for a, b in zip(upper, values)
                    ]
                else:
                    for index, value in enumerate(values):
                        if value is not None:
                            lower[index] = value
            elif node.operator in ("<=", ">=", "=="):
                # SIZE(bound_var) <op> threshold, in either operand order.
                if is_bound_size(node.left):
                    threshold, operator = resolve_threshold(node.right), node.operator
                elif is_bound_size(node.right):
                    threshold = resolve_threshold(node.left)
                    operator = {"<=": ">=", ">=": "<=", "==": "=="}[node.operator]
                else:
                    return
                if not isinstance(threshold, (int, float)):
                    return
                if operator in ("<=", "=="):
                    max_size = threshold if max_size is None else min(max_size, threshold)

        inspect(sigma_ast.formula)
        if initial and not prefer_full:
            return [lower[index] for index in range(size)]
        # Vary density between restarts, rather than always starting at BOTTOM.
        density = 1.0 if prefer_full else self.rng.choice((0.25, 0.5, 0.75, 1.0))
        free_indices = [index for index in range(size) if lower[index] is None]
        chosen = None
        if max_size is not None:
            # A tight SIZE cap makes a density-sampled start almost always
            # infeasible on a large domain: bound how many free positions a
            # random start can turn on so it starts near the feasible region.
            fixed_count = size - len(free_indices)
            quota = max(min(floor(max_size) - fixed_count, len(free_indices)), 0)
            target_count = max(0, round(quota * density))
            chosen = set(self.rng.sample(free_indices, target_count))
        candidate = []
        for index in range(size):
            if lower[index] is not None:
                value = lower[index]
            elif chosen is not None:
                value = (upper[index] if upper is not None else self.rng.choice((0, 1))) if index in chosen else None
            elif upper is not None:
                value = upper[index] if self.rng.random() < density else None
            else:
                value = self.rng.choice((0, 1)) if self.rng.random() < density else None
            candidate.append(value)
        return candidate

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

        The explicit seed is tried first. All initial scores, moves and
        restarts share the candidate budget. UCB1 chooses coordinates;
        sideways moves and occasional noisy moves escape local plateaus.
        A zero score proposes a witness, which the caller must verify.
        """
        size = len(reference_tuple)
        if size == 0 or budget <= 0:
            return None
        bound_var = sigma_ast.bound_var

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
                list(seed)
                if seed is not None and len(seed) == size
                else self._random_candidate(sigma_ast, context, size, initial=True)
            )
            best_score = score(candidate)
            if best_score is None:
                # Nested predicates may be uncertifiable at a random start.
                # BOTTOM remains a fallback, not the default initialization.
                if not self.contains_sigma(sigma_ast.formula):
                    return None
                candidate = [None] * size
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
                if stagnant >= size or evaluations - attempt_start >= attempt_budget:
                    candidate = self._random_candidate(sigma_ast, context, size)
                    best_score = score(candidate)
                    if best_score is None:
                        return None
                    tries = [0] * size
                    gains = [0.0] * size
                    stagnant = 0
                    attempt_start = evaluations
                    continue

                untried = [i for i in range(size) if tries[i] == 0]
                total = sum(tries)
                index = (
                    self.rng.choice(untried)
                    if untried
                    else max(
                        range(size),
                        key=lambda i: gains[i] / tries[i]
                        + sqrt(2 * log(total) / tries[i]),
                    )
                )
                current = candidate[index]
                trials = []
                for value in (0, 1, None):
                    if value == current or evaluations >= budget:
                        continue
                    candidate[index] = value
                    violation = score(candidate)
                    candidate[index] = current
                    if violation is not None:
                        if violation == 0:
                            candidate[index] = value
                            return tuple(candidate)
                        trials.append((value, violation))
                if not trials:
                    return None
                local_best = min(v for _, v in trials)
                choices = [(value, v) for value, v in trials if v == local_best]
                chosen, trial_score = self.rng.choice(choices)
                improvement = max(best_score - trial_score, 0.0)
                tries[index] += 1
                gains[index] += improvement
                if trial_score <= best_score or self.rng.random() < 0.1:
                    candidate[index] = chosen
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
        """Grow a partial witness monotonically while its score remains zero."""
        size = len(reference_tuple)
        if size == 0:
            return None
        bound_var = sigma_ast.bound_var

        with self.approximate_mode(sigma_ast.formula, inner_budget):

            def score(candidate_values: List[Any]) -> Optional[float]:
                local_context = context.copy()
                local_context[bound_var] = tuple(candidate_values)
                return self.soft_violation(sigma_ast.formula, local_context)

            candidate = (
                list(seed)
                if seed is not None and len(seed) == size
                else [None] * size
            )
            violation = score(candidate)
            if violation is None:
                return None

            remaining = [
                index for index, value in enumerate(candidate) if value is None
            ]
            evaluations = 0
            while remaining and evaluations < budget:
                already_witness = violation == 0
                pool = self.rng.sample(remaining, min(pool_size, len(remaining)))
                best_position = best_value = None
                best_violation = violation

                for position in pool:
                    for value in (0, 1):
                        if evaluations >= budget:
                            break
                        candidate[position] = value
                        trial = score(candidate)
                        evaluations += 1
                        candidate[position] = None
                        if trial is None:
                            continue
                        helps = trial < best_violation or (
                            already_witness and trial == 0
                        )
                        if helps and (
                            best_position is None or trial < best_violation
                        ):
                            best_violation = trial
                            best_position = position
                            best_value = value
                    if evaluations >= budget:
                        break

                if best_position is None:
                    break
                candidate[best_position] = best_value
                remaining.remove(best_position)
                violation = best_violation
            return tuple(candidate) if violation == 0 else None
