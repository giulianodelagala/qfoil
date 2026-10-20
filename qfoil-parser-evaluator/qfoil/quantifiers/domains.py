"""Lazy exact domains derived only from mandatory product factors.

Planning reads literals/free bindings without evaluating expressions. It
never extracts restrictions from NOT, sums, or nested quantifiers.
"""

from math import ceil, floor, isfinite
from typing import Any, Iterator

from ..language.ast import (
    BinaryOp,
    FunctionCall,
    NumberLiteral,
    Sigma,
    TupleLiteral,
    Variable,
)


def restricted_candidates(
    ast: Sigma, context: dict[str, Any], size: int
) -> Iterator[tuple]:
    choices = [[0, 1, None] for _ in range(size)]
    minimum, maximum = 0, size
    excluded = []

    def is_bound(node):
        return isinstance(node, Variable) and node.name == ast.bound_var

    def resolve(node):
        if isinstance(node, Variable) and not is_bound(node):
            return context.get(node.name)
        if isinstance(node, NumberLiteral):
            return node.value
        if isinstance(node, TupleLiteral) and all(
            isinstance(v, NumberLiteral) for v in node.elements
        ):
            return tuple(v.value for v in node.elements)
        return None

    def known_tuple(node):
        value = resolve(node)
        if (
            isinstance(value, tuple)
            and len(value) == size
            and all(v in (0, 1, None) for v in value)
        ):
            return value
        return None

    def upper(value):
        for i, fixed in enumerate(value):
            choices[i] = [v for v in choices[i] if v is None or v == fixed]

    def lower(value):
        for i, fixed in enumerate(value):
            if fixed is not None:
                choices[i] = [v for v in choices[i] if v == fixed]

    def full():
        for i in range(size):
            choices[i] = [v for v in choices[i] if v is not None]

    def is_size(node):
        return (
            isinstance(node, FunctionCall)
            and node.function_name.upper() == "SIZE"
            and len(node.arguments) == 1
            and is_bound(node.arguments[0])
        )

    def inspect(node):
        nonlocal minimum, maximum
        if isinstance(node, FunctionCall):
            name = node.function_name.upper()
            if (
                name == "FULL"
                and len(node.arguments) == 1
                and is_bound(node.arguments[0])
            ):
                full()
            elif (
                name == "COMP"
                and len(node.arguments) == 2
                and is_bound(node.arguments[1])
            ):
                fixed = known_tuple(node.arguments[0])
                if fixed is not None:
                    lower(fixed)
                    full()
            elif (
                name == "COMP"
                and len(node.arguments) == 2
                and is_bound(node.arguments[0])
            ):
                fixed = known_tuple(node.arguments[1])
                if fixed is not None:
                    upper(fixed)
                    if None in fixed:
                        minimum = size + 1
            return
        if not isinstance(node, BinaryOp):
            return
        if node.operator == "*":
            inspect(node.left)
            inspect(node.right)
            return
        operator = node.operator.upper()
        if operator in ("<<=", "SUBSETEQ", "<<", "SUBSET"):
            fixed = None
            if is_bound(node.left):
                fixed = known_tuple(node.right)
                if fixed is not None:
                    upper(fixed)
            elif is_bound(node.right):
                fixed = known_tuple(node.left)
                if fixed is not None:
                    lower(fixed)
            if fixed is not None and operator in ("<<", "SUBSET"):
                excluded.append(fixed)
            return
        if operator not in ("==", "<=", ">="):
            return
        if is_size(node.left):
            threshold = resolve(node.right)
        elif is_size(node.right):
            threshold = resolve(node.left)
            operator = {"<=": ">=", ">=": "<=", "==": "=="}[operator]
        else:
            return
        if not isinstance(threshold, (int, float)) or (
            isinstance(threshold, float) and not isfinite(threshold)
        ):
            return
        if operator in ("<=", "=="):
            maximum = min(maximum, floor(threshold))
        if operator in (">=", "=="):
            minimum = max(minimum, ceil(threshold))

    inspect(ast.formula)
    if any(not values for values in choices) or minimum > maximum:
        return
    suffix_min = [0] * (size + 1)
    suffix_max = [0] * (size + 1)
    for i in range(size - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + int(None not in choices[i])
        suffix_max[i] = suffix_max[i + 1] + int(
            any(v is not None for v in choices[i])
        )
    if suffix_min[0] > maximum or suffix_max[0] < minimum:
        return
    if size == 0:
        if () not in excluded:
            yield ()
        return
    # Iterative DFS preserves the original 0, 1, None ordering and avoids
    # recursion limits. Size bounds prune prefixes, not just finished tuples.
    candidate = [None] * size
    stack = [(iter(choices[0]), 0)]
    while stack:
        i = len(stack) - 1
        iterator, prefix_size = stack[-1]
        try:
            value = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        observed = prefix_size + int(value is not None)
        if (
            observed + suffix_min[i + 1] > maximum
            or observed + suffix_max[i + 1] < minimum
        ):
            continue
        candidate[i] = value
        if i + 1 == size:
            result = tuple(candidate)
            if result not in excluded:
                yield result
        else:
            stack.append((iter(choices[i + 1]), observed))
