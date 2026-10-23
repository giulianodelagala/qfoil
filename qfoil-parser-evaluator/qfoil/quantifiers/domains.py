"""Shared tuple restrictions derived only from mandatory product factors.

Planning reads literals/free bindings without evaluating expressions. It
never extracts restrictions from NOT, sums, or nested quantifiers.
"""

from dataclasses import dataclass
from math import ceil, floor, isfinite
from random import Random
from typing import Any, Iterator

from ..language.ast import (
    BinaryOp,
    FunctionCall,
    NumberLiteral,
    Sigma,
    TupleLiteral,
    Variable,
)


@dataclass(frozen=True)
class TupleDomain:
    """Necessary tuple constraints shared by enumeration and witness search."""

    choices: tuple[tuple, ...]
    minimum: int
    maximum: int
    excluded: tuple[tuple, ...] = ()

    @property
    def size_bounds(self) -> tuple[int, int]:
        minimum = max(self.minimum, sum(None not in v for v in self.choices))
        maximum = min(
            self.maximum,
            sum(any(x is not None for x in v) for v in self.choices),
        )
        return minimum, maximum

    @property
    def empty(self) -> bool:
        minimum, maximum = self.size_bounds
        return any(not values for values in self.choices) or minimum > maximum

    def accepts(self, candidate) -> bool:
        return (
            len(candidate) == len(self.choices)
            and all(
                value in values for value, values in zip(candidate, self.choices)
            )
            and self.minimum <= sum(v is not None for v in candidate) <= self.maximum
            and tuple(candidate) not in self.excluded
        )

    def move(self, candidate, index: int, value, rng: Random) -> list | None:
        """Make a legal move, swapping support at an active SIZE boundary.

        A single-coordinate search cannot traverse SIZE(y) == k without
        exchanging an observed position for an unobserved one.
        """
        if value not in self.choices[index] or candidate[index] == value:
            return None
        proposal = list(candidate)
        proposal[index] = value
        observed = sum(v is not None for v in proposal)
        if self.minimum <= observed <= self.maximum:
            return proposal if self.accepts(proposal) else None
        partners = [
            i for i, values in enumerate(self.choices)
            if i != index and (
                (observed > self.maximum and proposal[i] is not None and None in values)
                or (
                    observed < self.minimum and proposal[i] is None
                    and any(v is not None for v in values)
                )
            )
        ]
        rng.shuffle(partners)
        for partner in partners:
            original = proposal[partner]
            values = (
                [None] if observed > self.maximum
                else [v for v in self.choices[partner] if v is not None]
            )
            rng.shuffle(values)
            for replacement in values:
                proposal[partner] = replacement
                if self.accepts(proposal):
                    return proposal
            proposal[partner] = original
        return None

    def sample(
        self, rng: Random, *, initial: bool = False,
        prefer_full: bool = False, seed=None,
    ) -> list | None:
        """Construct a feasible start without enumerating the domain.

        Valid seeds are retained. Other seeds are projected onto the tuple
        and size bounds. Strict-subset exclusions use legal moves and a
        bounded number of retries; failure never certifies an empty domain.
        """
        if self.empty:
            return None
        if seed is not None and len(seed) != len(self.choices):
            seed = None
        if seed is not None and self.accepts(seed):
            return list(seed)
        minimum, maximum = self.size_bounds
        for attempt in range(len(self.excluded) + 2):
            candidate = [
                seed[i] if seed is not None and seed[i] in values
                else None if None in values
                else rng.choice(values)
                for i, values in enumerate(self.choices)
            ]
            observed = sum(v is not None for v in candidate)
            if seed is not None:
                target = max(minimum, min(maximum, observed))
            elif initial and attempt == 0 and not prefer_full:
                target = minimum
            else:
                density = 1.0 if prefer_full else rng.choice((0.25, 0.5, 0.75, 1.0))
                target = minimum + round((maximum - minimum) * density)
            if observed < target:
                positions = [
                    i for i, v in enumerate(candidate)
                    if v is None and any(x is not None for x in self.choices[i])
                ]
                for i in rng.sample(positions, target - observed):
                    candidate[i] = rng.choice([
                        v for v in self.choices[i] if v is not None
                    ])
            elif observed > target:
                positions = [
                    i for i, v in enumerate(candidate)
                    if v is not None and None in self.choices[i]
                ]
                for i in rng.sample(positions, observed - target):
                    candidate[i] = None
            if self.accepts(candidate):
                return candidate
            positions = list(range(len(candidate)))
            rng.shuffle(positions)
            for i in positions:
                values = list(self.choices[i])
                rng.shuffle(values)
                for value in values:
                    proposal = self.move(candidate, i, value, rng)
                    if proposal is not None:
                        return proposal
            seed = None
        return None


def plan_domain(ast: Sigma, context: dict[str, Any], size: int) -> TupleDomain:
    """Plan necessary constraints using literals and current free bindings."""
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
    return TupleDomain(
        tuple(tuple(v) for v in choices), minimum, maximum, tuple(excluded)
    )


def restricted_candidates(
    ast: Sigma, context: dict[str, Any], size: int
) -> Iterator[tuple]:
    domain = plan_domain(ast, context, size)
    choices = domain.choices
    minimum, maximum = domain.minimum, domain.maximum
    excluded = domain.excluded
    if domain.empty:
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
