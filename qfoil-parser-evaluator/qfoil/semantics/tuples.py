"""Deterministic semantics for QFOIL's partial-tuple lattice."""

from __future__ import annotations

from typing import Any, Callable


class TupleSemantics:
    """Evaluate native tuple functions independently of any runtime/backend."""

    def __init__(self, instance_len: int = 3) -> None:
        if (
            isinstance(instance_len, bool)
            or not isinstance(instance_len, int)
            or instance_len < 0
        ):
            raise ValueError("instance_len must be a non-negative integer")
        self.instance_len = instance_len

    @property
    def handlers(self) -> dict[str, Callable[[list[Any]], Any]]:
        """Native-function dispatch table supplied to a runtime registry."""
        return {
            "BOTTOM": self.bottom,
            "FULL": self.full,
            "L0": self.l0,
            "L1": self.feat,
            "FEAT": self.feat,
            "SIZE": self.size,
            "FREE": self.free,
            "MEET": self.meet,
            "JOIN": self.join,
            "PARTITION": self.partition,
            "OPPOSITE": self.opposite,
            "SUF": self.suf,
            "COMP": self.comp,
        }

    @staticmethod
    def _require_arity(func_name: str, args: list[Any], expected: int) -> None:
        if len(args) != expected:
            noun = "argument" if expected == 1 else "arguments"
            raise ValueError(f"{func_name} requires exactly {expected} {noun}")

    @staticmethod
    def _require_tuple(value: Any, func_name: str) -> tuple:
        if not isinstance(value, tuple):
            raise ValueError(
                f"{func_name} requires tuple argument, got {type(value)}: {value}"
            )
        return value

    @staticmethod
    def _require_equal_length(x: tuple, y: tuple, func_name: str) -> None:
        if len(x) != len(y):
            raise ValueError(f"{func_name} operation requires equal-length tuples")

    def bottom(self, args: list[Any]) -> tuple:
        if len(args) > 1:
            raise ValueError("BOTTOM requires 0 or 1 argument")
        if not args:
            return (None,) * self.instance_len
        length = args[0]
        if not isinstance(length, (int, float)) or length < 0 or length != int(length):
            raise ValueError(
                f"BOTTOM argument must be non-negative integer, got {length}"
            )
        return (None,) * int(length)

    def full(self, args: list[Any]) -> int:
        self._require_arity("FULL", args, 1)
        value = args[0]
        if not isinstance(value, tuple):
            raise ValueError(f"FULL argument must be tuple, got {type(value)}")
        return int(all(element is not None for element in value))

    def l0(self, args: list[Any]) -> int:
        self._require_arity("L0", args, 1)
        value = args[0]
        if not isinstance(value, tuple):
            raise ValueError(f"L0 argument must be tuple, got {type(value)}")
        return int(all(element is None for element in value))

    def feat(self, args: list[Any]) -> int:
        self._require_arity("FEAT", args, 1)
        value = args[0]
        if not isinstance(value, tuple):
            raise ValueError(f"FEAT argument must be tuple, got {type(value)}")
        return int(sum(element is not None for element in value) == 1)

    def size(self, args: list[Any]) -> int:
        self._require_arity("SIZE", args, 1)
        value = args[0]
        if not isinstance(value, tuple):
            raise ValueError(f"SIZE argument must be tuple, got {type(value)}")
        return sum(element is not None for element in value)

    def free(self, args: list[Any]) -> int:
        self._require_arity("FREE", args, 2)
        x = self._require_tuple(args[0], "FREE")
        y = self._require_tuple(args[1], "FREE")
        self._require_equal_length(x, y, "FREE")
        return sum(1 for xi, yi in zip(x, y) if xi is not None and yi != xi)

    def suf(self, args: list[tuple]) -> int:
        if len(args) != 2:
            raise ValueError("SUF requires exactly 2 arguments")
        x, y = args
        if not isinstance(x, tuple):
            raise ValueError(f"SUF first argument must be tuple, got {type(x)}")
        if not isinstance(y, tuple):
            raise ValueError(f"SUF second argument must be tuple, got {type(y)}")
        if len(x) != len(y):
            raise ValueError("SUF arguments must have same length")
        return int(all((a is None) == (b is None) for a, b in zip(x, y)))

    def opposite(self, args: list[tuple]) -> int:
        if len(args) != 2:
            raise ValueError("OPPOSITE requires exactly 2 arguments")
        x, y = args
        if not isinstance(x, tuple):
            raise ValueError(f"OPPOSITE first argument must be tuple, got {type(x)}")
        if not isinstance(y, tuple):
            raise ValueError(f"OPPOSITE second argument must be tuple, got {type(y)}")
        if len(x) != len(y):
            raise ValueError("OPPOSITE arguments must have same length")
        non_none_positions = 0
        has_different_values = False
        for x_value, y_value in zip(x, y):
            if x_value is not None and y_value is not None:
                non_none_positions += 1
                has_different_values |= x_value != y_value
                if non_none_positions > 1:
                    return 0
            elif (x_value is None) != (y_value is None):
                return 0
        return int(non_none_positions == 1 and has_different_values)

    def meet(self, args: list[tuple]) -> int:
        if len(args) != 3:
            raise ValueError("MEET requires exactly 3 arguments")
        if not all(isinstance(arg, tuple) for arg in args):
            raise ValueError("MEET arguments must be tuples")
        x, y, z = args
        if not (len(x) == len(y) == len(z)):
            raise ValueError("MEET arguments must be tuples of same length")
        return int(
            all(c == (a if a == b else None) for a, b, c in zip(x, y, z))
        )

    def join(self, args: list[tuple]) -> int:
        if len(args) != 3:
            raise ValueError("JOIN requires exactly 3 arguments")
        if not all(isinstance(arg, tuple) for arg in args):
            raise ValueError("JOIN arguments must be tuples")
        x, y, z = args
        if not (len(x) == len(y) == len(z)):
            raise ValueError("JOIN arguments must be tuples of same length")
        for a, b, c in zip(x, y, z):
            if a is not None and b is not None and a != b:
                return 0
            if c != (a if a is not None else b):
                return 0
        return 1

    def partition(self, args: list[tuple]) -> int:
        if len(args) != 2:
            raise ValueError("PARTITION requires exactly 2 arguments")
        if not all(isinstance(arg, tuple) for arg in args):
            raise ValueError("PARTITION arguments must be tuples")
        x, y = args
        if len(x) != len(y):
            raise ValueError("PARTITION arguments must be tuples of same length")
        return int(all(a is None or b is None for a, b in zip(x, y)))

    @staticmethod
    def subseteq(left: tuple, right: tuple) -> int:
        if len(left) != len(right):
            raise ValueError("SUBSETEQ operation requires equal-length tuples")
        return int(all(a is None or a == b for a, b in zip(left, right)))

    def subset(self, left: tuple, right: tuple) -> int:
        return int(bool(self.subseteq(left, right)) and left != right)

    def comp(self, args: list[tuple]) -> int:
        if len(args) != 2:
            raise ValueError("COMP requires exactly 2 arguments")
        if not all(isinstance(arg, tuple) for arg in args):
            raise ValueError("COMP arguments must be tuples")
        x, y = args
        if len(x) != len(y):
            raise ValueError("COMP operation requires equal-length tuples")
        return self.full([y]) if self.subseteq(x, y) else 0


__all__ = ["TupleSemantics"]
