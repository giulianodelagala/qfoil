"""Registry for native and QFOIL-defined functions."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

NativeHandler = Callable[[list[Any]], Any]


@dataclass(frozen=True)
class DefinedFunction:
    """A parameterized function whose body is written in QFOIL."""

    name: str
    parameters: tuple[str, ...]
    body: str


class FunctionRegistry:
    """Single source of truth for functions visible to a QFOIL runtime."""

    def __init__(self) -> None:
        self._native: dict[str, Optional[NativeHandler]] = {}
        self._defined: dict[str, DefinedFunction] = {}

    @staticmethod
    def normalize(name: str) -> str:
        return name.upper()

    def register_native(
        self,
        name: str,
        handler: Optional[NativeHandler],
    ) -> None:
        normalized = self.normalize(name)
        if normalized in self._defined:
            raise ValueError(
                f"Cannot register native function {normalized}: "
                "a QFOIL definition already uses that name"
            )
        self._native[normalized] = handler

    def register_defined(
        self,
        name: str,
        parameters: Iterable[str],
        body: str,
    ) -> DefinedFunction:
        normalized = self.normalize(name)
        if normalized in self._native:
            raise ValueError(
                f"Cannot redefine native function: {normalized}"
            )
        definition = DefinedFunction(normalized, tuple(parameters), body)
        self._defined[normalized] = definition
        return definition

    def has_native(self, name: str) -> bool:
        return self.normalize(name) in self._native

    def has_defined(self, name: str) -> bool:
        return self.normalize(name) in self._defined

    def native_handler(self, name: str) -> Optional[NativeHandler]:
        return self._native.get(self.normalize(name))

    def defined_function(self, name: str) -> DefinedFunction:
        return self._defined[self.normalize(name)]

    @property
    def native_names(self) -> set[str]:
        return set(self._native)

    @property
    def defined_names(self) -> set[str]:
        return set(self._defined)
