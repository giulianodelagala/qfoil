"""Runtime support for native and QFOIL-defined functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

from ..semantics.functions import DefinedFunction, FunctionRegistry, NativeHandler

if TYPE_CHECKING:
    from ..language.ast import ASTNode


class FunctionRuntime:
    """Register, compile, and invoke functions visible to QFOIL programs."""

    def __init__(
        self,
        native_handlers: dict[str, Optional[NativeHandler]],
        definitions: Iterable[DefinedFunction] = (),
        *,
        parser: Any = None,
    ) -> None:
        self.parser = parser
        self.registry = FunctionRegistry()
        self.compiled_cache: dict[str, ASTNode] = {}

        for name, handler in native_handlers.items():
            self.registry.register_native(name, handler)
        for definition in definitions:
            self.registry.register_defined(
                definition.name,
                definition.parameters,
                definition.body,
            )

    @staticmethod
    def _is_valid_identifier(value: Any) -> bool:
        return (
            isinstance(value, str)
            and value.isascii()
            and bool(value)
            and value[0].isalpha()
            and all(character.isalnum() or character == "_" for character in value)
        )

    def set_parser(self, parser: Any) -> None:
        self.parser = parser

    @property
    def native_names(self) -> set[str]:
        return self.registry.native_names

    def define(self, name: str, parameters: Sequence[str], body: str) -> None:
        """Register or replace a definition and eagerly compile its body."""
        if not self._is_valid_identifier(name):
            raise ValueError(f"Invalid logic function name: {name!r}")
        normalized_name = self.registry.normalize(name)
        if self.registry.has_native(normalized_name):
            raise ValueError(f"Cannot redefine native function {normalized_name}")

        if isinstance(parameters, str):
            raise ValueError("params must be a sequence of parameter names")
        normalized_parameters = list(parameters)
        if any(
            not self._is_valid_identifier(parameter)
            for parameter in normalized_parameters
        ):
            raise ValueError("Logic function parameters must be valid identifiers")
        if len(set(normalized_parameters)) != len(normalized_parameters):
            raise ValueError("Logic function parameters must be unique")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Logic function body must be a non-empty string")
        if self.parser is None:
            raise RuntimeError("Parser not set. Call set_parser() first.")

        compiled_body = self.parser.parse(body)
        self.registry.register_defined(
            normalized_name,
            normalized_parameters,
            body,
        )
        self.compiled_cache[normalized_name] = compiled_body

    def evaluate_native(self, name: str, args: list[Any]) -> Any:
        normalized_name = self.registry.normalize(name)
        handler = self.registry.native_handler(normalized_name)
        if handler is None:
            raise NotImplementedError(
                f"Function {normalized_name} is declared but not implemented"
            )
        return handler(args)

    def evaluate_defined(
        self,
        name: str,
        args: list[Any],
        context: dict[str, Any],
        evaluate_ast: Callable[[ASTNode, dict[str, Any]], Any],
    ) -> Any:
        normalized_name = self.registry.normalize(name)
        definition = self.registry.defined_function(normalized_name)
        if len(args) != len(definition.parameters):
            raise ValueError(
                f"Function {normalized_name} expects "
                f"{len(definition.parameters)} arguments, got {len(args)}"
            )

        if normalized_name not in self.compiled_cache:
            if self.parser is None:
                raise RuntimeError("Parser not set. Call set_parser() first.")
            self.compiled_cache[normalized_name] = self.parser.parse(definition.body)

        local_context = context.copy()
        local_context.update(zip(definition.parameters, args))
        return evaluate_ast(self.compiled_cache[normalized_name], local_context)


__all__ = ["FunctionRuntime"]
