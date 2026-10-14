"""FastAPI adapter around the QFOIL Python facade."""

from __future__ import annotations

import json
import math
from typing import Any, Callable, TypeVar

import uvicorn
from fastapi import Depends, FastAPI, Path, Query
from fastapi.responses import JSONResponse

from qfoil import (
    QFOIL,
    QFOILError,
    QFOILSyntaxError,
    QFOILValidationError,
    UnsupportedOperationError,
    Inconclusive,
)

from .model_catalog import load_catalog_from_env
from .supported_fragment import check_supported_fragment
from .model_manager import (
    BackendUnavailableError,
    ModelAlreadyLoadedError,
    ModelBusyError,
    ModelLoadError,
    ModelManager,
    ModelNotLoadedError,
    TestDatasetEmptyError,
    TestDatasetIndexError,
    TestDatasetSerializationError,
    TestDatasetUnavailableError,
    UnknownModelError,
)
from .schemas import (
    ConfigResponse,
    EvaluateRequest,
    EvaluateResponse,
    ExpressionRequest,
    HealthResponse,
    LatexResponse,
    ModelInfo,
    ModelListResponse,
    ModelStateResponse,
    ModelSummarySchema,
    TestDatasetSampleResponse,
    WitnessSearchRequest,
    WitnessSearchResponse,
)

MAX_ARRAY_LENGTH = 10_000
MODEL_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
T = TypeVar("T")


def _decode_json_value(value: Any) -> Any:
    """Convert JSON arrays to the tuple representation expected by QFOIL."""
    if isinstance(value, list):
        if len(value) > MAX_ARRAY_LENGTH:
            raise QFOILValidationError(
                f"arrays cannot contain more than {MAX_ARRAY_LENGTH} elements"
            )
        return tuple(_decode_json_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _decode_json_value(item) for key, item in value.items()}
    return value


def _call_qfoil(operation: Callable[[], T]) -> T:
    """Normalize legacy validation failures at the transport boundary."""
    try:
        return operation()
    except QFOILError:
        raise
    except (TypeError, ValueError) as exc:
        raise QFOILValidationError(str(exc)) from exc


def _error_response(status_code: int, code: str, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": str(exc)}},
    )


def _parse_decimal_integer(value: str, *, name: str) -> int:
    """Parse an HTTP integer without accepting floats or surrounding syntax."""
    try:
        if not value or value.strip() != value:
            raise ValueError
        return int(value, 10)
    except ValueError as exc:
        raise QFOILValidationError(f"{name} must be an integer") from exc


def _json_value(value: Any) -> Any:
    """Convert supported backend containers/scalars to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TestDatasetSerializationError()
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TestDatasetSerializationError()
        return {key: _json_value(item) for key, item in value.items()}

    # NumPy scalars and Torch tensors expose one of these conversion methods.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    raise TestDatasetSerializationError()


def _sample_response(raw: tuple[str, int, int, Any, Any]) -> dict[str, Any]:
    model_id, index, dataset_size, sample, expected = raw
    try:
        response = {
            "model_id": model_id,
            "split": "test",
            "index": index,
            "dataset_size": dataset_size,
            "sample": _json_value(sample),
            "expected": _json_value(expected),
        }
        json.dumps(response, allow_nan=False)
    except TestDatasetSerializationError:
        raise
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise TestDatasetSerializationError() from exc
    return response


def create_app(
    qfoil: QFOIL | None = None,
    model_manager: ModelManager | None = None,
) -> FastAPI:
    """Create an application around one model manager.

    Pass ``qfoil`` to customize the backend-independent runtime used before
    any model is loaded and after every unload. Pass ``model_manager``
    directly (typically in tests) to bypass catalog loading and PyJuice
    entirely; an injected manager never reads ``QFOIL_MODEL_CATALOG`` or
    imports PyJuice. The server always starts with no probabilistic model
    active -- a model is only loaded when a client calls the load endpoint.
    """
    manager = model_manager or ModelManager(
        default_runtime=qfoil,
        catalog=load_catalog_from_env(),
    )
    application = FastAPI(
        title="qfoil-server",
        version="0.1.0",
        description="Thin HTTP adapter for QFOIL expressions.",
    )

    async def get_model_manager() -> ModelManager:
        return manager

    @application.exception_handler(QFOILSyntaxError)
    async def syntax_error_handler(_request, exc: QFOILSyntaxError):
        return _error_response(400, "qfoil_syntax_error", exc)

    @application.exception_handler(QFOILValidationError)
    async def validation_error_handler(_request, exc: QFOILValidationError):
        return _error_response(422, "qfoil_validation_error", exc)

    @application.exception_handler(UnsupportedOperationError)
    async def unsupported_error_handler(_request, exc: UnsupportedOperationError):
        return _error_response(501, "unsupported_operation", exc)

    @application.exception_handler(QFOILError)
    async def qfoil_error_handler(_request, exc: QFOILError):
        return _error_response(500, "qfoil_error", exc)

    @application.exception_handler(UnknownModelError)
    async def unknown_model_handler(_request, exc: UnknownModelError):
        return _error_response(404, "unknown_model", exc)

    @application.exception_handler(ModelAlreadyLoadedError)
    async def model_already_loaded_handler(_request, exc: ModelAlreadyLoadedError):
        return _error_response(409, "model_already_loaded", exc)

    @application.exception_handler(ModelBusyError)
    async def model_busy_handler(_request, exc: ModelBusyError):
        return _error_response(409, "model_busy", exc)

    @application.exception_handler(BackendUnavailableError)
    async def backend_unavailable_handler(_request, exc: BackendUnavailableError):
        return _error_response(501, "backend_unavailable", exc)

    @application.exception_handler(ModelLoadError)
    async def model_load_error_handler(_request, exc: ModelLoadError):
        return _error_response(500, "model_load_error", exc)

    @application.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(_request, exc: ModelNotLoadedError):
        return _error_response(409, "model_not_loaded", exc)

    @application.exception_handler(TestDatasetUnavailableError)
    async def test_dataset_unavailable_handler(
        _request, exc: TestDatasetUnavailableError
    ):
        return _error_response(404, "test_dataset_unavailable", exc)

    @application.exception_handler(TestDatasetEmptyError)
    async def test_dataset_empty_handler(_request, exc: TestDatasetEmptyError):
        return _error_response(409, "test_dataset_empty", exc)

    @application.exception_handler(TestDatasetIndexError)
    async def test_dataset_index_handler(_request, exc: TestDatasetIndexError):
        return _error_response(404, "test_dataset_index_out_of_range", exc)

    @application.exception_handler(TestDatasetSerializationError)
    async def test_dataset_serialization_handler(
        _request, exc: TestDatasetSerializationError
    ):
        return _error_response(500, "test_dataset_serialization_error", exc)

    @application.get(
        "/health", response_model=HealthResponse, response_model_exclude_none=True
    )
    async def health(
        manager: ModelManager = Depends(get_model_manager),
    ) -> HealthResponse:
        metadata = await manager.current()
        return HealthResponse(
            status="ok",
            service="qfoil-server",
            model_status=metadata.status,
            model_id=metadata.id,
        )

    @application.get("/v1/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        """Expose static service defaults and limits without changing runtime state."""
        return ConfigResponse()

    @application.post("/v1/expressions/evaluate", response_model=EvaluateResponse)
    async def evaluate(
        request: EvaluateRequest,
        manager: ModelManager = Depends(get_model_manager),
    ) -> EvaluateResponse:
        bindings = _decode_json_value(request.bindings)

        def operation(runtime: QFOIL):
            runtime.evaluator.sigma.last_witness = None
            check_supported_fragment(request.expression, runtime)
            options = request.options
            if options.seed is not None and len(options.seed) != runtime.evaluator.instance_len:
                raise QFOILValidationError("seed length must match the active tuple domain")
            value = _call_qfoil(
                lambda: runtime.evaluate(
                    request.expression,
                    bindings=bindings,
                    sigma_mode="restricted" if options.mode == "exact" else options.mode,
                    evaluation_budget=options.evaluation_budget,
                    sigma_search_budget=options.sigma_search_budget,
                    rng_seed=options.rng_seed,
                    seed=tuple(options.seed) if options.seed is not None else None,
                )
            )
            witness = runtime.evaluator.sigma.last_witness
            return value, list(witness) if witness is not None else None

        value, last_witness = await manager.execute(operation)
        if isinstance(value, Inconclusive):
            return EvaluateResponse(
                status="inconclusive", value=None, last_witness=None,
                reason=value.reason,
            )
        return EvaluateResponse(value=value, last_witness=last_witness)

    @application.post("/v1/expressions/latex", response_model=LatexResponse)
    async def latex(
        request: ExpressionRequest,
        manager: ModelManager = Depends(get_model_manager),
    ) -> LatexResponse:
        value = await manager.execute_latex(
            lambda runtime: _call_qfoil(lambda: runtime.to_latex(request.expression))
        )
        return LatexResponse(latex=value)

    @application.post(
        "/v1/witnesses/search", response_model=WitnessSearchResponse
    )
    async def search_witness(
        request: WitnessSearchRequest,
        manager: ModelManager = Depends(get_model_manager),
    ) -> dict[str, Any]:
        bindings = _decode_json_value(request.bindings)
        options = request.options
        seed = tuple(options.seed) if options.seed is not None else None

        def operation(runtime: QFOIL):
            check_supported_fragment(request.expression, runtime)
            return _call_qfoil(
                lambda: runtime.search_witness(
                    request.expression,
                    var=request.variable,
                    context=bindings,
                    budget=options.budget,
                    pool_size=options.pool_size,
                    inner_budget=options.inner_budget,
                    seed=seed,
                )
            )

        result = await manager.execute(operation)
        return result.to_dict()

    @application.get("/v1/models", response_model=ModelListResponse)
    async def list_models(
        manager: ModelManager = Depends(get_model_manager),
    ) -> ModelListResponse:
        summaries = await manager.available()
        return ModelListResponse(
            models=[ModelSummarySchema(**summary.to_dict()) for summary in summaries]
        )

    @application.get(
        "/v1/models/current",
        response_model=ModelStateResponse,
        response_model_exclude_none=True,
    )
    async def current_model(
        manager: ModelManager = Depends(get_model_manager),
    ) -> ModelStateResponse:
        metadata = await manager.current()
        return ModelStateResponse(model=ModelInfo(**metadata.to_dict()))

    @application.get(
        "/v1/models/current/dataset/test/samples/random",
        response_model=TestDatasetSampleResponse,
    )
    async def random_test_sample(
        seed: str | None = Query(default=None),
        manager: ModelManager = Depends(get_model_manager),
    ) -> dict[str, Any]:
        parsed_seed = (
            _parse_decimal_integer(seed, name="seed") if seed is not None else None
        )
        return _sample_response(await manager.test_sample(seed=parsed_seed))

    @application.get(
        "/v1/models/current/dataset/test/samples/{index}",
        response_model=TestDatasetSampleResponse,
    )
    async def indexed_test_sample(
        index: str,
        manager: ModelManager = Depends(get_model_manager),
    ) -> dict[str, Any]:
        parsed_index = _parse_decimal_integer(index, name="index")
        if parsed_index < 0:
            raise QFOILValidationError("index must be non-negative")
        return _sample_response(await manager.test_sample(index=parsed_index))

    @application.post(
        "/v1/models/{model_id}/load",
        response_model=ModelStateResponse,
        response_model_exclude_none=True,
    )
    async def load_model(
        model_id: str = Path(pattern=MODEL_ID_PATTERN),
        manager: ModelManager = Depends(get_model_manager),
    ) -> ModelStateResponse:
        metadata = await manager.load(model_id)
        return ModelStateResponse(model=ModelInfo(**metadata.to_dict()))

    @application.delete(
        "/v1/models/current",
        response_model=ModelStateResponse,
        response_model_exclude_none=True,
    )
    async def unload_model(
        manager: ModelManager = Depends(get_model_manager),
    ) -> ModelStateResponse:
        metadata = await manager.unload()
        return ModelStateResponse(model=ModelInfo(**metadata.to_dict()))

    return application


app = create_app()


def run() -> None:
    """Run the development ASGI server."""
    uvicorn.run("qfoil_server.main:app", host="127.0.0.1", port=8000)
