"""HTTP request and response schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .config import (
    DEFAULT_EVALUATION_BUDGET,
    DEFAULT_INNER_BUDGET,
    DEFAULT_POOL_SIZE,
    DEFAULT_SEARCH_BUDGET,
    DEFAULT_SIGMA_SEARCH_BUDGET,
    MAX_EVALUATION_BUDGET,
    MAX_SIGMA_SEARCH_BUDGET,
    MAX_INNER_BUDGET,
    MAX_POOL_SIZE,
    MAX_SEARCH_BUDGET,
)

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictModel):
    status: str
    service: str
    model_status: str
    model_id: Optional[str] = None


class ExpressionRequest(StrictModel):
    expression: str = Field(min_length=1, max_length=10_000)


class EvaluationOptions(StrictModel):
    mode: Literal["heuristic", "exact"] = "heuristic"
    sigma_search_budget: int = Field(
        default=DEFAULT_SIGMA_SEARCH_BUDGET, ge=0, le=MAX_SIGMA_SEARCH_BUDGET, strict=True
    )
    evaluation_budget: Optional[int] = Field(
        default=None, ge=0, le=MAX_EVALUATION_BUDGET, strict=True
    )
    rng_seed: Optional[StrictInt] = None
    seed: Optional[list[Annotated[int, Field(strict=True, ge=0, le=1)] | None]] = Field(
        default=None, max_length=10_000
    )


class EvaluationConfig(StrictModel):
    defaults: EvaluationOptions = Field(default_factory=EvaluationOptions)
    modes: list[str] = Field(default_factory=lambda: ["heuristic", "exact"])
    max_sigma_search_budget: int = MAX_SIGMA_SEARCH_BUDGET
    max_evaluation_budget: int = MAX_EVALUATION_BUDGET


class EvaluateRequest(ExpressionRequest):
    bindings: dict[str, Any] = Field(default_factory=dict)
    options: EvaluationOptions = Field(default_factory=EvaluationOptions)


class EvaluateResponse(StrictModel):
    status: Literal["decided", "inconclusive"] = "decided"
    value: Any
    last_witness: list[int | None] | None = None
    reason: str | None = None


class LatexResponse(StrictModel):
    latex: str


class SearchOptions(StrictModel):
    budget: int = Field(default=DEFAULT_SEARCH_BUDGET, ge=0, le=MAX_SEARCH_BUDGET)
    pool_size: int = Field(default=DEFAULT_POOL_SIZE, ge=1, le=MAX_POOL_SIZE)
    inner_budget: int = Field(default=DEFAULT_INNER_BUDGET, ge=0, le=MAX_INNER_BUDGET)
    seed: Optional[list[int | None]] = None


class WitnessSearchRequest(ExpressionRequest):
    variable: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    bindings: dict[str, Any]
    options: SearchOptions = Field(default_factory=SearchOptions)


class SearchInfo(StrictModel):
    budget: int
    pool_size: int
    inner_budget: int


class SearchConfig(StrictModel):
    defaults: SearchInfo = Field(default_factory=lambda: SearchInfo(
        budget=DEFAULT_SEARCH_BUDGET,
        pool_size=DEFAULT_POOL_SIZE,
        inner_budget=DEFAULT_INNER_BUDGET,
    ))
    max: SearchInfo = Field(default_factory=lambda: SearchInfo(
        budget=MAX_SEARCH_BUDGET,
        pool_size=MAX_POOL_SIZE,
        inner_budget=MAX_INNER_BUDGET,
    ))


class ConfigResponse(StrictModel):
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    evaluation_budget: int = DEFAULT_EVALUATION_BUDGET
    sigma_search_budget: int = DEFAULT_SIGMA_SEARCH_BUDGET


class WitnessSearchResponse(StrictModel):
    status: Literal["found", "inconclusive"]
    variable: str
    witness: Optional[Any] = None
    verified: bool
    seed_used: Optional[Any] = None
    search: SearchInfo
    reason: str | None = None


class ModelSummarySchema(StrictModel):
    id: str
    backend: str


class ModelInfo(StrictModel):
    status: Literal["unloaded", "loading", "ready", "unloading", "error"]
    id: Optional[str] = None
    backend: Optional[str] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    device: Optional[str] = None
    loaded_at: Optional[str] = None


class ModelListResponse(StrictModel):
    models: list[ModelSummarySchema]


class ModelStateResponse(StrictModel):
    model: ModelInfo


class TestDatasetSampleResponse(StrictModel):
    model_id: str
    split: Literal["test"] = "test"
    index: int
    dataset_size: int
    sample: Any
    expected: Any


class ErrorDetail(StrictModel):
    code: str
    message: str


class ErrorResponse(StrictModel):
    error: ErrorDetail
