// Transport types and the fetch client for the qfoil-server HTTP contract.
// See qfoil-server/API_CONTRACT.md.

export type HealthResponse = {
  status: string;
  service: string;
  model_status: "unloaded" | "loading" | "ready" | "unloading" | "error";
  model_id?: string;
};

export type Bindings = Record<string, unknown>;

export type EvaluationOptions = {
  mode: "heuristic" | "exact";
  sigma_search_budget?: number;
  evaluation_budget?: number | null;
  rng_seed?: number | null;
  seed?: Witness | null;
};

export type EvaluateRequest = {
  expression: string;
  bindings: Bindings;
  options?: EvaluationOptions;
};

export type Witness = Array<0 | 1 | null>;

export type EvaluateResponse = {
  status?: "decided" | "inconclusive";
  reason?: string | null;
  value: unknown;
  last_witness: Witness | null;
};

export type LatexRequest = {
  expression: string;
};

export type LatexResponse = {
  latex: string;
};

export type SearchOptions = {
  budget?: number;
  pool_size?: number;
  inner_budget?: number;
  seed?: (number | null)[] | null;
};

export type WitnessSearchRequest = {
  expression: string;
  variable: string;
  bindings: Bindings;
  options?: SearchOptions;
};

export type WitnessSearchResponse = {
  status: "found" | "inconclusive";
  variable: string;
  witness: unknown;
  verified: boolean;
  seed_used: unknown;
  search: {
    budget: number;
    pool_size: number;
    inner_budget: number;
  };
  reason?: string | null;
};

// Read-only service configuration (GET /v1/config) -------------------------
//
// Defaults and server-side maximums for witness search options, plus the
// evaluation defaults and per-request budget limits.

export type SearchLimits = {
  budget: number;
  pool_size: number;
  inner_budget: number;
};

export type ConfigResponse = {
  evaluation: {
    defaults: EvaluationOptions;
    modes: Array<"heuristic" | "exact">;
    max_sigma_search_budget: number;
    max_evaluation_budget: number;
  };
  search: {
    defaults: SearchLimits;
    max: SearchLimits;
  };
  evaluation_budget: number;
  sigma_search_budget: number;
};

// Model lifecycle -----------------------------------------------------------

export type ModelSummary = {
  id: string;
  backend: string;
};

export type ModelListResponse = {
  models: ModelSummary[];
};

export type UnloadedModel = {
  status: "unloaded";
};

export type TransitionalModel = {
  status: "loading" | "unloading" | "error";
  id?: string;
  backend?: string;
};

export type ReadyModel = {
  status: "ready";
  id: string;
  backend: string;
  num_features: number;
  num_classes: number;
  device: string;
  loaded_at: string;
};

export type ModelInfo = UnloadedModel | TransitionalModel | ReadyModel;

export type ModelStateResponse = {
  model: ModelInfo;
};

export type ApiError = {
  kind: "network" | "http" | "qfoil" | "validation" | "unexpected";
  code?: string;
  message: string;
  status?: number;
};

export class ApiRequestError extends Error {
  readonly apiError: ApiError;

  constructor(apiError: ApiError) {
    super(apiError.message);
    this.apiError = apiError;
  }
}

const apiBaseUrl =
  import.meta.env.VITE_QFOIL_API_URL?.replace(/\/$/, "") ?? "/api";

function isQfoilErrorEnvelope(
  value: unknown,
): value is { error: { code: string; message: string } } {
  if (typeof value !== "object" || value === null || !("error" in value)) {
    return false;
  }
  const error = (value as { error: unknown }).error;
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    "message" in error &&
    typeof (error as { code: unknown }).code === "string" &&
    typeof (error as { message: unknown }).message === "string"
  );
}

function isFastApiValidationError(
  value: unknown,
): value is { detail: unknown } {
  return (
    typeof value === "object" &&
    value !== null &&
    "detail" in value &&
    !("error" in value)
  );
}

function describeFastApiDetail(detail: unknown): string {
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (
          typeof item === "object" &&
          item !== null &&
          "msg" in item &&
          typeof (item as { msg: unknown }).msg === "string"
        ) {
          return (item as { msg: string }).msg;
        }
        return null;
      })
      .filter((message): message is string => message !== null);
    if (messages.length > 0) {
      return messages.join("; ");
    }
  }
  return "The request was rejected as invalid.";
}

async function buildHttpError(response: Response): Promise<ApiRequestError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return new ApiRequestError({
      kind: "unexpected",
      message: `The server returned an unreadable ${response.status} response.`,
      status: response.status,
    });
  }

  if (isQfoilErrorEnvelope(body)) {
    return new ApiRequestError({
      kind: "qfoil",
      code: body.error.code,
      message: body.error.message,
      status: response.status,
    });
  }

  if (response.status === 422 && isFastApiValidationError(body)) {
    return new ApiRequestError({
      kind: "validation",
      message: describeFastApiDetail(body.detail),
      status: response.status,
    });
  }

  return new ApiRequestError({
    kind: "unexpected",
    message: `The server returned an unrecognized ${response.status} response.`,
    status: response.status,
  });
}

// Shared network handling: every request below (GET, POST with or without a
// body, DELETE) funnels through here so error normalization is defined once.
async function request<TResponse>(
  path: string,
  init: RequestInit,
  signal?: AbortSignal,
): Promise<TResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, { ...init, signal });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") {
      throw cause;
    }
    throw new ApiRequestError({
      kind: "network",
      message: "Could not reach qfoil-server. Check that it is running.",
    });
  }

  if (!response.ok) {
    throw await buildHttpError(response);
  }

  try {
    return (await response.json()) as TResponse;
  } catch {
    throw new ApiRequestError({
      kind: "unexpected",
      message: "The server returned an unreadable successful response.",
      status: response.status,
    });
  }
}

function getJson<TResponse>(
  path: string,
  signal?: AbortSignal,
): Promise<TResponse> {
  return request<TResponse>(path, { method: "GET" }, signal);
}

function postJson<TResponse>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<TResponse> {
  return request<TResponse>(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    signal,
  );
}

function postNoBody<TResponse>(
  path: string,
  signal?: AbortSignal,
): Promise<TResponse> {
  return request<TResponse>(path, { method: "POST" }, signal);
}

function deleteJson<TResponse>(
  path: string,
  signal?: AbortSignal,
): Promise<TResponse> {
  return request<TResponse>(path, { method: "DELETE" }, signal);
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return getJson<HealthResponse>("/health", signal);
}

export function getConfig(signal?: AbortSignal): Promise<ConfigResponse> {
  return getJson<ConfigResponse>("/v1/config", signal);
}

export async function evaluateExpression(
  request: EvaluateRequest,
  signal?: AbortSignal,
): Promise<EvaluateResponse> {
  const response = await postJson<EvaluateResponse>(
    "/v1/expressions/evaluate",
    request,
    signal,
  );

  if (typeof response !== "object" || response === null || !("last_witness" in response)) {
    throw new ApiRequestError({
      kind: "unexpected",
      message:
        "The server response is missing the required last_witness field.",
    });
  }

  return response;
}

export function renderLatex(
  request: LatexRequest,
  signal?: AbortSignal,
): Promise<LatexResponse> {
  return postJson<LatexResponse>("/v1/expressions/latex", request, signal);
}

export function searchWitness(
  request: WitnessSearchRequest,
  signal?: AbortSignal,
): Promise<WitnessSearchResponse> {
  return postJson<WitnessSearchResponse>(
    "/v1/witnesses/search",
    request,
    signal,
  );
}

export function listModels(signal?: AbortSignal): Promise<ModelListResponse> {
  return getJson<ModelListResponse>("/v1/models", signal);
}

export function getCurrentModel(
  signal?: AbortSignal,
): Promise<ModelStateResponse> {
  return getJson<ModelStateResponse>("/v1/models/current", signal);
}

export function loadModel(
  modelId: string,
  signal?: AbortSignal,
): Promise<ModelStateResponse> {
  return postNoBody<ModelStateResponse>(
    `/v1/models/${encodeURIComponent(modelId)}/load`,
    signal,
  );
}

export function unloadModel(
  signal?: AbortSignal,
): Promise<ModelStateResponse> {
  return deleteJson<ModelStateResponse>("/v1/models/current", signal);
}

// Test dataset samples -------------------------------------------------------
//
// Read-only inspection of the active model's test split. `sample` and
// `expected` are left as `unknown`: the server may return a nested
// list/object structure rather than a flat tuple, and the client must not
// reshape, truncate, or otherwise interpret it.

export type DatasetSampleResponse = {
  model_id: string;
  split: string;
  index: number;
  dataset_size: number;
  sample: unknown;
  expected: unknown;
};

function isDatasetSampleResponse(
  value: unknown,
): value is DatasetSampleResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "model_id" in value &&
    "split" in value &&
    "index" in value &&
    "dataset_size" in value &&
    "sample" in value &&
    "expected" in value
  );
}

async function getDatasetSample(
  path: string,
  signal?: AbortSignal,
): Promise<DatasetSampleResponse> {
  const response = await getJson<DatasetSampleResponse>(path, signal);
  if (!isDatasetSampleResponse(response)) {
    throw new ApiRequestError({
      kind: "unexpected",
      message:
        "The server response is missing required test dataset sample fields.",
    });
  }
  return response;
}

export function getTestSample(
  index: number,
  signal?: AbortSignal,
): Promise<DatasetSampleResponse> {
  return getDatasetSample(
    `/v1/models/current/dataset/test/samples/${index}`,
    signal,
  );
}

export function getRandomTestSample(
  seed: number | null = null,
  signal?: AbortSignal,
): Promise<DatasetSampleResponse> {
  const query = seed === null ? "" : `?seed=${seed}`;
  return getDatasetSample(
    `/v1/models/current/dataset/test/samples/random${query}`,
    signal,
  );
}
