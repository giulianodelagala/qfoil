import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiRequestError,
  evaluateExpression,
  getConfig,
  getCurrentModel,
  getHealth,
  getRandomTestSample,
  getTestSample,
  listModels,
  loadModel,
  renderLatex,
  searchWitness,
  unloadModel,
} from "../api";

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

describe("api client", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("builds the evaluate request with expression and bindings", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, { value: 1, last_witness: [null, 1, 0, null] }),
    );

    const result = await evaluateExpression({
      expression: "SIZE(x) <= 2",
      bindings: { x: [1, null, 0] },
    });

    expect(result).toEqual({ value: 1, last_witness: [null, 1, 0, null] });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/expressions/evaluate");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      expression: "SIZE(x) <= 2",
      bindings: { x: [1, null, 0] },
    });
  });

  it("accepts a null last_witness", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, { value: 0, last_witness: null }),
    );

    const result = await evaluateExpression({
      expression: "SIZE(x) <= 2",
      bindings: {},
    });

    expect(result).toEqual({ value: 0, last_witness: null });
  });

  it("treats a response missing last_witness as a contract error", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { value: 1 }));

    await expect(
      evaluateExpression({ expression: "SIZE(x) <= 2", bindings: {} }),
    ).rejects.toMatchObject({
      apiError: { kind: "unexpected" },
    });
  });

  it("builds the witness search request with expression, variable, bindings and options", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        status: "found",
        variable: "z",
        witness: [1, 0, 1],
        verified: true,
        seed_used: null,
        search: { budget: 100, pool_size: 3, inner_budget: 8 },
      }),
    );

    await searchWitness({
      expression: "FULL(z) * (z <<< x)",
      variable: "z",
      bindings: { x: [1, 0, 1] },
      options: { budget: 100, pool_size: 3, inner_budget: 8, seed: null },
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/witnesses/search");
    expect(JSON.parse(init.body)).toEqual({
      expression: "FULL(z) * (z <<< x)",
      variable: "z",
      bindings: { x: [1, 0, 1] },
      options: { budget: 100, pool_size: 3, inner_budget: 8, seed: null },
    });
  });

  it("builds the witness search request without an options field when omitted", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        status: "found",
        variable: "z",
        witness: [1, 0, 1],
        verified: true,
        seed_used: null,
        search: { budget: 1000, pool_size: 30, inner_budget: 8 },
      }),
    );

    await searchWitness({
      expression: "FULL(z) * (z <<< x)",
      variable: "z",
      bindings: { x: [1, 0, 1] },
    });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      expression: "FULL(z) * (z <<< x)",
      variable: "z",
      bindings: { x: [1, 0, 1] },
    });
  });

  it("getConfig uses GET /v1/config", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        search: {
          defaults: { budget: 1000, pool_size: 30, inner_budget: 8 },
          max: { budget: 100000, pool_size: 1000, inner_budget: 1000 },
        },
        evaluation_budget: 200000,
        sigma_search_budget: 64,
      }),
    );

    const result = await getConfig();

    expect(result.search.defaults).toEqual({
      budget: 1000,
      pool_size: 30,
      inner_budget: 8,
    });
    expect(result.evaluation_budget).toBe(200000);
    expect(result.sigma_search_budget).toBe(64);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/config");
    expect(init.method).toBe("GET");
  });

  it("processes a successful LaTeX response", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, { latex: "\\operatorname{SIZE}\\left(x\\right) \\leq 2" }),
    );

    const result = await renderLatex({ expression: "SIZE(x) <= 2" });

    expect(result.latex).toBe("\\operatorname{SIZE}\\left(x\\right) \\leq 2");
  });

  it("normalizes the QFOIL error envelope", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(400, {
        error: { code: "qfoil_syntax_error", message: "unexpected token" },
      }),
    );

    await expect(renderLatex({ expression: "SIZE(" })).rejects.toMatchObject({
      apiError: {
        kind: "qfoil",
        code: "qfoil_syntax_error",
        message: "unexpected token",
        status: 400,
      },
    });
  });

  it("normalizes the standard FastAPI 422 response", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(422, {
        detail: [{ loc: ["body", "expression"], msg: "field required", type: "value_error" }],
      }),
    );

    let caught: unknown;
    try {
      await evaluateExpression({ expression: "", bindings: {} });
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ApiRequestError);
    expect((caught as ApiRequestError).apiError.kind).toBe("validation");
    expect((caught as ApiRequestError).apiError.message).toMatch(/field required/);
  });

  it("normalizes a network failure", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));

    await expect(
      evaluateExpression({ expression: "SIZE(x) <= 2", bindings: {} }),
    ).rejects.toMatchObject({
      apiError: { kind: "network" },
    });
  });

  it("getHealth reflects the current health contract (model_status, model_id)", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        status: "ok",
        service: "qfoil-server",
        model_status: "ready",
        model_id: "mux",
      }),
    );

    const result = await getHealth();

    expect(result).toEqual({
      status: "ok",
      service: "qfoil-server",
      model_status: "ready",
      model_id: "mux",
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/health");
    expect(init.method).toBe("GET");
  });

  it("listModels uses GET /v1/models", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        models: [
          { id: "mux", backend: "pyjuice" },
          { id: "spam", backend: "pyjuice" },
        ],
      }),
    );

    const result = await listModels();

    expect(result.models).toHaveLength(2);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/models");
    expect(init.method).toBe("GET");
  });

  it("getCurrentModel uses GET /v1/models/current", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, { model: { status: "unloaded" } }),
    );

    const result = await getCurrentModel();

    expect(result).toEqual({ model: { status: "unloaded" } });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/models/current");
    expect(init.method).toBe("GET");
  });

  it("loadModel encodes the model_id and issues a POST with no body", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        model: {
          status: "ready",
          id: "mux spec/al",
          backend: "pyjuice",
          num_features: 8,
          num_classes: 2,
          device: "cuda:0",
          loaded_at: "2026-09-24T17:32:00Z",
        },
      }),
    );

    await loadModel("mux spec/al");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/models/mux%20spec%2Fal/load");
    expect(init.method).toBe("POST");
    expect(init.body).toBeUndefined();
  });

  it("unloadModel uses DELETE /v1/models/current", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, { model: { status: "unloaded" } }),
    );

    const result = await unloadModel();

    expect(result).toEqual({ model: { status: "unloaded" } });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/models/current");
    expect(init.method).toBe("DELETE");
  });

  it("all four lifecycle methods process the QFOIL error envelope", async () => {
    const errorBody = {
      error: { code: "unknown_model", message: "no such model" },
    };
    fetchMock.mockResolvedValue(jsonResponse(404, errorBody));

    await expect(listModels()).rejects.toMatchObject({
      apiError: { kind: "qfoil", code: "unknown_model" },
    });
    await expect(getCurrentModel()).rejects.toMatchObject({
      apiError: { kind: "qfoil", code: "unknown_model" },
    });
    await expect(loadModel("mux")).rejects.toMatchObject({
      apiError: { kind: "qfoil", code: "unknown_model" },
    });
    await expect(unloadModel()).rejects.toMatchObject({
      apiError: { kind: "qfoil", code: "unknown_model" },
    });
  });

  it("preserves code, status and message for model_already_loaded", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(409, {
        error: {
          code: "model_already_loaded",
          message: "A model is already active; unload it first",
        },
      }),
    );

    await expect(loadModel("spam")).rejects.toMatchObject({
      apiError: {
        kind: "qfoil",
        code: "model_already_loaded",
        message: "A model is already active; unload it first",
        status: 409,
      },
    });
  });

  it("getTestSample requests the indexed test dataset sample endpoint", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        model_id: "example-model",
        split: "test",
        index: 42,
        dataset_size: 10000,
        sample: [1, null, 0],
        expected: 1,
      }),
    );

    const result = await getTestSample(42);

    expect(result).toEqual({
      model_id: "example-model",
      split: "test",
      index: 42,
      dataset_size: 10000,
      sample: [1, null, 0],
      expected: 1,
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/models/current/dataset/test/samples/42");
    expect(init.method).toBe("GET");
  });

  it("getRandomTestSample omits the seed query parameter when no seed is given", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        model_id: "example-model",
        split: "test",
        index: 7,
        dataset_size: 10000,
        sample: [0, 1],
        expected: 0,
      }),
    );

    await getRandomTestSample();

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/models/current/dataset/test/samples/random");
  });

  it("getRandomTestSample includes the seed query parameter when given", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        model_id: "example-model",
        split: "test",
        index: 7,
        dataset_size: 10000,
        sample: [0, 1],
        expected: 0,
      }),
    );

    await getRandomTestSample(123);

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "/api/v1/models/current/dataset/test/samples/random?seed=123",
    );
  });

  it("getRandomTestSample supports a negative seed", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        model_id: "example-model",
        split: "test",
        index: 7,
        dataset_size: 10000,
        sample: [0, 1],
        expected: 0,
      }),
    );

    await getRandomTestSample(-7);

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "/api/v1/models/current/dataset/test/samples/random?seed=-7",
    );
  });

  it("tolerates additional properties on a test dataset sample response", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        model_id: "example-model",
        split: "test",
        index: 0,
        dataset_size: 10000,
        sample: [1],
        expected: 1,
        extra_field: "ignored by the client",
      }),
    );

    const result = await getTestSample(0);
    expect(result).toMatchObject({ index: 0, dataset_size: 10000 });
  });

  it("treats a test dataset sample response missing required fields as a contract error", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, { model_id: "example-model", split: "test" }),
    );

    await expect(getTestSample(0)).rejects.toMatchObject({
      apiError: { kind: "unexpected" },
    });
  });

  it("propagates the QFOIL error envelope for test dataset sample requests", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(404, {
        error: {
          code: "test_dataset_index_out_of_range",
          message: "index is outside the test dataset",
        },
      }),
    );

    await expect(getTestSample(999999)).rejects.toMatchObject({
      apiError: {
        kind: "qfoil",
        code: "test_dataset_index_out_of_range",
        status: 404,
      },
    });
  });
});
