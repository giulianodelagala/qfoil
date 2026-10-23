import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";

vi.mock("katex", () => ({
  default: { render: vi.fn() },
}));

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function flush(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function flushAll(times = 3): Promise<void> {
  for (let i = 0; i < times; i += 1) {
    await flush();
  }
}

type RouteHandler = (
  url: string,
  init?: RequestInit,
) => Response | Promise<Response>;

function defaultHealthResponse(): Response {
  return jsonResponse(200, {
    status: "ok",
    service: "qfoil-server",
    model_status: "unloaded",
  });
}

function defaultCurrentResponse(): Response {
  return jsonResponse(200, { model: { status: "unloaded" } });
}

function defaultModelsResponse(): Response {
  return jsonResponse(200, { models: [] });
}

function defaultConfigResponse(): Response {
  return jsonResponse(200, {
    evaluation: {
      defaults: { mode: "heuristic", sigma_search_budget: 64, rng_seed: null, seed: null },
      modes: ["heuristic", "exact"], max_sigma_search_budget: 20000000, max_evaluation_budget: 20000000,
    },
    search: {
      defaults: { budget: 1000, pool_size: 30, inner_budget: 8 },
      max: { budget: 100000, pool_size: 1000, inner_budget: 1000 },
    },
    evaluation_budget: 20000000,
    sigma_search_budget: 64,
  });
}

function buildFetchMock(options: {
  health?: RouteHandler;
  models?: RouteHandler;
  current?: RouteHandler;
  config?: RouteHandler;
  extra?: RouteHandler;
} = {}): ReturnType<typeof vi.fn> {
  return vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/health")) {
      return options.health ? options.health(url, init) : defaultHealthResponse();
    }
    if (url.endsWith("/v1/models/current")) {
      return options.current
        ? options.current(url, init)
        : defaultCurrentResponse();
    }
    if (url.endsWith("/v1/models")) {
      return options.models ? options.models(url, init) : defaultModelsResponse();
    }
    if (url.endsWith("/v1/config")) {
      return options.config ? options.config(url, init) : defaultConfigResponse();
    }
    if (options.extra) {
      return options.extra(url, init);
    }
    throw new Error(`Unhandled fetch: ${url}`);
  });
}

// Back-compat helper matching the original tests: health, models and current
// resolve to unremarkable defaults (connected, empty catalog, no model), and
// every other endpoint is routed to `handler`.
function healthOr(handler: RouteHandler): ReturnType<typeof vi.fn> {
  return buildFetchMock({ extra: handler });
}

async function loadMain(
  fetchMock: ReturnType<typeof vi.fn>,
): Promise<void> {
  document.body.innerHTML = '<div id="app"></div>';
  document.title = "";
  vi.stubGlobal("fetch", fetchMock);
  vi.resetModules();
  await import("../main");
  await flushAll();
}

function callsTo(fetchMock: ReturnType<typeof vi.fn>, suffix: string): number {
  return fetchMock.mock.calls.filter((call: unknown[]) =>
    (call[0] as string).endsWith(suffix),
  ).length;
}

type FakeModelInfo =
  | { status: "unloaded" }
  | {
      status: "ready";
      id: string;
      backend: string;
      num_features: number;
      num_classes: number;
      device: string;
      loaded_at: string;
    };

const READY_MODEL_DEFAULTS = {
  num_features: 8,
  num_classes: 2,
  device: "cuda:0",
  loaded_at: "2026-09-24T17:32:00Z",
};

// A minimal stateful stand-in for qfoil-server: GET current reflects
// whatever POST load / DELETE current last did, so tests that exercise a
// full load-then-confirm or unload-then-confirm cycle see consistent state
// instead of a static canned response silently overwriting it.
function createFakeServer(
  catalog: { id: string; backend: string }[] = [],
  initial: FakeModelInfo = { status: "unloaded" },
) {
  let current: FakeModelInfo = initial;

  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/health")) {
      return defaultHealthResponse();
    }
    if (url.endsWith("/v1/config")) {
      return defaultConfigResponse();
    }
    if (url.endsWith("/v1/models/current")) {
      if (init?.method === "DELETE") {
        current = { status: "unloaded" };
        return jsonResponse(200, { model: current });
      }
      return jsonResponse(200, { model: current });
    }
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, { models: catalog });
    }
    const loadMatch = /\/v1\/models\/([^/]+)\/load$/.exec(url);
    if (loadMatch) {
      const id = decodeURIComponent(loadMatch[1]);
      const entry = catalog.find((m) => m.id === id);
      current = {
        status: "ready",
        id,
        backend: entry?.backend ?? "pyjuice",
        ...READY_MODEL_DEFAULTS,
      };
      return jsonResponse(200, { model: current });
    }
    throw new Error(`Unhandled fetch: ${url}`);
  });

  return { fetchMock };
}

describe("evaluation settings", () => {
  afterEach(() => { vi.unstubAllGlobals(); sessionStorage.clear(); });
  beforeEach(() => { sessionStorage.clear(); });

  function edit(selector: string, value: string): void {
    const control = document.querySelector<HTMLInputElement>(selector)!;
    control.value = value;
    control.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function evaluationMock(): ReturnType<typeof vi.fn> {
    return healthOr(url => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { status: "decided", value: 1, last_witness: [1, 0, 1] });
      }
      throw new Error(`Unexpected request: ${url}`);
    });
  }

  function lastRequest(fetchMock: ReturnType<typeof vi.fn>): Record<string, any> {
    const calls = fetchMock.mock.calls.filter(call => String(call[0]).endsWith("/v1/expressions/evaluate"));
    return JSON.parse(calls.at(-1)![1]!.body as string);
  }

  it("puts configuration in its own accessible tab and removes witness search", async () => {
    await loadMain(buildFetchMock());
    expect(document.querySelector("#witness-form")).toBeNull();
    const panel = document.querySelector<HTMLDivElement>("#panel-settings")!;
    expect(panel.hidden).toBe(true);
    document.querySelector<HTMLButtonElement>("#tab-settings")!.click();
    expect(panel.hidden).toBe(false);
    expect(document.querySelector<HTMLDivElement>("#panel-playground")!.hidden).toBe(true);
    expect(document.querySelector<HTMLInputElement>("#evaluation-budget")!.value).toBe("64");
  });

  it("sends heuristic budget and both seeds through evaluate only", async () => {
    const mock = evaluationMock();
    await loadMain(mock);
    edit("#evaluation-budget", "1000");
    edit("#evaluation-rng-seed", "38");
    edit("#evaluation-seed-source", "tuple");
    edit("#evaluation-initial-tuple", "[1, null, 0]");
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(lastRequest(mock).options).toEqual({mode: "heuristic", evaluation_budget: 20000000, sigma_search_budget: 1000, rng_seed: 38, seed: [1, null, 0]});
    expect(callsTo(mock, "/v1/witnesses/search")).toBe(0);
    expect(document.querySelector("#last-witness-box")!.textContent).toBe("[1,0,1]");
  });

  it("sends the maximum heuristic and total budgets", async () => {
    const mock = evaluationMock();
    await loadMain(mock);
    edit("#evaluation-budget", "20000000");
    edit("#evaluation-work-budget", "20000000");
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(lastRequest(mock).options).toMatchObject({sigma_search_budget: 20000000, evaluation_budget: 20000000});
  });

  it("reuses the last witness as a heuristic initial tuple for the next evaluation", async () => {
    const mock = healthOr(url => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 1, last_witness: [1, null, 0] });
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    await loadMain(mock);
    const writeText = vi.fn();
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    edit("#evaluation-mode", "exact");
    edit("#evaluation-budget", "1000");
    edit("#evaluation-rng-seed", "38");
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();

    document.querySelector<HTMLButtonElement>("#last-witness-use-seed")!.click();

    expect(writeText).not.toHaveBeenCalled();
    expect(callsTo(mock, "/v1/expressions/evaluate")).toBe(1);
    expect(document.querySelector<HTMLDivElement>("#panel-settings")!.hidden).toBe(true);
    expect(document.querySelector<HTMLDivElement>("#panel-playground")!.hidden).toBe(false);
    expect(document.querySelector("#last-witness-feedback")!.textContent).toBe("Heuristic seed set for the next evaluation.");
    expect(document.querySelector<HTMLSelectElement>("#evaluation-mode")!.value).toBe("heuristic");
    expect(document.querySelector<HTMLSelectElement>("#evaluation-seed-source")!.value).toBe("tuple");
    const tuple = document.querySelector<HTMLTextAreaElement>("#evaluation-initial-tuple")!;
    expect(tuple.value).toBe("[1,null,0]");
    expect(tuple.disabled).toBe(false);
    expect(document.activeElement).not.toBe(tuple);

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(lastRequest(mock).options).toEqual({
      mode: "heuristic", evaluation_budget: 20000000,
      sigma_search_budget: 1000, rng_seed: 38, seed: [1, null, 0],
    });
  });

  it("disables witness reuse when evaluation settings could not be loaded", async () => {
    await loadMain(buildFetchMock({
      config: () => jsonResponse(500, {}),
      extra: () => jsonResponse(200, { value: 1, last_witness: [1, 0] }),
    }));
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(document.querySelector<HTMLDivElement>("#last-witness-block")!.hidden).toBe(false);
    expect(document.querySelector<HTMLButtonElement>("#last-witness-use-seed")!.disabled).toBe(true);
    expect(document.querySelector<HTMLButtonElement>("#last-witness-copy")!.disabled).toBe(false);
  });

  it("rejects an excessive total budget in exhaustive mode", async () => {
    const mock = evaluationMock();
    await loadMain(mock);
    edit("#evaluation-mode", "exact");
    edit("#evaluation-work-budget", "20000001");
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(callsTo(mock, "/v1/expressions/evaluate")).toBe(0);
  });

  it("sends exhaustive mode without inactive heuristic options", async () => {
    const mock = evaluationMock();
    await loadMain(mock);
    edit("#evaluation-budget", "invalid");
    edit("#evaluation-rng-seed", "invalid");
    edit("#evaluation-mode", "exact");
    expect(document.querySelector<HTMLDivElement>("#heuristic-settings")!.hidden).toBe(true);
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(lastRequest(mock).options).toEqual({mode: "exact", evaluation_budget: 20000000});
  });

  it.each([
    ["#evaluation-budget", "20000001"],
    ["#evaluation-work-budget", "20000001"],
    ["#evaluation-work-budget", "-1"],
    ["#evaluation-rng-seed", "1.5"],
    ["#evaluation-rng-seed", "9007199254740992"],
    ["#evaluation-initial-tuple", "[2, 0, null]"],
  ])("rejects invalid settings %s", async (selector, value) => {
    const mock = evaluationMock();
    await loadMain(mock);
    if (selector === "#evaluation-initial-tuple") edit("#evaluation-seed-source", "tuple");
    edit(selector, value);
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(callsTo(mock, "/v1/expressions/evaluate")).toBe(0);
    expect(document.querySelector<HTMLDivElement>("#panel-settings")!.hidden).toBe(false);
    expect(document.querySelector<HTMLParagraphElement>("#evaluation-settings-error")!.hidden).toBe(false);
  });

  it("persists settings but does not persist the initial tuple", async () => {
    await loadMain(buildFetchMock());
    edit("#evaluation-budget", "500");
    edit("#evaluation-work-budget", "15000000");
    edit("#evaluation-rng-seed", "7");
    edit("#evaluation-seed-source", "tuple");
    edit("#evaluation-initial-tuple", "[1, 0, 1]");
    await loadMain(buildFetchMock());
    expect(document.querySelector<HTMLInputElement>("#evaluation-budget")!.value).toBe("500");
    expect(document.querySelector<HTMLInputElement>("#evaluation-work-budget")!.value).toBe("15000000");
    expect(document.querySelector<HTMLInputElement>("#evaluation-rng-seed")!.value).toBe("7");
    expect(document.querySelector<HTMLSelectElement>("#evaluation-seed-source")!.value).toBe("random");
    document.querySelector<HTMLButtonElement>("#evaluation-restore-defaults")!.click();
    expect(document.querySelector<HTMLInputElement>("#evaluation-budget")!.value).toBe("64");
    expect(document.querySelector<HTMLInputElement>("#evaluation-rng-seed")!.value).toBe("");
    expect(document.querySelector<HTMLInputElement>("#evaluation-work-budget")!.value).toBe("20000000");
  });

  it("uses server defaults if configuration cannot load", async () => {
    const mock = buildFetchMock({config: () => jsonResponse(500, {}), extra: url => {
      if (url.endsWith("/v1/expressions/evaluate")) return jsonResponse(200, { value: 1, last_witness: null });
      throw new Error("Unexpected request");
    }});
    await loadMain(mock);
    expect(document.querySelector<HTMLParagraphElement>("#evaluation-config-error")!.hidden).toBe(false);
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(lastRequest(mock)).not.toHaveProperty("options");
  });

  it("shows inconclusive and its budget reason instead of numeric null", async () => {
    const mock = healthOr(() => jsonResponse(200, {status: "inconclusive", value: null, last_witness: null, reason: "evaluation_budget_exhausted"}));
    await loadMain(mock);
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(document.querySelector("#evaluation-output")!.textContent).toContain("work limit was reached");
    expect(document.querySelector<HTMLDivElement>("#last-witness-section")!.hidden).toBe(true);
  });
});

describe("concurrency", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("does not let a stale evaluate response overwrite a newer one", async () => {
    const resolvers: Array<(response: Response) => void> = [];
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return new Promise<Response>((resolve) => {
          resolvers.push(resolve);
        });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);

    const evaluateButton =
      document.querySelector<HTMLButtonElement>("#evaluate-button")!;
    const output = document.querySelector<HTMLDivElement>("#evaluation-output")!;

    evaluateButton.click();
    await flush();
    // The button disables itself while a request is in flight, which already
    // prevents duplicate clicks in the UI. Re-enable it to force a second,
    // overlapping request and exercise the requestId staleness guard itself.
    evaluateButton.disabled = false;
    evaluateButton.click();
    await flush();

    expect(resolvers).toHaveLength(2);

    // Resolve the second (latest) request first.
    resolvers[1](jsonResponse(200, { value: 2, last_witness: null }));
    await flush();
    expect(output.textContent).toBe("Value: 2");

    // The stale first request arrives later and must not override it.
    resolvers[0](jsonResponse(200, { value: 1, last_witness: null }));
    await flush();
    expect(output.textContent).toBe("Value: 2");
  });
});

describe("last witness", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubClipboard(
    writeText: (text: string) => Promise<void>,
  ): void {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
  }

  it("renders last_witness as exact JSON and copies it on click", async () => {
    const witness = [null, 1, 0, null];
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 1, last_witness: witness });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubClipboard(writeText);

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();

    const box = document.querySelector<HTMLDivElement>("#last-witness-box")!;
    expect(box.textContent).toBe("[null,1,0,null]");
    expect(JSON.parse(box.textContent!)).toEqual(witness);

    box.click();
    await flush();

    expect(writeText).toHaveBeenCalledWith("[null,1,0,null]");
    const feedback = document.querySelector<HTMLParagraphElement>(
      "#last-witness-feedback",
    )!;
    expect(feedback.hidden).toBe(false);
    expect(feedback.textContent).toBe("Copied");
  });

  it("copies the same content via Enter and Space", async () => {
    const witness = [1, 0];
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 1, last_witness: witness });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubClipboard(writeText);

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();

    const box = document.querySelector<HTMLDivElement>("#last-witness-box")!;
    box.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }),
    );
    await flush();
    expect(writeText).toHaveBeenCalledWith("[1,0]");

    writeText.mockClear();
    box.dispatchEvent(
      new KeyboardEvent("keydown", { key: " ", bubbles: true, cancelable: true }),
    );
    await flush();
    expect(writeText).toHaveBeenCalledWith("[1,0]");
  });

  it("does not truncate a long witness and keeps it on one line for horizontal scroll", async () => {
    const longWitness = Array.from({ length: 200 }, (_, i) =>
      i % 3 === 0 ? null : i % 2,
    );
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 1, last_witness: longWitness });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();

    const box = document.querySelector<HTMLDivElement>("#last-witness-box")!;
    const expected = JSON.stringify(longWitness);
    expect(box.textContent).toBe(expected);
    expect(box.classList.contains("witness-json-box")).toBe(true);
  });

  it("shows the empty state when last_witness is null, without an empty copyable box or the word null", async () => {
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 0, last_witness: null });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();

    const empty = document.querySelector<HTMLParagraphElement>(
      "#last-witness-empty",
    )!;
    const block = document.querySelector<HTMLDivElement>("#last-witness-block")!;
    expect(empty.hidden).toBe(false);
    expect(empty.textContent).toContain(
      "No witness was produced during this evaluation.",
    );
    expect(block.hidden).toBe(true);
  });

  it("treats a response missing last_witness as a contract error and shows no witness box", async () => {
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 1 });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();

    const error = document.querySelector<HTMLParagraphElement>(
      "#evaluation-error",
    )!;
    expect(error.hidden).toBe(false);
    const section = document.querySelector<HTMLDivElement>(
      "#last-witness-section",
    )!;
    expect(section.hidden).toBe(true);
  });

  it("clears the previous witness immediately when a new evaluation starts", async () => {
    const resolvers: Array<(response: Response) => void> = [];
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return new Promise<Response>((resolve) => resolvers.push(resolve));
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);
    const evaluateButton =
      document.querySelector<HTMLButtonElement>("#evaluate-button")!;
    const box = document.querySelector<HTMLDivElement>("#last-witness-box")!;
    const section = document.querySelector<HTMLDivElement>(
      "#last-witness-section",
    )!;

    evaluateButton.click();
    await flush();
    resolvers[0](jsonResponse(200, { value: 1, last_witness: [1, 0] }));
    await flush();
    expect(box.textContent).toBe("[1,0]");
    expect(section.hidden).toBe(false);

    evaluateButton.disabled = false;
    evaluateButton.click();
    await flush();

    expect(section.hidden).toBe(true);
    expect(box.textContent).toBe("");

    resolvers[1](jsonResponse(200, { value: 2, last_witness: null }));
    await flush();
  });

  it("clears the witness when evaluation fails", async () => {
    let shouldFail = false;
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        if (shouldFail) {
          return jsonResponse(400, {
            error: { code: "qfoil_syntax_error", message: "bad expression" },
          });
        }
        return jsonResponse(200, { value: 1, last_witness: [1, 0] });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);
    const evaluateButton =
      document.querySelector<HTMLButtonElement>("#evaluate-button")!;
    evaluateButton.click();
    await flush();

    const box = document.querySelector<HTMLDivElement>("#last-witness-box")!;
    const section = document.querySelector<HTMLDivElement>(
      "#last-witness-section",
    )!;
    expect(box.textContent).toBe("[1,0]");

    shouldFail = true;
    evaluateButton.disabled = false;
    evaluateButton.click();
    await flush();

    expect(section.hidden).toBe(true);
    expect(box.textContent).toBe("");
    const error = document.querySelector<HTMLParagraphElement>(
      "#evaluation-error",
    )!;
    expect(error.hidden).toBe(false);
  });

  it("does not call /v1/witnesses/search to obtain or display last_witness", async () => {
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 1, last_witness: [1, 0] });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();

    expect(callsTo(fetchMock, "/v1/witnesses/search")).toBe(0);
  });

  it("keeps the JSON visible and shows a non-destructive error when the clipboard write fails", async () => {
    const witness = [1, 0];
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 1, last_witness: witness });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);
    stubClipboard(() => Promise.reject(new Error("denied")));

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();

    const box = document.querySelector<HTMLDivElement>("#last-witness-box")!;
    box.click();
    await flush();

    const clipboardError = document.querySelector<HTMLParagraphElement>(
      "#last-witness-clipboard-error",
    )!;
    expect(clipboardError.hidden).toBe(false);
    expect(box.textContent).toBe("[1,0]");
  });

  it("preserves the existing value display alongside last_witness", async () => {
    const fetchMock = healthOr((url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { value: 1, last_witness: [1, 0] });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);

    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();

    const output = document.querySelector<HTMLDivElement>("#evaluation-output")!;
    expect(output.textContent).toBe("Value: 1");
  });
});

describe("model lifecycle: unloaded state", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the selector and 'No model loaded' when no model is active", async () => {
    const fetchMock = buildFetchMock({
      models: () =>
        jsonResponse(200, {
          models: [{ id: "mux", backend: "pyjuice" }],
        }),
    });

    await loadMain(fetchMock);

    expect(
      document.querySelector<HTMLSpanElement>("#model-indicator-text")!
        .textContent,
    ).toBe("No model loaded");

    const select = document.querySelector<HTMLSelectElement>("#model-select")!;
    expect(select.options).toHaveLength(1);
    expect(select.options[0].value).toBe("mux");
    expect(select.options[0].textContent).toBe("mux (pyjuice)");

    const loadButton =
      document.querySelector<HTMLButtonElement>("#model-load-button")!;
    expect(loadButton.disabled).toBe(false);
  });

  it("shows 'No models are configured' and disables Load when the catalog is empty", async () => {
    const fetchMock = buildFetchMock();

    await loadMain(fetchMock);

    const empty = document.querySelector<HTMLParagraphElement>(
      "#model-catalog-empty",
    )!;
    expect(empty.hidden).toBe(false);
    expect(empty.textContent).toContain(
      "No models are configured on qfoil-server.",
    );

    const form = document.querySelector<HTMLDivElement>("#model-catalog-form")!;
    expect(form.hidden).toBe(true);

    const loadButton =
      document.querySelector<HTMLButtonElement>("#model-load-button")!;
    expect(loadButton.disabled).toBe(true);
  });

  it("keeps Render and Evaluate available without a model", async () => {
    const fetchMock = buildFetchMock();
    await loadMain(fetchMock);

    expect(document.querySelector<HTMLButtonElement>("#render-button")!.disabled).toBe(
      false,
    );
    expect(
      document.querySelector<HTMLButtonElement>("#evaluate-button")!.disabled,
    ).toBe(false);
  });
});

describe("model lifecycle: ready state", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function readyModel() {
    return jsonResponse(200, {
      model: {
        status: "ready",
        id: "mux",
        backend: "pyjuice",
        num_features: 8,
        num_classes: 2,
        device: "cuda:0",
        loaded_at: "2026-09-24T17:32:00Z",
      },
    });
  }

  it("shows metadata and a persistent indicator", async () => {
    const fetchMock = buildFetchMock({ current: readyModel });
    await loadMain(fetchMock);

    expect(
      document.querySelector<HTMLSpanElement>("#model-indicator-text")!
        .textContent,
    ).toBe("mux · PyJuice · cuda:0");

    // Metadata is populated even though the Model tab isn't active yet
    // (Playground is the default active tab).
    expect(document.querySelector("#panel-model")!.hasAttribute("hidden")).toBe(
      true,
    );

    document.querySelector<HTMLButtonElement>("#tab-model")!.click();

    expect(document.querySelector("#panel-model")!.hasAttribute("hidden")).toBe(
      false,
    );
    expect(document.querySelector<HTMLElement>("#model-meta-id")!.textContent).toBe(
      "mux",
    );
    expect(
      document.querySelector<HTMLElement>("#model-meta-status")!.textContent,
    ).toBe("Ready");
    expect(
      document.querySelector<HTMLElement>("#model-meta-backend")!.textContent,
    ).toBe("PyJuice");
    expect(
      document.querySelector<HTMLElement>("#model-meta-features")!.textContent,
    ).toBe("8");
    expect(
      document.querySelector<HTMLElement>("#model-meta-classes")!.textContent,
    ).toBe("2");
    expect(
      document.querySelector<HTMLElement>("#model-meta-device")!.textContent,
    ).toBe("cuda:0");
    const loaded = document.querySelector<HTMLTimeElement>("#model-meta-loaded")!;
    expect(loaded.dateTime).toBe("2026-09-24T17:32:00Z");
  });

  it("keeps the compact indicator visible regardless of the active tab", async () => {
    const fetchMock = buildFetchMock({ current: readyModel });
    await loadMain(fetchMock);

    const indicator = document.querySelector<HTMLSpanElement>(
      "#model-indicator",
    )!;
    // Playground is the default active tab; the Model tabpanel is hidden,
    // but the compact status readout is not part of either tabpanel.
    expect(document.querySelector("#panel-model")!.hasAttribute("hidden")).toBe(
      true,
    );
    expect(indicator.hidden).toBe(false);
    expect(indicator.textContent).toContain("mux");

    document.querySelector<HTMLButtonElement>("#tab-model")!.click();
    expect(indicator.hidden).toBe(false);
    expect(indicator.textContent).toContain("mux");
  });

  it("is a plain status readout, not a button", async () => {
    const fetchMock = buildFetchMock({ current: readyModel });
    await loadMain(fetchMock);

    const indicator = document.querySelector("#model-indicator")!;
    expect(indicator.tagName).not.toBe("BUTTON");
    const serverIndicator = document.querySelector("#status-indicator")!;
    expect(serverIndicator.tagName).not.toBe("BUTTON");
  });

  it("hides the selector and shows Unload; no direct loading of a second model", async () => {
    const fetchMock = buildFetchMock({ current: readyModel });
    await loadMain(fetchMock);

    expect(
      document.querySelector<HTMLDivElement>("#model-panel-unloaded")!.hidden,
    ).toBe(true);
    const readySection = document.querySelector<HTMLDivElement>(
      "#model-panel-ready",
    )!;
    expect(readySection.hidden).toBe(false);

    const unloadButton = document.querySelector<HTMLButtonElement>(
      "#model-unload-button",
    )!;
    expect(unloadButton.disabled).toBe(false);
    expect(unloadButton.textContent).toBe("Unload");

    // The selector/Load controls exist but are not reachable to trigger a
    // second load while a model is active.
    const select = document.querySelector<HTMLSelectElement>("#model-select")!;
    const loadButton =
      document.querySelector<HTMLButtonElement>("#model-load-button")!;
    expect(select.disabled).toBe(true);
    expect(loadButton.disabled).toBe(true);
  });

  it("never renders server-internal fields such as filesystem paths", async () => {
    const fetchMock = buildFetchMock({
      current: () =>
        jsonResponse(200, {
          model: {
            status: "ready",
            id: "mux",
            backend: "pyjuice",
            num_features: 8,
            num_classes: 2,
            device: "cuda:0",
            loaded_at: "2026-09-24T17:32:00Z",
            // Hypothetical extra server fields that must never be surfaced.
            path: "/srv/qfoil/models/mux",
            min_batch_size: 4,
          },
        }),
    });

    await loadMain(fetchMock);

    const panelText = document.querySelector<HTMLElement>("#model-panel")!
      .textContent!;
    expect(panelText).not.toContain("/srv/qfoil");
    expect(panelText.toLowerCase()).not.toContain("min_batch_size");
    expect(
      document.querySelector<HTMLSpanElement>("#model-indicator-text")!
        .textContent,
    ).not.toContain("/srv/qfoil");
  });
});

describe("model lifecycle: document.title", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reflects unloaded", async () => {
    const fetchMock = buildFetchMock();
    await loadMain(fetchMock);
    expect(document.title).toBe("QFOIL Playground · No model");
  });

  it("reflects ready", async () => {
    const fetchMock = buildFetchMock({
      current: () =>
        jsonResponse(200, {
          model: {
            status: "ready",
            id: "mux",
            backend: "pyjuice",
            num_features: 8,
            num_classes: 2,
            device: "cuda:0",
            loaded_at: "2026-09-24T17:32:00Z",
          },
        }),
    });
    await loadMain(fetchMock);
    expect(document.title).toBe("QFOIL Playground · mux");
  });

  it("reflects loading and unloading during the respective requests", async () => {
    // A stateful mock whose GET /v1/models/current always reflects `current`,
    // so the app's post-load/post-unload confirmation fetch sees consistent
    // state instead of a stale canned response overwriting it back.
    let current: FakeModelInfo = { status: "unloaded" };
    let resolveLoad: ((response: Response) => void) | null = null;
    let resolveUnload: ((response: Response) => void) | null = null;

    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.endsWith("/health")) return defaultHealthResponse();
      if (url.endsWith("/v1/config")) return defaultConfigResponse();
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { models: [{ id: "mux", backend: "pyjuice" }] });
      }
      if (url.endsWith("/v1/models/current")) {
        if (init?.method === "DELETE") {
          return new Promise<Response>((resolve) => {
            resolveUnload = (response) => {
              current = { status: "unloaded" };
              resolve(response);
            };
          });
        }
        return jsonResponse(200, { model: current });
      }
      if (url.endsWith("/v1/models/mux/load")) {
        return new Promise<Response>((resolve) => {
          resolveLoad = (response) => {
            current = {
              status: "ready",
              id: "mux",
              backend: "pyjuice",
              ...READY_MODEL_DEFAULTS,
            };
            resolve(response);
          };
        });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    });

    await loadMain(fetchMock);

    const select = document.querySelector<HTMLSelectElement>("#model-select")!;
    select.value = "mux";
    document.querySelector<HTMLButtonElement>("#model-load-button")!.click();
    await flush();

    expect(document.title).toBe("QFOIL Playground · Loading mux…");

    resolveLoad!(
      jsonResponse(200, {
        model: {
          status: "ready",
          id: "mux",
          backend: "pyjuice",
          ...READY_MODEL_DEFAULTS,
        },
      }),
    );
    await flushAll();
    expect(document.title).toBe("QFOIL Playground · mux");

    vi.spyOn(window, "confirm").mockReturnValue(true);
    document.querySelector<HTMLButtonElement>("#model-unload-button")!.click();
    await flush();

    expect(document.title).toBe("QFOIL Playground · Unloading mux…");

    resolveUnload!(jsonResponse(200, { model: { status: "unloaded" } }));
    await flushAll();
    expect(document.title).toBe("QFOIL Playground · No model");
  });
});

describe("model lifecycle: load flow", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("disables the selector, Load and Evaluate while loading", async () => {
    let resolveLoad: ((response: Response) => void) | null = null;
    const fetchMock = buildFetchMock({
      models: () =>
        jsonResponse(200, { models: [{ id: "mux", backend: "pyjuice" }] }),
      extra: (url) => {
        if (url.endsWith("/v1/models/mux/load")) {
          return new Promise<Response>((resolve) => {
            resolveLoad = resolve;
          });
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });

    await loadMain(fetchMock);

    const select = document.querySelector<HTMLSelectElement>("#model-select")!;
    select.value = "mux";
    document.querySelector<HTMLButtonElement>("#model-load-button")!.click();
    await flush();

    expect(select.disabled).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>("#model-load-button")!.disabled,
    ).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>("#evaluate-button")!.disabled,
    ).toBe(true);
    // Render formula remains available; it does not depend on the model.
    expect(
      document.querySelector<HTMLButtonElement>("#render-button")!.disabled,
    ).toBe(false);

    resolveLoad!(
      jsonResponse(200, {
        model: {
          status: "ready",
          id: "mux",
          backend: "pyjuice",
          num_features: 8,
          num_classes: 2,
          device: "cuda:0",
          loaded_at: "2026-09-24T17:32:00Z",
        },
      }),
    );
    await flushAll();
  });

  it("hides the selector and shows Unload after a successful load", async () => {
    const server = createFakeServer([{ id: "mux", backend: "pyjuice" }]);

    await loadMain(server.fetchMock);

    const select = document.querySelector<HTMLSelectElement>("#model-select")!;
    select.value = "mux";
    document.querySelector<HTMLButtonElement>("#model-load-button")!.click();
    await flushAll();

    expect(
      document.querySelector<HTMLDivElement>("#model-panel-unloaded")!.hidden,
    ).toBe(true);
    expect(
      document.querySelector<HTMLDivElement>("#model-panel-ready")!.hidden,
    ).toBe(false);
    expect(
      document.querySelector<HTMLButtonElement>("#model-unload-button")!.disabled,
    ).toBe(false);
  });

  it("refreshes current after a failed load", async () => {
    let currentCallCount = 0;
    const fetchMock = buildFetchMock({
      models: () =>
        jsonResponse(200, { models: [{ id: "mux", backend: "pyjuice" }] }),
      current: () => {
        currentCallCount += 1;
        return defaultCurrentResponse();
      },
      extra: (url) => {
        if (url.endsWith("/v1/models/mux/load")) {
          return jsonResponse(500, {
            error: { code: "model_load_error", message: "the backend rejected the model" },
          });
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });

    await loadMain(fetchMock);
    const callsAfterStartup = currentCallCount;

    const select = document.querySelector<HTMLSelectElement>("#model-select")!;
    select.value = "mux";
    document.querySelector<HTMLButtonElement>("#model-load-button")!.click();
    await flushAll();

    expect(currentCallCount).toBeGreaterThan(callsAfterStartup);
    expect(
      document.querySelector<HTMLParagraphElement>("#model-panel-error")!.hidden,
    ).toBe(false);
    expect(
      document.querySelector<HTMLDivElement>("#model-panel-unloaded")!.hidden,
    ).toBe(false);
  });

  it("shows a comprehensible message for model_already_loaded and refreshes current", async () => {
    let currentCallCount = 0;
    const fetchMock = buildFetchMock({
      models: () =>
        jsonResponse(200, { models: [{ id: "mux", backend: "pyjuice" }] }),
      current: () => {
        currentCallCount += 1;
        return defaultCurrentResponse();
      },
      extra: (url) => {
        if (url.endsWith("/v1/models/mux/load")) {
          return jsonResponse(409, {
            error: {
              code: "model_already_loaded",
              message: "A model is already active; unload it first",
            },
          });
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });

    await loadMain(fetchMock);
    const callsAfterStartup = currentCallCount;

    const select = document.querySelector<HTMLSelectElement>("#model-select")!;
    select.value = "mux";
    document.querySelector<HTMLButtonElement>("#model-load-button")!.click();
    await flushAll();

    const errorEl = document.querySelector<HTMLParagraphElement>(
      "#model-panel-error",
    )!;
    expect(errorEl.hidden).toBe(false);
    expect(errorEl.textContent).toContain("A model is already active");
    expect(currentCallCount).toBeGreaterThan(callsAfterStartup);
  });
});

describe("model lifecycle: unload flow", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function readyModel() {
    return jsonResponse(200, {
      model: {
        status: "ready",
        id: "mux",
        backend: "pyjuice",
        num_features: 8,
        num_classes: 2,
        device: "cuda:0",
        loaded_at: "2026-09-24T17:32:00Z",
      },
    });
  }

  it("shows the selector again after a successful unload", async () => {
    const server = createFakeServer([{ id: "mux", backend: "pyjuice" }], {
      status: "ready",
      id: "mux",
      backend: "pyjuice",
      ...READY_MODEL_DEFAULTS,
    });

    await loadMain(server.fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    document.querySelector<HTMLButtonElement>("#model-unload-button")!.click();
    await flushAll();

    expect(
      document.querySelector<HTMLDivElement>("#model-panel-unloaded")!.hidden,
    ).toBe(false);
    expect(
      document.querySelector<HTMLDivElement>("#model-panel-ready")!.hidden,
    ).toBe(true);
  });

  it("does nothing when the user declines the confirmation", async () => {
    const fetchMock = buildFetchMock({ current: readyModel });
    await loadMain(fetchMock);
    vi.spyOn(window, "confirm").mockReturnValue(false);

    document.querySelector<HTMLButtonElement>("#model-unload-button")!.click();
    await flush();

    expect(callsTo(fetchMock, "/v1/models/current")).toBeLessThanOrEqual(2);
    expect(
      document.querySelector<HTMLDivElement>("#model-panel-ready")!.hidden,
    ).toBe(false);
  });
});

describe("model lifecycle: server connectivity", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("refreshes current when the window regains focus", async () => {
    const fetchMock = buildFetchMock();
    await loadMain(fetchMock);

    const before = callsTo(fetchMock, "/v1/models/current");
    window.dispatchEvent(new Event("focus"));
    await flush();

    expect(callsTo(fetchMock, "/v1/models/current")).toBeGreaterThan(before);
  });

  it("disables lifecycle and Evaluate when the server is unavailable", async () => {
    const fetchMock = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    await loadMain(fetchMock);

    expect(
      document.querySelector<HTMLSpanElement>("#status-text")!.textContent,
    ).toBe("Unavailable");
    expect(
      document.querySelector<HTMLSpanElement>("#model-indicator-text")!
        .textContent,
    ).toBe("Model status unavailable");
    expect(
      document.querySelector<HTMLButtonElement>("#render-button")!.disabled,
    ).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>("#evaluate-button")!.disabled,
    ).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>("#model-load-button")!.disabled,
    ).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>("#model-unload-button")!.disabled,
    ).toBe(true);
    expect(document.title).toBe("QFOIL Playground");
  });
});

describe("test dataset samples", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function readySampleModel(): Response {
    return jsonResponse(200, {
      model: {
        status: "ready",
        id: "mux",
        backend: "pyjuice",
        num_features: 8,
        num_classes: 2,
        device: "cuda:0",
        loaded_at: "2026-09-24T17:32:00Z",
      },
    });
  }

  function sampleResponseBody(overrides: Partial<Record<string, unknown>> = {}) {
    return {
      model_id: "example-model",
      split: "test",
      index: 42,
      dataset_size: 10000,
      sample: [1, null, 0],
      expected: 1,
      ...overrides,
    };
  }

  function stubClipboard(writeText: (text: string) => Promise<void>): void {
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });
  }

  function sampleCallCount(fetchMock: ReturnType<typeof vi.fn>): number {
    return fetchMock.mock.calls.filter((call: unknown[]) =>
      (call[0] as string).includes("/dataset/test/samples"),
    ).length;
  }

  it("appears, enabled, when the model is ready", async () => {
    const fetchMock = buildFetchMock({ current: readySampleModel });
    await loadMain(fetchMock);

    const panel = document.querySelector<HTMLElement>("#test-dataset-panel")!;
    expect(panel.hidden).toBe(false);
    expect(
      document.querySelector<HTMLButtonElement>("#sample-get-button")!.disabled,
    ).toBe(false);
    expect(
      document.querySelector<HTMLButtonElement>("#sample-random-button")!
        .disabled,
    ).toBe(false);
    expect(
      document.querySelector<HTMLParagraphElement>("#sample-empty")!.hidden,
    ).toBe(false);
  });

  it("is hidden and disabled without an active model, and never queries the sample endpoints", async () => {
    const fetchMock = buildFetchMock();
    await loadMain(fetchMock);

    const panel = document.querySelector<HTMLElement>("#test-dataset-panel")!;
    expect(panel.hidden).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>("#sample-get-button")!.disabled,
    ).toBe(true);
    expect(
      document.querySelector<HTMLButtonElement>("#sample-random-button")!
        .disabled,
    ).toBe(true);
    expect(sampleCallCount(fetchMock)).toBe(0);
  });

  it("Get sample fetches the indexed sample", async () => {
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.endsWith("/dataset/test/samples/42")) {
          return jsonResponse(200, sampleResponseBody());
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);

    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    expect(sampleCallCount(fetchMock)).toBe(1);
    const call = fetchMock.mock.calls.find((c: unknown[]) =>
      (c[0] as string).includes("/dataset/test/samples"),
    )!;
    expect(call[0]).toBe("/api/v1/models/current/dataset/test/samples/42");
  });

  it.each([
    ["", "empty"],
    ["-1", "negative"],
    ["1.5", "non-integer"],
    ["abc", "non-numeric"],
  ])("rejects a %s (%s) sample index without calling the server", async (value) => {
    const fetchMock = buildFetchMock({ current: readySampleModel });
    await loadMain(fetchMock);

    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      value;
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    expect(sampleCallCount(fetchMock)).toBe(0);
    const error = document.querySelector<HTMLParagraphElement>(
      "#sample-error",
    )!;
    expect(error.hidden).toBe(false);
    expect(error.textContent).toMatch(/non-negative integer/i);
  });

  it("Random sample fetches without a seed query parameter when Seed is empty", async () => {
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.endsWith("/dataset/test/samples/random")) {
          return jsonResponse(200, sampleResponseBody({ index: 7 }));
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);

    document.querySelector<HTMLButtonElement>("#sample-random-button")!.click();
    await flush();

    const call = fetchMock.mock.calls.find((c: unknown[]) =>
      (c[0] as string).includes("/dataset/test/samples"),
    )!;
    expect(call[0]).toBe("/api/v1/models/current/dataset/test/samples/random");
  });

  it("Random sample includes the seed query parameter when Seed is set", async () => {
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.includes("/dataset/test/samples/random")) {
          return jsonResponse(200, sampleResponseBody({ index: 7 }));
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);

    document.querySelector<HTMLInputElement>("#sample-seed-input")!.value =
      "123";
    document.querySelector<HTMLButtonElement>("#sample-random-button")!.click();
    await flush();

    const call = fetchMock.mock.calls.find((c: unknown[]) =>
      (c[0] as string).includes("/dataset/test/samples"),
    )!;
    expect(call[0]).toBe(
      "/api/v1/models/current/dataset/test/samples/random?seed=123",
    );
  });

  it("rejects a malformed seed without calling the server", async () => {
    const fetchMock = buildFetchMock({ current: readySampleModel });
    await loadMain(fetchMock);

    document.querySelector<HTMLInputElement>("#sample-seed-input")!.value =
      "not-a-number";
    document.querySelector<HTMLButtonElement>("#sample-random-button")!.click();
    await flush();

    expect(sampleCallCount(fetchMock)).toBe(0);
    const error = document.querySelector<HTMLParagraphElement>(
      "#sample-error",
    )!;
    expect(error.hidden).toBe(false);
    expect(error.textContent).toMatch(/base-10 integer/i);
  });

  it("renders index, dataset size, sample and expected, preserving null", async () => {
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.includes("/dataset/test/samples")) {
          return jsonResponse(200, sampleResponseBody());
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);

    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    expect(
      document.querySelector<HTMLParagraphElement>("#sample-index-line")!
        .textContent,
    ).toBe("Index: 42 of 10000");
    const sampleJson = document.querySelector<HTMLPreElement>("#sample-json")!;
    expect(sampleJson.textContent).toBe(
      JSON.stringify([1, null, 0], null, 2),
    );
    expect(sampleJson.textContent).toContain("null");
    // Pretty-printed samples can run to hundreds of lines; the box scrolls
    // instead of stretching the card indefinitely.
    expect(sampleJson.classList.contains("sample-json-box")).toBe(true);
    const expectedJson = document.querySelector<HTMLPreElement>(
      "#sample-expected-json",
    )!;
    expect(expectedJson.textContent).toBe(JSON.stringify(1, null, 2));
    expect(expectedJson.classList.contains("sample-json-box")).toBe(true);
    expect(
      document.querySelector<HTMLParagraphElement>("#sample-empty")!.hidden,
    ).toBe(true);
  });

  it("copies only JSON.stringify(sample) via Copy sample", async () => {
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.includes("/dataset/test/samples")) {
          return jsonResponse(200, sampleResponseBody());
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubClipboard(writeText);

    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    document.querySelector<HTMLButtonElement>("#sample-copy-button")!.click();
    await flush();

    expect(writeText).toHaveBeenCalledWith("[1,null,0]");
    const feedback = document.querySelector<HTMLParagraphElement>(
      "#sample-copy-feedback",
    )!;
    expect(feedback.hidden).toBe(false);
    expect(feedback.textContent).toBe("Copied");
  });

  it("copies the full response object via Copy full response", async () => {
    const body = sampleResponseBody();
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.includes("/dataset/test/samples")) {
          return jsonResponse(200, body);
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);
    const writeText = vi.fn().mockResolvedValue(undefined);
    stubClipboard(writeText);

    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    document
      .querySelector<HTMLButtonElement>("#sample-copy-full-button")!
      .click();
    await flush();

    expect(writeText).toHaveBeenCalledWith(JSON.stringify(body));
    const feedback = document.querySelector<HTMLParagraphElement>(
      "#sample-copy-full-feedback",
    )!;
    expect(feedback.hidden).toBe(false);
    expect(feedback.textContent).toBe("Copied");
  });

  it("keeps the JSON visible and shows a non-destructive error when copy fails", async () => {
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.includes("/dataset/test/samples")) {
          return jsonResponse(200, sampleResponseBody());
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);
    stubClipboard(() => Promise.reject(new Error("denied")));

    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    document.querySelector<HTMLButtonElement>("#sample-copy-button")!.click();
    await flush();

    const error = document.querySelector<HTMLParagraphElement>(
      "#sample-copy-error",
    )!;
    expect(error.hidden).toBe(false);
    const sampleJson = document.querySelector<HTMLPreElement>("#sample-json")!;
    expect(sampleJson.textContent).toBe(
      JSON.stringify([1, null, 0], null, 2),
    );
  });

  it("keeps the last valid result visible and shows the new error when a later request fails", async () => {
    let shouldFail = false;
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.includes("/dataset/test/samples")) {
          if (shouldFail) {
            return jsonResponse(404, {
              error: {
                code: "test_dataset_index_out_of_range",
                message: "index is outside the test dataset",
              },
            });
          }
          return jsonResponse(200, sampleResponseBody());
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);

    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    const sampleJson = document.querySelector<HTMLPreElement>("#sample-json")!;
    expect(sampleJson.textContent).toBe(
      JSON.stringify([1, null, 0], null, 2),
    );

    shouldFail = true;
    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "999999";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    const error = document.querySelector<HTMLParagraphElement>(
      "#sample-error",
    )!;
    expect(error.hidden).toBe(false);
    expect(error.textContent).toContain("index is outside the test dataset");
    // The previous valid result must still be shown, not replaced with
    // partial or empty data.
    expect(sampleJson.textContent).toBe(
      JSON.stringify([1, null, 0], null, 2),
    );
    expect(
      document.querySelector<HTMLDivElement>("#sample-result")!.hidden,
    ).toBe(false);
  });

  it("clears the sample when the model is unloaded", async () => {
    const server = createFakeServer(
      [{ id: "mux", backend: "pyjuice" }],
      { status: "ready", id: "mux", backend: "pyjuice", ...READY_MODEL_DEFAULTS },
    );
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.includes("/dataset/test/samples")) {
        return jsonResponse(200, sampleResponseBody());
      }
      return server.fetchMock(url, init);
    });

    await loadMain(fetchMock);
    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();
    expect(
      document.querySelector<HTMLDivElement>("#sample-result")!.hidden,
    ).toBe(false);

    vi.spyOn(window, "confirm").mockReturnValue(true);
    document.querySelector<HTMLButtonElement>("#model-unload-button")!.click();
    await flushAll();

    expect(
      document.querySelector<HTMLElement>("#test-dataset-panel")!.hidden,
    ).toBe(true);
    expect(
      document.querySelector<HTMLDivElement>("#sample-result")!.hidden,
    ).toBe(true);
    expect(
      document.querySelector<HTMLParagraphElement>("#sample-empty")!.hidden,
    ).toBe(false);
  });

  it("discards a stale response when a newer sample request has already resolved", async () => {
    const resolvers: Array<(response: Response) => void> = [];
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.includes("/dataset/test/samples")) {
          return new Promise<Response>((resolve) => resolvers.push(resolve));
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);

    const getButton =
      document.querySelector<HTMLButtonElement>("#sample-get-button")!;
    const indexInput =
      document.querySelector<HTMLInputElement>("#sample-index-input")!;

    indexInput.value = "1";
    getButton.click();
    await flush();
    // The button disables itself while in flight; force it back on to
    // exercise the requestId staleness guard with a genuine second request.
    getButton.disabled = false;
    indexInput.value = "2";
    getButton.click();
    await flush();

    expect(resolvers).toHaveLength(2);

    resolvers[1](jsonResponse(200, sampleResponseBody({ index: 2 })));
    await flush();
    expect(
      document.querySelector<HTMLParagraphElement>("#sample-index-line")!
        .textContent,
    ).toBe("Index: 2 of 10000");

    resolvers[0](jsonResponse(200, sampleResponseBody({ index: 1 })));
    await flush();
    expect(
      document.querySelector<HTMLParagraphElement>("#sample-index-line")!
        .textContent,
    ).toBe("Index: 2 of 10000");
  });

  it("discards a sample response that arrives after the model has changed", async () => {
    let resolveSample: ((response: Response) => void) | null = null;
    const server = createFakeServer(
      [{ id: "mux", backend: "pyjuice" }],
      { status: "ready", id: "mux", backend: "pyjuice", ...READY_MODEL_DEFAULTS },
    );
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.includes("/dataset/test/samples")) {
        return new Promise<Response>((resolve) => {
          resolveSample = resolve;
        });
      }
      return server.fetchMock(url, init);
    });

    await loadMain(fetchMock);
    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    vi.spyOn(window, "confirm").mockReturnValue(true);
    document.querySelector<HTMLButtonElement>("#model-unload-button")!.click();
    await flushAll();

    resolveSample!(jsonResponse(200, sampleResponseBody()));
    await flush();

    expect(
      document.querySelector<HTMLDivElement>("#sample-result")!.hidden,
    ).toBe(true);
    expect(
      document.querySelector<HTMLParagraphElement>("#sample-empty")!.hidden,
    ).toBe(false);
  });

  it("tolerates additional properties on the sample response", async () => {
    const fetchMock = buildFetchMock({
      current: readySampleModel,
      extra: (url) => {
        if (url.includes("/dataset/test/samples")) {
          return jsonResponse(
            200,
            sampleResponseBody({ extra_field: "ignored" }),
          );
        }
        throw new Error(`Unhandled fetch: ${url}`);
      },
    });
    await loadMain(fetchMock);

    document.querySelector<HTMLInputElement>("#sample-index-input")!.value =
      "42";
    document.querySelector<HTMLButtonElement>("#sample-get-button")!.click();
    await flush();

    expect(
      document.querySelector<HTMLParagraphElement>("#sample-index-line")!
        .textContent,
    ).toBe("Index: 42 of 10000");
    expect(
      document.querySelector<HTMLParagraphElement>("#sample-error")!.hidden,
    ).toBe(true);
  });

  it("has accessible labels, button names and status regions", async () => {
    const fetchMock = buildFetchMock({ current: readySampleModel });
    await loadMain(fetchMock);

    const indexInput =
      document.querySelector<HTMLInputElement>("#sample-index-input")!;
    const indexLabel = document.querySelector<HTMLLabelElement>(
      `label[for="${indexInput.id}"]`,
    );
    expect(indexLabel).not.toBeNull();
    expect(indexLabel!.textContent).toBe("Sample index");

    const seedInput =
      document.querySelector<HTMLInputElement>("#sample-seed-input")!;
    const seedLabel = document.querySelector<HTMLLabelElement>(
      `label[for="${seedInput.id}"]`,
    );
    expect(seedLabel).not.toBeNull();
    expect(seedLabel!.textContent).toBe("Seed");

    expect(
      document.querySelector<HTMLButtonElement>("#sample-get-button")!
        .textContent,
    ).toBe("Get sample");
    expect(
      document.querySelector<HTMLButtonElement>("#sample-random-button")!
        .textContent,
    ).toBe("Random sample");
    expect(
      document.querySelector<HTMLButtonElement>("#sample-copy-button")!
        .textContent,
    ).toBe("Copy sample");
    expect(
      document.querySelector<HTMLButtonElement>("#sample-copy-full-button")!
        .textContent,
    ).toBe("Copy full response");

    expect(
      document.querySelector("#sample-error")!.getAttribute("role"),
    ).toBe("alert");
    expect(
      document.querySelector("#sample-loading")!.getAttribute("aria-live"),
    ).toBe("polite");
    expect(
      document.querySelector("#sample-copy-feedback")!.getAttribute(
        "aria-live",
      ),
    ).toBe("polite");
  });
});

describe("tabs", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows Playground by default with Model's tabpanel hidden", async () => {
    const fetchMock = buildFetchMock();
    await loadMain(fetchMock);

    const tabPlayground = document.querySelector<HTMLButtonElement>(
      "#tab-playground",
    )!;
    const tabModel = document.querySelector<HTMLButtonElement>("#tab-model")!;
    expect(tabPlayground.getAttribute("aria-selected")).toBe("true");
    expect(tabModel.getAttribute("aria-selected")).toBe("false");
    expect(
      document.querySelector("#panel-playground")!.hasAttribute("hidden"),
    ).toBe(false);
    expect(document.querySelector("#panel-model")!.hasAttribute("hidden")).toBe(
      true,
    );
  });

  it("switches tabs on click", async () => {
    const fetchMock = buildFetchMock();
    await loadMain(fetchMock);

    document.querySelector<HTMLButtonElement>("#tab-model")!.click();

    expect(
      document.querySelector("#tab-model")!.getAttribute("aria-selected"),
    ).toBe("true");
    expect(
      document.querySelector("#tab-playground")!.getAttribute("aria-selected"),
    ).toBe("false");
    expect(document.querySelector("#panel-model")!.hasAttribute("hidden")).toBe(
      false,
    );
    expect(
      document.querySelector("#panel-playground")!.hasAttribute("hidden"),
    ).toBe(true);

    document.querySelector<HTMLButtonElement>("#tab-playground")!.click();
    expect(
      document.querySelector("#panel-playground")!.hasAttribute("hidden"),
    ).toBe(false);
    expect(document.querySelector("#panel-model")!.hasAttribute("hidden")).toBe(
      true,
    );
  });

  it("switches tabs with ArrowRight/ArrowLeft and moves focus", async () => {
    const fetchMock = buildFetchMock();
    await loadMain(fetchMock);

    const tabPlayground = document.querySelector<HTMLButtonElement>(
      "#tab-playground",
    )!;
    const tabModel = document.querySelector<HTMLButtonElement>("#tab-model")!;

    tabPlayground.focus();
    tabPlayground.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }),
    );

    expect(tabModel.getAttribute("aria-selected")).toBe("true");
    expect(document.activeElement).toBe(tabModel);
    expect(document.querySelector("#panel-model")!.hasAttribute("hidden")).toBe(
      false,
    );

    tabModel.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }),
    );
    expect(tabPlayground.getAttribute("aria-selected")).toBe("true");
    expect(document.activeElement).toBe(tabPlayground);
  });
});


describe("Random Forest UI", () => {
  afterEach(() => vi.unstubAllGlobals());
  const current = () => jsonResponse(200, { model: {
    status: "ready", id: "mnist_rf", backend: "random_forest",
    num_features: 196, num_classes: 2, device: "cpu",
    loaded_at: "2026-10-06T20:00:00Z",
  }});

  it("explains numeric VAL semantics and the FULL example", async () => {
    await loadMain(buildFetchMock({ current }));
    expect(document.querySelector("#model-meta-backend")!.textContent).toBe("Random Forest");
    expect(document.querySelector("#model-indicator-text")!.textContent).toContain("Random Forest");
    const hint = document.querySelector("#expression-model-hint")!.textContent;
    expect(hint).toContain("integer label");
    expect(hint).toContain("numeric 0");
    expect(hint).toContain("FULL(x) * (VAL(x) == 3)");
    expect(hint).toContain("unsupported");
  });

  it.each([0, 3, 8])("keeps returned %i numeric in the evaluation panel", async (label) => {
    await loadMain(buildFetchMock({ current, extra: (url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(200, { status: "decided", value: label, last_witness: null, reason: null });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    }}));
    document.querySelector<HTMLTextAreaElement>("#expression-input")!.value = "VAL(x)";
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(document.querySelector("#evaluation-output")!.textContent).toBe(`Value: ${label}`);
  });

  it("sends unsupported queries and displays the server's 501 error", async () => {
    const fetchMock = buildFetchMock({ current, extra: (url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return jsonResponse(501, { error: { code: "unsupported_operation", message: "Consulta no soportada por Random Forest" }});
      }
      throw new Error(`Unhandled fetch: ${url}`);
    }});
    await loadMain(fetchMock);
    document.querySelector<HTMLTextAreaElement>("#expression-input")!.value = "EP(x,3)";
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flushAll();
    expect(callsTo(fetchMock, "/v1/expressions/evaluate")).toBe(1);
    expect(document.querySelector<HTMLElement>("#evaluation-error")!.hidden).toBe(false);
    expect(document.querySelector("#evaluation-error")!.textContent).toContain("Consulta no soportada por Random Forest");
  });
});


describe("input history integration", () => {
  afterEach(() => {
    localStorage.clear();
    vi.unstubAllGlobals();
  });

  it("records both inputs and ignores a pending response after history selection", async () => {
    localStorage.clear();
    let finish!: (response: Response) => void;
    const fetchMock = buildFetchMock({ extra: (url) => {
      if (url.endsWith("/v1/expressions/evaluate")) {
        return new Promise<Response>((resolve) => { finish = resolve; });
      }
      throw new Error(`Unhandled fetch: ${url}`);
    }});
    await loadMain(fetchMock);
    const expression = document.querySelector<HTMLTextAreaElement>("#expression-input")!;
    const bindings = document.querySelector<HTMLTextAreaElement>("#bindings-input")!;
    expression.value = "FULL(x)";
    bindings.value = '{"x": [1, 0]}';
    document.querySelector<HTMLButtonElement>("#evaluate-button")!.click();
    await flush();
    expect(JSON.parse(localStorage.getItem("qfoil.history.expression-input")!)[0]).toBe("FULL(x)");
    expect(JSON.parse(localStorage.getItem("qfoil.history.bindings-input")!)[0]).toBe(bindings.value);
    const select = expression.nextElementSibling!.querySelector("select")!;
    select.value = "1";
    select.dispatchEvent(new Event("change"));
    expect(bindings.value).toBe('{"x": [1, 0]}');
    finish(jsonResponse(200, { value: 99, last_witness: [1, 0] }));
    await flushAll();
    expect(document.querySelector("#evaluation-output")!.textContent).not.toContain("99");
    expect(document.querySelector<HTMLElement>("#last-witness-section")!.hidden).toBe(true);
    expect(callsTo(fetchMock, "/v1/expressions/evaluate")).toBe(1);
  });
});
