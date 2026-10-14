import { beforeEach, describe, expect, it, vi } from "vitest";

const katexRender = vi.hoisted(() => vi.fn());
vi.mock("katex", () => ({
  default: { render: katexRender },
}));

import { formatBackendLabel, renderFormula, renderScalarOrJson, renderWitnessResult } from "../render";
import type { WitnessSearchResponse } from "../api";

describe("renderFormula", () => {
  beforeEach(() => {
    katexRender.mockClear();
  });

  it("passes exactly the server-provided LaTeX and the required options to KaTeX", () => {
    const container = document.createElement("div");
    const latex = "\\operatorname{SIZE}\\left(x\\right) \\leq 2";

    renderFormula(container, latex);

    expect(katexRender).toHaveBeenCalledTimes(1);
    expect(katexRender).toHaveBeenCalledWith(latex, container, {
      throwOnError: false,
      strict: "warn",
      trust: false,
    });
  });
});

describe("renderScalarOrJson", () => {
  it("renders scalars as 'Value: <x>'", () => {
    const container = document.createElement("div");
    renderScalarOrJson(container, 1);
    expect(container.textContent).toBe("Value: 1");
  });

  it("renders structured values as formatted JSON", () => {
    const container = document.createElement("div");
    renderScalarOrJson(container, { a: [1, null] });
    expect(container.textContent).toBe(JSON.stringify({ a: [1, null] }, null, 2));
  });
});

describe("renderWitnessResult", () => {
  it("shows a found witness with its details", () => {
    const container = document.createElement("div");
    const response: WitnessSearchResponse = {
      status: "found",
      variable: "z",
      witness: [1, 0, 1],
      verified: true,
      seed_used: null,
      search: { budget: 1000, pool_size: 30, inner_budget: 8 },
    };

    renderWitnessResult(container, response);

    expect(container.textContent).toContain("found");
    expect(container.textContent).toContain("[1,0,1]");
    expect(container.textContent).toContain("yes");
    expect(container.textContent).not.toContain("No verified witness was found");
  });

  it("shows the mandated inconclusive explanation", () => {
    const container = document.createElement("div");
    const response: WitnessSearchResponse = {
      status: "inconclusive",
      variable: "z",
      witness: null,
      verified: false,
      seed_used: null,
      search: { budget: 1000, pool_size: 30, inner_budget: 8 },
      reason: "bounded_search_incomplete",
    };

    renderWitnessResult(container, response);

    expect(container.textContent).toContain("inconclusive");
    expect(container.textContent).toContain(
      "No verified witness was found within the search budget.",
    );
    expect(container.textContent).toContain(
      "This does not prove that no witness exists.",
    );
  });
});


describe("Random Forest presentation", () => {
  it("uses the backend display name", () => {
    expect(formatBackendLabel("random_forest")).toBe("Random Forest");
  });
  it.each([0, 3, 8])("renders label %i as a numeric value", (label) => {
    const container = document.createElement("div");
    renderScalarOrJson(container, label);
    expect(container.textContent).toBe(`Value: ${label}`);
  });
});
