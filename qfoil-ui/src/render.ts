// Safe DOM updates and KaTeX rendering. Nothing here uses innerHTML with
// server-provided or user-provided content.

import katex from "katex";
import type { ApiError, WitnessSearchResponse } from "./api";

export type StatusState = "checking" | "connected" | "unavailable";

export function setStatusBadge(
  indicator: HTMLElement,
  text: HTMLElement,
  state: StatusState,
  message: string,
): void {
  indicator.classList.remove(
    "status--checking",
    "status--connected",
    "status--unavailable",
  );
  indicator.classList.add(`status--${state}`);
  text.textContent = message;
}

export function renderFormula(container: HTMLElement, latex: string): void {
  katex.render(latex, container, {
    throwOnError: false,
    strict: "warn",
    trust: false,
  });
}

function clearElement(el: HTMLElement): void {
  el.replaceChildren();
}

export function renderScalarOrJson(container: HTMLElement, value: unknown): void {
  clearElement(container);
  const isScalar =
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean";

  if (isScalar) {
    const p = document.createElement("p");
    p.className = "value-line";
    p.textContent = `Value: ${String(value)}`;
    container.appendChild(p);
    return;
  }

  const pre = document.createElement("pre");
  pre.className = "mono json-output";
  pre.textContent = JSON.stringify(value, null, 2);
  container.appendChild(pre);
}

const WITNESS_COPY_FEEDBACK_MS = 1500;

function buildWitnessValueBox(witness: unknown): HTMLDivElement {
  const witnessJson = JSON.stringify(witness);

  const wrap = document.createElement("div");
  wrap.className = "witness-value-wrap";

  const label = document.createElement("span");
  label.className = "field-label";
  label.textContent = "Witness";
  wrap.appendChild(label);

  const box = document.createElement("div");
  box.className = "mono witness-json-box";
  box.tabIndex = 0;
  box.setAttribute("role", "button");
  box.setAttribute("aria-label", "Copy witness JSON to clipboard");
  box.textContent = witnessJson;
  wrap.appendChild(box);

  const feedback = document.createElement("p");
  feedback.className = "witness-copy-feedback";
  feedback.setAttribute("role", "status");
  feedback.setAttribute("aria-live", "polite");
  feedback.hidden = true;
  wrap.appendChild(feedback);

  const clipboardError = document.createElement("p");
  clipboardError.className = "error-text";
  clipboardError.setAttribute("role", "alert");
  clipboardError.hidden = true;
  wrap.appendChild(clipboardError);

  let feedbackTimeout: number | undefined;

  async function copyWitness(): Promise<void> {
    clipboardError.hidden = true;
    clipboardError.textContent = "";
    try {
      await navigator.clipboard.writeText(witnessJson);
      if (feedbackTimeout !== undefined) {
        window.clearTimeout(feedbackTimeout);
      }
      feedback.hidden = false;
      feedback.textContent = "Copied";
      feedbackTimeout = window.setTimeout(() => {
        feedback.hidden = true;
        feedback.textContent = "";
        feedbackTimeout = undefined;
      }, WITNESS_COPY_FEEDBACK_MS);
    } catch {
      clipboardError.hidden = false;
      clipboardError.textContent =
        "Could not copy automatically. Select the text above and copy it manually.";
    }
  }

  box.addEventListener("click", () => void copyWitness());
  box.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      void copyWitness();
    }
  });

  return wrap;
}

export function renderWitnessResult(
  container: HTMLElement,
  result: WitnessSearchResponse,
): void {
  clearElement(container);

  const list = document.createElement("dl");
  list.className = "witness-result-list";

  const addRow = (term: string, value: string): void => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.className = "mono";
    dd.textContent = value;
    list.appendChild(dt);
    list.appendChild(dd);
  };

  addRow("Status", result.status);
  addRow("Variable", result.variable);
  addRow("Verified", result.verified ? "yes" : "no");
  addRow("Seed used", JSON.stringify(result.seed_used));
  addRow(
    "Search options",
    `budget=${result.search.budget}, pool_size=${result.search.pool_size}, inner_budget=${result.search.inner_budget}`,
  );

  container.appendChild(list);
  container.appendChild(buildWitnessValueBox(result.witness));

  if (result.status === "inconclusive") {
    const note = document.createElement("p");
    note.className = "witness-note";
    note.textContent =
      "No verified witness was found within the search budget. " +
      "This does not prove that no witness exists.";
    container.appendChild(note);
  }
}

export function describeApiError(error: ApiError): string {
  if (error.kind === "qfoil" && error.code) {
    return `${error.message} (${error.code})`;
  }
  return error.message;
}

export function setErrorText(el: HTMLElement, message: string | null): void {
  if (message === null) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = message;
}

export function setStale(containers: HTMLElement[], stale: boolean): void {
  for (const container of containers) {
    container.classList.toggle("is-stale", stale);
  }
}

export function formatBackendLabel(backend: string): string {
  if (backend.toLowerCase() === "random_forest") {
    return "Random Forest";
  }
  if (backend.toLowerCase() === "pyjuice") {
    return "PyJuice";
  }
  return backend.length > 0 ? backend[0].toUpperCase() + backend.slice(1) : backend;
}

export function setLoadedTime(el: HTMLTimeElement, isoString: string): void {
  el.dateTime = isoString;
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) {
    el.textContent = isoString;
    return;
  }
  el.textContent = new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
