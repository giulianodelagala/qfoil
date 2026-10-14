import "katex/dist/katex.min.css";
import "./styles.css";

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
  unloadModel,
  type ConfigResponse,
  type DatasetSampleResponse,
  type ModelInfo,
  type ModelSummary,
  type EvaluationOptions,
  type Witness,
} from "./api";
import {
  parseBindings,
  parseIntegerInRange,
  parseIntegerSeed,
  parseNonNegativeInteger,
  parseSeed,
} from "./validation";
import {
  describeApiError,
  formatBackendLabel,
  renderFormula,
  renderScalarOrJson,
  setErrorText,
  setLoadedTime,
  setStale,
  setStatusBadge,
} from "./render";
import { mountVisualizeTab } from "./visualize";
import { INITIAL_BINDINGS, INITIAL_EXPRESSIONS, mountHistory } from "./history";

const DEFAULT_EXPRESSION = "SIZE(x) <= 2";
const DEFAULT_BINDINGS = '{\n  "x": [1, null, 0]\n}';
const BASE_TITLE = "QFOIL Playground";

// Single on/off switch for the experimental tuple visualization tab. Flip to
// false (or delete this line, the import above, and the mountVisualizeTab
// call further down) to remove the feature entirely.
const FEATURE_VISUALIZE_TUPLES = true;

const app = document.querySelector<HTMLDivElement>("#app");
if (app === null) {
  throw new Error("Missing #app root element");
}

app.innerHTML = `
  <header class="site-header">
    <div>
      <p class="eyebrow">Proof of concept</p>
      <h1>QFOIL Playground</h1>
    </div>
    <div class="status-bar" aria-live="polite">
      <span id="status-indicator" class="status-item status-item--checking">
        <span class="status-dot" aria-hidden="true"></span>
        <span id="status-text">Checking…</span>
      </span>
      <button id="retry-button" type="button" class="retry-button" hidden>Retry</button>
      <span id="model-indicator" class="status-item status-item--unavailable">
        <span id="model-indicator-icon" class="status-item-icon" aria-hidden="true">◌</span>
        <span id="model-indicator-text">Checking…</span>
      </span>
    </div>
  </header>

  <div id="model-live-region" class="sr-only" aria-live="polite" role="status"></div>

  <div id="tablist" class="tablist" role="tablist" aria-label="Views">
    <button
      id="tab-playground"
      type="button"
      class="tab"
      role="tab"
      aria-selected="true"
      aria-controls="panel-playground"
    >Playground</button>
    <button
      id="tab-model"
      type="button"
      class="tab"
      role="tab"
      aria-selected="false"
      aria-controls="panel-model"
      tabindex="-1"
    >Model</button>
    <button id="tab-settings" type="button" class="tab" role="tab"
      aria-selected="false" aria-controls="panel-settings" tabindex="-1"
    >Evaluation settings</button>
  </div>

  <div id="panel-settings" class="tabpanel" role="tabpanel"
    aria-labelledby="tab-settings" tabindex="0" hidden>
    <section class="card evaluation-settings" aria-labelledby="settings-heading">
      <h2 id="settings-heading">Evaluation settings</h2>
      <p class="model-panel-note">These settings apply when you evaluate an expression in Playground.</p>
      <div id="evaluation-options" hidden>
        <div class="field">
          <label for="evaluation-mode">Mode</label>
          <select id="evaluation-mode">
            <option value="heuristic">Heuristic</option>
            <option value="exact">Exhaustive</option>
          </select>
          <p id="evaluation-mode-note" class="model-panel-note"></p>
        </div>
        <div class="field">
          <label for="evaluation-work-budget">Total evaluation budget</label>
          <input id="evaluation-work-budget" class="mono" type="text" inputmode="numeric" />
          <p id="evaluation-global-budget-note" class="model-panel-note"></p>
        </div>
        <div id="heuristic-settings">
          <div class="field">
            <label for="evaluation-budget">Search budget per SIGMA attempt</label>
            <input id="evaluation-budget" class="mono" type="text" inputmode="numeric" />
            <p id="evaluation-budget-note" class="model-panel-note"></p>
          </div>
          <div class="field">
            <label for="evaluation-rng-seed">Random seed (optional integer)</label>
            <input id="evaluation-rng-seed" class="mono" type="text" placeholder="38" />
            <p class="model-panel-note">Use the same seed to reproduce random starts and moves.</p>
          </div>
          <div class="field">
            <label for="evaluation-seed-source">Starting tuple</label>
            <select id="evaluation-seed-source">
              <option value="random">Random starts</option>
              <option value="tuple">Use an initial tuple</option>
            </select>
          </div>
          <div id="evaluation-initial-tuple-field" class="field" hidden>
            <label for="evaluation-initial-tuple">Initial tuple (JSON)</label>
            <textarea id="evaluation-initial-tuple" class="mono" rows="3"
              placeholder="[1, null, 0]" spellcheck="false"></textarea>
            <p class="model-panel-note">Used for a direct SIGMA comparison. The tuple must match the model's number of features.</p>
          </div>
        </div>
        <div class="actions">
          <button id="evaluation-restore-defaults" type="button" class="secondary-button">Restore defaults</button>
        </div>
      </div>
      <p id="evaluation-config-error" class="error-text" role="alert" hidden></p>
      <button id="evaluation-retry-config" type="button" class="secondary-button" hidden>Reload settings</button>
      <p id="evaluation-settings-error" class="error-text" role="alert" hidden></p>
    </section>
  </div>

  <div
    id="panel-model"
    class="tabpanel"
    role="tabpanel"
    aria-labelledby="tab-model"
    tabindex="0"
    hidden
  >
  <section id="model-panel" class="card model-panel" aria-labelledby="model-panel-heading">
    <h2 id="model-panel-heading">Model</h2>

    <div id="model-panel-unloaded" class="model-panel-section" hidden>
      <p id="model-catalog-empty" class="empty" hidden>
        No models are configured on qfoil-server.
      </p>
      <div id="model-catalog-form">
        <div class="field">
          <label for="model-select">Model</label>
          <select id="model-select"></select>
        </div>
      </div>
      <p class="model-panel-note">
        No model is loaded.<br />
        SIZE, FULL and formula rendering remain available.<br />
        EP, CEP, VAL and COND require an active model.
      </p>
    </div>

    <div id="model-panel-ready" class="model-panel-section" hidden>
      <dl class="model-meta-list">
        <dt>Model</dt>
        <dd id="model-meta-id" class="mono"></dd>
        <dt>Status</dt>
        <dd id="model-meta-status"></dd>
        <dt>Backend</dt>
        <dd id="model-meta-backend"></dd>
        <dt>Features</dt>
        <dd id="model-meta-features"></dd>
        <dt>Classes</dt>
        <dd id="model-meta-classes"></dd>
        <dt>Device</dt>
        <dd id="model-meta-device" class="mono"></dd>
        <dt>Loaded</dt>
        <dd><time id="model-meta-loaded"></time></dd>
      </dl>
      <p class="model-panel-note">
        Unload the current model before loading another.
      </p>
    </div>

    <div id="model-panel-error-state" class="model-panel-section" hidden>
      <p class="empty">The model lifecycle reported an inconsistent state.</p>
    </div>

    <p id="model-panel-error" class="error-text" role="alert" hidden></p>

    <div class="actions">
      <button id="model-load-button" type="button" hidden>Load</button>
      <button id="model-unload-button" type="button" hidden>Unload</button>
      <button id="model-refresh-button" type="button" class="secondary-button">Refresh</button>
    </div>
  </section>

  <section
    id="test-dataset-panel"
    class="card model-panel"
    aria-labelledby="test-dataset-heading"
    hidden
  >
    <h2 id="test-dataset-heading">Test dataset</h2>
    <p class="model-panel-note">
      Read-only inspection of the active model's test split. It does not
      change the expression, bindings or active tab.
    </p>

    <div class="field">
      <label for="sample-index-input">Sample index</label>
      <input id="sample-index-input" class="mono" type="text" inputmode="numeric" />
    </div>

    <div class="actions">
      <button id="sample-get-button" type="button">Get sample</button>
      <button id="sample-random-button" type="button" class="secondary-button">Random sample</button>
    </div>

    <div class="field">
      <label for="sample-seed-input">Seed</label>
      <input id="sample-seed-input" class="mono" type="text" />
    </div>
    <p class="model-panel-note">
      Optional. Used only by Random sample; must be a base-10 integer (e.g.
      123 or -7). Leaving it empty omits the seed.
    </p>

    <p id="sample-loading" class="empty" aria-live="polite" hidden>Loading sample…</p>
    <p id="sample-empty" class="empty">No sample loaded yet</p>
    <p id="sample-error" class="error-text" role="alert" hidden></p>

    <div id="sample-result" hidden>
      <p id="sample-index-line" class="value-line"></p>

      <div>
        <span class="field-label">Sample</span>
        <pre id="sample-json" class="mono json-output sample-json-box"></pre>
        <div class="actions">
          <button id="sample-copy-button" type="button" class="secondary-button">Copy sample</button>
        </div>
        <p id="sample-copy-feedback" class="witness-copy-feedback" role="status" aria-live="polite" hidden></p>
        <p id="sample-copy-error" class="error-text" role="alert" hidden></p>
      </div>

      <div>
        <span class="field-label">Expected</span>
        <pre id="sample-expected-json" class="mono json-output sample-json-box"></pre>
      </div>

      <div class="actions">
        <button id="sample-copy-full-button" type="button" class="secondary-button">Copy full response</button>
      </div>
      <p id="sample-copy-full-feedback" class="witness-copy-feedback" role="status" aria-live="polite" hidden></p>
      <p id="sample-copy-full-error" class="error-text" role="alert" hidden></p>
    </div>
  </section>
  </div>

  <div
    id="panel-playground"
    class="tabpanel"
    role="tabpanel"
    aria-labelledby="tab-playground"
    tabindex="0"
  >
  <main>
    <div class="columns">
      <section class="card" aria-labelledby="expression-heading">
        <h2 id="expression-heading">Expression</h2>
        <p id="evaluation-settings-summary" class="model-panel-note" aria-live="polite">Heuristic mode · server defaults</p>
        <p id="expression-model-hint" class="model-panel-note" aria-live="polite"></p>
        <form id="expression-form">
          <div class="field">
            <label for="expression-input">Expression</label>
            <textarea id="expression-input" class="mono" rows="4" spellcheck="false"></textarea>
          </div>
          <div class="field">
            <label for="bindings-input">Bindings</label>
            <textarea id="bindings-input" class="mono" rows="7" spellcheck="false"></textarea>
          </div>
          <div class="actions">
            <button id="render-button" type="submit">Render formula</button>
            <button id="evaluate-button" type="button">Evaluate</button>
          </div>
        </form>
      </section>

      <div class="column-stack">
        <section class="card" aria-labelledby="formula-heading">
          <h2 id="formula-heading">Formula</h2>
          <p id="formula-empty" class="empty">No formula rendered yet</p>
          <div id="formula-output" class="formula-output" aria-live="polite" hidden></div>
          <div id="latex-source-wrap" class="latex-source-wrap" hidden>
            <span class="field-label">LaTeX source</span>
            <pre id="latex-source" class="mono latex-source" tabindex="0"></pre>
          </div>
          <p id="formula-error" class="error-text" role="alert" hidden></p>
        </section>

        <section class="card" aria-labelledby="evaluation-heading">
          <h2 id="evaluation-heading">Evaluation result</h2>
          <div id="evaluation-output" aria-live="polite">
            <p id="evaluation-empty" class="empty">No evaluation yet</p>
          </div>
          <div id="last-witness-section" hidden>
            <p id="last-witness-empty" class="empty" hidden>
              No witness was produced during this evaluation.
            </p>
            <div id="last-witness-block" hidden>
              <span class="field-label">Last witness</span>
              <div
                id="last-witness-box"
                class="mono witness-json-box"
                tabindex="0"
                role="button"
                aria-label="Copy last witness JSON to clipboard"
              ></div>
              <p class="witness-copy-hint">
                Verified witness produced while evaluating the SIGMA
                comparison. Click to copy it as JSON.
              </p>
              <p
                id="last-witness-feedback"
                class="witness-copy-feedback"
                role="status"
                aria-live="polite"
                hidden
              ></p>
              <p
                id="last-witness-clipboard-error"
                class="error-text"
                role="alert"
                hidden
              ></p>
            </div>
          </div>
          <p id="evaluation-error" class="error-text" role="alert" hidden></p>
        </section>
      </div>
    </div>

  </main>
  </div>
`;

function requireElement<T extends Element>(selector: string): T {
  const el = document.querySelector<T>(selector);
  if (el === null) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return el;
}

const statusIndicator = requireElement<HTMLSpanElement>("#status-indicator");
const statusText = requireElement<HTMLSpanElement>("#status-text");
const retryButton = requireElement<HTMLButtonElement>("#retry-button");

const modelIndicatorStatus = requireElement<HTMLSpanElement>(
  "#model-indicator",
);
const modelIndicatorIcon = requireElement<HTMLSpanElement>(
  "#model-indicator-icon",
);
const modelIndicatorText = requireElement<HTMLSpanElement>(
  "#model-indicator-text",
);
const tablist = requireElement<HTMLDivElement>("#tablist");
const tabPlayground = requireElement<HTMLButtonElement>("#tab-playground");
const tabModel = requireElement<HTMLButtonElement>("#tab-model");
const panelPlayground = requireElement<HTMLDivElement>("#panel-playground");
const panelModel = requireElement<HTMLDivElement>("#panel-model");
const modelPanelUnloadedSection = requireElement<HTMLDivElement>(
  "#model-panel-unloaded",
);
const modelCatalogEmpty = requireElement<HTMLParagraphElement>(
  "#model-catalog-empty",
);
const modelCatalogForm = requireElement<HTMLDivElement>("#model-catalog-form");
const modelSelect = requireElement<HTMLSelectElement>("#model-select");
const modelLoadButton = requireElement<HTMLButtonElement>(
  "#model-load-button",
);
const modelPanelReadySection = requireElement<HTMLDivElement>(
  "#model-panel-ready",
);
const modelMetaId = requireElement<HTMLElement>("#model-meta-id");
const modelMetaStatus = requireElement<HTMLElement>("#model-meta-status");
const modelMetaBackend = requireElement<HTMLElement>("#model-meta-backend");
const modelMetaFeatures = requireElement<HTMLElement>("#model-meta-features");
const modelMetaClasses = requireElement<HTMLElement>("#model-meta-classes");
const modelMetaDevice = requireElement<HTMLElement>("#model-meta-device");
const modelMetaLoaded = requireElement<HTMLTimeElement>("#model-meta-loaded");
const modelUnloadButton = requireElement<HTMLButtonElement>(
  "#model-unload-button",
);
const modelPanelErrorSection = requireElement<HTMLDivElement>(
  "#model-panel-error-state",
);
const modelPanelError = requireElement<HTMLParagraphElement>(
  "#model-panel-error",
);
const modelRefreshButton = requireElement<HTMLButtonElement>(
  "#model-refresh-button",
);
const modelLiveRegion = requireElement<HTMLDivElement>("#model-live-region");

const testDatasetPanel = requireElement<HTMLElement>("#test-dataset-panel");
const sampleIndexInput = requireElement<HTMLInputElement>("#sample-index-input");
const sampleSeedInput = requireElement<HTMLInputElement>("#sample-seed-input");
const sampleGetButton = requireElement<HTMLButtonElement>("#sample-get-button");
const sampleRandomButton = requireElement<HTMLButtonElement>(
  "#sample-random-button",
);
const sampleLoading = requireElement<HTMLParagraphElement>("#sample-loading");
const sampleEmpty = requireElement<HTMLParagraphElement>("#sample-empty");
const sampleError = requireElement<HTMLParagraphElement>("#sample-error");
const sampleResult = requireElement<HTMLDivElement>("#sample-result");
const sampleIndexLine = requireElement<HTMLParagraphElement>("#sample-index-line");
const sampleJson = requireElement<HTMLPreElement>("#sample-json");
const sampleCopyButton = requireElement<HTMLButtonElement>("#sample-copy-button");
const sampleCopyFeedback = requireElement<HTMLParagraphElement>(
  "#sample-copy-feedback",
);
const sampleCopyError = requireElement<HTMLParagraphElement>("#sample-copy-error");
const sampleExpectedJson = requireElement<HTMLPreElement>("#sample-expected-json");
const sampleCopyFullButton = requireElement<HTMLButtonElement>(
  "#sample-copy-full-button",
);
const sampleCopyFullFeedback = requireElement<HTMLParagraphElement>(
  "#sample-copy-full-feedback",
);
const sampleCopyFullError = requireElement<HTMLParagraphElement>(
  "#sample-copy-full-error",
);

const expressionForm = requireElement<HTMLFormElement>("#expression-form");
const expressionInput = requireElement<HTMLTextAreaElement>("#expression-input");
const expressionModelHint = requireElement<HTMLParagraphElement>(
  "#expression-model-hint",
);
const bindingsInput = requireElement<HTMLTextAreaElement>("#bindings-input");
const renderButton = requireElement<HTMLButtonElement>("#render-button");
const evaluateButton = requireElement<HTMLButtonElement>("#evaluate-button");

const formulaEmpty = requireElement<HTMLParagraphElement>("#formula-empty");
const formulaOutput = requireElement<HTMLDivElement>("#formula-output");
const latexSourceWrap = requireElement<HTMLDivElement>("#latex-source-wrap");
const latexSource = requireElement<HTMLPreElement>("#latex-source");
const formulaError = requireElement<HTMLParagraphElement>("#formula-error");

const evaluationOutput = requireElement<HTMLDivElement>("#evaluation-output");
const evaluationError = requireElement<HTMLParagraphElement>("#evaluation-error");

const lastWitnessSection = requireElement<HTMLDivElement>("#last-witness-section");
const lastWitnessEmpty = requireElement<HTMLParagraphElement>("#last-witness-empty");
const lastWitnessBlock = requireElement<HTMLDivElement>("#last-witness-block");
const lastWitnessBox = requireElement<HTMLDivElement>("#last-witness-box");
const lastWitnessFeedback = requireElement<HTMLParagraphElement>(
  "#last-witness-feedback",
);
const lastWitnessClipboardError = requireElement<HTMLParagraphElement>(
  "#last-witness-clipboard-error",
);

const tabSettings = requireElement<HTMLButtonElement>("#tab-settings");
const panelSettings = requireElement<HTMLDivElement>("#panel-settings");
const evaluationOptionsSection = requireElement<HTMLDivElement>("#evaluation-options");
const evaluationMode = requireElement<HTMLSelectElement>("#evaluation-mode");
const evaluationBudget = requireElement<HTMLInputElement>("#evaluation-budget");
const evaluationWorkBudget = requireElement<HTMLInputElement>("#evaluation-work-budget");
const evaluationRngSeed = requireElement<HTMLInputElement>("#evaluation-rng-seed");
const evaluationSeedSource = requireElement<HTMLSelectElement>("#evaluation-seed-source");
const evaluationInitialTuple = requireElement<HTMLTextAreaElement>("#evaluation-initial-tuple");
const evaluationSettingsError = requireElement<HTMLParagraphElement>("#evaluation-settings-error");
const evaluationConfigError = requireElement<HTMLParagraphElement>("#evaluation-config-error");
const evaluationRetryConfig = requireElement<HTMLButtonElement>("#evaluation-retry-config");
const evaluationRestoreDefaults = requireElement<HTMLButtonElement>("#evaluation-restore-defaults");
let evaluationConfig: ConfigResponse | null = null;

expressionInput.value = DEFAULT_EXPRESSION;
bindingsInput.value = DEFAULT_BINDINGS;
const expressionHistory = mountHistory(expressionInput, INITIAL_EXPRESSIONS);
const bindingsHistory = mountHistory(bindingsInput, INITIAL_BINDINGS);

function resolveErrorMessage(error: unknown): string {
  if (error instanceof ApiRequestError) {
    return describeApiError(error.apiError);
  }
  return "Unexpected error.";
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function announce(message: string): void {
  modelLiveRegion.textContent = message;
}

// --- Server health -------------------------------------------------------

type ServerStatus = "checking" | "connected" | "unavailable";
let serverStatus: ServerStatus = "checking";

let healthRequestId = 0;
let healthAbortController: AbortController | null = null;

async function checkHealth(): Promise<void> {
  const requestId = ++healthRequestId;
  healthAbortController?.abort();
  const controller = new AbortController();
  healthAbortController = controller;

  serverStatus = "checking";
  setStatusBadge(statusIndicator, statusText, "checking", "Checking…");
  retryButton.hidden = true;
  renderModelIndicatorAndTitle();
  updateControlAvailability();

  try {
    await getHealth(controller.signal);
    if (requestId !== healthRequestId) return;
    serverStatus = "connected";
    setStatusBadge(statusIndicator, statusText, "connected", "Connected");
    renderModelIndicatorAndTitle();
    updateControlAvailability();
    void refreshCatalog();
    void refreshCurrentModel();
  } catch (error) {
    if (requestId !== healthRequestId || isAbortError(error)) return;
    serverStatus = "unavailable";
    setStatusBadge(statusIndicator, statusText, "unavailable", "Unavailable");
    retryButton.hidden = false;
    renderModelIndicatorAndTitle();
    updateControlAvailability();
  }
}

retryButton.addEventListener("click", () => void checkHealth());

// --- Model lifecycle -------------------------------------------------------

type ModelKind = "unloaded" | "loading" | "ready" | "unloading" | "error";

type ModelUiState = {
  kind: ModelKind;
  id?: string;
  backend?: string;
  num_features?: number;
  num_classes?: number;
  device?: string;
  loaded_at?: string;
};

let modelState: ModelUiState = { kind: "unloaded" };
let modelCatalog: ModelSummary[] = [];
let catalogError: string | null = null;

function mapModelInfo(info: ModelInfo): ModelUiState {
  if (info.status === "ready") {
    return {
      kind: "ready",
      id: info.id,
      backend: info.backend,
      num_features: info.num_features,
      num_classes: info.num_classes,
      device: info.device,
      loaded_at: info.loaded_at,
    };
  }
  if (info.status === "unloaded") {
    return { kind: "unloaded" };
  }
  return { kind: info.status, id: info.id, backend: info.backend };
}

function applyModelState(next: ModelUiState): void {
  const changed = modelState.kind !== next.kind || modelState.id !== next.id;
  modelState = next;
  if (changed) {
    clearLastWitness();
    resetSampleOperation();
    clearSampleResult();
  }
  renderModelIndicatorAndTitle();
  syncModelPanel();
  syncSampleDatasetPanel();
  updateControlAvailability();
}

function renderModelIndicatorAndTitle(): void {
  if (serverStatus !== "connected") {
    const text = serverStatus === "checking" ? "Checking…" : "Model status unavailable";
    modelIndicatorStatus.className = "status-item status-item--unavailable";
    modelIndicatorIcon.textContent = "◌";
    modelIndicatorText.textContent = text;
    document.title = BASE_TITLE;
    expressionModelHint.textContent =
      serverStatus === "checking"
        ? "Checking server…"
        : "Server unavailable — evaluation is disabled.";
    return;
  }

  switch (modelState.kind) {
    case "unloaded":
      modelIndicatorStatus.className = "status-item status-item--unloaded";
      modelIndicatorIcon.textContent = "○";
      modelIndicatorText.textContent = "No model loaded";
      document.title = `${BASE_TITLE} · No model`;
      expressionModelHint.textContent =
        "No model loaded — SIZE, FULL and formula rendering work here; " +
        "EP, CEP, VAL and COND need a model (see the Model tab).";
      break;
    case "loading":
      modelIndicatorStatus.className = "status-item status-item--loading";
      modelIndicatorIcon.textContent = "◌";
      modelIndicatorText.textContent = `Loading ${modelState.id}…`;
      document.title = `${BASE_TITLE} · Loading ${modelState.id}…`;
      expressionModelHint.textContent = `Loading ${modelState.id}… Backend operations will be available once it's ready.`;
      break;
    case "ready": {
      const summary = [
        modelState.id,
        modelState.backend ? formatBackendLabel(modelState.backend) : null,
        modelState.device,
      ]
        .filter((part): part is string => Boolean(part))
        .join(" · ");
      modelIndicatorStatus.className = "status-item status-item--ready";
      modelIndicatorIcon.textContent = "●";
      modelIndicatorText.textContent = summary;
      document.title = `${BASE_TITLE} · ${modelState.id}`;
      expressionModelHint.textContent = modelState.backend === "random_forest"
        ? `Model ${modelState.id} ready — Random Forest: VAL(x) returns the integer label for complete binary instances and numeric 0 for partial instances. Example: FULL(x) * (VAL(x) == 3). VAL(x,c), EP, CEP and COND are unsupported.`
        : `Model ${modelState.id} ready — SIZE, FULL, EP, CEP, VAL and COND are available for PyJuice.`;
      break;
    }
    case "unloading":
      modelIndicatorStatus.className = "status-item status-item--unloading";
      modelIndicatorIcon.textContent = "◌";
      modelIndicatorText.textContent = `Unloading ${modelState.id}…`;
      document.title = `${BASE_TITLE} · Unloading ${modelState.id}…`;
      expressionModelHint.textContent = `Unloading ${modelState.id}… Backend operations will stop working.`;
      break;
    case "error":
      modelIndicatorStatus.className = "status-item status-item--error";
      modelIndicatorIcon.textContent = "⚠";
      modelIndicatorText.textContent = "Model error";
      document.title = `${BASE_TITLE} · Model error`;
      expressionModelHint.textContent =
        "Model error — check the Model tab.";
      break;
  }
}

function populateModelSelect(): void {
  const previous = modelSelect.value;
  modelSelect.replaceChildren();
  for (const entry of modelCatalog) {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = `${entry.id} (${entry.backend})`;
    modelSelect.appendChild(option);
  }
  if (modelCatalog.some((entry) => entry.id === previous)) {
    modelSelect.value = previous;
  }
}

function syncModelPanel(): void {
  const showUnloaded = modelState.kind === "unloaded" || modelState.kind === "loading";
  const showReady = modelState.kind === "ready" || modelState.kind === "unloading";
  const showError = modelState.kind === "error";

  modelPanelUnloadedSection.hidden = !showUnloaded;
  modelPanelReadySection.hidden = !showReady;
  modelPanelErrorSection.hidden = !showError;

  let hasCatalog = false;
  if (showUnloaded) {
    populateModelSelect();
    hasCatalog = modelCatalog.length > 0;
    modelCatalogEmpty.hidden = hasCatalog;
    modelCatalogForm.hidden = !hasCatalog;
    modelLoadButton.textContent =
      modelState.kind === "loading" ? "Loading…" : "Load";
  }
  modelLoadButton.hidden = !showUnloaded || !hasCatalog;
  modelUnloadButton.hidden = !showReady;

  if (showReady) {
    if (modelState.kind === "ready") {
      modelMetaId.textContent = modelState.id ?? "";
      modelMetaStatus.textContent = "Ready";
      modelMetaBackend.textContent = modelState.backend
        ? formatBackendLabel(modelState.backend)
        : "";
      modelMetaFeatures.textContent = String(modelState.num_features ?? "");
      modelMetaClasses.textContent = String(modelState.num_classes ?? "");
      modelMetaDevice.textContent = modelState.device ?? "";
      if (modelState.loaded_at) {
        setLoadedTime(modelMetaLoaded, modelState.loaded_at);
      }
    } else {
      modelMetaStatus.textContent = "Unloading…";
    }
    modelUnloadButton.textContent =
      modelState.kind === "unloading" ? "Unloading…" : "Unload";
  }

  if (catalogError !== null && showUnloaded) {
    setErrorText(modelPanelError, catalogError);
  }
}

function updateControlAvailability(): void {
  const connected = serverStatus === "connected";
  const busy = modelState.kind === "loading" || modelState.kind === "unloading";

  renderButton.disabled = !connected;
  evaluateButton.disabled = !connected || busy;

  const canLoad = connected && modelState.kind === "unloaded" && modelCatalog.length > 0;
  modelSelect.disabled = !canLoad;
  modelLoadButton.disabled = !canLoad;

  modelUnloadButton.disabled = !(connected && modelState.kind === "ready");

  const sampleReady = connected && modelState.kind === "ready";
  sampleIndexInput.disabled = !sampleReady;
  sampleSeedInput.disabled = !sampleReady;
  sampleGetButton.disabled = !sampleReady || sampleOperationInFlight;
  sampleRandomButton.disabled = !sampleReady || sampleOperationInFlight;
}

let currentRequestId = 0;

async function refreshCurrentModel(): Promise<void> {
  const requestId = ++currentRequestId;
  try {
    const response = await getCurrentModel();
    if (requestId !== currentRequestId) return;
    applyModelState(mapModelInfo(response.model));
  } catch (error) {
    if (requestId !== currentRequestId || isAbortError(error)) return;
    if (error instanceof ApiRequestError && error.apiError.kind === "network") {
      void checkHealth();
      return;
    }
    setErrorText(modelPanelError, resolveErrorMessage(error));
  }
}

async function refreshCatalog(): Promise<void> {
  try {
    const response = await listModels();
    modelCatalog = response.models;
    catalogError = null;
  } catch (error) {
    if (isAbortError(error)) return;
    modelCatalog = [];
    catalogError = resolveErrorMessage(error);
  }
  syncModelPanel();
  updateControlAvailability();
}

async function handleLoadModel(): Promise<void> {
  if (modelState.kind !== "unloaded") return;
  const modelId = modelSelect.value;
  if (modelId === "") return;
  const entry = modelCatalog.find((item) => item.id === modelId);

  setErrorText(modelPanelError, null);
  applyModelState({ kind: "loading", id: modelId, backend: entry?.backend });
  announce(`Loading ${modelId}…`);

  try {
    const response = await loadModel(modelId);
    applyModelState(mapModelInfo(response.model));
    announce(
      response.model.status === "ready"
        ? `Model ${modelId} is ready.`
        : `Model ${modelId} load finished.`,
    );
    void refreshCurrentModel();
  } catch (error) {
    const message = resolveErrorMessage(error);
    setErrorText(modelPanelError, message);
    announce(message);
    await refreshCurrentModel();
  }
}

async function handleUnloadModel(): Promise<void> {
  if (modelState.kind !== "ready") return;
  const id = modelState.id;
  const backend = modelState.backend;

  const confirmed = window.confirm(
    `Unload model "${id}"? Backend operations will stop working until another model is loaded.`,
  );
  if (!confirmed) return;

  setErrorText(modelPanelError, null);
  applyModelState({ kind: "unloading", id, backend });
  announce(`Unloading ${id}…`);

  try {
    const response = await unloadModel();
    applyModelState(mapModelInfo(response.model));
    announce("Model unloaded.");
    void refreshCurrentModel();
    void refreshCatalog();
  } catch (error) {
    const message = resolveErrorMessage(error);
    setErrorText(modelPanelError, message);
    announce(message);
    await refreshCurrentModel();
  }
}

async function handleModelRefresh(): Promise<void> {
  await Promise.all([refreshCatalog(), refreshCurrentModel()]);
}

interface TabEntry {
  id: string;
  button: HTMLButtonElement;
  panel: HTMLDivElement;
}

const tabs: TabEntry[] = [];

function registerTab(entry: TabEntry): void {
  tabs.push(entry);
  entry.button.addEventListener("click", () => setActiveTab(entry.id));
}

function setActiveTab(id: string, options: { focusTab?: boolean } = {}): void {
  let activeEntry: TabEntry | undefined;
  for (const entry of tabs) {
    const isActive = entry.id === id;
    entry.button.setAttribute("aria-selected", String(isActive));
    entry.button.tabIndex = isActive ? 0 : -1;
    entry.panel.hidden = !isActive;
    if (isActive) {
      activeEntry = entry;
    }
  }

  if (options.focusTab) {
    activeEntry?.button.focus();
  }
}

registerTab({ id: "playground", button: tabPlayground, panel: panelPlayground });
registerTab({ id: "model", button: tabModel, panel: panelModel });
registerTab({ id: "settings", button: tabSettings, panel: panelSettings });

if (FEATURE_VISUALIZE_TUPLES) {
  registerTab(mountVisualizeTab(tablist, app));
}

setActiveTab("playground");

tablist.addEventListener("keydown", (event) => {
  const currentIndex = tabs.findIndex((entry) => entry.button === document.activeElement);
  if (currentIndex === -1) return;

  let nextIndex: number | null = null;
  if (event.key === "ArrowRight") {
    nextIndex = (currentIndex + 1) % tabs.length;
  } else if (event.key === "ArrowLeft") {
    nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
  } else if (event.key === "Home") {
    nextIndex = 0;
  } else if (event.key === "End") {
    nextIndex = tabs.length - 1;
  }
  if (nextIndex === null) return;

  event.preventDefault();
  setActiveTab(tabs[nextIndex].id, { focusTab: true });
});

modelLoadButton.addEventListener("click", () => void handleLoadModel());
modelUnloadButton.addEventListener("click", () => void handleUnloadModel());
modelRefreshButton.addEventListener("click", () => void handleModelRefresh());

window.addEventListener("focus", () => {
  if (serverStatus === "connected") {
    void refreshCurrentModel();
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && serverStatus === "connected") {
    void refreshCurrentModel();
  }
});

let sampleRequestId = 0;
let sampleAbortController: AbortController | null = null;
let sampleOperationInFlight = false;
let currentSampleJson: string | null = null;
let currentFullResponseJson: string | null = null;

const sampleCopy = makeCopyHandler(
  () => currentSampleJson,
  sampleCopyFeedback,
  sampleCopyError,
);
const fullResponseCopy = makeCopyHandler(
  () => currentFullResponseJson,
  sampleCopyFullFeedback,
  sampleCopyFullError,
);

renderModelIndicatorAndTitle();
syncModelPanel();
syncSampleDatasetPanel();
updateControlAvailability();

void checkHealth();
void loadEvaluationConfig();

// --- Render formula --------------------------------------------------------

let latexRequestId = 0;
let latexAbortController: AbortController | null = null;

async function handleRenderFormula(): Promise<void> {
  const expression = expressionInput.value;
  if (expression.trim() === "") {
    setErrorText(formulaError, "Expression must not be empty.");
    return;
  }

  const requestId = ++latexRequestId;
  expressionHistory.add(expression);
  latexAbortController?.abort();
  const controller = new AbortController();
  latexAbortController = controller;

  setErrorText(formulaError, null);
  setStale([formulaOutput, latexSourceWrap], true);
  renderButton.disabled = true;
  renderButton.textContent = "Rendering…";

  try {
    const response = await renderLatex({ expression }, controller.signal);
    if (requestId !== latexRequestId) return;
    formulaEmpty.hidden = true;
    formulaOutput.hidden = false;
    renderFormula(formulaOutput, response.latex);
    latexSourceWrap.hidden = false;
    latexSource.textContent = response.latex;
  } catch (error) {
    if (requestId !== latexRequestId || isAbortError(error)) return;
    setErrorText(formulaError, resolveErrorMessage(error));
  } finally {
    if (requestId === latexRequestId) {
      setStale([formulaOutput, latexSourceWrap], false);
      renderButton.textContent = "Render formula";
      updateControlAvailability();
    }
  }
}

expressionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void handleRenderFormula();
});

// --- Last witness ------------------------------------------------------

const COPY_FEEDBACK_MS = 1500;

let currentLastWitnessJson: string | null = null;
let copyFeedbackTimeout: number | undefined;

function setCopyFeedback(message: string | null): void {
  if (copyFeedbackTimeout !== undefined) {
    window.clearTimeout(copyFeedbackTimeout);
    copyFeedbackTimeout = undefined;
  }
  if (message === null) {
    lastWitnessFeedback.hidden = true;
    lastWitnessFeedback.textContent = "";
    return;
  }
  lastWitnessFeedback.hidden = false;
  lastWitnessFeedback.textContent = message;
  copyFeedbackTimeout = window.setTimeout(() => {
    lastWitnessFeedback.hidden = true;
    lastWitnessFeedback.textContent = "";
    copyFeedbackTimeout = undefined;
  }, COPY_FEEDBACK_MS);
}

function clearLastWitness(): void {
  currentLastWitnessJson = null;
  setCopyFeedback(null);
  setErrorText(lastWitnessClipboardError, null);
  lastWitnessSection.hidden = true;
  lastWitnessEmpty.hidden = true;
  lastWitnessBlock.hidden = true;
  lastWitnessBox.textContent = "";
}

function showLastWitness(witness: Witness | null): void {
  lastWitnessSection.hidden = false;
  setCopyFeedback(null);
  setErrorText(lastWitnessClipboardError, null);

  if (witness === null) {
    lastWitnessEmpty.hidden = false;
    lastWitnessBlock.hidden = true;
    currentLastWitnessJson = null;
    return;
  }

  lastWitnessEmpty.hidden = true;
  lastWitnessBlock.hidden = false;
  currentLastWitnessJson = JSON.stringify(witness);
  lastWitnessBox.textContent = currentLastWitnessJson;
}

async function copyLastWitness(): Promise<void> {
  if (currentLastWitnessJson === null) return;
  setErrorText(lastWitnessClipboardError, null);
  try {
    await navigator.clipboard.writeText(currentLastWitnessJson);
    setCopyFeedback("Copied");
  } catch {
    setErrorText(
      lastWitnessClipboardError,
      "Could not copy automatically. Select the text above and copy it manually.",
    );
  }
}

lastWitnessBox.addEventListener("click", () => void copyLastWitness());
lastWitnessBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    void copyLastWitness();
  }
});

expressionInput.addEventListener("input", clearLastWitness);
bindingsInput.addEventListener("input", clearLastWitness);

// --- Evaluate ----------------------------------------------------------

let evaluateRequestId = 0;
let evaluateAbortController: AbortController | null = null;

async function handleEvaluate(): Promise<void> {
  const bindingsResult = parseBindings(bindingsInput.value);
  if (!bindingsResult.ok) {
    setErrorText(evaluationError, bindingsResult.message);
    return;
  }

  const expression = expressionInput.value;
  if (expression.trim() === "") {
    setErrorText(evaluationError, "Expression must not be empty.");
    return;
  }

  const options = readEvaluationOptions();
  if (options === null) return;
  setErrorText(evaluationSettingsError, null);
  const requestId = ++evaluateRequestId;
  expressionHistory.add(expression);
  bindingsHistory.add(bindingsInput.value);
  evaluateAbortController?.abort();
  const controller = new AbortController();
  evaluateAbortController = controller;

  setErrorText(evaluationError, null);
  setStale([evaluationOutput], true);
  clearLastWitness();
  evaluateButton.disabled = true;
  evaluateButton.textContent = "Evaluating…";

  try {
    const response = await evaluateExpression(
      { expression, bindings: bindingsResult.value, ...(options === undefined ? {} : { options }) },
      controller.signal,
    );
    if (requestId !== evaluateRequestId) return;
    if (response.status === "inconclusive") {
      const message = document.createElement("p");
      message.className = "empty";
      message.textContent = response.reason === "evaluation_budget_exhausted"
        ? "Inconclusive: the evaluation work limit was reached."
        : "Inconclusive: the search could not certify a result. This does not prove that no witness exists.";
      evaluationOutput.replaceChildren(message);
      clearLastWitness();
    } else {
      renderScalarOrJson(evaluationOutput, response.value);
      showLastWitness(response.last_witness);
    }
  } catch (error) {
    if (requestId !== evaluateRequestId || isAbortError(error)) return;
    setErrorText(evaluationError, resolveErrorMessage(error));
  } finally {
    if (requestId === evaluateRequestId) {
      setStale([evaluationOutput], false);
      evaluateButton.textContent = "Evaluate";
      updateControlAvailability();
    }
  }
}

evaluateButton.addEventListener("click", () => void handleEvaluate());

// --- Test dataset samples (Model tab) ---------------------------------------

function syncSampleDatasetPanel(): void {
  testDatasetPanel.hidden = modelState.kind !== "ready";
}

function clearSampleResult(): void {
  currentSampleJson = null;
  currentFullResponseJson = null;
  sampleCopy.reset();
  fullResponseCopy.reset();
  setErrorText(sampleCopyError, null);
  setErrorText(sampleCopyFullError, null);
  setErrorText(sampleError, null);
  sampleResult.hidden = true;
  sampleEmpty.hidden = false;
}

function resetSampleOperation(): void {
  sampleRequestId += 1;
  sampleAbortController?.abort();
  sampleAbortController = null;
  sampleOperationInFlight = false;
  sampleLoading.hidden = true;
  setStale([sampleResult], false);
}

function makeCopyHandler(
  getText: () => string | null,
  feedbackEl: HTMLElement,
  errorEl: HTMLParagraphElement,
): { copy: () => Promise<void>; reset: () => void } {
  let timeout: number | undefined;

  function reset(): void {
    if (timeout !== undefined) {
      window.clearTimeout(timeout);
      timeout = undefined;
    }
    feedbackEl.hidden = true;
    feedbackEl.textContent = "";
  }

  async function copy(): Promise<void> {
    const text = getText();
    if (text === null) return;
    setErrorText(errorEl, null);
    try {
      await navigator.clipboard.writeText(text);
      reset();
      feedbackEl.hidden = false;
      feedbackEl.textContent = "Copied";
      timeout = window.setTimeout(() => {
        feedbackEl.hidden = true;
        feedbackEl.textContent = "";
        timeout = undefined;
      }, COPY_FEEDBACK_MS);
    } catch {
      setErrorText(
        errorEl,
        "Could not copy automatically. Select the text above and copy it manually.",
      );
    }
  }

  return { copy, reset };
}

function renderSampleResult(response: DatasetSampleResponse): void {
  sampleEmpty.hidden = true;
  sampleResult.hidden = false;
  sampleIndexLine.textContent = `Index: ${response.index} of ${response.dataset_size}`;
  sampleJson.textContent = JSON.stringify(response.sample, null, 2);
  sampleExpectedJson.textContent = JSON.stringify(response.expected, null, 2);

  currentSampleJson = JSON.stringify(response.sample);
  currentFullResponseJson = JSON.stringify(response);
  sampleCopy.reset();
  fullResponseCopy.reset();
  setErrorText(sampleCopyError, null);
  setErrorText(sampleCopyFullError, null);
}

async function runSampleRequest(
  fetchSample: (signal: AbortSignal) => Promise<DatasetSampleResponse>,
): Promise<void> {
  const requestId = ++sampleRequestId;
  sampleAbortController?.abort();
  const controller = new AbortController();
  sampleAbortController = controller;

  setErrorText(sampleError, null);
  sampleOperationInFlight = true;
  sampleLoading.hidden = false;
  setStale([sampleResult], true);
  updateControlAvailability();

  try {
    const response = await fetchSample(controller.signal);
    if (requestId !== sampleRequestId) return;
    renderSampleResult(response);
  } catch (error) {
    if (requestId !== sampleRequestId || isAbortError(error)) return;
    setErrorText(sampleError, resolveErrorMessage(error));
  } finally {
    if (requestId === sampleRequestId) {
      sampleOperationInFlight = false;
      sampleLoading.hidden = true;
      setStale([sampleResult], false);
      updateControlAvailability();
    }
  }
}

async function handleGetSample(): Promise<void> {
  if (modelState.kind !== "ready") return;
  const indexResult = parseNonNegativeInteger(
    sampleIndexInput.value,
    "Sample index",
  );
  if (!indexResult.ok) {
    setErrorText(sampleError, indexResult.message);
    return;
  }

  await runSampleRequest((signal) => getTestSample(indexResult.value, signal));
}

async function handleRandomSample(): Promise<void> {
  if (modelState.kind !== "ready") return;
  const seedResult = parseIntegerSeed(sampleSeedInput.value);
  if (!seedResult.ok) {
    setErrorText(sampleError, seedResult.message);
    return;
  }

  await runSampleRequest((signal) =>
    getRandomTestSample(seedResult.value, signal),
  );
}

sampleGetButton.addEventListener("click", () => void handleGetSample());
sampleRandomButton.addEventListener("click", () => void handleRandomSample());
sampleCopyButton.addEventListener("click", () => void sampleCopy.copy());
sampleCopyFullButton.addEventListener("click", () => void fullResponseCopy.copy());

// --- Evaluation settings ------------------------------------------------
const EVALUATION_OPTIONS_STORAGE_KEY = "qfoil.evaluation-options";

function updateEvaluationSettings(): void {
  const exact = evaluationMode.value === "exact";
  requireElement<HTMLDivElement>("#heuristic-settings").hidden = exact;
  evaluationBudget.disabled = exact;
  evaluationRngSeed.disabled = exact;
  evaluationSeedSource.disabled = exact;
  evaluationInitialTuple.disabled = exact || evaluationSeedSource.value !== "tuple";
  requireElement<HTMLDivElement>("#evaluation-initial-tuple-field").hidden =
    evaluationSeedSource.value !== "tuple";
  requireElement<HTMLParagraphElement>("#evaluation-mode-note").textContent = exact
    ? "Enumerates candidates compatible with required tuple and size conditions. Evaluation may remain inconclusive if the work limit is reached."
    : "Searches from random starts and verifies candidates. An inconclusive result does not prove absence.";
  requireElement<HTMLParagraphElement>("#evaluation-settings-summary").textContent =
    evaluationConfig === null ? "Heuristic mode · server defaults"
    : exact ? "Exhaustive mode" : `Heuristic mode · search budget ${evaluationBudget.value}`;
}

function readEvaluationOptions(): EvaluationOptions | undefined | null {
  if (evaluationConfig === null) return undefined;
  function invalid(message: string): null {
    setErrorText(evaluationSettingsError, message);
    setErrorText(evaluationError, "Check the options in Evaluation settings.");
    setActiveTab("settings");
    return null;
  }
  const workBudget = parseIntegerInRange(evaluationWorkBudget.value, "Total evaluation budget", 0,
    evaluationConfig.evaluation.max_evaluation_budget);
  if (!workBudget.ok) return invalid(workBudget.message);
  if (evaluationMode.value === "exact") return { mode: "exact", evaluation_budget: workBudget.value };
  const budget = parseIntegerInRange(evaluationBudget.value, "Search budget", 0,
    evaluationConfig.evaluation.max_sigma_search_budget);
  if (!budget.ok) return invalid(budget.message);
  const rng = parseIntegerSeed(evaluationRngSeed.value);
  if (!rng.ok) return invalid(rng.message);
  if (rng.value !== null && !Number.isSafeInteger(rng.value)) {
    return invalid("Random seed must be a safe integer.");
  }
  let seed: Witness | null = null;
  if (evaluationSeedSource.value === "tuple") {
    const parsed = parseSeed(evaluationInitialTuple.value);
    if (!parsed.ok) return invalid(parsed.message);
    if (parsed.value === null || !parsed.value.every(v => v === null || v === 0 || v === 1)) {
      return invalid("Initial tuple must be a JSON array of 0, 1 or null.");
    }
    if (modelState.kind === "ready" && parsed.value.length !== modelState.num_features) {
      return invalid(`Initial tuple must contain ${modelState.num_features} entries.`);
    }
    seed = parsed.value as Witness;
  }
  return { mode: "heuristic", evaluation_budget: workBudget.value, sigma_search_budget: budget.value, rng_seed: rng.value, seed };
}

function persistEvaluationOptions(): void {
  if (evaluationConfig === null) return;
  try {
    // Initial tuples belong to the current expression/model, not saved settings.
    sessionStorage.setItem(EVALUATION_OPTIONS_STORAGE_KEY, JSON.stringify({
      mode: evaluationMode.value,
      sigma_search_budget: evaluationBudget.value,
      evaluation_budget: evaluationWorkBudget.value,
      rng_seed: evaluationRngSeed.value,
    }));
  } catch { /* Session storage is optional. */ }
}

function applyEvaluationDefaults(): void {
  if (evaluationConfig === null) return;
  const defaults = evaluationConfig.evaluation.defaults;
  evaluationMode.value = defaults.mode;
  evaluationBudget.value = String(defaults.sigma_search_budget);
  evaluationWorkBudget.value = String(defaults.evaluation_budget ?? evaluationConfig.evaluation_budget);
  evaluationRngSeed.value = defaults.rng_seed === null || defaults.rng_seed === undefined ? "" : String(defaults.rng_seed);
  evaluationSeedSource.value = "random";
  evaluationInitialTuple.value = "";
  updateEvaluationSettings();
}

async function loadEvaluationConfig(): Promise<void> {
  try {
    const config = await getConfig();
    if (!config.evaluation || !config.evaluation.modes.includes("heuristic") ||
      !config.evaluation.modes.includes("exact") ||
      !Number.isInteger(config.evaluation.max_evaluation_budget) ||
      !Number.isInteger(config.evaluation_budget) ||
      !Number.isInteger(config.evaluation.max_sigma_search_budget) ||
      !Number.isInteger(config.evaluation.defaults.sigma_search_budget)) {
      throw new Error("Evaluation options are unavailable.");
    }
    evaluationConfig = config;
    applyEvaluationDefaults();
    try {
      const raw = sessionStorage.getItem(EVALUATION_OPTIONS_STORAGE_KEY);
      const stored = raw === null ? null : JSON.parse(raw);
      if (stored !== null) {
        if (stored.mode === "heuristic" || stored.mode === "exact") evaluationMode.value = stored.mode;
        if (typeof stored.sigma_search_budget === "string") {
          const parsed = parseIntegerInRange(stored.sigma_search_budget, "Search budget", 0, config.evaluation.max_sigma_search_budget);
          if (parsed.ok) evaluationBudget.value = String(parsed.value);
        }
        if (typeof stored.evaluation_budget === "string") {
          const parsed = parseIntegerInRange(stored.evaluation_budget, "Total evaluation budget", 0, config.evaluation.max_evaluation_budget);
          if (parsed.ok) evaluationWorkBudget.value = String(parsed.value);
        }
        if (typeof stored.rng_seed === "string") {
          const parsed = parseIntegerSeed(stored.rng_seed);
          if (parsed.ok && (parsed.value === null || Number.isSafeInteger(parsed.value))) evaluationRngSeed.value = stored.rng_seed;
        }
      }
    } catch { /* Ignore invalid or unavailable session storage. */ }
    requireElement<HTMLParagraphElement>("#evaluation-budget-note").textContent =
      `From 0 to ${config.evaluation.max_sigma_search_budget.toLocaleString()}. Shared across starts and restarts within each attempt.`;
    requireElement<HTMLParagraphElement>("#evaluation-global-budget-note").textContent =
      `From 0 to ${config.evaluation.max_evaluation_budget.toLocaleString()}. Shared by all nested queries in either mode; reaching this budget returns inconclusive.`;
    evaluationOptionsSection.hidden = false;
    evaluationRetryConfig.hidden = true;
    setErrorText(evaluationConfigError, null);
  } catch {
    evaluationConfig = null;
    evaluationOptionsSection.hidden = true;
    evaluationRetryConfig.hidden = false;
    setErrorText(evaluationConfigError, "Could not load evaluation settings. Evaluation uses server defaults until settings are available.");
  }
  updateEvaluationSettings();
}

evaluationRestoreDefaults.addEventListener("click", () => {
  applyEvaluationDefaults();
  persistEvaluationOptions();
  setErrorText(evaluationSettingsError, null);
  invalidateInputResults(false);
  clearLastWitness();
});
evaluationRetryConfig.addEventListener("click", () => void loadEvaluationConfig());
for (const control of [evaluationMode, evaluationWorkBudget, evaluationBudget, evaluationRngSeed, evaluationSeedSource, evaluationInitialTuple]) {
  control.addEventListener("input", () => {
    updateEvaluationSettings();
    persistEvaluationOptions();
    setErrorText(evaluationSettingsError, null);
    invalidateInputResults(false);
    clearLastWitness();
  });
}

// History selection emits input, just like editing the field by hand.
// Invalidate in-flight responses so they cannot appear to describe new inputs.
function invalidateInputResults(expressionChanged: boolean): void {
  if (expressionChanged) {
    ++latexRequestId;
    latexAbortController?.abort();
    setStale([formulaOutput, latexSourceWrap], true);
    setErrorText(formulaError, null);
    renderButton.textContent = "Render formula";
  }
  ++evaluateRequestId;
  evaluateAbortController?.abort();
  setStale([evaluationOutput], true);
  setErrorText(evaluationError, null);
  evaluateButton.textContent = "Evaluate";
  updateControlAvailability();
}

expressionInput.addEventListener("input", () => invalidateInputResults(true));
bindingsInput.addEventListener("input", () => invalidateInputResults(false));
