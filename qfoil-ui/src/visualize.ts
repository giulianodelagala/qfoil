// Experimental tuple visualization tab. Self-contained on purpose: this
// module builds its own tab button, panel, and styles (via ./visualize.css)
// at runtime instead of relying on static markup in index.html or shared
// entries in validation.ts/render.ts. To remove the feature entirely, delete
// this file, visualize.css, and the two call sites in main.ts (the import
// and the mountVisualizeTab call).
import "./visualize.css";

import { setErrorText } from "./render";

export interface MountedTab {
  id: string;
  button: HTMLButtonElement;
  panel: HTMLDivElement;
}

type TupleValue = 0 | 1 | null;

interface ColorConfig {
  zero: string;
  one: string;
  null: string;
  alpha: number;
}

interface RenderState {
  pixelSize: number;
  gap: number;
  columns: number;
  showOriginal: boolean;
  showExplanation: boolean;
  originalColors: ColorConfig;
  explanationColors: ColorConfig;
}

const TAB_ID = "visualize";

const DEFAULT_ORIGINAL_TUPLE = "[1, null, 0]";
const DEFAULT_EXPLANATION_TUPLE = "[1, null, null]";
const DEFAULT_PIXEL_SIZE = 24;
const DEFAULT_GAP = 4;
const DEFAULT_COLUMNS = 8;
const DEFAULT_ORIGINAL_COLORS: ColorConfig = {
  zero: "#cbd5e1",
  one: "#2c5cc5",
  null: "#ffffff",
  alpha: 1,
};
const DEFAULT_EXPLANATION_COLORS: ColorConfig = {
  zero: "#f2994a",
  one: "#1b7f4d",
  null: "#ffffff",
  alpha: 0.6,
};

type ParseResult =
  | { ok: true; value: TupleValue[] }
  | { ok: false; message: string };

function parseTuple(raw: string): ParseResult {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return { ok: false, message: "Enter a tuple as a JSON array." };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : "invalid JSON";
    return { ok: false, message: `Tuple must be valid JSON: ${detail}` };
  }

  if (!Array.isArray(parsed)) {
    return { ok: false, message: "Tuple must be a JSON array." };
  }
  if (!parsed.every((item) => item === 0 || item === 1 || item === null)) {
    return { ok: false, message: "Tuple values must be 0, 1, or null." };
  }

  return { ok: true, value: parsed as TupleValue[] };
}

function hexToRgba(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "");
  const r = parseInt(normalized.slice(0, 2), 16);
  const g = parseInt(normalized.slice(2, 4), 16);
  const b = parseInt(normalized.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function colorForValue(value: TupleValue, colors: ColorConfig): string {
  const hex = value === 0 ? colors.zero : value === 1 ? colors.one : colors.null;
  return hexToRgba(hex, colors.alpha);
}

function createField(labelText: string, input: HTMLElement): HTMLDivElement {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  label.textContent = labelText;
  if (input.id) {
    label.htmlFor = input.id;
  }
  wrapper.append(label, input);
  return wrapper;
}

function createTextarea(id: string, defaultValue: string): HTMLTextAreaElement {
  const textarea = document.createElement("textarea");
  textarea.id = id;
  textarea.className = "mono";
  textarea.rows = 4;
  textarea.spellcheck = false;
  textarea.value = defaultValue;
  return textarea;
}

function createErrorText(): HTMLParagraphElement {
  const p = document.createElement("p");
  p.className = "error-text";
  p.setAttribute("role", "alert");
  p.hidden = true;
  return p;
}

function createCheckboxField(
  labelText: string,
  id: string,
  checked: boolean,
): { wrapper: HTMLDivElement; input: HTMLInputElement } {
  const input = document.createElement("input");
  input.type = "checkbox";
  input.id = id;
  input.checked = checked;

  const label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = labelText;

  const wrapper = document.createElement("div");
  wrapper.className = "visualize-checkbox-row";
  wrapper.append(input, label);
  return { wrapper, input };
}

function createNumberField(
  labelText: string,
  id: string,
  defaultValue: number,
  min: number,
): { wrapper: HTMLDivElement; input: HTMLInputElement } {
  const input = document.createElement("input");
  input.type = "number";
  input.id = id;
  input.min = String(min);
  input.value = String(defaultValue);
  return { wrapper: createField(labelText, input), input };
}

function createColorField(
  labelText: string,
  id: string,
  defaultHex: string,
): { wrapper: HTMLDivElement; input: HTMLInputElement } {
  const input = document.createElement("input");
  input.type = "color";
  input.id = id;
  input.value = defaultHex;

  const label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = labelText;

  const wrapper = document.createElement("div");
  wrapper.className = "visualize-color-row";
  wrapper.append(label, input);
  return { wrapper, input };
}

function createAlphaField(
  id: string,
  defaultValue: number,
): { wrapper: HTMLDivElement; input: HTMLInputElement; output: HTMLSpanElement } {
  const input = document.createElement("input");
  input.type = "range";
  input.id = id;
  input.min = "0";
  input.max = "1";
  input.step = "0.05";
  input.value = String(defaultValue);

  const label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = "Alpha";

  const output = document.createElement("span");
  output.className = "mono";
  output.textContent = defaultValue.toFixed(2);

  const wrapper = document.createElement("div");
  wrapper.className = "visualize-alpha-row";
  wrapper.append(label, input, output);
  return { wrapper, input, output };
}

interface ColorCard {
  card: HTMLElement;
  zeroInput: HTMLInputElement;
  oneInput: HTMLInputElement;
  nullInput: HTMLInputElement;
  alphaInput: HTMLInputElement;
  alphaOutput: HTMLSpanElement;
}

function buildColorCard(title: string, idPrefix: string, colors: ColorConfig): ColorCard {
  const card = document.createElement("section");
  card.className = "card";

  const heading = document.createElement("h2");
  heading.textContent = title;

  const zero = createColorField("Value 0", `${idPrefix}-color-0`, colors.zero);
  const one = createColorField("Value 1", `${idPrefix}-color-1`, colors.one);
  const nullField = createColorField("Value null", `${idPrefix}-color-null`, colors.null);
  const alpha = createAlphaField(`${idPrefix}-alpha`, colors.alpha);

  card.append(heading, zero.wrapper, one.wrapper, nullField.wrapper, alpha.wrapper);

  return {
    card,
    zeroInput: zero.input,
    oneInput: one.input,
    nullInput: nullField.input,
    alphaInput: alpha.input,
    alphaOutput: alpha.output,
  };
}

function renderLegend(legendEl: HTMLDivElement, state: RenderState): void {
  legendEl.replaceChildren();
  const entries: { label: string; color: string }[] = [];

  if (state.showOriginal) {
    entries.push(
      { label: "0 (original)", color: state.originalColors.zero },
      { label: "1 (original)", color: state.originalColors.one },
      { label: "null (original) ○", color: state.originalColors.null },
    );
  }
  if (state.showExplanation) {
    entries.push(
      { label: "0 (explanation)", color: state.explanationColors.zero },
      { label: "1 (explanation)", color: state.explanationColors.one },
      { label: "null (explanation) ○", color: state.explanationColors.null },
    );
  }

  for (const entry of entries) {
    const item = document.createElement("span");
    item.className = "visualize-legend-item";

    const swatch = document.createElement("span");
    swatch.className = "visualize-legend-swatch";
    swatch.style.background = entry.color;
    swatch.setAttribute("aria-hidden", "true");

    const text = document.createElement("span");
    text.textContent = entry.label;

    item.append(swatch, text);
    legendEl.appendChild(item);
  }
}

function renderGrid(
  gridEl: HTMLDivElement,
  legendEl: HTMLDivElement,
  original: TupleValue[],
  explanation: TupleValue[],
  state: RenderState,
): void {
  gridEl.replaceChildren();

  const length = original.length;
  gridEl.style.gridTemplateColumns = `repeat(${state.columns}, ${state.pixelSize}px)`;
  gridEl.style.gap = `${state.gap}px`;
  gridEl.setAttribute(
    "aria-label",
    `Tuple view, ${length} features` +
      (state.showOriginal ? ", showing original" : "") +
      (state.showExplanation ? ", showing explanation" : ""),
  );

  for (let i = 0; i < length; i++) {
    const cell = document.createElement("div");
    cell.className = "visualize-cell";
    cell.style.width = `${state.pixelSize}px`;
    cell.style.height = `${state.pixelSize}px`;

    const originalValue = original[i];
    const explanationValue = explanation[i];
    const titleParts: string[] = [];

    if (state.showOriginal) {
      const layer = document.createElement("div");
      layer.className = "visualize-cell-layer";
      layer.style.background = colorForValue(originalValue, state.originalColors);
      cell.appendChild(layer);
      if (originalValue === null) {
        const mark = document.createElement("span");
        mark.className = "visualize-null-mark visualize-null-mark--original";
        mark.textContent = "○";
        cell.appendChild(mark);
      }
      titleParts.push(`original=${originalValue === null ? "null" : originalValue}`);
    }

    if (state.showExplanation) {
      const layer = document.createElement("div");
      layer.className = "visualize-cell-layer";
      layer.style.background = colorForValue(explanationValue, state.explanationColors);
      cell.appendChild(layer);
      if (explanationValue === null) {
        const mark = document.createElement("span");
        mark.className = "visualize-null-mark visualize-null-mark--explanation";
        mark.textContent = "○";
        cell.appendChild(mark);
      }
      titleParts.push(`explanation=${explanationValue === null ? "null" : explanationValue}`);
    }

    cell.title = `Feature ${i}: ${titleParts.length > 0 ? titleParts.join(", ") : "hidden"}`;
    gridEl.appendChild(cell);
  }

  renderLegend(legendEl, state);
}

function buildPanelBody(panel: HTMLDivElement): void {
  const introSection = document.createElement("section");
  introSection.className = "card";
  introSection.setAttribute("aria-labelledby", "visualize-heading");

  const heading = document.createElement("h2");
  heading.id = "visualize-heading";
  heading.textContent = "Visualize";

  const intro = document.createElement("p");
  intro.className = "model-panel-note";
  intro.textContent =
    "Experimental: compare an original tuple and an explanation tuple as " +
    "overlaid colored grids. Values must be 0, 1, or null.";

  introSection.append(heading, intro);

  const instancesRow = document.createElement("div");
  instancesRow.className = "columns";

  const originalField = createField(
    "Original instance",
    createTextarea("visualize-original-input", DEFAULT_ORIGINAL_TUPLE),
  );
  const originalInput = originalField.querySelector("textarea") as HTMLTextAreaElement;
  const originalError = createErrorText();
  const { wrapper: originalCheckboxRow, input: showOriginalInput } = createCheckboxField(
    "Show original",
    "visualize-show-original",
    true,
  );
  originalField.append(originalError, originalCheckboxRow);

  const explanationField = createField(
    "Explanation instance",
    createTextarea("visualize-explanation-input", DEFAULT_EXPLANATION_TUPLE),
  );
  const explanationInput = explanationField.querySelector("textarea") as HTMLTextAreaElement;
  const explanationError = createErrorText();
  const { wrapper: explanationCheckboxRow, input: showExplanationInput } = createCheckboxField(
    "Show explanation",
    "visualize-show-explanation",
    true,
  );
  explanationField.append(explanationError, explanationCheckboxRow);

  instancesRow.append(originalField, explanationField);

  const settingsCard = document.createElement("section");
  settingsCard.className = "card";
  settingsCard.setAttribute("aria-labelledby", "visualize-settings-heading");

  const settingsHeading = document.createElement("h2");
  settingsHeading.id = "visualize-settings-heading";
  settingsHeading.textContent = "Display settings";

  const settingsRow = document.createElement("div");
  settingsRow.className = "visualize-settings-row";
  const { wrapper: pixelSizeWrapper, input: pixelSizeInput } = createNumberField(
    "Pixel size (px)",
    "visualize-pixel-size",
    DEFAULT_PIXEL_SIZE,
    4,
  );
  const { wrapper: gapWrapper, input: gapInput } = createNumberField(
    "Gap (px)",
    "visualize-gap",
    DEFAULT_GAP,
    0,
  );
  const { wrapper: columnsWrapper, input: columnsInput } = createNumberField(
    "Columns",
    "visualize-columns",
    DEFAULT_COLUMNS,
    1,
  );
  settingsRow.append(pixelSizeWrapper, gapWrapper, columnsWrapper);
  settingsCard.append(settingsHeading, settingsRow);

  const colorsRow = document.createElement("div");
  colorsRow.className = "columns";
  const originalColorCard = buildColorCard(
    "Original colors",
    "visualize-original",
    DEFAULT_ORIGINAL_COLORS,
  );
  const explanationColorCard = buildColorCard(
    "Explanation colors",
    "visualize-explanation",
    DEFAULT_EXPLANATION_COLORS,
  );
  colorsRow.append(originalColorCard.card, explanationColorCard.card);

  const viewCard = document.createElement("section");
  viewCard.className = "card";
  viewCard.setAttribute("aria-labelledby", "visualize-view-heading");

  const viewHeading = document.createElement("h2");
  viewHeading.id = "visualize-view-heading";
  viewHeading.textContent = "Tuple view";

  const viewEmpty = document.createElement("p");
  viewEmpty.className = "empty";
  viewEmpty.textContent = "No tuple rendered yet";

  const viewError = createErrorText();

  const gridWrap = document.createElement("div");
  gridWrap.className = "visualize-grid-wrap";
  gridWrap.hidden = true;
  const grid = document.createElement("div");
  grid.className = "visualize-grid";
  const legend = document.createElement("div");
  legend.className = "visualize-legend";
  gridWrap.append(grid, legend);

  viewCard.append(viewHeading, viewEmpty, viewError, gridWrap);

  panel.append(introSection, instancesRow, settingsCard, colorsRow, viewCard);

  function render(): void {
    const showOriginal = showOriginalInput.checked;
    const showExplanation = showExplanationInput.checked;

    const originalResult = parseTuple(originalInput.value);
    setErrorText(originalError, originalResult.ok ? null : originalResult.message);

    const explanationResult = parseTuple(explanationInput.value);
    setErrorText(explanationError, explanationResult.ok ? null : explanationResult.message);

    if (!originalResult.ok || !explanationResult.ok) {
      setErrorText(viewError, null);
      gridWrap.hidden = true;
      viewEmpty.hidden = false;
      viewEmpty.textContent = "No tuple rendered yet";
      return;
    }

    if (originalResult.value.length !== explanationResult.value.length) {
      setErrorText(viewError, "Original and explanation must have the same length.");
      gridWrap.hidden = true;
      viewEmpty.hidden = true;
      return;
    }
    setErrorText(viewError, null);

    if (!showOriginal && !showExplanation) {
      gridWrap.hidden = true;
      viewEmpty.hidden = false;
      viewEmpty.textContent = "Both tuples are hidden.";
      return;
    }

    viewEmpty.hidden = true;
    gridWrap.hidden = false;

    const state: RenderState = {
      pixelSize: Math.max(1, Number(pixelSizeInput.value) || DEFAULT_PIXEL_SIZE),
      gap: Math.max(0, Number(gapInput.value) || 0),
      columns: Math.max(1, Math.floor(Number(columnsInput.value)) || 1),
      showOriginal,
      showExplanation,
      originalColors: {
        zero: originalColorCard.zeroInput.value,
        one: originalColorCard.oneInput.value,
        null: originalColorCard.nullInput.value,
        alpha: Number(originalColorCard.alphaInput.value),
      },
      explanationColors: {
        zero: explanationColorCard.zeroInput.value,
        one: explanationColorCard.oneInput.value,
        null: explanationColorCard.nullInput.value,
        alpha: Number(explanationColorCard.alphaInput.value),
      },
    };

    originalColorCard.alphaOutput.textContent = state.originalColors.alpha.toFixed(2);
    explanationColorCard.alphaOutput.textContent = state.explanationColors.alpha.toFixed(2);

    renderGrid(grid, legend, originalResult.value, explanationResult.value, state);
  }

  const watchedInputs: HTMLElement[] = [
    originalInput,
    explanationInput,
    showOriginalInput,
    showExplanationInput,
    pixelSizeInput,
    gapInput,
    columnsInput,
    originalColorCard.zeroInput,
    originalColorCard.oneInput,
    originalColorCard.nullInput,
    originalColorCard.alphaInput,
    explanationColorCard.zeroInput,
    explanationColorCard.oneInput,
    explanationColorCard.nullInput,
    explanationColorCard.alphaInput,
  ];
  for (const input of watchedInputs) {
    input.addEventListener("input", render);
  }

  render();
}

export function mountVisualizeTab(
  tablist: HTMLElement,
  panelContainer: HTMLElement,
): MountedTab {
  const button = document.createElement("button");
  button.id = "tab-visualize";
  button.type = "button";
  button.className = "tab";
  button.setAttribute("role", "tab");
  button.setAttribute("aria-selected", "false");
  button.setAttribute("aria-controls", "panel-visualize");
  button.tabIndex = -1;
  button.textContent = "Visualize";
  tablist.appendChild(button);

  const panel = document.createElement("div");
  panel.id = "panel-visualize";
  panel.className = "tabpanel";
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", "tab-visualize");
  panel.tabIndex = 0;
  panel.hidden = true;
  panelContainer.appendChild(panel);

  buildPanelBody(panel);

  return { id: TAB_ID, button, panel };
}
