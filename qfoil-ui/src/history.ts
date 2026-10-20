const HISTORY_LIMIT = 20;

export const INITIAL_EXPRESSIONS = [
  "FULL(x) * (y <<= x) * (EP(y, k) >= delta)",
  "FULL(x) * (y <<= x) * (CEP(y, k) >= delta)",
  "SIGMA y. ( (y <<= x) * (SIGMA z. (COMP(y,z) * (VAL(z) == 0)) == 0) * (SIZE(y) <= 7)) >= 1",
  "VAL(x)",
  "VAL(x, k)",
  "EP(x, k)",
  "CEP(x, k)",
  "COND(event, given, k)",
];

export const INITIAL_BINDINGS = [
  JSON.stringify({ x: [0, 1, 0, 1, 1, 0, 0, 1], y: [0, 1, null, null, 1, null, null, null], k: 1, delta: 0.9 }, null, 2),
  JSON.stringify({ x: [0, 0, 1, 0, 0, 0, 0, 1], y: [0, 0, null, 0, 0, 0, 0, 1], k: 1, delta: 0.8 }, null, 2),
  JSON.stringify({ x: [0, 1, null, null, 1, null, null, null], k: 1 }, null, 2),
  JSON.stringify({ event: [null, null, null, null, 1, null, null, null], given: [0, 1, null, null, null, null, null, null], k: 1 }, null, 2),
];

export function mountHistory(input: HTMLTextAreaElement, seeds: string[]) {
  const key = `qfoil.history.${input.id}`;
  let entries = [...seeds];
  try {
    const raw = localStorage.getItem(key);
    if (raw !== null) {
      const parsed: unknown = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.every((item) => typeof item === "string")) {
        // Upgrade saved expressions to the current language syntax.
        const saved = input.id === "expression-input"
          ? parsed.map((item) => item.replaceAll("<<<", "<<="))
          : parsed;
        entries = [...new Set(saved.filter((item) => item.trim() !== ""))].slice(0, HISTORY_LIMIT);
      }
    }
  } catch {
    // Storage unavailable or damaged: keep a usable in-memory history.
  }

  const details = document.createElement("details");
  details.className = "input-history";
  const label = input.id === "expression-input" ? "Expression" : "Bindings";
  details.innerHTML = `
    <summary>History</summary>
    <div class="history-controls">
      <select aria-label="${label} history"></select>
      <button type="button" class="secondary-button">Copy</button>
      <button type="button" class="secondary-button">Clear</button>
    </div>
    <pre class="mono history-preview" hidden></pre>
    <p class="model-panel-note" role="status" aria-live="polite"></p>
  `;
  input.after(details);
  const select = details.querySelector("select")!;
  const [copy, clear] = details.querySelectorAll("button");
  const preview = details.querySelector("pre")!;
  const status = details.querySelector("p")!;
  let selected: string | null = null;
  let copyRequest = 0;

  function persist() {
    try {
      localStorage.setItem(key, JSON.stringify(entries));
    } catch {
      // Persistence is optional; the current session remains usable.
    }
  }

  function render() {
    select.replaceChildren(new Option(entries.length ? "Select to use…" : "No history yet", ""));
    entries.forEach((entry, index) => {
      const option = new Option(entry.replace(/\s+/g, " ").slice(0, 100), String(index));
      option.title = entry;
      select.add(option);
    });
    selected = null;
    copyRequest++;
    copy.disabled = true;
    clear.disabled = entries.length === 0;
    preview.hidden = true;
    preview.textContent = "";
    status.textContent = "";
  }

  select.addEventListener("change", () => {
    copyRequest++;
    selected = select.value === "" ? null : entries[Number(select.value)] ?? null;
    copy.disabled = selected === null;
    preview.hidden = selected === null;
    preview.textContent = selected ?? "";
    status.textContent = "";
    if (selected !== null) {
      input.value = selected;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.focus();
    }
  });

  copy.addEventListener("click", async () => {
    if (selected === null) return;
    const request = ++copyRequest;
    try {
      await navigator.clipboard.writeText(selected);
      if (request === copyRequest) status.textContent = "Copied";
    } catch {
      if (request === copyRequest) status.textContent = "Could not copy. Select the text in the field and copy it manually.";
    }
  });
  clear.addEventListener("click", () => {
    entries = [];
    persist();
    render();
  });
  persist();
  render();

  return {
    add(text: string) {
      if (text.trim() === "") return;
      entries = [text, ...entries.filter((entry) => entry !== text)].slice(0, HISTORY_LIMIT);
      persist();
      render();
    },
  };
}
