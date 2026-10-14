import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mountHistory } from "../history";

function mount(id = "expression-input", seeds = ["EP(x, k)"]) {
  const input = document.createElement("textarea");
  input.id = id;
  document.body.append(input);
  const history = mountHistory(input, seeds);
  const details = input.nextElementSibling!;
  return { input, history, select: details.querySelector("select")!, buttons: details.querySelectorAll("button") };
}

describe("independent input histories", () => {
  beforeEach(() => {
    localStorage.clear();
    document.body.replaceChildren();
  });
  afterEach(() => vi.restoreAllMocks());

  it("seeds once, persists exact text, deduplicates and bounds entries", () => {
    const { history } = mount();
    for (let i = 0; i < 25; i++) history.add(`SIZE(x) <= ${i}`);
    history.add("SIZE(x) <= 24");
    const reloaded = mount();
    expect(reloaded.select.options).toHaveLength(21);
    expect(reloaded.select.options[1].text).toBe("SIZE(x) <= 24");
    expect(JSON.parse(localStorage.getItem("qfoil.history.expression-input")!)).toHaveLength(20);
  });

  it("selects and copies only its own field, without executing a request", async () => {
    const expression = mount();
    const bindings = mount("bindings-input", ['{\n  "k": 1\n}']);
    const onInput = vi.fn();
    expression.input.addEventListener("input", onInput);
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    expression.select.value = "0";
    expression.select.dispatchEvent(new Event("change"));
    expect(expression.input.value).toBe("EP(x, k)");
    expect(bindings.input.value).toBe("");
    expect(onInput).toHaveBeenCalledOnce();
    expression.buttons[0].click();
    await Promise.resolve();
    expect(writeText).toHaveBeenCalledWith("EP(x, k)");
    expect(expression.input.nextElementSibling!.textContent).toContain("Copied");
  });

  it("keeps cleared histories empty after reload and leaves the other history intact", () => {
    const expression = mount();
    mount("bindings-input", ['{"k": 1}']);
    expression.buttons[1].click();
    expect(mount().select.options).toHaveLength(1);
    expect(mount("bindings-input").select.options[1].text).toBe('{"k": 1}');
  });

  it("recovers from malformed storage and continues when storage is unavailable", () => {
    localStorage.setItem("qfoil.history.expression-input", "{}");
    expect(mount().select.options[1].text).toBe("EP(x, k)");
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("unavailable"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("unavailable"); });
    const { history, select } = mount();
    history.add("FULL(x)");
    expect(select.options[1].text).toBe("FULL(x)");
  });

  it("reports clipboard failure without losing the selected text", async () => {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined });
    const { input, select, buttons } = mount();
    select.value = "0";
    select.dispatchEvent(new Event("change"));
    buttons[0].click();
    await Promise.resolve();
    expect(input.value).toBe("EP(x, k)");
    expect(input.nextElementSibling!.textContent).toContain("Could not copy");
  });
});
