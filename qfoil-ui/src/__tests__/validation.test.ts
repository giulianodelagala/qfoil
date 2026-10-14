import { describe, expect, it } from "vitest";

import { parseBindings, parseIntegerSeed, parseSeed } from "../validation";

describe("parseBindings", () => {
  it("accepts a valid JSON object", () => {
    const result = parseBindings('{ "x": [1, null, 0] }');
    expect(result).toEqual({ ok: true, value: { x: [1, null, 0] } });
  });

  it("rejects invalid JSON", () => {
    const result = parseBindings("{ x: 1 ");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.message).toMatch(/valid JSON/i);
    }
  });

  it("rejects a top-level JSON array", () => {
    const result = parseBindings("[1, 2, 3]");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.message).toMatch(/object/i);
    }
  });

  it("rejects a top-level JSON scalar", () => {
    const result = parseBindings("42");
    expect(result.ok).toBe(false);
  });

  it("treats an empty string as empty bindings", () => {
    const result = parseBindings("   ");
    expect(result).toEqual({ ok: true, value: {} });
  });
});

describe("parseSeed", () => {
  it("accepts an empty string as no seed", () => {
    expect(parseSeed("")).toEqual({ ok: true, value: null });
  });

  it("accepts a valid array of numbers and nulls", () => {
    expect(parseSeed("[1, null, 0]")).toEqual({
      ok: true,
      value: [1, null, 0],
    });
  });

  it("rejects invalid JSON", () => {
    const result = parseSeed("[1, ");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.message).toMatch(/valid JSON/i);
    }
  });

  it("rejects a seed that is not an array", () => {
    const result = parseSeed('{ "a": 1 }');
    expect(result.ok).toBe(false);
  });

  it("rejects an array containing non-numeric, non-null values", () => {
    const result = parseSeed('["a", 1]');
    expect(result.ok).toBe(false);
  });
});

describe("parseIntegerSeed", () => {
  it("accepts an empty string as no seed", () => {
    expect(parseIntegerSeed("")).toEqual({ ok: true, value: null });
  });

  it("accepts a positive base-10 integer", () => {
    expect(parseIntegerSeed("123")).toEqual({ ok: true, value: 123 });
  });

  it("accepts a negative base-10 integer", () => {
    expect(parseIntegerSeed("-7")).toEqual({ ok: true, value: -7 });
  });

  it("rejects a non-integer value", () => {
    const result = parseIntegerSeed("1.5");
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.message).toMatch(/base-10 integer/i);
    }
  });

  it("rejects a non-numeric value", () => {
    const result = parseIntegerSeed("abc");
    expect(result.ok).toBe(false);
  });
});
