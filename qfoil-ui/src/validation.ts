// Local, syntactic-only validation. No QFOIL syntax or semantics are checked
// here; that belongs to the server.

export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; message: string };

export function parseBindings(raw: string): ValidationResult<Record<string, unknown>> {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return { value: {}, ok: true };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : "invalid JSON";
    return { ok: false, message: `Bindings must be valid JSON: ${detail}` };
  }

  if (
    typeof parsed !== "object" ||
    parsed === null ||
    Array.isArray(parsed)
  ) {
    return { ok: false, message: "Bindings must be a JSON object." };
  }

  return { ok: true, value: parsed as Record<string, unknown> };
}

export function parseSeed(
  raw: string,
): ValidationResult<(number | null)[] | null> {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return { ok: true, value: null };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : "invalid JSON";
    return { ok: false, message: `Seed must be valid JSON: ${detail}` };
  }

  if (parsed === null) {
    return { ok: true, value: null };
  }

  if (
    !Array.isArray(parsed) ||
    !parsed.every((item) => item === null || typeof item === "number")
  ) {
    return { ok: false, message: "Seed must be a JSON array of numbers or null." };
  }

  return { ok: true, value: parsed as (number | null)[] };
}

export function parseNonNegativeInteger(
  raw: string,
  fieldName: string,
): ValidationResult<number> {
  const trimmed = raw.trim();
  if (trimmed === "" || !/^\d+$/.test(trimmed)) {
    return { ok: false, message: `${fieldName} must be a non-negative integer.` };
  }
  return { ok: true, value: Number(trimmed) };
}

// Like parseNonNegativeInteger, but also enforced against a server-reported
// [min, max] range (e.g. the witness search limits from GET /v1/config).
export function parseIntegerInRange(
  raw: string,
  fieldName: string,
  min: number,
  max: number,
): ValidationResult<number> {
  const trimmed = raw.trim();
  if (trimmed === "" || !/^\d+$/.test(trimmed)) {
    return { ok: false, message: `${fieldName} must be a non-negative integer.` };
  }
  const value = Number(trimmed);
  if (value < min || value > max) {
    return {
      ok: false,
      message: `${fieldName} must be between ${min} and ${max}.`,
    };
  }
  return { ok: true, value };
}

// Optional base-10 integer, as accepted by the test dataset random-sample
// endpoint's `seed` query parameter. Unlike parseNonNegativeInteger, a sign
// is allowed and an empty string means "no seed" rather than an error.
export function parseIntegerSeed(raw: string): ValidationResult<number | null> {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return { ok: true, value: null };
  }
  if (!/^-?\d+$/.test(trimmed)) {
    return { ok: false, message: "Seed must be a base-10 integer." };
  }
  return { ok: true, value: Number(trimmed) };
}
