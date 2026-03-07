import { describe, expect, it } from "vitest";
import { DEFAULT_MODEL_BY_PROVIDER, MODEL_OPTIONS_BY_PROVIDER } from "@t3tools/contracts";

import {
  getDefaultModel,
  getDefaultReasoningEffort,
  getModelOptions,
  getReasoningEffortOptions,
  normalizeModelSlug,
  resolveModelSlug,
} from "./model";

describe("normalizeModelSlug", () => {
  it("maps known aliases to canonical slugs", () => {
    expect(normalizeModelSlug("5.3")).toBe("gpt-5.3-codex");
    expect(normalizeModelSlug("gpt-5.3")).toBe("gpt-5.3-codex");
  });

  it("returns null for empty or missing values", () => {
    expect(normalizeModelSlug("")).toBeNull();
    expect(normalizeModelSlug("   ")).toBeNull();
    expect(normalizeModelSlug(null)).toBeNull();
    expect(normalizeModelSlug(undefined)).toBeNull();
  });

  it("preserves non-aliased model slugs", () => {
    expect(normalizeModelSlug("gpt-5.2")).toBe("gpt-5.2");
    expect(normalizeModelSlug("gpt-5.2-codex")).toBe("gpt-5.2-codex");
  });

  it("does not leak prototype properties as aliases", () => {
    expect(normalizeModelSlug("toString")).toBe("toString");
    expect(normalizeModelSlug("constructor")).toBe("constructor");
  });
});

describe("resolveModelSlug", () => {
  it("returns default only when the model is missing", () => {
    expect(resolveModelSlug(undefined)).toBe(DEFAULT_MODEL_BY_PROVIDER.codex);
    expect(resolveModelSlug(null)).toBe(DEFAULT_MODEL_BY_PROVIDER.codex);
  });

  it("preserves unknown custom models", () => {
    expect(resolveModelSlug("gpt-4.1")).toBe(DEFAULT_MODEL_BY_PROVIDER.codex);
    expect(resolveModelSlug("custom/internal-model")).toBe(DEFAULT_MODEL_BY_PROVIDER.codex);
  });

  it("resolves only supported model options", () => {
    for (const model of MODEL_OPTIONS_BY_PROVIDER.codex) {
      expect(resolveModelSlug(model.slug)).toBe(model.slug);
    }
  });

  it("passes through dynamic provider models when provider catalog is empty", () => {
    expect(MODEL_OPTIONS_BY_PROVIDER.opencode).toHaveLength(0);
    expect(resolveModelSlug("opencode/big-pickle", "opencode")).toBe("opencode/big-pickle");
  });

  it("keeps provider default when dynamic provider model is missing", () => {
    expect(resolveModelSlug(undefined, "opencode")).toBe(DEFAULT_MODEL_BY_PROVIDER.opencode);
    expect(resolveModelSlug("   ", "opencode")).toBe(DEFAULT_MODEL_BY_PROVIDER.opencode);
  });

  it("keeps codex defaults for backward compatibility", () => {
    expect(getDefaultModel()).toBe(DEFAULT_MODEL_BY_PROVIDER.codex);
    expect(getModelOptions()).toEqual(MODEL_OPTIONS_BY_PROVIDER.codex);
  });
});

describe("getReasoningEffortOptions", () => {
  it("returns codex reasoning options for codex", () => {
    expect(getReasoningEffortOptions("codex")).toEqual(["xhigh", "high", "medium", "low"]);
  });
});

describe("getDefaultReasoningEffort", () => {
  it("returns provider-scoped defaults", () => {
    expect(getDefaultReasoningEffort("codex")).toBe("high");
  });
});
