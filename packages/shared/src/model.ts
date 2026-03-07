import {
  CODEX_REASONING_EFFORT_OPTIONS,
  DEFAULT_MODEL_BY_PROVIDER,
  MODEL_OPTIONS_BY_PROVIDER,
  MODEL_SLUG_ALIASES_BY_PROVIDER,
  type CodexReasoningEffort,
  type ModelSlug,
  type ProviderKind,
  type ReasoningEffort,
} from "@t3tools/contracts";

type CatalogProvider = keyof typeof MODEL_OPTIONS_BY_PROVIDER;

const MODEL_SLUG_SET_BY_PROVIDER: Record<CatalogProvider, ReadonlySet<ModelSlug>> = {
  codex: new Set(MODEL_OPTIONS_BY_PROVIDER.codex.map((option) => option.slug)),
  opencode: new Set(MODEL_OPTIONS_BY_PROVIDER.opencode.map((option) => option.slug)),
};

export function getModelOptions(provider: ProviderKind = "codex") {
  return MODEL_OPTIONS_BY_PROVIDER[provider];
}

export function getDefaultModel(provider: ProviderKind = "codex"): ModelSlug {
  return DEFAULT_MODEL_BY_PROVIDER[provider];
}

export function normalizeModelSlug(
  model: string | null | undefined,
  provider: ProviderKind = "codex",
): ModelSlug | null {
  if (typeof model !== "string") {
    return null;
  }

  const trimmed = model.trim();
  if (!trimmed) {
    return null;
  }

  const aliases = MODEL_SLUG_ALIASES_BY_PROVIDER[provider] as Record<string, ModelSlug>;
  const aliased = aliases[trimmed];
  return typeof aliased === "string" ? aliased : (trimmed as ModelSlug);
}

export function resolveModelSlug(
  model: string | null | undefined,
  provider: ProviderKind = "codex",
): ModelSlug {
  const normalized = normalizeModelSlug(model, provider);
  if (!normalized) {
    return getDefaultModel(provider);
  }

  const catalog = MODEL_SLUG_SET_BY_PROVIDER[provider];
  if (catalog.size === 0) {
    return normalized;
  }

  return catalog.has(normalized) ? normalized : getDefaultModel(provider);
}

export function resolveModelSlugForProvider(
  provider: ProviderKind,
  model: string | null | undefined,
): ModelSlug {
  return resolveModelSlug(model, provider);
}

export function getReasoningEffortOptions(
  provider: ProviderKind = "codex",
): ReadonlyArray<ReasoningEffort> {
  return provider === "codex" ? CODEX_REASONING_EFFORT_OPTIONS : [];
}

export function getDefaultReasoningEffort(provider: "codex"): CodexReasoningEffort;
export function getDefaultReasoningEffort(provider: ProviderKind): ReasoningEffort | null;
export function getDefaultReasoningEffort(
  provider: ProviderKind = "codex",
): ReasoningEffort | null {
  return provider === "codex" ? "high" : null;
}

export { CODEX_REASONING_EFFORT_OPTIONS };
