// Frontend mirror of backend/llm_gateways.py.
//
// These are OpenAI-compatible gateways: one API key each, and the backend talks
// to them with the same OpenAI client it already uses. Keep this file in sync
// with the registry - `gateways.test.ts` fails if a model listed here is
// missing from CodeGenerationModel (which is how drift with llm.py shows up).

import { CodeGenerationModel } from "./models";

export interface GatewayModelOption {
  /** Value of the backend `Llm` enum member. */
  model: CodeGenerationModel;
  /** The gateway can read the screenshot, so it is usable for image/video input. */
  supportsVision: boolean;
}

export interface GatewayOption {
  id: string;
  name: string;
  /** Settings field that stores the key; sent to the backend on every generation. */
  apiKeyField:
    | "openRouterApiKey"
    | "nvidiaApiKey"
    | "kiloApiKey"
    | "zenApiKey"
    | "zenmuxApiKey";
  apiKeyPlaceholder: string;
  baseUrlField:
    | "openRouterBaseUrl"
    | "nvidiaBaseUrl"
    | "kiloBaseUrl"
    | "zenBaseUrl"
    | "zenmuxBaseUrl";
  /** What to prefill when the user asks for the default endpoint. */
  defaultBaseUrl: string;
  signupUrl: string;
  blurb: string;
  models: GatewayModelOption[];
}

export const GATEWAY_OPTIONS: GatewayOption[] = [
  {
    id: "openrouter",
    name: "OpenRouter",
    apiKeyField: "openRouterApiKey",
    apiKeyPlaceholder: "OpenRouter API key (sk-or-...)",
    baseUrlField: "openRouterBaseUrl",
    defaultBaseUrl: "https://openrouter.ai/api/v1",
    signupUrl: "https://openrouter.ai/keys",
    blurb:
      "One key for many models, including a rotating set of :free ones. Free models are rate-limited and markedly weaker at screenshot-to-code than GPT, Claude, or Gemini.",
    models: [
      {
        model: CodeGenerationModel.OPENROUTER_GEMMA_4_31B_FREE,
        supportsVision: true,
      },
      {
        model: CodeGenerationModel.OPENROUTER_NEMOTRON_OMNI_30B_FREE,
        supportsVision: true,
      },
      {
        model: CodeGenerationModel.OPENROUTER_NEMOTRON_NANO_12B_VL_FREE,
        supportsVision: true,
      },
      { model: CodeGenerationModel.OPENROUTER_AUTO_FREE, supportsVision: true },
    ],
  },
  {
    id: "nvidia",
    name: "NVIDIA NIM",
    apiKeyField: "nvidiaApiKey",
    apiKeyPlaceholder: "NVIDIA API key (nvapi-...)",
    baseUrlField: "nvidiaBaseUrl",
    defaultBaseUrl: "https://integrate.api.nvidia.com/v1",
    signupUrl: "https://build.nvidia.com/models",
    blurb:
      "Free hosted inference against a large model catalog. The Llama Vision models here answer in plain HTML instead of using agent tools.",
    models: [
      { model: CodeGenerationModel.NVIDIA_LLAMA_3_2_90B_VISION, supportsVision: true },
      { model: CodeGenerationModel.NVIDIA_LLAMA_3_2_11B_VISION, supportsVision: true },
    ],
  },
  {
    id: "kilo",
    name: "Kilo Gateway",
    apiKeyField: "kiloApiKey",
    apiKeyPlaceholder: "Kilo API key",
    baseUrlField: "kiloBaseUrl",
    // Not `/v1` - Kilo's chat route is https://api.kilo.ai/api/gateway/chat/completions.
    defaultBaseUrl: "https://api.kilo.ai/api/gateway",
    signupUrl: "https://app.kilo.ai/profile",
    blurb:
      "Coding-agent gateway with free tiers and an auto router. New models are free to try for a while after release.",
    models: [
      { model: CodeGenerationModel.KILO_NEMOTRON_OMNI_30B_FREE, supportsVision: true },
      { model: CodeGenerationModel.KILO_AUTO, supportsVision: true },
      { model: CodeGenerationModel.KILO_MINIMAX_M2_5_FREE, supportsVision: false },
    ],
  },
  {
    id: "zen",
    name: "OpenCode Zen",
    apiKeyField: "zenApiKey",
    apiKeyPlaceholder: "OpenCode API key",
    baseUrlField: "zenBaseUrl",
    defaultBaseUrl: "https://opencode.ai/zen/v1",
    signupUrl: "https://opencode.ai/auth",
    blurb:
      "Curated pay-per-use gateway with a few free models. Its free catalog rotates, so a model can disappear without notice.",
    models: [
      { model: CodeGenerationModel.ZEN_GPT_5_NANO, supportsVision: true },
      { model: CodeGenerationModel.ZEN_BIG_PICKLE, supportsVision: false },
    ],
  },
  {
    id: "zenmux",
    name: "ZenMux",
    apiKeyField: "zenmuxApiKey",
    apiKeyPlaceholder: "ZenMux API key",
    baseUrlField: "zenmuxBaseUrl",
    defaultBaseUrl: "https://zenmux.ai/api/v1",
    signupUrl: "https://zenmux.ai",
    blurb:
      "A different router than OpenCode Zen — wired as its own provider because \"Zen\" means either depending on where you read about it. Pay-per-use, no free slugs.",
    models: [
      { model: CodeGenerationModel.ZENMUX_GPT_5, supportsVision: true },
      { model: CodeGenerationModel.ZENMUX_GEMINI_2_5_PRO, supportsVision: true },
    ],
  },
];

/** Gateways whose curated models can all see an image - i.e. usable for screenshots. */
export function gatewaysSupportingScreenshots(): GatewayOption[] {
  return GATEWAY_OPTIONS.filter((gateway) =>
    gateway.models.some((option) => option.supportsVision)
  );
}
