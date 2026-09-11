// Keep in sync with backend (llm.py)
// Order here matches dropdown order
export enum CodeGenerationModel {
  CLAUDE_OPUS_5_LOW = "claude-opus-5 (low effort)",
  CLAUDE_OPUS_5_MEDIUM = "claude-opus-5 (medium effort)",
  CLAUDE_OPUS_5_HIGH = "claude-opus-5 (high effort)",
  CLAUDE_OPUS_5_XHIGH = "claude-opus-5 (xhigh effort)",
  CLAUDE_OPUS_5_MAX = "claude-opus-5 (max effort)",
  CLAUDE_OPUS_4_8_LOW = "claude-opus-4-8 (low effort)",
  CLAUDE_OPUS_4_8_MEDIUM = "claude-opus-4-8 (medium effort)",
  CLAUDE_OPUS_4_8_HIGH = "claude-opus-4-8 (high effort)",
  CLAUDE_OPUS_4_8_XHIGH = "claude-opus-4-8 (xhigh effort)",
  CLAUDE_OPUS_4_8_MAX = "claude-opus-4-8 (max effort)",
  CLAUDE_FABLE_5_MAX = "claude-fable-5 (max effort)",
  CLAUDE_SONNET_4_6 = "claude-sonnet-4-6",
  GPT_5_5_NONE = "gpt-5.5 (no thinking)",
  GPT_5_5_LOW = "gpt-5.5 (low thinking)",
  GPT_5_5_MEDIUM = "gpt-5.5 (medium thinking)",
  GPT_5_5_HIGH = "gpt-5.5 (high thinking)",
  GPT_5_6_SOL_NONE = "gpt-5.6-sol (no thinking)",
  GPT_5_6_SOL_LOW = "gpt-5.6-sol (low thinking)",
  GPT_5_6_SOL_MEDIUM = "gpt-5.6-sol (medium thinking)",
  GPT_5_6_SOL_HIGH = "gpt-5.6-sol (high thinking)",
  GPT_5_6_SOL_XHIGH = "gpt-5.6-sol (xhigh thinking)",
  GPT_5_6_SOL_MAX = "gpt-5.6-sol (max thinking)",
  GPT_5_6_TERRA_LOW = "gpt-5.6-terra (low thinking)",
  GPT_5_5_XHIGH = "gpt-5.5 (xhigh thinking)",
  GPT_5_4_MINI_LOW = "gpt-5.4-mini (low thinking)",
  GEMINI_3_FLASH_PREVIEW_HIGH = "gemini-3-flash-preview (high thinking)",
  GEMINI_3_FLASH_PREVIEW_MINIMAL = "gemini-3-flash-preview (minimal thinking)",
  GEMINI_3_1_PRO_PREVIEW_HIGH = "gemini-3.1-pro-preview (high thinking)",
  GEMINI_3_1_PRO_PREVIEW_MEDIUM = "gemini-3.1-pro-preview (medium thinking)",
  GEMINI_3_1_PRO_PREVIEW_LOW = "gemini-3.1-pro-preview (low thinking)",
  GEMINI_3_5_FLASH_HIGH = "gemini-3.5-flash (high thinking)",
  GEMINI_3_5_FLASH_MEDIUM = "gemini-3.5-flash (medium thinking)",
  GEMINI_3_5_FLASH_LOW = "gemini-3.5-flash (low thinking)",
  GEMINI_3_5_FLASH_MINIMAL = "gemini-3.5-flash (minimal thinking)",
  GEMINI_3_6_FLASH_HIGH = "gemini-3.6-flash (high thinking)",
  GEMINI_3_6_FLASH_MEDIUM = "gemini-3.6-flash (medium thinking)",
  GEMINI_3_6_FLASH_LOW = "gemini-3.6-flash (low thinking)",
  GEMINI_3_6_FLASH_MINIMAL = "gemini-3.6-flash (minimal thinking)",
  // OpenAI-compatible gateways. Values are `<gateway>/<model id>` to match the
  // backend `Llm` enum, because the same upstream model can be listed by two
  // gateways. See lib/gateways.ts.
  OPENROUTER_GEMMA_4_31B_FREE = "openrouter/google/gemma-4-31b-it:free",
  OPENROUTER_NEMOTRON_OMNI_30B_FREE = "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
  OPENROUTER_NEMOTRON_NANO_12B_VL_FREE = "openrouter/nvidia/nemotron-nano-12b-v2-vl:free",
  OPENROUTER_AUTO_FREE = "openrouter/openrouter/free",
  NVIDIA_LLAMA_3_2_90B_VISION = "nvidia/meta/llama-3.2-90b-vision-instruct",
  NVIDIA_LLAMA_3_2_11B_VISION = "nvidia/meta/llama-3.2-11b-vision-instruct",
  KILO_NEMOTRON_OMNI_30B_FREE = "kilo/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
  KILO_AUTO = "kilo/kilo/auto",
  KILO_MINIMAX_M2_5_FREE = "kilo/minimax/minimax-m2.5:free",
  ZEN_GPT_5_NANO = "zen/gpt-5-nano",
  ZEN_BIG_PICKLE = "zen/big-pickle",
}

export type VariantLabelTone = "fast" | "max" | "free";

export interface VariantLabel {
  text: string;
  tone: VariantLabelTone;
}

export interface VariantLabelContext {
  inputMode: "image" | "video" | "text";
  generationType: "create" | "update";
}

// Per-model badge text. Only these models are labelled. Heavyweight
// variants read "Max" (sol max anchors image create; 3.1 Pro high anchors
// video); Flash-minimal is the only variant fast enough to earn "Fast";
// anything served by an OpenAI-compatible gateway reads "Free" so a
// free-tier take is not mistaken for a frontier one.
const VARIANT_LABELS: Partial<Record<CodeGenerationModel, VariantLabel>> = {
  // Gateway variants are flagged so a free-tier take is never mistaken for a
  // frontier one.
  [CodeGenerationModel.OPENROUTER_GEMMA_4_31B_FREE]: { text: "Free", tone: "free" },
  [CodeGenerationModel.OPENROUTER_NEMOTRON_OMNI_30B_FREE]: {
    text: "Free",
    tone: "free",
  },
  [CodeGenerationModel.OPENROUTER_NEMOTRON_NANO_12B_VL_FREE]: {
    text: "Free",
    tone: "free",
  },
  [CodeGenerationModel.OPENROUTER_AUTO_FREE]: { text: "Free", tone: "free" },
  [CodeGenerationModel.KILO_NEMOTRON_OMNI_30B_FREE]: {
    text: "Free",
    tone: "free",
  },
  [CodeGenerationModel.KILO_MINIMAX_M2_5_FREE]: { text: "Free", tone: "free" },
  [CodeGenerationModel.ZEN_BIG_PICKLE]: { text: "Free", tone: "free" },
  [CodeGenerationModel.GEMINI_3_FLASH_PREVIEW_MINIMAL]: { text: "Fast", tone: "fast" },
  [CodeGenerationModel.GEMINI_3_1_PRO_PREVIEW_HIGH]: { text: "Max", tone: "max" },
  [CodeGenerationModel.GPT_5_5_HIGH]: { text: "Max", tone: "max" },
  [CodeGenerationModel.GPT_5_6_SOL_MAX]: { text: "Max", tone: "max" },
};

// Badges are only shown on create flows and on any video flow. In particular
// image/text update runs reuse Flash-minimal but should stay unlabelled.
export function getVariantLabel(
  model: string | undefined,
  context: VariantLabelContext
): VariantLabel | null {
  if (!model) return null;
  const showLabels =
    context.generationType === "create" || context.inputMode === "video";
  if (!showLabels) return null;
  return VARIANT_LABELS[model as CodeGenerationModel] ?? null;
}

// Will generate a static error if a model in the enum above is not in the descriptions
export const CODE_GENERATION_MODEL_DESCRIPTIONS: {
  [key in CodeGenerationModel]: { name: string };
} = {
  "gpt-5.6-sol (no thinking)": {
    name: "GPT 5.6 Sol (none)",
  },
  "gpt-5.6-sol (low thinking)": {
    name: "GPT 5.6 Sol (low)",
  },
  "gpt-5.6-sol (medium thinking)": {
    name: "GPT 5.6 Sol (medium)",
  },
  "gpt-5.6-sol (high thinking)": {
    name: "GPT 5.6 Sol (high)",
  },
  "gpt-5.6-sol (xhigh thinking)": {
    name: "GPT 5.6 Sol (xhigh)",
  },
  "gpt-5.6-sol (max thinking)": {
    name: "GPT 5.6 Sol (max)",
  },
  "gpt-5.6-terra (low thinking)": {
    name: "GPT 5.6 Terra (low)",
  },
  "gpt-5.5 (no thinking)": {
    name: "GPT 5.5 (none)",
  },
  "gpt-5.5 (low thinking)": {
    name: "GPT 5.5 (low)",
  },
  "gpt-5.5 (medium thinking)": {
    name: "GPT 5.5 (medium)",
  },
  "gpt-5.5 (high thinking)": {
    name: "GPT 5.5 (high)",
  },
  "gpt-5.5 (xhigh thinking)": {
    name: "GPT 5.5 (xhigh)",
  },
  "gpt-5.4-mini (low thinking)": {
    name: "GPT 5.4 Mini (low)",
  },
  "claude-opus-5 (low effort)": {
    name: "Claude Opus 5 (low)",
  },
  "claude-opus-5 (medium effort)": {
    name: "Claude Opus 5 (medium)",
  },
  "claude-opus-5 (high effort)": {
    name: "Claude Opus 5 (high)",
  },
  "claude-opus-5 (xhigh effort)": {
    name: "Claude Opus 5 (xhigh)",
  },
  "claude-opus-5 (max effort)": {
    name: "Claude Opus 5 (max)",
  },
  "claude-opus-4-8 (low effort)": {
    name: "Claude Opus 4.8 (low)",
  },
  "claude-opus-4-8 (medium effort)": {
    name: "Claude Opus 4.8 (medium)",
  },
  "claude-opus-4-8 (high effort)": {
    name: "Claude Opus 4.8 (high)",
  },
  "claude-opus-4-8 (xhigh effort)": {
    name: "Claude Opus 4.8 (xhigh)",
  },
  "claude-opus-4-8 (max effort)": {
    name: "Claude Opus 4.8 (max)",
  },
  "claude-fable-5 (max effort)": {
    name: "Claude Fable 5 (max)",
  },
  "claude-sonnet-4-6": { name: "Claude Sonnet 4.6" },
  "gemini-3.5-flash (high thinking)": {
    name: "Gemini 3.5 Flash (high)",
  },
  "gemini-3.5-flash (medium thinking)": {
    name: "Gemini 3.5 Flash (medium)",
  },
  "gemini-3.5-flash (low thinking)": {
    name: "Gemini 3.5 Flash (low)",
  },
  "gemini-3.5-flash (minimal thinking)": {
    name: "Gemini 3.5 Flash (minimal)",
  },
  "gemini-3.6-flash (high thinking)": {
    name: "Gemini 3.6 Flash (high)",
  },
  "gemini-3.6-flash (medium thinking)": {
    name: "Gemini 3.6 Flash (medium)",
  },
  "gemini-3.6-flash (low thinking)": {
    name: "Gemini 3.6 Flash (low)",
  },
  "gemini-3.6-flash (minimal thinking)": {
    name: "Gemini 3.6 Flash (minimal)",
  },
  "gemini-3-flash-preview (high thinking)": {
    name: "Gemini 3 Flash (high)",
  },
  "gemini-3-flash-preview (minimal thinking)": {
    name: "Gemini 3 Flash (minimal)",
  },
  "gemini-3.1-pro-preview (high thinking)": {
    name: "Gemini 3.1 Pro (high)",
  },
  "gemini-3.1-pro-preview (medium thinking)": {
    name: "Gemini 3.1 Pro (medium)",
  },
  "gemini-3.1-pro-preview (low thinking)": {
    name: "Gemini 3.1 Pro (low)",
  },
  "openrouter/google/gemma-4-31b-it:free": {
    name: "OpenRouter: Gemma 4 31B (free)",
  },
  "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": {
    name: "OpenRouter: Nemotron Nano Omni 30B (free)",
  },
  "openrouter/nvidia/nemotron-nano-12b-v2-vl:free": {
    name: "OpenRouter: Nemotron Nano 12B VL (free)",
  },
  "openrouter/openrouter/free": { name: "OpenRouter: auto (free)" },
  "nvidia/meta/llama-3.2-90b-vision-instruct": {
    name: "NVIDIA NIM: Llama 3.2 90B Vision",
  },
  "nvidia/meta/llama-3.2-11b-vision-instruct": {
    name: "NVIDIA NIM: Llama 3.2 11B Vision",
  },
  "kilo/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": {
    name: "Kilo: Nemotron Nano Omni 30B (free)",
  },
  "kilo/kilo/auto": { name: "Kilo: auto" },
  "kilo/minimax/minimax-m2.5:free": { name: "Kilo: MiniMax M2.5 (free)" },
  "zen/gpt-5-nano": { name: "OpenCode Zen: GPT-5 Nano" },
  "zen/big-pickle": { name: "OpenCode Zen: Big Pickle (free)" },
};
