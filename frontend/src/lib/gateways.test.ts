import {
  GATEWAY_OPTIONS,
  gatewaysSupportingScreenshots,
} from "./gateways";
import {
  CODE_GENERATION_MODEL_DESCRIPTIONS,
  CodeGenerationModel,
  getVariantLabel,
} from "./models";

const CREATE = { inputMode: "image" as const, generationType: "create" as const };

describe("gateway registry", () => {
  test("every gateway model exists in the backend-mirrored enum and descriptions", () => {
    for (const gateway of GATEWAY_OPTIONS) {
      for (const option of gateway.models) {
        // Fails loudly when llm.py and lib/gateways.ts drift apart, which is the
        // only guard here that the ids the backend sends are known to the UI.
        expect(Object.values(CodeGenerationModel)).toContain(option.model);
        expect(CODE_GENERATION_MODEL_DESCRIPTIONS[option.model].name).toBeTruthy();
      }
    }
  });

  test("gateway ids and settings fields are unique", () => {
    const ids = GATEWAY_OPTIONS.map((gateway) => gateway.id);
    const keyFields = GATEWAY_OPTIONS.map((gateway) => gateway.apiKeyField);
    const urlFields = GATEWAY_OPTIONS.map((gateway) => gateway.baseUrlField);

    expect(new Set(ids).size).toBe(ids.length);
    expect(new Set(keyFields).size).toBe(keyFields.length);
    expect(new Set(urlFields).size).toBe(urlFields.length);
  });

  test("every gateway can serve a screenshot", () => {
    // A gateway with nothing vision-capable could never run the core flow, so
    // adding one means adding an image-capable model to it.
    expect(gatewaysSupportingScreenshots().map((g) => g.id)).toEqual(
      GATEWAY_OPTIONS.map((g) => g.id)
    );
  });

  test("base URLs are OpenAI-compatible roots, without the completions path", () => {
    for (const gateway of GATEWAY_OPTIONS) {
      expect(gateway.defaultBaseUrl).not.toMatch(/\/chat\/completions\/?$/);
      expect(gateway.defaultBaseUrl).toMatch(/^https:\/\//);
      expect(gateway.defaultBaseUrl).not.toMatch(/\/$/);
    }
    // Kilo's route is under /api/gateway, not the usual /v1.
    expect(
      GATEWAY_OPTIONS.find((gateway) => gateway.id === "kilo")?.defaultBaseUrl
    ).toBe("https://api.kilo.ai/api/gateway");
  });

  test("free-tier variants are badged so they read differently from frontier ones", () => {
    const label = getVariantLabel(
      CodeGenerationModel.OPENROUTER_GEMMA_4_31B_FREE,
      CREATE
    );
    expect(label).toEqual({ text: "Free", tone: "free" });
    // Kilo's paid auto-router is a gateway model but not a free one.
    expect(getVariantLabel(CodeGenerationModel.KILO_AUTO, CREATE)).toBeNull();
  });
});
