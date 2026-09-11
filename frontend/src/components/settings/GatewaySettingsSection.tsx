import React from "react";
import { Input } from "../ui/input";
import { IS_RUNNING_ON_CLOUD } from "../../config";
import { Settings } from "../../types";
import { GATEWAY_OPTIONS, GatewayOption } from "../../lib/gateways";
import { CODE_GENERATION_MODEL_DESCRIPTIONS } from "../../lib/models";

interface Props {
  settings: Settings;
  setSettings: React.Dispatch<React.SetStateAction<Settings>>;
}

// Derived from the registry rather than re-listed, so a new gateway field can
// never be added to one place and forgotten in the other.
type GatewayField = GatewayOption["apiKeyField"] | GatewayOption["baseUrlField"];

// Explicit per-field update rather than a computed key, so a typo in a field
// name is a compile error instead of a setting that silently never persists.
function makeSetter(
  setSettings: Props["setSettings"],
  field: GatewayField
): (value: string) => void {
  return (value: string) => {
    setSettings((s) => ({ ...s, [field]: value }));
  };
}

function GatewayRow({
  gateway,
  settings,
  setSettings,
}: {
  gateway: GatewayOption;
  settings: Settings;
  setSettings: Props["setSettings"];
}) {
  const setApiKey = makeSetter(setSettings, gateway.apiKeyField);
  const setBaseUrl = makeSetter(setSettings, gateway.baseUrlField);
  const apiKey = settings[gateway.apiKeyField];
  const baseUrl = settings[gateway.baseUrlField];
  const textOnly = gateway.models.filter((model) => !model.supportsVision);

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-sm font-medium text-gray-700 dark:text-zinc-300">
          {gateway.name} API key
        </p>
        <a
          href={gateway.signupUrl}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-violet-600 hover:text-violet-700 dark:text-violet-400 dark:hover:text-violet-300"
        >
          Get a key
        </a>
      </div>
      <p className="text-xs text-gray-500 dark:text-zinc-400">{gateway.blurb}</p>
      <Input
        id={`${gateway.id}-api-key`}
        placeholder={gateway.apiKeyPlaceholder}
        value={apiKey || ""}
        onChange={(e) => setApiKey(e.target.value)}
      />
      <p className="text-xs text-gray-500 dark:text-zinc-400">
        Models used:{" "}
        {gateway.models
          .map(
            (model) =>
              CODE_GENERATION_MODEL_DESCRIPTIONS[model.model]?.name ?? model.model
          )
          .join(", ")}
        {textOnly.length > 0 && (
          <span className="text-gray-400 dark:text-zinc-500">
            {" "}
            — {textOnly.length} of these take text only, so they are skipped for
            screenshot inputs.
          </span>
        )}
      </p>
      {!IS_RUNNING_ON_CLOUD && (
        <div>
          <p className="text-xs text-gray-500 dark:text-zinc-400">
            Base URL (optional) — leave blank for {gateway.defaultBaseUrl}
          </p>
          <Input
            id={`${gateway.id}-base-url`}
            className="mt-1"
            placeholder={gateway.defaultBaseUrl}
            value={baseUrl || ""}
            onChange={(e) => setBaseUrl(e.target.value)}
          />
        </div>
      )}
    </div>
  );
}

/**
 * Keys for the OpenAI-compatible gateways. A key here lets a generation run
 * without OpenAI/Anthropic/Gemini, and costs nothing on the `:free` slugs.
 */
const GatewaySettingsSection: React.FC<Props> = ({ settings, setSettings }) => {
  const anyConfigured = GATEWAY_OPTIONS.some(
    (gateway) => !!settings[gateway.apiKeyField]
  );

  return (
    <div className="rounded-lg border border-gray-200 bg-white dark:border-zinc-700 dark:bg-zinc-800/60">
      <div className="border-b border-gray-100 px-4 py-3 dark:border-zinc-700">
        <h2 className="text-sm font-medium text-gray-900 dark:text-white">
          Free &amp; low-cost gateways
        </h2>
      </div>
      <div className="space-y-4 p-4">
        <p className="text-xs text-gray-500 dark:text-zinc-400">
          These speak the OpenAI API, so each one just needs a key. They are used
          when no OpenAI, Anthropic, or Gemini key is set; with a first-party key
          present, the first-party models win. Only stored in your browser.
        </p>
        {GATEWAY_OPTIONS.map((gateway) => (
          <GatewayRow
            key={gateway.id}
            gateway={gateway}
            settings={settings}
            setSettings={setSettings}
          />
        ))}
        {anyConfigured && (
          <p className="rounded-md border border-amber-300 bg-amber-50 p-2.5 text-xs text-amber-800 dark:border-amber-700/60 dark:bg-amber-900/20 dark:text-amber-200">
            Free-tier models are much weaker at screenshot-to-code than GPT,
            Claude, or Gemini: they drop elements, invent copy, and truncate long
            HTML. Compare a variant before you judge the result.
          </p>
        )}
      </div>
    </div>
  );
};

export default GatewaySettingsSection;
