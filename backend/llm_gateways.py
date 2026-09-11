"""Registry of OpenAI-compatible gateways (OpenRouter, NVIDIA NIM, Kilo, Zen).

These providers all speak the OpenAI *Chat Completions* wire format, so no new
SDK is needed: the existing ``AsyncOpenAI`` client is pointed at a different
``base_url`` with a per-provider API key and, for the OpenRouter-style routers,
a couple of attribution headers.

This module is deliberately import-light so both the backend and the tests can
reason about it in isolation. It owns:

- which gateway serves which ``Llm`` member, and the bare model id to send
- what each model can actually do (vision input, tool calling)
- where each gateway's key and base URL come from (settings dialog > env > default)

`Llm` values for gateway models are ``"<gateway id>/<model id>"``, and this
module *derives* the enum member from that rule, so a typo in either table fails
at import time instead of silently becoming an unsupported model.

Gateways expose their chat endpoint at ``<base_url>/chat/completions``, so a
base URL must not include the path; ``https://openrouter.ai/api/v1`` is right.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import config
from llm import Llm

# Sent with the app's identity so gateways that rank traffic by app
# (OpenRouter, and the OpenRouter-compatible Kilo gateway) can see where a
# request came from. Both title spellings are sent: `X-Title` is the
# long-standing name, `X-OpenRouter-Title` the one OpenRouter's docs use now.
APP_ATTRIBUTION_HEADERS: Dict[str, str] = {
    "HTTP-Referer": "https://github.com/abi/screenshot-to-code",
    "X-Title": "screenshot-to-code",
    "X-OpenRouter-Title": "screenshot-to-code",
}


@dataclass(frozen=True)
class GatewayModel:
    """One model as offered by one gateway."""

    # Bare model id the gateway expects on the wire, e.g. `google/gemma-4-31b-it:free`.
    api_name: str
    # Whether the model accepts image parts in `messages`. Screenshot-to-code is
    # a vision task, so models without this can only serve text -> code runs.
    supports_vision: bool
    # Whether function/tool calling is usable. Without it the agent runs in
    # "answer only" mode and the final HTML is read out of the assistant text.
    supports_tools: bool

    def member(self, gateway_id: str) -> Llm:
        return Llm(f"{gateway_id}/{self.api_name}")


@dataclass(frozen=True)
class Gateway:
    """Connection details for one OpenAI-compatible provider."""

    id: str
    display_name: str
    default_base_url: str
    # Env var names, also used as attribute names on `config` so the .env file,
    # the README and the tests all refer to the same thing.
    api_key_env: str
    base_url_env: str
    # Field names the frontend Settings dialog sends as websocket params.
    settings_key: str
    settings_base_url_key: str
    docs_url: str
    models: Tuple[GatewayModel, ...] = ()
    # Ask the gateway to report token usage on the final streamed chunk
    # (`stream_options: {"include_usage": true}`). Optional in the OpenAI spec
    # and rejected outright by some stricter proxies, so it is opt-out per
    # gateway and retried once without it when the request is refused.
    send_stream_usage: bool = True
    extra_headers: Mapping[str, str] = field(default_factory=dict[str, str])

    def _env(self, name: str) -> Optional[str]:
        """Read a config value lazily so tests can monkeypatch `config`."""
        value = getattr(config, name, None) or os.environ.get(name)
        return str(value) if value else None

    def resolve_api_key(self, settings_value: Optional[str] = None) -> Optional[str]:
        """Settings dialog value wins, then `.env` / the process environment."""
        if settings_value:
            return settings_value
        return self._env(self.api_key_env)

    def resolve_base_url(self, settings_value: Optional[str] = None) -> str:
        """Base URL for `AsyncOpenAI`; overridable for proxies and self-hosting."""
        for candidate in (settings_value, self._env(self.base_url_env)):
            if candidate:
                return str(candidate).rstrip("/")
        return self.default_base_url.rstrip("/")

    def request_headers(self) -> Dict[str, str]:
        return dict(self.extra_headers)

    def api_names(self) -> List[str]:
        return [entry.api_name for entry in self.models]

    def vision_models(self) -> List[Llm]:
        return [entry.member(self.id) for entry in self.models if entry.supports_vision]


_OPENROUTER = Gateway(
    id="openrouter",
    display_name="OpenRouter",
    default_base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY",
    base_url_env="OPENROUTER_BASE_URL",
    settings_key="openRouterApiKey",
    settings_base_url_key="openRouterBaseUrl",
    docs_url="https://openrouter.ai/models",
    extra_headers=APP_ATTRIBUTION_HEADERS,
    models=(
        GatewayModel(
            api_name="google/gemma-4-31b-it:free",
            supports_vision=True,
            supports_tools=True,
        ),
        GatewayModel(
            api_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            supports_vision=True,
            supports_tools=True,
        ),
        # Small/fast lane: reads layouts, but the weakest of the free set.
        GatewayModel(
            api_name="nvidia/nemotron-nano-12b-v2-vl:free",
            supports_vision=True,
            supports_tools=False,
        ),
        # OpenRouter's "any available free model" route: survives slug churn and
        # a rate-limited favourite.
        GatewayModel(
            api_name="openrouter/free",
            supports_vision=True,
            supports_tools=True,
        ),
    ),
)

_NVIDIA = Gateway(
    id="nvidia",
    display_name="NVIDIA NIM",
    default_base_url="https://integrate.api.nvidia.com/v1",
    api_key_env="NVIDIA_API_KEY",
    base_url_env="NVIDIA_BASE_URL",
    settings_key="nvidiaApiKey",
    settings_base_url_key="nvidiaBaseUrl",
    docs_url="https://build.nvidia.com/models",
    models=(
        GatewayModel(
            api_name="meta/llama-3.2-90b-vision-instruct",
            supports_vision=True,
            # The hosted Llama 3.2 Vision builds do not reliably honour tool
            # calls, so let them write the HTML directly instead.
            supports_tools=False,
        ),
        GatewayModel(
            api_name="meta/llama-3.2-11b-vision-instruct",
            supports_vision=True,
            supports_tools=False,
        ),
    ),
)

_KILO = Gateway(
    id="kilo",
    display_name="Kilo Gateway",
    # No `/v1`: Kilo's OpenAI-compatible endpoint is
    # `https://api.kilo.ai/api/gateway/chat/completions`.
    default_base_url="https://api.kilo.ai/api/gateway",
    api_key_env="KILO_API_KEY",
    base_url_env="KILO_BASE_URL",
    settings_key="kiloApiKey",
    settings_base_url_key="kiloBaseUrl",
    docs_url="https://kilo.ai/docs/gateway/api-reference",
    extra_headers=APP_ATTRIBUTION_HEADERS,
    models=(
        GatewayModel(
            api_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            supports_vision=True,
            supports_tools=True,
        ),
        # Kilo's own router (advertises image input + tool calling); costs a
        # little but keeps working when a `:free` slug is retired.
        GatewayModel(api_name="kilo/auto", supports_vision=True, supports_tools=True),
        GatewayModel(
            api_name="minimax/minimax-m2.5:free",
            # Text-only: fine for text -> code, blind to screenshots.
            supports_vision=False,
            supports_tools=True,
        ),
    ),
)

_ZEN = Gateway(
    id="zen",
    display_name="OpenCode Zen",
    default_base_url="https://opencode.ai/zen/v1",
    api_key_env="ZEN_API_KEY",
    base_url_env="ZEN_BASE_URL",
    settings_key="zenApiKey",
    settings_base_url_key="zenBaseUrl",
    docs_url="https://opencode.ai/docs/providers/",
    models=(
        GatewayModel(
            api_name="gpt-5-nano",
            supports_vision=True,
            supports_tools=True,
        ),
        GatewayModel(
            api_name="big-pickle",
            # Free while OpenCode collects feedback on it; text input only.
            supports_vision=False,
            supports_tools=False,
        ),
    ),
)

# Order matters: it is the tie-break for variant selection.
GATEWAYS: Dict[str, Gateway] = {
    gateway.id: gateway for gateway in (_OPENROUTER, _NVIDIA, _KILO, _ZEN)
}

GATEWAY_FOR_MODEL: Dict[Llm, Gateway] = {}
GATEWAY_MODEL_FOR_MODEL: Dict[Llm, GatewayModel] = {}

for _gateway in GATEWAYS.values():
    for _entry in _gateway.models:
        try:
            _member = _entry.member(_gateway.id)
        except ValueError as exc:  # pragma: no cover - import-time guard
            raise ValueError(
                f"llm_gateways lists `{_gateway.id}/{_entry.api_name}` but llm.py "
                f"has no such Llm member (add it or fix the api_name): {exc}"
            ) from exc
        GATEWAY_FOR_MODEL[_member] = _gateway
        GATEWAY_MODEL_FOR_MODEL[_member] = _entry


def gateway_for_model(model: Llm) -> Optional[Gateway]:
    return GATEWAY_FOR_MODEL.get(model)


def is_gateway_model(model: Llm) -> bool:
    return model in GATEWAY_FOR_MODEL


def gateway_api_name(model: Llm) -> str:
    entry = GATEWAY_MODEL_FOR_MODEL.get(model)
    if entry is None:
        raise ValueError(f"{model.value} is not served by a registered gateway")
    return entry.api_name


def supports_vision(model: Llm) -> bool:
    """False only for gateway models explicitly marked text-only.

    First-party models (GPT, Claude, Gemini) all accept image input, which is
    why they need no entry here.
    """
    entry = GATEWAY_MODEL_FOR_MODEL.get(model)
    return True if entry is None else entry.supports_vision


def supports_tools(model: Llm) -> bool:
    entry = GATEWAY_MODEL_FOR_MODEL.get(model)
    return True if entry is None else entry.supports_tools


def gateway_settings_values(params: Mapping[str, object]) -> Dict[str, str]:
    """Pick the gateway fields out of a websocket params payload."""
    values: Dict[str, str] = {}
    for gateway in GATEWAYS.values():
        for key in (gateway.settings_key, gateway.settings_base_url_key):
            raw = params.get(key)
            if isinstance(raw, str) and raw.strip():
                values[key] = raw.strip()
    return values


def configured_gateways(
    api_key_overrides: Optional[Mapping[str, str]] = None,
) -> List[Gateway]:
    """Gateways that have a usable key.

    `api_key_overrides` are the Settings-dialog values keyed by gateway id (as
    extracted by the websocket route); each gateway falls back to its `.env`
    key. Returned in registry order so model selection stays deterministic.
    """
    api_key_overrides = api_key_overrides or {}
    return [
        gateway
        for gateway in GATEWAYS.values()
        if gateway.resolve_api_key(api_key_overrides.get(gateway.id))
    ]


def select_gateway_models(
    gateways: Sequence[Gateway],
    *,
    vision_required: bool,
    limit: int = 4,
) -> List[Llm]:
    """Model list to cycle variants through, best-supported models first.

    With a screenshot in the request, text-only models are dropped outright —
    they would return a confident guess about an image they never saw.
    """

    selected: List[Llm] = []
    for gateway in gateways:
        usable = [
            entry
            for entry in gateway.models
            if not vision_required or entry.supports_vision
        ]
        # Tool-capable models first; `sorted` is stable, so the registry order
        # breaks ties.
        ordered = sorted(usable, key=lambda entry: 0 if entry.supports_tools else 1)
        selected.extend(entry.member(gateway.id) for entry in ordered)
    return selected[:limit] if limit > 0 else selected
