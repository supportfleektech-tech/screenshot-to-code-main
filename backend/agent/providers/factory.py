from typing import Dict, Optional, Tuple

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.providers.anthropic import AnthropicProviderSession, serialize_anthropic_tools
from agent.providers.base import ProviderSession
from agent.providers.gemini import GeminiProviderSession, serialize_gemini_tools
from agent.providers.openai import OpenAIProviderSession, serialize_openai_tools
from agent.providers.openai_compatible import (
    OpenAICompatibleProviderSession,
    serialize_chat_completions_tools,
)
from agent.tools import canonical_tool_definitions
from config import IS_PROD, REPLICATE_API_KEY
from fs_logging.agent_runs import AgentRunRecorder
from llm import ANTHROPIC_MODELS, GEMINI_MODELS, OPENAI_MODELS, Llm
from llm_gateways import Gateway, gateway_for_model, supports_tools
from preview_screenshot import is_screenshot_preview_available


def _resolve_gateway_credentials(
    gateway: Gateway,
    api_keys: Optional[Dict[str, str]],
    base_urls: Optional[Dict[str, str]],
) -> Tuple[str, str]:
    """Pick (api_key, base_url) for a gateway: request settings, then `.env`.

    Base URLs are the only user-tunable part: they exist for proxies and
    self-hosted endpoints, and are disabled in prod (the hosted app talks to
    the real gateway) exactly like OPENAI_BASE_URL is. The route already drops
    dialog values in prod; this covers the `.env` side.
    """
    api_key = gateway.resolve_api_key((api_keys or {}).get(gateway.id))
    if not api_key:
        raise Exception(f"{gateway.display_name} API key is missing.")

    if IS_PROD:
        return api_key, gateway.default_base_url.rstrip("/")
    return api_key, gateway.resolve_base_url((base_urls or {}).get(gateway.id))


def create_provider_session(
    model: Llm,
    prompt_messages: list[ChatCompletionMessageParam],
    should_generate_images: bool,
    openai_api_key: Optional[str],
    openai_base_url: Optional[str],
    anthropic_api_key: Optional[str],
    gemini_api_key: Optional[str],
    replicate_api_key: Optional[str],
    should_extract_assets: bool = True,
    recorder: Optional[AgentRunRecorder] = None,
    gateway_api_keys: Optional[Dict[str, str]] = None,
    gateway_base_urls: Optional[Dict[str, str]] = None,
) -> ProviderSession:
    canonical_tools = canonical_tool_definitions(
        image_generation_enabled=should_generate_images,
        # The edit_images tool calls Replicate, so don't offer it without a key.
        image_editing_enabled=bool(replicate_api_key or REPLICATE_API_KEY),
        # The extract_assets tool calls Gemini, so don't offer it without a key.
        asset_extraction_enabled=should_extract_assets and bool(gemini_api_key),
        # screenshot_preview needs headless Chromium; skip it if it can't launch.
        screenshot_enabled=is_screenshot_preview_available(),
    )

    if model in OPENAI_MODELS:
        if not openai_api_key:
            raise Exception("OpenAI API key is missing.")

        client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        return OpenAIProviderSession(
            client=client,
            model=model,
            prompt_messages=prompt_messages,
            tools=serialize_openai_tools(canonical_tools),
            recorder=recorder,
        )

    if model in ANTHROPIC_MODELS:
        if not anthropic_api_key:
            raise Exception("Anthropic API key is missing.")

        client = AsyncAnthropic(api_key=anthropic_api_key)
        return AnthropicProviderSession(
            client=client,
            model=model,
            prompt_messages=prompt_messages,
            tools=serialize_anthropic_tools(canonical_tools),
            recorder=recorder,
        )

    if model in GEMINI_MODELS:
        if not gemini_api_key:
            raise Exception("Gemini API key is missing.")

        client = genai.Client(api_key=gemini_api_key)
        return GeminiProviderSession(
            client=client,
            model=model,
            prompt_messages=prompt_messages,
            tools=serialize_gemini_tools(canonical_tools),
            recorder=recorder,
        )

    gateway = gateway_for_model(model)
    if gateway is not None:
        api_key, base_url = _resolve_gateway_credentials(
            gateway, gateway_api_keys, gateway_base_urls
        )
        # Gateways speak Chat Completions rather than the Responses API, and
        # several free vision models cannot drive tool calls, so tools are
        # offered per model instead of unconditionally.
        tools = (
            serialize_chat_completions_tools(canonical_tools)
            if supports_tools(model)
            else []
        )
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=gateway.request_headers(),
        )
        return OpenAICompatibleProviderSession(
            client=client,
            gateway=gateway,
            model=model,
            prompt_messages=prompt_messages,
            tools=tools,
            recorder=recorder,
        )

    raise ValueError(f"Unsupported model: {model.value}")
