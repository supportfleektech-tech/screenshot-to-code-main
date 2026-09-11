# pyright: reportUnknownVariableType=false
"""Provider session for OpenAI-compatible gateways (OpenRouter, NVIDIA, Kilo, Zen).

The first-party OpenAI session uses the Responses API (`client.responses`),
which these gateways do not implement; they all implement Chat Completions.
So this session reuses the same `AsyncOpenAI` client with a different `base_url`
and speaks `messages` + `tool` roles instead of Responses input items.
"""

import base64
import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderSession,
    ProviderTurn,
    StreamEvent,
)
from agent.tools import (
    CanonicalToolDefinition,
    ToolCall,
    parse_json_arguments,
)
from agent.state import ensure_str
from costs.pricing import MODEL_PRICING
from costs.token_usage import TokenUsage
from fs_logging.agent_runs import AgentRunRecorder
from fs_logging.prompt_reports import PromptReportLogger
from llm import Llm
from llm_gateways import Gateway, gateway_api_name

# The optional request fields we are willing to send but not willing to die on:
# strict validators on free-tier gateways 400 on unknown or unsupported keys, so
# a refusal that names one of these gets the field dropped and one retry.
_RECOVERABLE_FIELDS = ("stream_options", "max_tokens", "tool_choice")


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key off either a pydantic chunk or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def serialize_chat_completions_tools(
    tools: List[CanonicalToolDefinition],
) -> List[Dict[str, Any]]:
    """Canonical tools -> Chat Completions `tools` entries.

    Deliberately not "strict": the Responses-style schema rewrite would make
    every property nullable, and Chat Completions does not require it.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": copy.deepcopy(tool.parameters),
            },
        }
        for tool in tools
    ]


def _image_part(url: str) -> Dict[str, Any]:
    # Only `url` is sent: `detail` is an OpenAI-specific key and some
    # gateways reject unknown fields on content parts.
    return {"type": "image_url", "image_url": {"url": url}}


def to_chat_completions_messages(
    prompt_messages: List[ChatCompletionMessageParam],
) -> List[Dict[str, Any]]:
    """Normalize prompt messages for a Chat Completions gateway.

    Keeps text and image parts, drops anything a gateway is unlikely to accept,
    and maps the `developer` role onto `system` (older/stricter Chat
    Completions validators only know the classic roles).
    """
    messages: List[Dict[str, Any]] = []
    for message in prompt_messages:
        role = ensure_str(message.get("role", "user"))
        if role == "developer":
            role = "system"
        content = message.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": ensure_str(content)})
            continue

        parts: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text = ensure_str(part.get("text"))
                if text:
                    parts.append({"type": "text", "text": text})
            elif part_type == "image_url":
                image_url = part.get("image_url")
                url = (
                    image_url.get("url")
                    if isinstance(image_url, dict)
                    else ensure_str(image_url)
                )
                if url:
                    parts.append(_image_part(ensure_str(url)))
        messages.append({"role": role, "content": parts})
    return messages


@dataclass
class ChatCompletionParseState:
    assistant_text: str = ""
    tool_calls: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    usage: TokenUsage | None = None


def _extract_chat_usage(usage: Any) -> TokenUsage:
    """Unified token usage from a Chat Completions `usage` object.

    Cached tokens are included in `prompt_tokens` by OpenAI-compatible APIs, so
    they are subtracted to keep `input` cache-exclusive like every other
    provider here.
    """
    if usage is None:
        return TokenUsage()
    prompt_tokens = _attr(usage, "prompt_tokens", 0) or 0
    completion_tokens = _attr(usage, "completion_tokens", 0) or 0
    total_tokens = _attr(usage, "total_tokens", 0) or 0
    details = _attr(usage, "prompt_tokens_details") or {}
    cached_tokens = _attr(details, "cached_tokens", 0) or 0
    return TokenUsage(
        input=prompt_tokens - cached_tokens,
        output=completion_tokens,
        cache_read=cached_tokens,
        cache_write=0,
        total=total_tokens,
    )


async def parse_chat_completion_chunk(
    chunk: Any,
    state: ChatCompletionParseState,
    on_event: EventSink,
) -> None:
    """Fold one streamed chunk into `state`, forwarding deltas to `on_event`."""
    usage = _attr(chunk, "usage")
    if usage is not None:
        state.usage = _extract_chat_usage(usage)

    choices = _attr(chunk, "choices") or []
    if not choices:
        return

    delta = _attr(choices[0], "delta")
    if delta is None:
        return

    text = _attr(delta, "content")
    if text:
        state.assistant_text += text
        await on_event(StreamEvent(type="assistant_delta", text=ensure_str(text)))

    # Reasoning stream names differ across gateways: OpenRouter normalizes to
    # `reasoning`, DeepSeek/Qwen-style upstreams emit `reasoning_content`, and
    # some just prefix the model's thinking into `content`.
    for key in ("reasoning", "reasoning_content", "thinking"):
        thinking = _attr(delta, key)
        if thinking:
            await on_event(
                StreamEvent(type="thinking_delta", text=ensure_str(thinking))
            )
            break

    for call in _attr(delta, "tool_calls") or []:
        index = _attr(call, "index", 0)
        # Only the first chunk of a call carries its id; later deltas must not
        # replace it with the fallback.
        call_id = _attr(call, "id")
        function = _attr(call, "function") or {}
        name = _attr(function, "name")
        arguments_delta = _attr(function, "arguments")
        if not isinstance(index, int):
            index = 0

        entry = state.tool_calls.setdefault(
            index,
            {"id": call_id or f"call-{index}", "name": name or "", "arguments": ""},
        )
        if call_id:
            entry["id"] = call_id
        if name:
            entry["name"] = name
        if arguments_delta:
            entry["arguments"] += ensure_str(arguments_delta)

        await on_event(
            StreamEvent(
                type="tool_call_delta",
                tool_call_id=entry["id"],
                tool_name=entry.get("name") or None,
                tool_arguments=entry["arguments"],
            )
        )


def build_provider_turn(state: ChatCompletionParseState) -> ProviderTurn:
    tool_calls: List[ToolCall] = []
    for entry in state.tool_calls.values():
        args, error = parse_json_arguments(entry.get("arguments"))
        if error:
            args = {"INVALID_JSON": ensure_str(entry.get("arguments"))}
        tool_calls.append(
            ToolCall(
                id=entry.get("id") or f"call-{len(tool_calls)}",
                name=entry.get("name") or "unknown_tool",
                arguments=args,
            )
        )
    return ProviderTurn(
        assistant_text=state.assistant_text,
        tool_calls=tool_calls,
        assistant_turn=None,
    )


# Output caps that are documented for a gateway's model and tight enough to
# matter. Unlisted models omit `max_tokens` and inherit the gateway default,
# which beats guessing a number a given host rejects.
_MAX_OUTPUT_TOKENS: Dict[Llm, int] = {
    Llm.ZEN_BIG_PICKLE: 32000,
    Llm.KILO_AUTO: 128000,
}


def _max_output_tokens(model: Llm) -> Optional[int]:
    return _MAX_OUTPUT_TOKENS.get(model)


def _image_ref(part: Any) -> str | None:
    """A public URL is sent as-is; local bytes become a base64 data URL."""
    if part.image_url:
        return part.image_url
    if part.data is not None:
        encoded = base64.b64encode(part.data).decode("ascii")
        return f"data:{part.mime_type};base64,{encoded}"
    return None


class OpenAICompatibleProviderSession(ProviderSession):
    def __init__(
        self,
        client: AsyncOpenAI,
        gateway: Gateway,
        model: Llm,
        prompt_messages: List[ChatCompletionMessageParam],
        tools: List[Dict[str, Any]],
        recorder: Optional[AgentRunRecorder] = None,
    ):
        self._client = client
        self._gateway = gateway
        self._model = model
        self._api_name = gateway_api_name(model)
        self._tools = tools
        self._headers = gateway.request_headers()
        self._total_usage = TokenUsage()
        self._recorder = recorder
        self._prompt_report_logger = PromptReportLogger(
            provider=gateway.id,
            model=model,
            api_model_name=self._api_name,
        )
        self._messages = to_chat_completions_messages(prompt_messages)

    async def _create_stream(self, params: Dict[str, Any]) -> Any:
        """Request the stream, dropping optional fields a gateway refuses.

        Free-tier gateways share one OpenAI-shaped endpoint but disagree on
        which optional fields they accept; a 400 that names one of ours is
        treated as "don't send that", not as a hard failure.
        """
        # One attempt per recoverable field, plus the original request.
        attempts = len(_RECOVERABLE_FIELDS) + 1
        for attempt in range(attempts):
            try:
                return await self._client.chat.completions.create(  # type: ignore
                    **params,
                    extra_headers=self._headers or None,
                )
            except openai.BadRequestError as exc:
                dropped = [
                    key
                    for key in _RECOVERABLE_FIELDS
                    if key in params and key in str(exc)
                ]
                if not dropped or attempt == attempts - 1:
                    raise
                for key in dropped:
                    params.pop(key, None)
                print(
                    f"[{self._gateway.display_name}] gateway rejected "
                    f"{', '.join(dropped)}; retrying without it"
                )
        raise AssertionError("unreachable")  # pragma: no cover

    async def stream_turn(self, on_event: EventSink) -> ProviderTurn:
        params: Dict[str, Any] = {
            "model": self._api_name,
            "messages": self._messages,
            "stream": True,
        }
        if self._tools:
            params["tools"] = self._tools
            params["tool_choice"] = "auto"
        max_output_tokens = _max_output_tokens(self._model)
        if max_output_tokens is not None:
            params["max_tokens"] = max_output_tokens
        if self._gateway.send_stream_usage:
            params["stream_options"] = {"include_usage": True}

        self._prompt_report_logger.record_request(params)
        if self._recorder is not None:
            self._recorder.record_llm_request(self._gateway.id, self._api_name, params)

        state = ChatCompletionParseState()
        stream = await self._create_stream(params)
        async for chunk in stream:  # type: ignore
            await parse_chat_completion_chunk(chunk, state, on_event)

        if state.usage is not None:
            self._prompt_report_logger.record_usage(state.usage)
            self._total_usage.accumulate(state.usage)

        turn = build_provider_turn(state)
        if self._recorder is not None:
            self._recorder.record_llm_response(
                turn.assistant_text, turn.tool_calls, state.usage
            )
        return turn

    def total_cost_usd(self) -> float | None:
        pricing = MODEL_PRICING.get(self._api_name)
        if pricing is None:
            return None
        return self._total_usage.cost(pricing)

    async def append_tool_results(
        self,
        turn: ProviderTurn,
        executed_tool_calls: List[ExecutedToolCall],
    ) -> None:
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": turn.assistant_text or None,
        }
        if turn.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in turn.tool_calls
            ]
        self._messages.append(assistant_message)

        # Chat Completions `tool` messages carry text only, so rendered images
        # a tool produced ride in a follow-up user turn (every tool call is
        # answered first, which is what the API requires).
        images: List[Tuple[str, str]] = []
        for executed in executed_tool_calls:
            result_json = json.dumps(executed.result.result)
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": executed.tool_call.id,
                    "content": result_json,
                }
            )
            if not executed.result.ok:
                continue
            for part in executed.result.multimodal_parts or []:
                url = _image_ref(part)
                if url is not None:
                    images.append((part.display_name, url))

        if images:
            parts: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": "Rendered images for the tool results above.",
                }
            ]
            for display_name, url in images:
                parts.append({"type": "text", "text": display_name})
                parts.append(_image_part(url))
            self._messages.append({"role": "user", "content": parts})

    async def close(self) -> None:
        usage = self._total_usage
        pricing = MODEL_PRICING.get(self._api_name)
        cost_str = f" cost=${usage.cost(pricing):.4f}" if pricing else ""
        print(
            f"[TOKEN USAGE] provider={self._gateway.id} model={self._api_name} | "
            f"input={usage.input} output={usage.output} "
            f"cache_read={usage.cache_read} cache_write={usage.cache_write} "
            f"total={usage.total}{cost_str}"
        )
        await self._client.close()
