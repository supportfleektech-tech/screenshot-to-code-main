"""Tests for the Chat Completions session used by OpenAI-compatible gateways."""

import copy
from types import SimpleNamespace
from typing import Any, Dict, List, cast

import httpx
import openai
import pytest

from agent.providers.openai_compatible import (
    ChatCompletionParseState,
    OpenAICompatibleProviderSession,
    build_provider_turn,
    parse_chat_completion_chunk,
    serialize_chat_completions_tools,
    to_chat_completions_messages,
)
from agent.tools import CanonicalToolDefinition
from openai.types.chat import ChatCompletionMessageParam
from agent.tools.types import ToolCall, ToolExecutionResult, ToolMultimodalPart
from agent.providers.base import ExecutedToolCall, StreamEvent
from llm import Llm
from llm_gateways import GATEWAYS

OPENROUTER = GATEWAYS["openrouter"]
NVIDIA = GATEWAYS["nvidia"]


# --------------------------------------------------------------------------
# helpers


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_calls: List[Dict[str, Any]] | None = None,
    usage: Dict[str, Any] | None = None,
    finish: bool = False,
) -> SimpleNamespace:
    """Shape a streamed chunk like the OpenAI SDK's pydantic events."""
    delta: Dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    if tool_calls is not None:
        delta["tool_calls"] = [
            SimpleNamespace(
                index=call.get("index", 0),
                id=call.get("id"),
                function=SimpleNamespace(
                    name=call.get("name"),
                    arguments=call.get("arguments"),
                ),
            )
            for call in tool_calls
        ]
    choice = SimpleNamespace(
        delta=SimpleNamespace(**delta) if delta else None,
        finish_reason="stop" if finish else None,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(**usage) if usage else None,
    )


class _FakeStream:
    def __init__(self, chunks: List[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> Any:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeChatCompletions:
    def __init__(self, responses: List[Any]) -> None:
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(copy.deepcopy(kwargs))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeClient:
    def __init__(self, responses: List[Any]) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(responses))
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _bad_request(message: str) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://gateway.example/v1/chat/completions")
    return openai.BadRequestError(
        message,
        response=httpx.Response(400, request=request, json={"error": message}),
        body=None,
    )


def _session(
    client: Any,
    model: Llm = Llm.OPENROUTER_GEMMA_4_31B_FREE,
    tools: List[Dict[str, Any]] | None = None,
    gateway: Any = OPENROUTER,
) -> OpenAICompatibleProviderSession:
    return OpenAICompatibleProviderSession(
        client=client,  # type: ignore[arg-type]
        gateway=gateway,  # type: ignore[arg-type]
        model=model,
        prompt_messages=[{"role": "user", "content": "Build this page."}],
        tools=tools or [],
    )


async def _collect(events: List[StreamEvent]) -> Any:
    async def sink(event: StreamEvent) -> None:
        events.append(event)

    return sink


def _tool_def() -> CanonicalToolDefinition:
    return CanonicalToolDefinition(
        name="create_file",
        description="Write the file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["content"],
        },
    )


# --------------------------------------------------------------------------
# chunk parsing


@pytest.mark.asyncio
async def test_content_deltas_accumulate_and_stream() -> None:
    state = ChatCompletionParseState()
    events: List[StreamEvent] = []
    sink = await _collect(events)

    await parse_chat_completion_chunk(_chunk(content="<html"), state, sink)
    await parse_chat_completion_chunk(_chunk(content=">"), state, sink)

    assert state.assistant_text == "<html>"
    assert [event.text for event in events] == ["<html", ">"]
    assert all(event.type == "assistant_delta" for event in events)


@pytest.mark.asyncio
async def test_reasoning_deltas_are_reported_as_thinking() -> None:
    for key in ("reasoning", "reasoning_content"):
        state = ChatCompletionParseState()
        events: List[StreamEvent] = []
        sink = await _collect(events)
        chunk = _chunk(reasoning="checking the layout")
        if key == "reasoning_content":
            chunk.choices[0].delta = SimpleNamespace(reasoning_content="checking the layout")  # type: ignore[union-attr]

        await parse_chat_completion_chunk(chunk, state, sink)

        assert [event.type for event in events] == ["thinking_delta"]
        assert events[0].text == "checking the layout"


@pytest.mark.asyncio
async def test_tool_call_arguments_accumulate_across_chunks() -> None:
    state = ChatCompletionParseState()
    events: List[StreamEvent] = []
    sink = await _collect(events)

    await parse_chat_completion_chunk(
        _chunk(
            tool_calls=[
                {"index": 0, "id": "call_1", "name": "create_file", "arguments": ""}
            ]
        ),
        state,
        sink,
    )
    await parse_chat_completion_chunk(
        _chunk(tool_calls=[{"index": 0, "arguments": '{"path":'}]),
        state,
        sink,
    )
    await parse_chat_completion_chunk(
        _chunk(tool_calls=[{"index": 0, "arguments": ' "index.html"}'}]),
        state,
        sink,
    )

    turn = build_provider_turn(state)
    assert turn.tool_calls == [
        ToolCall(id="call_1", name="create_file", arguments={"path": "index.html"})
    ]
    # The engine streams a live preview off each delta, so every event must
    # carry the arguments accumulated so far, not just the newest fragment.
    assert [event.tool_arguments for event in events] == [
        "",
        '{"path":',
        '{"path": "index.html"}',
    ]
    assert all(event.tool_call_id == "call_1" for event in events)


@pytest.mark.asyncio
async def test_truncated_tool_arguments_become_invalid_json_payload() -> None:
    state = ChatCompletionParseState()
    await parse_chat_completion_chunk(
        _chunk(
            tool_calls=[
                {"index": 0, "id": "c", "name": "create_file", "arguments": '{"cont'}
            ]
        ),
        state,
        (await _collect([])),
    )

    turn = build_provider_turn(state)
    assert turn.tool_calls[0].arguments == {"INVALID_JSON": '{"cont'}


@pytest.mark.asyncio
async def test_usage_only_chunk_is_captured_without_a_choice() -> None:
    state = ChatCompletionParseState()
    events: List[StreamEvent] = []

    await parse_chat_completion_chunk(
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=400),
            ),
        ),
        state,
        await _collect(events),
    )

    assert events == []
    assert state.usage is not None
    # Cached tokens are subtracted so `input` stays cache-exclusive.
    assert state.usage.input == 600
    assert state.usage.cache_read == 400
    assert state.usage.output == 200


# --------------------------------------------------------------------------
# request shaping


def test_messages_keep_text_and_images_but_drop_openai_only_keys() -> None:
    prompt_messages = cast(
        List[ChatCompletionMessageParam],
        [
            {"role": "system", "content": "You build HTML."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "recreate this"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AAA",
                            "detail": "original",
                        },
                    },
                    {"type": "other", "text": "ignored"},
                ],
            },
        ],
    )

    messages = to_chat_completions_messages(prompt_messages)

    assert messages[0] == {"role": "system", "content": "You build HTML."}
    assert messages[1]["content"] == [
        {"type": "text", "text": "recreate this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]


def test_developer_role_maps_to_system() -> None:
    messages = to_chat_completions_messages(
        cast(
            List[ChatCompletionMessageParam],
            [{"role": "developer", "content": "think step by step"}],
        )
    )
    assert messages == [{"role": "system", "content": "think step by step"}]


def test_tools_use_the_nested_chat_completions_shape() -> None:
    definition = _tool_def()
    tools = serialize_chat_completions_tools([definition])

    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Write the file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["content"],
                },
            },
        }
    ]
    # Responses-style strict schemas would null out every property type.
    assert "strict" not in tools[0]["function"]
    # Deep-copied, so a gateway cannot mutate the shared tool definitions.
    assert tools[0]["function"]["parameters"] is not definition.parameters


@pytest.mark.asyncio
async def test_stream_request_includes_usage_and_gateway_defaults() -> None:
    client = _FakeClient([_FakeStream([_chunk(content="ok", finish=True)])])
    session = _session(client, tools=serialize_chat_completions_tools([_tool_def()]))

    turn = await session.stream_turn(await _collect([]))

    params = client.chat.completions.calls[0]
    assert turn.assistant_text == "ok"
    assert params["model"] == "google/gemma-4-31b-it:free"
    assert params["stream"] is True
    assert params["tool_choice"] == "auto"
    assert params["stream_options"] == {"include_usage": True}
    # OpenRouter attribution rides on the per-request headers.
    assert params["extra_headers"]["HTTP-Referer"] == (
        "https://github.com/abi/screenshot-to-code"
    )
    # The bare slug goes on the wire, not the `<gateway>/<id>` enum value.
    assert params["messages"][-1]["content"] == "Build this page."


@pytest.mark.asyncio
async def test_no_tools_means_no_tool_choice() -> None:
    client = _FakeClient([_FakeStream([_chunk(content="<html>", finish=True)])])
    session = _session(client, tools=[])

    await session.stream_turn(await _collect([]))

    params = client.chat.completions.calls[0]
    assert "tools" not in params
    assert "tool_choice" not in params


@pytest.mark.asyncio
async def test_gateway_rejection_of_an_optional_field_is_retried_without_it() -> None:
    client = _FakeClient(
        [
            _bad_request("Invalid request body: Unknown parameter: 'stream_options'"),
            _FakeStream([_chunk(content="ok", finish=True)]),
        ]
    )
    session = _session(client)

    turn = await session.stream_turn(await _collect([]))

    assert turn.assistant_text == "ok"
    assert len(client.chat.completions.calls) == 2
    assert "stream_options" in client.chat.completions.calls[0]
    assert "stream_options" not in client.chat.completions.calls[1]


@pytest.mark.asyncio
async def test_unrelated_bad_request_is_not_swallowed() -> None:
    client = _FakeClient([_bad_request("model not found")])
    session = _session(client)

    with pytest.raises(openai.BadRequestError):
        await session.stream_turn(await _collect([]))
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_documented_output_caps_are_sent_and_omitted_otherwise() -> None:
    client = _FakeClient([_FakeStream([_chunk(content="ok", finish=True)])])
    await _session(
        client, model=Llm.ZEN_BIG_PICKLE, gateway=GATEWAYS["zen"]
    ).stream_turn(await _collect([]))
    assert client.chat.completions.calls[0]["max_tokens"] == 32000

    client = _FakeClient([_FakeStream([_chunk(content="ok", finish=True)])])
    await _session(client).stream_turn(await _collect([]))
    assert "max_tokens" not in client.chat.completions.calls[0]


# --------------------------------------------------------------------------
# conversation continuation


@pytest.mark.asyncio
async def test_tool_results_append_assistant_and_tool_messages() -> None:
    client = _FakeClient([_FakeStream([])])
    session = _session(client)
    turn = build_provider_turn(
        await _state_with(
            content="working on it",
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_9",
                    "name": "create_file",
                    "arguments": '{"content": "<html></html>"}',
                }
            ],
        )
    )

    await session.append_tool_results(
        turn,
        [
            ExecutedToolCall(
                tool_call=ToolCall(
                    id="call_9", name="create_file", arguments={"content": "x"}
                ),
                result=ToolExecutionResult(
                    ok=True, result={"path": "index.html"}, summary={}
                ),
            )
        ],
    )

    messages = session._messages  # noqa: SLF001 - asserting on the request body
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"] == [
        {
            "id": "call_9",
            "type": "function",
            "function": {
                "name": "create_file",
                "arguments": '{"content": "<html></html>"}',
            },
        }
    ]
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_9",
        "content": '{"path": "index.html"}',
    }


@pytest.mark.asyncio
async def test_tool_images_follow_in_a_user_turn() -> None:
    """Chat `tool` messages carry text only, so rendered screenshots ride along
    in a follow-up user message after every call is answered."""
    client = _FakeClient([_FakeStream([])])
    session = _session(client)
    turn = build_provider_turn(
        await _state_with(
            tool_calls=[
                {
                    "index": 0,
                    "id": "a",
                    "name": "screenshot_preview",
                    "arguments": "{}",
                },
                {
                    "index": 1,
                    "id": "b",
                    "name": "screenshot_preview",
                    "arguments": "{}",
                },
            ]
        )
    )

    await session.append_tool_results(
        turn,
        [
            ExecutedToolCall(
                tool_call=ToolCall(id="a", name="screenshot_preview", arguments={}),
                result=ToolExecutionResult(
                    ok=True,
                    result={"status": "ok"},
                    summary={},
                    multimodal_parts=[
                        ToolMultimodalPart(
                            display_name="shot.png",
                            mime_type="image/png",
                            data=b"png-bytes",
                        )
                    ],
                ),
            ),
            ExecutedToolCall(
                tool_call=ToolCall(id="b", name="screenshot_preview", arguments={}),
                result=ToolExecutionResult(
                    ok=False,
                    result={"error": "chromium missing"},
                    summary={},
                    multimodal_parts=None,
                ),
            ),
        ],
    )

    messages = session._messages  # noqa: SLF001
    assert [message["role"] for message in messages[1:]] == [
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    image_parts = messages[-1]["content"]
    assert image_parts[0] == {
        "type": "text",
        "text": "Rendered images for the tool results above.",
    }
    assert image_parts[2]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_gateway_models_are_unpriced_so_budget_checks_stay_disabled() -> None:
    client = _FakeClient(
        [
            _FakeStream(
                [
                    _chunk(content="<html></html>", finish=True),
                    _chunk(usage={"prompt_tokens": 10, "completion_tokens": 2}),
                ]
            )
        ]
    )
    session = _session(client)

    await session.stream_turn(await _collect([]))

    assert session.total_cost_usd() is None
    await session.close()
    assert client.closed is True


@pytest.mark.asyncio
async def test_nvidia_gateway_sends_no_attribution_headers() -> None:
    client = _FakeClient([_FakeStream([_chunk(content="ok", finish=True)])])
    session = _session(client, model=Llm.NVIDIA_LLAMA_3_2_90B_VISION, gateway=NVIDIA)

    await session.stream_turn(await _collect([]))

    assert client.chat.completions.calls[0]["extra_headers"] is None


async def _state_with(
    content: str = "", tool_calls: List[Dict[str, Any]] | None = None
) -> ChatCompletionParseState:
    state = ChatCompletionParseState()
    await parse_chat_completion_chunk(
        _chunk(content=content or None, tool_calls=tool_calls),
        state,
        await _collect([]),
    )
    return state
