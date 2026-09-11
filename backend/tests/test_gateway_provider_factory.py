"""Gateway dispatch in the provider factory: credentials, tools, and keys."""

import pytest

import config
from agent.providers.factory import create_provider_session
from agent.providers.openai_compatible import OpenAICompatibleProviderSession
from agent.providers.openai import OpenAIProviderSession
from llm import Llm

GATEWAY_KWARGS = dict(
    should_generate_images=False,
    openai_api_key=None,
    openai_base_url=None,
    anthropic_api_key=None,
    gemini_api_key=None,
    replicate_api_key=None,
)


def _raw_session(**overrides: object) -> object:
    kwargs: dict[str, object] = {
        "prompt_messages": [{"role": "user", "content": "hi"}],
        **GATEWAY_KWARGS,
        **overrides,
    }
    return create_provider_session(**kwargs)  # type: ignore[arg-type]


def _session(**overrides: object) -> OpenAICompatibleProviderSession:
    session = _raw_session(**overrides)
    assert isinstance(session, OpenAICompatibleProviderSession)
    return session


def _clear_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENROUTER_API_KEY",
        "NVIDIA_API_KEY",
        "KILO_API_KEY",
        "ZEN_API_KEY",
        "ZENMUX_API_KEY",
        "OPENROUTER_BASE_URL",
        "NVIDIA_BASE_URL",
        "KILO_BASE_URL",
        "ZEN_BASE_URL",
        "ZENMUX_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(config, name, None, raising=False)


def test_gateway_model_uses_the_chat_completions_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gateway_env(monkeypatch)

    session = _session(
        model=Llm.OPENROUTER_GEMMA_4_31B_FREE,
        gateway_api_keys={"openrouter": "sk-or-from-ui"},
    )

    assert isinstance(session, OpenAICompatibleProviderSession)
    client = session._client
    assert str(client.base_url) == "https://openrouter.ai/api/v1/"
    assert client.api_key == "sk-or-from-ui"
    assert client.default_headers["X-Title"] == "screenshot-to-code"


def test_env_key_is_used_when_the_dialog_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gateway_env(monkeypatch)
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-from-env")
    monkeypatch.setattr(config, "NVIDIA_BASE_URL", "https://nim.internal:8000/v1")

    session = _session(model=Llm.NVIDIA_LLAMA_3_2_90B_VISION)

    client = session._client
    assert str(client.base_url) == "https://nim.internal:8000/v1/"


def test_missing_gateway_key_names_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gateway_env(monkeypatch)

    with pytest.raises(Exception, match="NVIDIA NIM API key is missing"):
        _session(model=Llm.NVIDIA_LLAMA_3_2_11B_VISION)


def test_gateway_base_url_override_is_ignored_in_prod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_gateway_env(monkeypatch)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-from-env")
    monkeypatch.setattr("agent.providers.factory.IS_PROD", True)
    monkeypatch.setattr(config, "OPENROUTER_BASE_URL", "https://attacker.example/v1")

    session = _session(model=Llm.OPENROUTER_AUTO_FREE)

    client = session._client
    assert str(client.base_url) == "https://openrouter.ai/api/v1/"


def test_models_without_tool_support_get_no_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NVIDIA vision builds answer in plain text; offering them tools would
    burn the turn on a tool call they cannot emit."""
    _clear_gateway_env(monkeypatch)
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-x")

    session = _session(model=Llm.NVIDIA_LLAMA_3_2_90B_VISION)
    assert session._tools == []

    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-x")
    session = _session(model=Llm.OPENROUTER_GEMMA_4_31B_FREE)
    assert session._tools
    assert {tool["type"] for tool in session._tools} == {"function"}


def test_first_party_models_keep_using_their_own_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gateway key must not silently reroute a GPT request."""
    _clear_gateway_env(monkeypatch)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-x")

    session = _raw_session(model=Llm.GPT_5_5_HIGH, openai_api_key="sk-openai")

    assert isinstance(session, OpenAIProviderSession)
