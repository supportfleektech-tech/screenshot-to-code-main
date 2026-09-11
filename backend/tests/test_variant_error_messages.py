"""Variant error copy must name the provider that actually failed.

A free-tier 429 and an OpenAI quota 429 need different advice, and the
first-party wording is locked so existing users' muscle memory (and the hosted
support macros) keep working.
"""

from typing import Any, List

import httpx
import openai
import pytest

from llm import Llm
from routes.generate_code import AgenticGenerationStage, _describe_provider

CREDITS_HINT = (
    " Alternatively, you can purchase code generation credits directly on this "
    "website."
)


def _error(cls: type, message: str) -> Any:
    request = httpx.Request("POST", "https://gateway.example/v1/chat/completions")
    return cls(
        message,
        response=httpx.Response(400, request=request, json={"error": message}),
        body=None,
    )


def test_first_party_labels_are_unchanged() -> None:
    assert _describe_provider(Llm.GPT_5_5_HIGH).label == "OpenAI"
    assert _describe_provider(Llm.CLAUDE_OPUS_5_HIGH).label == "Anthropic"
    assert _describe_provider(Llm.GEMINI_3_5_FLASH_HIGH).label == "Gemini"


def test_openai_messages_keep_their_original_wording() -> None:
    provider = _describe_provider(Llm.GPT_5_5_HIGH)

    assert provider.auth_error() == (
        "Incorrect OpenAI key. Please make sure your OpenAI API key is correct, "
        "or create a new OpenAI API key on your OpenAI dashboard."
    )
    assert provider.rate_limit_error() == (
        "OpenAI error - 'You exceeded your current quota, please check your plan "
        "and billing details.'"
    )
    assert (
        "https://github.com/abi/screenshot-to-code/blob/main/Troubleshooting.md"
        in provider.not_found_error("Model not found")
    )


def test_gateway_messages_point_at_the_gateway_instead_of_openai() -> None:
    provider = _describe_provider(Llm.OPENROUTER_GEMMA_4_31B_FREE)

    assert provider.label == "OpenRouter"
    assert "OpenAI" not in provider.auth_error()
    assert "OPENROUTER_API_KEY" in provider.auth_error()

    rate_limited = provider.rate_limit_error()
    assert "OpenRouter is rate-limiting" in rate_limited
    # The useful advice for a saturated :free model is to wait or use a
    # first-party key, not to go look for a billing page.
    assert "billing details" not in rate_limited

    missing = provider.not_found_error("The model `foo/bar` does not exist")
    assert "google/gemma-4-31b-it:free" in missing
    assert "openrouter.ai" in missing


def test_gateway_models_report_the_gateway_that_serves_them() -> None:
    assert _describe_provider(Llm.NVIDIA_LLAMA_3_2_90B_VISION).label == "NVIDIA NIM"
    assert _describe_provider(Llm.KILO_AUTO).env_var == "KILO_API_KEY"


def test_nvidia_omni_and_kilo_omni_are_distinct_ids() -> None:
    """The same upstream slug on two gateways must not collapse into one."""
    openrouter = _describe_provider(Llm.OPENROUTER_NEMOTRON_OMNI_30B_FREE)
    kilo = _describe_provider(Llm.KILO_NEMOTRON_OMNI_30B_FREE)

    assert (
        openrouter.model_id
        == kilo.model_id
        == ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
    )
    assert openrouter.label != kilo.label


class _RaisingAgent:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def run(self, model: Llm, prompt_messages: Any) -> str:
        raise self._error


def _stage(send_message: Any) -> AgenticGenerationStage:
    return AgenticGenerationStage(
        send_message=send_message,
        openai_api_key=None,
        openai_base_url=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        replicate_api_key=None,
        should_generate_images=False,
        file_state=None,
        asset_base_url="",
        option_codes=[],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "error", "expected_fragment"),
    [
        (
            Llm.OPENROUTER_GEMMA_4_31B_FREE,
            _error(openai.RateLimitError, "Too many requests"),
            "OpenRouter is rate-limiting this request",
        ),
        (
            Llm.GPT_5_5_HIGH,
            _error(openai.RateLimitError, "Too many requests"),
            "OpenAI error - 'You exceeded your current quota",
        ),
        (
            Llm.NVIDIA_LLAMA_3_2_90B_VISION,
            _error(openai.AuthenticationError, "bad key"),
            "NVIDIA NIM API key",
        ),
    ],
)
async def test_variant_failures_send_provider_specific_copy(
    model: Llm, error: Exception, expected_fragment: str, monkeypatch
) -> None:
    sent: List[Any] = []

    async def send_message(*args: Any) -> None:
        sent.append(args)

    monkeypatch.setattr(
        "routes.generate_code.Agent", lambda **kwargs: _RaisingAgent(error)
    )

    completion = await _stage(send_message)._run_variant(0, model, [])

    assert completion == ""
    errors = [call for call in sent if call[0] == "variantError"]
    assert len(errors) == 1
    assert errors[0][2] == 0  # variant index
    assert expected_fragment in errors[0][1]


@pytest.mark.asyncio
async def test_hosted_runs_append_the_credits_hint_for_gateway_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hosted "buy credits" escape hatch is not OpenAI-only."""
    sent: List[Any] = []

    async def send_message(*args: Any) -> None:
        sent.append(args)

    monkeypatch.setattr("routes.generate_code.IS_PROD", True)
    monkeypatch.setattr(
        "routes.generate_code.Agent",
        lambda **kwargs: _RaisingAgent(_error(openai.RateLimitError, "429")),
    )

    await _stage(send_message)._run_variant(0, Llm.OPENROUTER_AUTO_FREE, [])

    message = next(call for call in sent if call[0] == "variantError")[1]
    assert message.endswith(CREDITS_HINT.strip())
    assert "rate-limiting" in message
