"""Tests for the OpenAI-compatible gateway registry."""

import pytest

import config
from llm import (
    MODEL_PROVIDER,
    OPENAI_COMPATIBLE_MODELS,
    OPENAI_COMPATIBLE_PROVIDER_IDS,
    Llm,
)
from llm_gateways import (
    APP_ATTRIBUTION_HEADERS,
    GATEWAYS,
    GATEWAY_FOR_MODEL,
    configured_gateways,
    gateway_api_name,
    gateway_for_model,
    select_gateway_models,
    supports_tools,
    supports_vision,
)


def test_every_gateway_model_is_registered_in_the_provider_map() -> None:
    """llm.py and llm_gateways.py must not drift apart.

    MODEL_PROVIDER is what labels run logs and prompt reports, so a gateway
    model missing from it would be reported as having no provider at all.
    """
    for model in GATEWAY_FOR_MODEL:
        assert MODEL_PROVIDER[model] == GATEWAY_FOR_MODEL[model].id
    assert set(GATEWAY_FOR_MODEL) == OPENAI_COMPATIBLE_MODELS


def test_gateway_model_ids_are_derived_from_the_gateway_id() -> None:
    for model, gateway in GATEWAY_FOR_MODEL.items():
        assert model.value.startswith(f"{gateway.id}/")
        assert model.value == f"{gateway.id}/{gateway_api_name(model)}"


def test_provider_ids_in_llm_match_the_registry() -> None:
    assert set(GATEWAYS) == set(OPENAI_COMPATIBLE_PROVIDER_IDS)


def test_every_gateway_offers_at_least_one_vision_capable_model() -> None:
    """Screenshot-to-code is a vision task; a gateway with no image-capable
    model can only ever serve text prompts, which the selection logic has to
    know about explicitly rather than discover mid-run."""
    visionless = [
        gateway.id for gateway in GATEWAYS.values() if not gateway.vision_models()
    ]
    assert visionless == []


def test_openrouter_needs_attribution_headers_and_kilo_is_gateway_pathed() -> None:
    openrouter = GATEWAYS["openrouter"]
    assert openrouter.default_base_url == "https://openrouter.ai/api/v1"
    assert openrouter.request_headers() == APP_ATTRIBUTION_HEADERS
    assert "HTTP-Referer" in openrouter.request_headers()

    # Kilo's OpenAI-compatible route is not under /v1.
    kilo = GATEWAYS["kilo"]
    assert kilo.default_base_url == "https://api.kilo.ai/api/gateway"
    assert kilo.resolve_base_url().endswith("/api/gateway")


def test_base_url_prefers_the_request_then_env_then_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = GATEWAYS["nvidia"]
    assert gateway.resolve_base_url("https://proxy.example/v1/") == (
        "https://proxy.example/v1"
    )

    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    assert gateway.resolve_base_url(None) == "https://integrate.api.nvidia.com/v1"

    monkeypatch.setattr(config, "NVIDIA_BASE_URL", "https://nim.internal/v1")
    assert gateway.resolve_base_url(None) == "https://nim.internal/v1"
    assert gateway.resolve_base_url("https://dialog.example/v1") == (
        "https://dialog.example/v1"
    )


def test_api_key_prefers_the_request_then_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = GATEWAYS["kilo"]
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    monkeypatch.setattr(config, "KILO_API_KEY", None)

    assert gateway.resolve_api_key(None) is None
    assert gateway.resolve_api_key("sk-kilo-from-ui") == "sk-kilo-from-ui"

    monkeypatch.setattr(config, "KILO_API_KEY", "sk-kilo-from-env")
    assert gateway.resolve_api_key(None) == "sk-kilo-from-env"
    assert gateway.resolve_api_key("sk-kilo-from-ui") == "sk-kilo-from-ui"


def test_configured_gateways_follows_request_and_env_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENROUTER_API_KEY",
        "NVIDIA_API_KEY",
        "KILO_API_KEY",
        "ZEN_API_KEY",
        "ZENMUX_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(config, name, None)

    assert configured_gateways() == []
    assert [g.id for g in configured_gateways({"zen": "sk-zen"})] == ["zen"]
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or")
    # Registry order, not insertion order, decides who goes first.
    assert [g.id for g in configured_gateways({"zen": "sk-zen"})] == [
        "openrouter",
        "zen",
    ]


def test_image_selection_drops_text_only_models() -> None:
    gateways = [GATEWAYS["openrouter"], GATEWAYS["kilo"], GATEWAYS["zen"]]

    image_models = select_gateway_models(gateways, vision_required=True, limit=99)
    text_models = select_gateway_models(gateways, vision_required=False, limit=99)

    assert Llm.ZEN_BIG_PICKLE not in image_models
    assert Llm.KILO_MINIMAX_M2_5_FREE not in image_models
    assert Llm.ZEN_BIG_PICKLE in text_models
    assert all(supports_vision(model) for model in image_models)


def test_tool_capable_models_lead_models_that_need_plain_output() -> None:
    models = select_gateway_models([GATEWAYS["nvidia"]], vision_required=True, limit=4)
    assert all(not supports_tools(model) for model in models)

    models = select_gateway_models(
        [GATEWAYS["openrouter"]], vision_required=True, limit=4
    )
    assert models[0] == Llm.OPENROUTER_GEMMA_4_31B_FREE
    # The 12B VL model is the only non-tool-caller in the OpenRouter set, so it
    # is last.
    assert models[-1] == Llm.OPENROUTER_NEMOTRON_NANO_12B_VL_FREE


def test_selection_respects_the_variant_limit() -> None:
    models = select_gateway_models(
        [GATEWAYS["openrouter"], GATEWAYS["kilo"]], vision_required=True, limit=2
    )
    assert len(models) == 2


def test_gateway_api_name_rejects_first_party_models() -> None:
    assert gateway_for_model(Llm.GPT_5_5_HIGH) is None
    with pytest.raises(ValueError, match="not served by a registered gateway"):
        gateway_api_name(Llm.GPT_5_5_HIGH)
    # First-party models default to capable, since they all accept images and
    # tool calls here.
    assert supports_vision(Llm.GPT_5_5_HIGH) is True
    assert supports_tools(Llm.CLAUDE_OPUS_5_HIGH) is True
