from unittest.mock import AsyncMock

import pytest

from routes.generate_code import ParameterExtractionStage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_value", "expected"),
    [(None, True), (False, False), (True, True)],
)
async def test_asset_extraction_preference_defaults_on_and_supports_opt_out(
    request_value: bool | None, expected: bool
) -> None:
    stage = ParameterExtractionStage(AsyncMock())
    params: dict[str, object] = {
        "generatedCodeConfig": "html_tailwind",
        "inputMode": "text",
        "prompt": {"text": "hello"},
    }
    if request_value is not None:
        params["isAssetExtractionEnabled"] = request_value

    extracted = await stage.extract_and_validate(params)

    assert extracted.should_extract_assets is expected


@pytest.mark.asyncio
async def test_extracts_gemini_api_key_from_settings_dialog() -> None:
    stage = ParameterExtractionStage(AsyncMock())

    extracted = await stage.extract_and_validate(
        {
            "generatedCodeConfig": "html_tailwind",
            "inputMode": "text",
            "openAiApiKey": "",
            "anthropicApiKey": "",
            "geminiApiKey": "gemini-from-ui",
            "prompt": {"text": "hello"},
        }
    )

    assert extracted.gemini_api_key == "gemini-from-ui"


@pytest.mark.asyncio
async def test_extracts_gemini_api_key_from_env_when_not_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("routes.generate_code.GEMINI_API_KEY", "gemini-from-env")
    stage = ParameterExtractionStage(AsyncMock())

    extracted = await stage.extract_and_validate(
        {
            "generatedCodeConfig": "html_tailwind",
            "inputMode": "text",
            "prompt": {"text": "hello"},
        }
    )

    assert extracted.gemini_api_key == "gemini-from-env"


@pytest.mark.asyncio
async def test_extracts_replicate_api_key_from_settings_dialog() -> None:
    stage = ParameterExtractionStage(AsyncMock())

    extracted = await stage.extract_and_validate(
        {
            "generatedCodeConfig": "html_tailwind",
            "inputMode": "text",
            "replicateApiKey": "replicate-from-ui",
            "prompt": {"text": "hello"},
        }
    )

    assert extracted.replicate_api_key == "replicate-from-ui"


@pytest.mark.asyncio
async def test_extracts_replicate_api_key_from_env_when_not_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("routes.generate_code.REPLICATE_API_KEY", "replicate-from-env")
    stage = ParameterExtractionStage(AsyncMock())

    extracted = await stage.extract_and_validate(
        {
            "generatedCodeConfig": "html_tailwind",
            "inputMode": "text",
            "prompt": {"text": "hello"},
        }
    )

    assert extracted.replicate_api_key == "replicate-from-env"


@pytest.mark.asyncio
async def test_extracts_design_system_from_request() -> None:
    stage = ParameterExtractionStage(AsyncMock())

    extracted = await stage.extract_and_validate(
        {
            "generatedCodeConfig": "html_css",
            "inputMode": "text",
            "prompt": {"text": "hello"},
            "designSystem": "  Reuse .mockup-frame  ",
        }
    )

    assert extracted.design_system == "Reuse .mockup-frame"


@pytest.mark.asyncio
async def test_extracts_gateway_keys_and_base_urls_from_settings_dialog() -> None:
    stage = ParameterExtractionStage(AsyncMock())

    extracted = await stage.extract_and_validate(
        {
            "generatedCodeConfig": "html_tailwind",
            "inputMode": "image",
            "prompt": {"text": "recreate this"},
            "openRouterApiKey": "sk-or-from-ui",
            "openRouterBaseUrl": "https://or.internal/v1",
            "nvidiaApiKey": "nvapi-from-ui",
            "kiloApiKey": "",
            "zenApiKey": "  ",
        }
    )

    # Blank/whitespace values are treated as "not set" so an empty settings
    # field cannot shadow a key in backend/.env.
    assert extracted.gateway_api_keys == {
        "openrouter": "sk-or-from-ui",
        "nvidia": "nvapi-from-ui",
    }
    assert extracted.gateway_base_urls == {"openrouter": "https://or.internal/v1"}


@pytest.mark.asyncio
async def test_gateway_base_url_overrides_are_dropped_in_prod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("routes.generate_code.IS_PROD", True)
    stage = ParameterExtractionStage(AsyncMock())

    extracted = await stage.extract_and_validate(
        {
            "generatedCodeConfig": "html_tailwind",
            "inputMode": "text",
            "prompt": {"text": "hello"},
            "openRouterApiKey": "sk-or-from-ui",
            "openRouterBaseUrl": "https://attacker.example/v1",
        }
    )

    assert extracted.gateway_api_keys == {"openrouter": "sk-or-from-ui"}
    assert extracted.gateway_base_urls == {}


@pytest.mark.asyncio
async def test_gateway_fields_default_to_empty_for_older_clients() -> None:
    extracted = await ParameterExtractionStage(AsyncMock()).extract_and_validate(
        {
            "generatedCodeConfig": "html_tailwind",
            "inputMode": "text",
            "prompt": {"text": "hello"},
        }
    )

    assert extracted.gateway_api_keys == {}
    assert extracted.gateway_base_urls == {}
