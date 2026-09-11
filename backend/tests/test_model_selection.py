import pytest
from unittest.mock import AsyncMock
from routes.generate_code import ModelSelectionStage
from llm import Llm


class TestModelSelectionAllKeys:
    """Test model selection when Gemini, Anthropic, and OpenAI API keys are present."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_throw_error = AsyncMock()
        self.model_selector = ModelSelectionStage(mock_throw_error)

    @pytest.mark.asyncio
    async def test_gemini_anthropic_create(self):
        """All keys text create: fixed order for four variants."""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="text",
            openai_api_key="key",
            anthropic_api_key="key",
            gemini_api_key="key",
        )

        expected = [
            Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
            Llm.GPT_5_6_SOL_HIGH,
            Llm.CLAUDE_OPUS_5_HIGH,
            Llm.GEMINI_3_1_PRO_PREVIEW_LOW,
        ]
        assert models == expected

    @pytest.mark.asyncio
    async def test_gemini_anthropic_create_image(self):
        """All keys image create: fixed order for four variants."""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="image",
            openai_api_key="key",
            anthropic_api_key="key",
            gemini_api_key="key",
        )

        expected = [
            Llm.CLAUDE_OPUS_5_MEDIUM,
            Llm.GEMINI_3_FLASH_PREVIEW_HIGH,
            Llm.GEMINI_3_1_PRO_PREVIEW_HIGH,
            Llm.GPT_5_6_SOL_MAX,
        ]
        assert models == expected

    @pytest.mark.asyncio
    async def test_gemini_anthropic_update_text(self):
        """All keys text update: uses two fast edit variants."""
        models = await self.model_selector.select_models(
            generation_type="update",
            input_mode="text",
            openai_api_key="key",
            anthropic_api_key="key",
            gemini_api_key="key",
        )

        expected = [
            Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
            Llm.GPT_5_6_TERRA_LOW,
        ]
        assert models == expected

    @pytest.mark.asyncio
    async def test_gemini_anthropic_update(self):
        """All keys image update: uses two fast edit variants."""
        models = await self.model_selector.select_models(
            generation_type="update",
            input_mode="image",
            openai_api_key="key",
            anthropic_api_key="key",
            gemini_api_key="key",
        )

        expected = [
            Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
            Llm.GPT_5_6_TERRA_LOW,
        ]
        assert models == expected

    @pytest.mark.asyncio
    async def test_video_create_prefers_gemini_minimal_then_3_1_high(self):
        """Video create always uses two Gemini variants in fixed order."""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="video",
            openai_api_key="key",
            anthropic_api_key="key",
            gemini_api_key="key",
        )

        expected = [
            Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
            Llm.GEMINI_3_1_PRO_PREVIEW_HIGH,
        ]
        assert models == expected

    @pytest.mark.asyncio
    async def test_video_update_prefers_gemini_minimal_then_3_1_high(self):
        """Video update always uses the same two Gemini variants as video create."""
        models = await self.model_selector.select_models(
            generation_type="update",
            input_mode="video",
            openai_api_key="key",
            anthropic_api_key="key",
            gemini_api_key="key",
        )

        expected = [
            Llm.GEMINI_3_FLASH_PREVIEW_MINIMAL,
            Llm.GEMINI_3_1_PRO_PREVIEW_HIGH,
        ]
        assert models == expected


class TestModelSelectionOpenAIAnthropic:
    """Test model selection when only OpenAI and Anthropic keys are present."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_throw_error = AsyncMock()
        self.model_selector = ModelSelectionStage(mock_throw_error)

    @pytest.mark.asyncio
    async def test_openai_anthropic(self):
        """OpenAI + Anthropic: Claude Opus 4.8 medium, GPT 5.5 high, GPT 5.5 low, cycling"""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="text",
            openai_api_key="key",
            anthropic_api_key="key",
            gemini_api_key=None,
        )

        expected = [
            Llm.CLAUDE_OPUS_4_8_MEDIUM,
            Llm.GPT_5_5_HIGH,
            Llm.GPT_5_5_LOW,
            Llm.CLAUDE_OPUS_4_8_MEDIUM,
        ]
        assert models == expected


class TestModelSelectionAnthropicOnly:
    """Test model selection when only Anthropic key is present."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_throw_error = AsyncMock()
        self.model_selector = ModelSelectionStage(mock_throw_error)

    @pytest.mark.asyncio
    async def test_anthropic_only(self):
        """Anthropic only: Claude Opus 4.8 medium and Claude Sonnet 4.6 cycling"""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="text",
            openai_api_key=None,
            anthropic_api_key="key",
            gemini_api_key=None,
        )

        expected = [
            Llm.CLAUDE_OPUS_4_8_MEDIUM,
            Llm.CLAUDE_SONNET_4_6,
            Llm.CLAUDE_OPUS_4_8_MEDIUM,
            Llm.CLAUDE_SONNET_4_6,
        ]
        assert models == expected


class TestModelSelectionOpenAIOnly:
    """Test model selection when only OpenAI key is present."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_throw_error = AsyncMock()
        self.model_selector = ModelSelectionStage(mock_throw_error)

    @pytest.mark.asyncio
    async def test_openai_only(self):
        """OpenAI only: GPT 5.5 high and low, cycling"""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="text",
            openai_api_key="key",
            anthropic_api_key=None,
            gemini_api_key=None,
        )

        expected = [
            Llm.GPT_5_5_HIGH,
            Llm.GPT_5_5_LOW,
            Llm.GPT_5_5_HIGH,
            Llm.GPT_5_5_LOW,
        ]
        assert models == expected


class TestModelSelectionNoKeys:
    """Test model selection when no API keys are present."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_throw_error = AsyncMock()
        self.model_selector = ModelSelectionStage(mock_throw_error)

    @pytest.mark.asyncio
    async def test_no_keys_raises_error(self):
        """No keys: Should raise an exception"""
        with pytest.raises(Exception, match="No API key"):
            await self.model_selector.select_models(
                generation_type="create",
                input_mode="text",
                openai_api_key=None,
                anthropic_api_key=None,
                gemini_api_key=None,
            )


class TestModelSelectionGatewayOnly:
    """A key for an OpenAI-compatible gateway is enough to generate."""

    def setup_method(self):
        self.throw_error = AsyncMock()
        self.model_selector = ModelSelectionStage(self.throw_error)

    @pytest.mark.asyncio
    async def test_openrouter_key_covers_all_variants(self):
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="image",
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            gateway_api_keys={"openrouter": "sk-or-test"},
        )

        # Tool-capable models lead; the tool-less VL model lands last.
        assert models == [
            Llm.OPENROUTER_GEMMA_4_31B_FREE,
            Llm.OPENROUTER_NEMOTRON_OMNI_30B_FREE,
            Llm.OPENROUTER_AUTO_FREE,
            Llm.OPENROUTER_NEMOTRON_NANO_12B_VL_FREE,
        ]

    @pytest.mark.asyncio
    async def test_update_narrows_to_two_variants(self):
        models = await self.model_selector.select_models(
            generation_type="update",
            input_mode="image",
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            gateway_api_keys={"nvidia": "nvapi-test"},
        )

        assert models == [
            Llm.NVIDIA_LLAMA_3_2_90B_VISION,
            Llm.NVIDIA_LLAMA_3_2_11B_VISION,
        ]

    @pytest.mark.asyncio
    async def test_text_only_models_are_kept_for_text_prompts(self):
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="text",
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            gateway_api_keys={"zen": "sk-zen-test"},
        )

        assert set(models) == {Llm.ZEN_GPT_5_NANO, Llm.ZEN_BIG_PICKLE}

    @pytest.mark.asyncio
    async def test_screenshots_drop_models_that_cannot_see_them(self):
        """Kilo's minimax and Zen's big-pickle are text-only: with a screenshot
        in the request they would hallucinate a layout from the prompt alone."""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="image",
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            gateway_api_keys={"kilo": "sk-kilo-test", "zen": "sk-zen-test"},
        )

        assert Llm.KILO_MINIMAX_M2_5_FREE not in models
        assert Llm.ZEN_BIG_PICKLE not in models
        # Three usable models across two gateways, cycled to four variants.
        assert models == [
            Llm.KILO_NEMOTRON_OMNI_30B_FREE,
            Llm.KILO_AUTO,
            Llm.ZEN_GPT_5_NANO,
            Llm.KILO_NEMOTRON_OMNI_30B_FREE,
        ]

    @pytest.mark.asyncio
    async def test_frontier_keys_still_win_over_gateways(self):
        """Free-tier gateways are the fallback, never a downgrade of a run that
        has a first-party key."""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="text",
            openai_api_key="sk-openai",
            anthropic_api_key=None,
            gemini_api_key=None,
            gateway_api_keys={"openrouter": "sk-or-test"},
        )

        assert models == [
            Llm.GPT_5_5_HIGH,
            Llm.GPT_5_5_LOW,
            Llm.GPT_5_5_HIGH,
            Llm.GPT_5_5_LOW,
        ]

    @pytest.mark.asyncio
    async def test_unusable_gateway_selection_shows_its_own_message(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The actionable "no vision-capable model" error must not be masked by
        the generic 'add an API key' message."""

        def empty(*args: object, **kwargs: object) -> list[Llm]:
            return []

        monkeypatch.setattr("routes.generate_code.select_gateway_models", empty)

        with pytest.raises(Exception, match="accept image input"):
            await self.model_selector.select_models(
                generation_type="create",
                input_mode="image",
                openai_api_key=None,
                anthropic_api_key=None,
                gemini_api_key=None,
                gateway_api_keys={"zen": "sk-zen-test"},
            )

        assert self.throw_error.await_args is not None
        message = self.throw_error.await_args.args[0]
        assert "accept image input" in message
        # Not the generic "add a key" text, which would send the user to look
        # for a key they already have.
        assert "OpenAI-compatible gateway keys" not in message

    @pytest.mark.asyncio
    async def test_no_frontier_and_no_gateway_key_still_raises(self):
        with pytest.raises(Exception, match="No API key"):
            await self.model_selector.select_models(
                generation_type="create",
                input_mode="text",
                openai_api_key=None,
                anthropic_api_key=None,
                gemini_api_key=None,
                gateway_api_keys={},
            )
