"""LangChain standard integration tests for CopilotChatModel.

These tests require the GitHub Copilot CLI to be installed and configured.
They are marked with @pytest.mark.integration and excluded from regular test runs.

To run these tests:
    uv run pytest tests/integration_tests/ -v -m integration
"""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_copilot import CopilotChatModel


class TestCopilotChatModelIntegration(ChatModelIntegrationTests):
    """Standard integration tests for CopilotChatModel.

    This class implements LangChain's standard test suite for chat models.
    Tests are automatically discovered and run by pytest.

    The test suite validates that CopilotChatModel correctly implements
    the BaseChatModel interface and behaves as expected in real-world usage.
    """

    @property
    def chat_model_class(self) -> Type[CopilotChatModel]:
        """Return the chat model class to test."""
        return CopilotChatModel

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the model.

        These parameters will be used to instantiate CopilotChatModel
        for each test. Using gpt-4.1 with temperature=0 for deterministic
        responses in tests.
        """
        return {
            "model_name": "gpt-4.1",
            "temperature": 0,
        }

    # Feature support flags
    # Most features are disabled by default - override to enable specific tests

    @property
    def has_tool_calling(self) -> bool:
        """CopilotChatModel supports tool calling via bind_tools."""
        return True

    @property
    def has_tool_choice(self) -> bool:
        """CopilotChatModel does not support tool choice (yet)."""
        return False

    @property
    def has_structured_output(self) -> bool:
        """CopilotChatModel does not support structured output (yet)."""
        return False

    @property
    def supports_json_mode(self) -> bool:
        """CopilotChatModel does not support JSON mode (yet)."""
        return False

    @property
    def supports_image_inputs(self) -> bool:
        """CopilotChatModel supports base64 and data URL image inputs."""
        return True

    @property
    def supports_image_urls(self) -> bool:
        """CopilotChatModel does not support remote image URLs yet."""
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        """CopilotChatModel does not support PDF inputs."""
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        """CopilotChatModel does not support audio inputs."""
        return False

    @property
    def supports_video_inputs(self) -> bool:
        """CopilotChatModel does not support video inputs."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """CopilotChatModel does not return usage metadata (yet)."""
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        """CopilotChatModel does not support Anthropic-style inputs."""
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        """CopilotChatModel supports image tool messages for final tool outputs."""
        return True

    @property
    def supports_pdf_tool_message(self) -> bool:
        """CopilotChatModel does not support PDF tool messages."""
        return False

    @property
    def supports_model_override(self) -> bool:
        """CopilotChatModel supports overriding model at runtime."""
        return True

    @property
    def model_override_value(self) -> str:
        """Alternative model to test model override functionality."""
        return "gpt-5-mini"
