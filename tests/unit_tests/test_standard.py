"""LangChain standard unit tests for CopilotChatModel.

These tests validate the model's behavior in isolation without requiring
external dependencies like the Copilot CLI.

To run these tests:
    uv run pytest tests/unit_tests/ -v
"""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_copilot import CopilotChatModel


class TestCopilotChatModelUnit(ChatModelUnitTests):
    """Standard unit tests for CopilotChatModel.
    
    This class implements LangChain's standard unit test suite for chat models.
    Tests validate that CopilotChatModel correctly implements the BaseChatModel
    interface and handles initialization, serialization, and basic operations.
    
    Unit tests run in isolation and do not require the Copilot CLI to be installed.
    """

    @property
    def chat_model_class(self) -> Type[CopilotChatModel]:
        """Return the chat model class to test."""
        return CopilotChatModel

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the model.
        
        These parameters will be used to instantiate CopilotChatModel
        for each test. Using gpt-4o with temperature=0 for deterministic
        responses in tests.
        """
        return {
            "model_name": "gpt-4o",
            "temperature": 0,
        }

    # Feature support flags
    # Most features are disabled by default - override to enable specific tests

    @property
    def has_tool_calling(self) -> bool:
        """CopilotChatModel does not support tool calling (yet)."""
        return False

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
        """CopilotChatModel does not support image inputs (yet)."""
        return False

    @property
    def supports_image_urls(self) -> bool:
        """CopilotChatModel does not support image URLs (yet)."""
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
        """CopilotChatModel does not support image tool messages."""
        return False

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
        return "gpt-4o-mini"

    # Note: init_from_env_params is not implemented yet as CopilotChatModel
    # doesn't currently support initialization from environment variables
    # for API keys (it relies on the Copilot CLI for authentication).
