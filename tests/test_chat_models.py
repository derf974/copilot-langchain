"""Tests for CopilotChatModel."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_copilot import CopilotChatModel


class TestCopilotChatModel:
    """Test suite for CopilotChatModel."""

    def test_initialization(self):
        """Test basic model initialization."""
        model = CopilotChatModel(model_name="gpt-4o")
        assert model.model_name == "gpt-4o"
        assert model.streaming is False
        assert model._llm_type == "copilot-chat"

    def test_initialization_with_streaming(self):
        """Test model initialization with streaming enabled."""
        model = CopilotChatModel(model_name="gpt-5", streaming=True)
        assert model.model_name == "gpt-5"
        assert model.streaming is True

    def test_initialization_with_temperature(self):
        """Test model initialization with temperature."""
        model = CopilotChatModel(model_name="gpt-4o", temperature=0.7, max_tokens=1000)
        assert model.temperature == 0.7
        assert model.max_tokens == 1000

    def test_convert_messages(self):
        """Test message conversion from LangChain to Copilot format."""
        model = CopilotChatModel()

        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
        ]

        converted = model._convert_messages(messages)

        assert len(converted) == 3
        assert converted[0] == {"role": "system", "content": "You are helpful."}
        assert converted[1] == {"role": "user", "content": "Hello!"}
        assert converted[2] == {"role": "assistant", "content": "Hi there!"}

    def test_create_session_config(self):
        """Test session configuration creation."""
        model = CopilotChatModel(
            model_name="gpt-4o", streaming=True, temperature=0.5, max_tokens=500
        )

        config = model._create_session_config()

        assert config["model"] == "gpt-4o"
        assert config["streaming"] is True
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 500

    def test_create_session_config_defaults(self):
        """Test session configuration with default values."""
        model = CopilotChatModel()

        config = model._create_session_config()

        assert config["model"] == "gpt-4o"
        assert config["streaming"] is False
        assert "temperature" not in config
        assert "max_tokens" not in config

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test that _get_client creates and starts a client."""
        # Reset shared client
        CopilotChatModel._shared_client = None

        with patch("langchain_copilot.chat_models.CopilotClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            model = CopilotChatModel()
            client = await model._get_client()

            # Verify client was created and started
            mock_client_class.assert_called_once()
            mock_client.start.assert_called_once()
            assert client == mock_client
            assert CopilotChatModel._shared_client == mock_client

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Test that _get_client reuses existing client."""
        # Setup existing client
        existing_client = AsyncMock()
        CopilotChatModel._shared_client = existing_client

        model = CopilotChatModel()
        client = await model._get_client()

        # Should return the same client
        assert client == existing_client

    @pytest.mark.asyncio
    async def test_agenerate(self):
        """Test async generation."""
        CopilotChatModel._shared_client = None

        with patch("langchain_copilot.chat_models.CopilotClient") as mock_client_class:
            # Setup mocks
            mock_client = AsyncMock()
            mock_session = AsyncMock()

            # Store the callback to trigger it later
            stored_callback = None

            def mock_on(callback):
                nonlocal stored_callback
                stored_callback = callback

            # Configure session.send to trigger the event
            async def mock_send(message):
                # Simulate receiving a message after send
                if stored_callback:
                    # Create a mock event object
                    class MockEvent:
                        class Type:
                            value = "assistant.message"

                        type = Type()

                        class Data:
                            content = "Hello from Copilot!"

                        data = Data()

                    await asyncio.sleep(0.01)  # Small delay to simulate async
                    stored_callback(MockEvent())

            mock_session.on = mock_on
            mock_session.send = mock_send
            mock_client_class.return_value = mock_client
            mock_client.create_session = AsyncMock(return_value=mock_session)

            model = CopilotChatModel()
            messages = [HumanMessage(content="Hi")]

            result = await model._agenerate(messages)

            # Verify result
            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Hello from Copilot!"

            # Verify session was created and destroyed
            mock_client.create_session.assert_called_once()
            mock_session.destroy.assert_called_once()
            mock_client.stop.assert_called_once()

    def test_model_alias(self):
        """Test that 'model' alias works for model_name."""
        model = CopilotChatModel(model="gpt-5")
        assert model.model_name == "gpt-5"


class TestCopilotChatModelIntegration:
    """Integration tests (require actual Copilot CLI setup)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_invocation(self):
        """Test real invocation (requires Copilot CLI)."""
        model = CopilotChatModel(model_name="gpt-4o")
        messages = [HumanMessage(content="Say 'test passed' and nothing else.")]

        result = await model._agenerate(messages)

        assert len(result.generations) == 1
        assert "test passed" in result.generations[0].message.content.lower()

    @pytest.mark.integration
    def test_real_invoke_sync(self):
        """Test real synchronous invocation (requires Copilot CLI)."""
        model = CopilotChatModel(model_name="gpt-4o")
        messages = [HumanMessage(content="Say 'test passed' and nothing else.")]

        result = model.invoke(messages)

        assert "test passed" in result.content.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_streaming(self):
        """Test real streaming (requires Copilot CLI)."""
        model = CopilotChatModel(model_name="gpt-4o", streaming=True)
        messages = [HumanMessage(content="Count from 1 to 3.")]

        chunks = []
        async for chunk in model._astream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Concatenate all chunks
        full_content = "".join(c.message.content for c in chunks)
        assert len(full_content) > 0
