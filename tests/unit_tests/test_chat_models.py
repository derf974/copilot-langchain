"""Tests for CopilotChatModel."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_copilot import CopilotChatModel
from copilot import Tool, define_tool
from pydantic import BaseModel, Field


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
        assert "system_message" not in config

    def test_create_session_config_defaults(self):
        """Test session configuration with default values."""
        model = CopilotChatModel()

        config = model._create_session_config()

        assert config["model"] == "gpt-4o"
        assert config["streaming"] is False
        assert "temperature" not in config
        assert "max_tokens" not in config
        assert "system_message" not in config

    def test_create_session_config_with_system_message(self):
        """Test session configuration with system messages."""
        model = CopilotChatModel()

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
        ]

        config = model._create_session_config(messages)

        assert config["model"] == "gpt-4o"
        assert "system_message" in config
        assert config["system_message"]["content"] == "You are a helpful assistant."

    def test_create_session_config_with_multiple_system_messages(self):
        """Test session configuration with multiple system messages."""
        model = CopilotChatModel()

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            SystemMessage(content="You speak French."),
            HumanMessage(content="Hello!"),
        ]

        config = model._create_session_config(messages)

        assert config["model"] == "gpt-4o"
        assert "system_message" in config
        assert (
            config["system_message"]["content"]
            == "You are a helpful assistant.\nYou speak French."
        )

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
    async def test_get_client_with_cli_url(self):
        """Test that _get_client passes cli_url as options dict to CopilotClient."""
        CopilotChatModel._shared_client = None

        with patch("langchain_copilot.chat_models.CopilotClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            model = CopilotChatModel(cli_url="http://localhost:1234")
            await model._get_client()

            # CopilotClient should be called with a dict containing cli_url,
            # not with cli_url as a keyword argument
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            assert call_args[0][0] == {"cli_url": "http://localhost:1234"}

    @pytest.mark.asyncio
    async def test_get_client_with_cli_path(self):
        """Test that _get_client passes cli_path as options dict to CopilotClient."""
        CopilotChatModel._shared_client = None

        with patch("langchain_copilot.chat_models.CopilotClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            model = CopilotChatModel(cli_path="/usr/local/bin/copilot")
            await model._get_client()

            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            assert call_args[0][0] == {"cli_path": "/usr/local/bin/copilot"}

    @pytest.mark.asyncio
    async def test_get_client_without_options(self):
        """Test that _get_client passes None to CopilotClient when no options set."""
        CopilotChatModel._shared_client = None

        with patch("langchain_copilot.chat_models.CopilotClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            model = CopilotChatModel()
            await model._get_client()

            mock_client_class.assert_called_once_with(None)

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
                    # Create a mock event object for assistant message
                    class MockMessageEvent:
                        class Type:
                            value = "assistant.message"

                        type = Type()

                        class Data:
                            content = "Hello from Copilot!"

                        data = Data()

                    # Create a mock event for session idle
                    class MockIdleEvent:
                        class Type:
                            value = "session.idle"

                        type = Type()

                    await asyncio.sleep(0.01)  # Small delay to simulate async
                    stored_callback(MockMessageEvent())
                    await asyncio.sleep(0.01)
                    stored_callback(MockIdleEvent())

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

    @pytest.mark.asyncio
    async def test_agenerate_with_system_message(self):
        """Test async generation with system messages."""
        CopilotChatModel._shared_client = None

        with patch("langchain_copilot.chat_models.CopilotClient") as mock_client_class:
            # Setup mocks
            mock_client = AsyncMock()
            mock_session = AsyncMock()

            # Store the callback to trigger it later
            stored_callback = None
            send_called_with = None

            def mock_on(callback):
                nonlocal stored_callback
                stored_callback = callback

            # Configure session.send to trigger the event
            async def mock_send(message):
                nonlocal send_called_with
                send_called_with = message
                # Simulate receiving a message after send
                if stored_callback:
                    # Create a mock event object for assistant message
                    class MockMessageEvent:
                        class Type:
                            value = "assistant.message"

                        type = Type()

                        class Data:
                            content = "Bonjour!"

                        data = Data()

                    # Create a mock event for session idle
                    class MockIdleEvent:
                        class Type:
                            value = "session.idle"

                        type = Type()

                    await asyncio.sleep(0.01)  # Small delay to simulate async
                    stored_callback(MockMessageEvent())
                    await asyncio.sleep(0.01)
                    stored_callback(MockIdleEvent())

            mock_session.on = mock_on
            mock_session.send = mock_send
            mock_client_class.return_value = mock_client
            mock_client.create_session = AsyncMock(return_value=mock_session)

            model = CopilotChatModel()
            messages = [
                SystemMessage(content="You are a French translator."),
                HumanMessage(content="Say hello"),
            ]

            result = await model._agenerate(messages)

            # Verify result
            assert len(result.generations) == 1
            assert result.generations[0].message.content == "Bonjour!"

            # Verify session was created with system message
            mock_client.create_session.assert_called_once()
            call_args = mock_client.create_session.call_args
            session_config = call_args[0][0]
            assert "system_message" in session_config
            assert (
                session_config["system_message"]["content"]
                == "You are a French translator."
            )

            assert send_called_with is not None

            # Verify cleanup
            mock_session.destroy.assert_called_once()
            mock_client.stop.assert_called_once()

    def test_initialization_with_tools(self):
        """Test model initialization with tools."""
        from copilot import Tool

        async def mock_tool_handler(invocation):
            return {"textResultForLlm": "Result", "resultType": "success"}

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
            },
            handler=mock_tool_handler,
        )

        model = CopilotChatModel(model_name="gpt-4o", tools=[tool])
        assert model.tools is not None
        assert len(model.tools) == 1
        assert model.tools[0].name == "test_tool"

    def test_create_session_config_with_tools(self):
        """Test session configuration includes tools."""
        from copilot import Tool

        async def mock_tool_handler(invocation):
            return {"textResultForLlm": "Result", "resultType": "success"}

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
            },
            handler=mock_tool_handler,
        )

        model = CopilotChatModel(model_name="gpt-4o", tools=[tool])
        config = model._create_session_config()

        assert "tools" in config
        assert len(config["tools"]) == 1
        assert config["tools"][0].name == "test_tool"

    def test_create_session_config_without_tools(self):
        """Test session configuration without tools."""
        model = CopilotChatModel()
        config = model._create_session_config()

        assert config["tools"] is None

    @pytest.mark.asyncio
    async def test_agenerate_with_tools(self):
        """Test async generation with tools."""
        from copilot import Tool

        CopilotChatModel._shared_client = None

        async def mock_tool_handler(invocation):
            return {"textResultForLlm": "42", "resultType": "success"}

        tool = Tool(
            name="calculator",
            description="Performs calculations",
            parameters={
                "type": "object",
                "properties": {"operation": {"type": "string"}},
            },
            handler=mock_tool_handler,
        )

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
                    # Create a mock event object for assistant message
                    class MockMessageEvent:
                        class Type:
                            value = "assistant.message"

                        type = Type()

                        class Data:
                            content = "The result is 42"

                        data = Data()

                    # Create a mock event for session idle
                    class MockIdleEvent:
                        class Type:
                            value = "session.idle"

                        type = Type()

                    await asyncio.sleep(0.01)  # Small delay to simulate async
                    stored_callback(MockMessageEvent())
                    await asyncio.sleep(0.01)
                    stored_callback(MockIdleEvent())

            mock_session.on = mock_on
            mock_session.send = mock_send
            mock_client_class.return_value = mock_client
            mock_client.create_session = AsyncMock(return_value=mock_session)

            model = CopilotChatModel(tools=[tool])
            messages = [HumanMessage(content="What is 21 + 21?")]

            result = await model._agenerate(messages)

            # Verify result
            assert len(result.generations) == 1
            assert "42" in result.generations[0].message.content

            # Verify session was created with tools
            mock_client.create_session.assert_called_once()
            call_args = mock_client.create_session.call_args
            session_config = call_args[0][0]
            assert session_config["tools"] is not None
            assert len(session_config["tools"]) == 1
            assert session_config["tools"][0].name == "calculator"

            # Verify cleanup
            mock_session.destroy.assert_called_once()
            mock_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_astream_with_tools(self):
        """Test async streaming with tools."""
        from copilot import Tool

        CopilotChatModel._shared_client = None

        async def mock_tool_handler(invocation):
            return {"textResultForLlm": "The weather is sunny", "resultType": "success"}

        tool = Tool(
            name="get_weather",
            description="Gets weather information",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
            handler=mock_tool_handler,
        )

        with patch("langchain_copilot.chat_models.CopilotClient") as mock_client_class:
            # Setup mocks
            mock_client = AsyncMock()
            mock_session = AsyncMock()

            # Store the callback to trigger it later
            stored_callback = None

            def mock_on(callback):
                nonlocal stored_callback
                stored_callback = callback

            # Configure session.send to trigger streaming events
            async def mock_send(message):
                # Simulate receiving streaming chunks
                if stored_callback:

                    class MockDeltaEvent:
                        class Type:
                            value = "assistant.message_delta"

                        type = Type()

                        class Data:
                            delta_content = None

                        data = Data()

                    class MockFinalEvent:
                        class Type:
                            value = "assistant.message"

                        type = Type()

                        class Data:
                            content = "The weather is sunny in Paris"

                        data = Data()

                    # Send chunks
                    await asyncio.sleep(0.01)
                    event1 = MockDeltaEvent()
                    event1.data.delta_content = "The weather "
                    stored_callback(event1)

                    await asyncio.sleep(0.01)
                    event2 = MockDeltaEvent()
                    event2.data.delta_content = "is sunny "
                    stored_callback(event2)

                    await asyncio.sleep(0.01)
                    event3 = MockDeltaEvent()
                    event3.data.delta_content = "in Paris"
                    stored_callback(event3)

                    # Send final message
                    await asyncio.sleep(0.01)
                    stored_callback(MockFinalEvent())

            mock_session.on = mock_on
            mock_session.send = mock_send
            mock_client_class.return_value = mock_client
            mock_client.create_session = AsyncMock(return_value=mock_session)

            model = CopilotChatModel(streaming=True, tools=[tool])
            messages = [HumanMessage(content="What's the weather in Paris?")]

            # Collect chunks
            chunks = []
            async for chunk in model._astream(messages):
                chunks.append(chunk.message.content)

            # Verify chunks were received
            assert len(chunks) == 3
            assert chunks[0] == "The weather "
            assert chunks[1] == "is sunny "
            assert chunks[2] == "in Paris"

            # Verify session was created with tools and streaming enabled
            mock_client.create_session.assert_called_once()
            call_args = mock_client.create_session.call_args
            session_config = call_args[0][0]
            assert session_config["streaming"] is True
            assert "tools" in session_config
            assert len(session_config["tools"]) == 1
            assert session_config["tools"][0].name == "get_weather"

            # Verify cleanup
            mock_session.destroy.assert_called_once()
            mock_client.stop.assert_called_once()

    def test_bind_tools_with_copilot_tool(self):
        """Test bind_tools with Copilot SDK Tool instances."""

        async def mock_handler(invocation):
            return {"textResultForLlm": "42", "resultType": "success"}

        tool = Tool(
            name="calculator",
            description="Performs calculations",
            parameters={
                "type": "object",
                "properties": {"operation": {"type": "string"}},
            },
            handler=mock_handler,
        )

        model = CopilotChatModel(model="gpt-4o")
        bound_model = model.bind_tools([tool])

        # Should return a RunnableBinding
        from langchain_core.runnables import RunnableBinding

        assert isinstance(bound_model, RunnableBinding)
        # The underlying bound model should have tools in kwargs
        assert "tools" in bound_model.kwargs
        assert len(bound_model.kwargs["tools"]) == 1
        assert bound_model.kwargs["tools"][0].name == "calculator"

    def test_bind_tools_with_define_tool(self):
        """Test bind_tools with @define_tool decorated functions."""

        class CalcParams(BaseModel):
            a: float = Field(description="First number")
            b: float = Field(description="Second number")

        @define_tool(description="Add two numbers")
        async def add_numbers(params: CalcParams) -> str:
            return str(params.a + params.b)

        model = CopilotChatModel(model="gpt-4o")
        bound_model = model.bind_tools([add_numbers])

        from langchain_core.runnables import RunnableBinding

        assert isinstance(bound_model, RunnableBinding)
        assert "tools" in bound_model.kwargs
        assert len(bound_model.kwargs["tools"]) == 1

    def test_bind_tools_with_langchain_tool(self):
        """Test bind_tools with LangChain @tool decorated functions."""

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        model = CopilotChatModel(model="gpt-4o")
        bound_model = model.bind_tools([search])

        from langchain_core.runnables import RunnableBinding

        assert isinstance(bound_model, RunnableBinding)
        assert "tools" in bound_model.kwargs
        assert len(bound_model.kwargs["tools"]) == 1
        # The tool should have been converted from LangChain format
        assert bound_model.kwargs["tools"][0].name == "search"

    def test_bind_tools_with_plain_callable(self):
        """Test that bind_tools supports plain callables."""

        def plain_function(x: int) -> int:
            """Double a number."""
            return x * 2

        model = CopilotChatModel(model="gpt-4o")
        bound_model = model.bind_tools([plain_function])

        from langchain_core.runnables import RunnableBinding

        assert isinstance(bound_model, RunnableBinding)
        assert "tools" in bound_model.kwargs
        assert len(bound_model.kwargs["tools"]) == 1
        assert bound_model.kwargs["tools"][0].name == "plain_function"

    def test_bind_tools_rejects_dict_without_handler(self):
        """Test that bind_tools accepts dict schemas (they are skipped)."""
        tool_dict = {
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
            }
        }

        model = CopilotChatModel(model="gpt-4o")

        # Dict schemas are now skipped (no error raised)
        bound_model = model.bind_tools([tool_dict])

        from langchain_core.runnables import RunnableBinding

        assert isinstance(bound_model, RunnableBinding)
        # No tools should be bound since dict schemas are skipped
        assert bound_model.kwargs.get("tools", []) == []

    def test_bind_tools_preserves_model_config(self):
        """Test that bind_tools preserves other model configuration."""
        model = CopilotChatModel(
            model="gpt-4o", temperature=0.7, max_tokens=1000, streaming=True
        )

        class TestParams(BaseModel):
            value: str = Field(description="Test value")

        @define_tool(description="Test")
        async def test_tool(params: TestParams) -> str:
            return "test"

        bound_model = model.bind_tools([test_tool])

        from langchain_core.runnables import RunnableBinding

        assert isinstance(bound_model, RunnableBinding)
        # The bound object wraps the original model
        assert bound_model.bound.model_name == "gpt-4o"
        assert bound_model.bound.streaming is True

    def test_messages_to_prompt_with_only_system_messages(self):
        """Test that _messages_to_prompt raises ValueError when only system messages are provided."""
        model = CopilotChatModel()
        messages = [SystemMessage(content="You are helpful.")]

        # SystemMessage are filtered out to prevent prompt injection
        with pytest.raises(ValueError) as exc_info:
            model._messages_to_prompt(messages)

        assert "No valid messages to send" in str(exc_info.value)

    def test_messages_to_prompt_with_mixed_messages(self):
        """Test that _messages_to_prompt filters out system messages to prevent injection."""
        model = CopilotChatModel()
        messages = [
            SystemMessage(content="System instruction"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]

        result = model._messages_to_prompt(messages)

        # SystemMessage should be filtered out (handled separately in session config)
        expected_prompt = "User: Hello\n\nAssistant: Hi there"
        assert result == expected_prompt
        # Ensure no system instruction leaked into the conversational prompt
        assert "System instruction" not in result
