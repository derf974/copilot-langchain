"""LangChain ChatModel implementation using GitHub Copilot SDK."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, AsyncIterator, ClassVar, Iterator, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, model_validator

from copilot import CopilotClient
from copilot.session import PermissionHandler, SystemMessageReplaceConfig
from copilot.tools import Tool
from copilot.client import ExternalServerConfig, SubprocessConfig

import logging

# Suppress AssertionError logging from the Copilot SDK's event deserialization
# This is a workaround for a bug in the SDK where some events have unexpected context types
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


class CopilotChatModel(BaseChatModel):
    """LangChain chat model using GitHub Copilot SDK.

    This model provides a LangChain interface to the GitHub Copilot SDK,
    supporting both synchronous and asynchronous operations, as well as streaming.

    The Copilot client is shared across instances and lazily initialized on first use.

    Example:
        ```python
        from langchain_copilot import CopilotChatModel
        from langchain_core.messages import HumanMessage

        model = CopilotChatModel(model_name="gpt-4o")
        messages = [HumanMessage(content="Hello!")]
        response = model.invoke(messages)
        print(response.content)
        ```

    Attributes:
        model_name: The name of the model to use (e.g., "gpt-4o", "gpt-5")
        streaming: Whether to enable streaming mode
        cli_path: Optional path to the Copilot CLI executable
        cli_url: Optional URL of an existing Copilot CLI server
        temperature: Temperature for response generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
    """

    model_name: str = Field(default="gpt-4o", alias="model")
    streaming: bool = Field(default=False)
    cli_path: Optional[str] = Field(default=None)
    cli_url: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    tools: Optional[list[Tool]] = Field(default=None)

    # Internal shared client (class variable)
    _shared_client: ClassVar[Optional[CopilotClient]] = None
    _client_lock: ClassVar[Optional[asyncio.Lock]] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def _initialize_lock(self) -> "CopilotChatModel":
        """Initialize the async lock for client management."""
        if CopilotChatModel._client_lock is None:
            CopilotChatModel._client_lock = asyncio.Lock()
        return self

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "copilot-chat"

    async def _get_client(self) -> CopilotClient:
        """Get or create the shared Copilot client (lazy initialization).

        Returns:
            The shared CopilotClient instance
        """
        if CopilotChatModel._shared_client is None:
            async with CopilotChatModel._client_lock:
                if CopilotChatModel._shared_client is None:
                    options = None
                    if self.cli_url:
                        options = ExternalServerConfig(url=self.cli_url)
                    elif self.cli_path:
                        options = SubprocessConfig(cli_path=self.cli_path)

                    CopilotChatModel._shared_client = CopilotClient(options or None)

                    # Set up custom exception handler for asyncio loop to suppress
                    # AssertionErrors from Copilot SDK event deserialization
                    loop = asyncio.get_event_loop()

                    def custom_exception_handler(loop, context):
                        exception = context.get("exception")
                        # Suppress AssertionError from Copilot SDK's session_events.py
                        if isinstance(exception, AssertionError):
                            # Ignore this specific error from the SDK
                            return
                        # For other exceptions, use default handling
                        loop.default_exception_handler(context)

                    loop.set_exception_handler(custom_exception_handler)

                    await CopilotChatModel._shared_client.start()

        return CopilotChatModel._shared_client

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
        """Convert LangChain messages to Copilot SDK format.

        Args:
            messages: List of LangChain BaseMessage objects

        Returns:
            List of message dictionaries in Copilot format
        """
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                # Default to user for unknown message types
                role = "user"

            converted.append({"role": role, "content": msg.content})

        return converted

    def _create_session_config(
        self, messages: Optional[list[BaseMessage]] = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Create session configuration for Copilot SDK.

        Args:
            messages: Optional list of messages to extract system message from
            **kwargs: Additional arguments (e.g., tools from bind_tools)

        Returns:
            Configuration dictionary for creating a Copilot session
        """
        config_params = {
            "model": self.model_name,
            "streaming": self.streaming,
            "tools": kwargs.get("tools", self.tools),
            "on_permission_request": PermissionHandler.approve_all,
        }

        # Extract system messages if provided
        if messages:
            system_messages = [
                msg for msg in messages if isinstance(msg, SystemMessage)
            ]
            if system_messages:
                # Concatenate all system messages
                system_content = "\n".join(str(msg.content) for msg in system_messages)
                config_params["system_message"] = SystemMessageReplaceConfig(
                    mode="replace",
                    content=system_content,
                )

        return config_params

    def _messages_to_prompt(self, messages: list[BaseMessage]) -> str:
        parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                parts.append(f"Tool: {msg.content}")
            elif isinstance(msg, SystemMessage):
                # Skip SystemMessage - they are extracted separately in _create_session_config
                # to avoid prompt injection risks
                continue
            else:
                # Fallback for other BaseMessage types to avoid dropping content
                role = getattr(msg, "type", msg.__class__.__name__)
                parts.append(f"{role.capitalize()}: {msg.content}")
        if not parts:
            raise ValueError(
                "No valid messages to send. Messages must contain at least one "
                "HumanMessage, AIMessage, or ToolMessage. SystemMessage instances "
                "are automatically extracted and passed to the session configuration, "
                "but at least one conversational message is required to start the interaction."
            )
        return "\n\n".join(parts)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response synchronously.

        Args:
            messages: List of messages to send
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments

        Returns:
            ChatResult containing the generated response
        """
        # Run async version in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new event loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self._agenerate(messages, stop, run_manager, **kwargs)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                self._agenerate(messages, stop, run_manager, **kwargs)
            )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response asynchronously.

        Args:
            messages: List of messages to send
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments

        Returns:
            ChatResult containing the generated response
        """
        client = await self._get_client()
        # Pass messages and kwargs to extract system messages and tools for session config
        session_config = self._create_session_config(messages, **kwargs)

        # Create a session
        session = await client.create_session(**session_config)

        try:
            full_prompt = self._messages_to_prompt(messages)

            last_event = await session.send_and_wait(full_prompt)

            response_content = ""
            if last_event is not None and last_event.data is not None:
                response_content = last_event.data.content or ""

            # Create response
            message = AIMessage(content=response_content)
            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        finally:
            # Clean up session
            await session.destroy()
            await client.stop()

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response synchronously.

        Args:
            messages: List of messages to send
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments

        yields:
            ChatGenerationChunk for each chunk of the response
        """
        # Run async version in sync context
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError(
                "Cannot use sync streaming from an async context. "
                "Use astream() instead."
            )

        async_gen = self._astream(messages, stop, run_manager, **kwargs)

        # Convert async generator to sync
        while True:
            try:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream response asynchronously.

        Args:
            messages: List of messages to send
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional arguments

        Yields:
            ChatGenerationChunk for each chunk of the response
        """
        client = await self._get_client()
        # Pass messages and kwargs to extract system messages and tools for session config
        session_config = self._create_session_config(messages, **kwargs)
        session_config["streaming"] = True  # Force streaming mode

        # Create a session
        session = await client.create_session(**session_config)

        try:
            full_prompt = self._messages_to_prompt(messages)

            # Queue to collect chunks
            chunk_queue: asyncio.Queue = asyncio.Queue()
            complete = asyncio.Event()

            def on_event(event):
                try:
                    if event.type.value == "assistant.message_delta":
                        # Streaming message chunk - print incrementally
                        content = event.data.delta_content or ""
                        asyncio.create_task(chunk_queue.put(content))
                    elif event.type.value == "assistant.reasoning_delta":
                        # Streaming reasoning chunk (if model supports reasoning)
                        # content = event.data.delta_content or ""
                        # asyncio.create_task(chunk_queue.put(content))
                        pass
                    elif event.type.value == "assistant.message":
                        # Final message - complete content
                        asyncio.create_task(chunk_queue.put(None))
                        complete.set()
                    elif event.type.value == "assistant.reasoning":
                        # Final reasoning content (if model supports reasoning)
                        pass
                    elif event.type.value == "session.idle":
                        # Session finished processing
                        complete.set()
                except (AttributeError, KeyError):
                    # Ignore malformed events
                    pass

            # Register event listener
            session.on(on_event)

            # Send the prompt
            await session.send(full_prompt)

            # Yield chunks as they arrive
            while not complete.is_set() or not chunk_queue.empty():
                try:
                    chunk_content = await asyncio.wait_for(
                        chunk_queue.get(), timeout=0.1
                    )

                    if chunk_content is None:
                        # End of stream
                        break

                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=chunk_content)
                    )

                    if run_manager:
                        await run_manager.on_llm_new_token(chunk_content)

                    yield chunk

                except asyncio.TimeoutError:
                    continue

        finally:
            # Clean up session
            await session.destroy()
            await client.stop()

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to the model.

        Transforms LangChain tools into Copilot SDK Tool format and creates
        a new model instance with these tools bound.

        Args:
            tools: Sequence of tools to bind. Can be:
                - Dict representing an OpenAI-style tool schema
                - Pydantic class/BaseModel
                - Python callable/function
                - LangChain BaseTool
            tool_choice: Tool choice strategy (not currently used by Copilot SDK)
            **kwargs: Additional arguments

        Returns:
            A new CopilotChatModel instance with tools bound

        Example:
            ```python
            from pydantic import BaseModel, Field
            from copilot import define_tool

            class WeatherParams(BaseModel):
                location: str = Field(description="City name")

            @define_tool(description="Get weather")
            async def get_weather(params: WeatherParams) -> str:
                return f"Weather in {params.location}: sunny"

            model = CopilotChatModel(model="gpt-4o")
            model_with_tools = model.bind_tools([get_weather])
            ```
        """
        # Convert LangChain tools to Copilot SDK Tool format
        copilot_tools = []

        for tool in tools:
            if isinstance(tool, Tool):
                # Already a Copilot SDK Tool
                copilot_tools.append(tool)
            elif callable(tool):
                # Check if it's already a defined tool (has _tool_info attribute)
                if hasattr(tool, "_tool_info"):
                    copilot_tools.append(tool)
                else:
                    # It's a plain callable - convert it using convert_to_openai_tool
                    try:
                        tool_schema = convert_to_openai_tool(tool)

                        # Create handler for the callable
                        def create_callable_handler(func: Callable):
                            async def handler(invocation):
                                args = invocation.get("arguments", {})
                                try:
                                    # Check if it's async
                                    if asyncio.iscoroutinefunction(func):
                                        result = await func(**args)
                                    else:
                                        result = func(**args)

                                    return {
                                        "textResultForLlm": str(result),
                                        "resultType": "success",
                                    }
                                except Exception as e:
                                    return {
                                        "textResultForLlm": f"Error: {str(e)}",
                                        "resultType": "error",
                                    }

                            return handler

                        copilot_tool = Tool(
                            name=tool_schema["function"]["name"],
                            description=tool_schema["function"]["description"],
                            parameters=tool_schema["function"]["parameters"],
                            handler=create_callable_handler(tool),
                            overrides_built_in_tool=True,
                        )
                        copilot_tools.append(copilot_tool)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert callable {tool} to Copilot tool: {e}"
                        )
            elif isinstance(tool, BaseTool):
                # LangChain BaseTool - convert to Copilot format
                tool_schema = convert_to_openai_tool(tool)

                # Create an async handler that wraps the BaseTool
                # We need to capture the tool instance in the closure
                def create_handler(base_tool: BaseTool):
                    async def handler(invocation):
                        args = invocation.get("arguments", {})
                        try:
                            # Try async invoke first
                            if hasattr(base_tool, "ainvoke"):
                                result = await base_tool.ainvoke(args)
                            else:
                                # Fall back to sync invoke
                                result = base_tool.invoke(args)

                            return {
                                "textResultForLlm": str(result),
                                "resultType": "success",
                            }
                        except Exception as e:
                            return {
                                "textResultForLlm": f"Error: {str(e)}",
                                "resultType": "error",
                            }

                    return handler

                copilot_tool = Tool(
                    name=tool_schema["function"]["name"],
                    description=tool_schema["function"]["description"],
                    parameters=tool_schema["function"]["parameters"],
                    handler=create_handler(tool),
                    overrides_built_in_tool=True,
                )
                copilot_tools.append(copilot_tool)
            elif isinstance(tool, dict):
                # Dict-style tool schema - could be OpenAI format or JSON schema
                if "function" in tool:
                    # OpenAI-style tool schema
                    # We can't create a handler from just a schema
                    # This is for tools that are schema-only (like for structured output)
                    # Skip them as they can't be called
                    continue
                elif "type" in tool and tool.get("type") == "object":
                    # JSON schema format (Pydantic schema dict)
                    # Similar to above, schema-only without handler
                    # Skip as it can't be called
                    continue
                else:
                    raise ValueError(
                        f"Unsupported dict tool format: {tool}. "
                        "Expected OpenAI function format or JSON schema."
                    )
            elif isinstance(tool, type):
                # Pydantic class - convert to tool schema
                try:
                    tool_schema = convert_to_openai_tool(tool)

                    # For a Pydantic class without a handler, we can't execute it
                    # But we can still register it as a schema for structured output
                    # For now, raise a helpful error
                    raise ValueError(
                        f"Pydantic class {tool.__name__} cannot be used directly as a tool. "
                        "Wrap it with @define_tool decorator or use it in with_structured_output() instead."
                    )
                except ValueError:
                    # Re-raise our custom error
                    raise
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert Pydantic class {tool} to tool schema: {e}"
                    )
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

        # Return a RunnableBinding with the tools bound as kwargs
        # This is the standard LangChain pattern
        return RunnableBinding(
            bound=self,
            kwargs={"tools": copilot_tools} if copilot_tools else {},
            config={},
        )
