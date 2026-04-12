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
import json

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    get_config_list,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, model_validator

from copilot import CopilotClient
from copilot.session import PermissionHandler, SystemMessageReplaceConfig
from copilot.tools import Tool, ToolResult
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

        model = CopilotChatModel(model_name="gpt-5-mini")
        messages = [HumanMessage(content="Hello!")]
        response = model.invoke(messages)
        print(response.content)
        ```

    Attributes:
        model_name: The name of the model to use (e.g., "gpt-5-mini", "gpt-5")
        streaming: Whether to enable streaming mode
        cli_path: Optional path to the Copilot CLI executable
        cli_url: Optional URL of an existing Copilot CLI server
        temperature: Temperature for response generation (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
    """

    model_name: str = Field(default="gpt-5-mini", alias="model")
    streaming: bool = Field(default=False)
    cli_path: Optional[str] = Field(default=None)
    cli_url: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    tools: Optional[list[Tool]] = Field(default=None)

    # Internal shared client (class variable)
    _shared_client: ClassVar[Optional[CopilotClient]] = None
    _client_lock: ClassVar[Optional[asyncio.Lock]] = None
    _shared_loop: ClassVar[Optional[asyncio.AbstractEventLoop]] = None

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

        The client is tied to the event loop it was started on.  When pytest-asyncio
        (or asyncio.run) creates a fresh loop for each test, we detect the loop change
        and transparently recreate the client so the underlying I/O transports are
        always valid.

        Returns:
            The shared CopilotClient instance
        """
        current_loop = asyncio.get_running_loop()

        # If the client was initialised on a different (now closed) event loop,
        # discard it so we start fresh on the current loop.
        if (
            CopilotChatModel._shared_client is not None
            and CopilotChatModel._shared_loop is not current_loop
        ):
            CopilotChatModel._shared_client = None
            CopilotChatModel._shared_loop = None
            # Recreate the lock bound to the current loop
            CopilotChatModel._client_lock = asyncio.Lock()

        if CopilotChatModel._shared_client is None:
            async with CopilotChatModel._client_lock:
                if CopilotChatModel._shared_client is None:
                    options = None
                    if self.cli_url:
                        options = ExternalServerConfig(url=self.cli_url)
                    elif self.cli_path:
                        options = SubprocessConfig(cli_path=self.cli_path)

                    CopilotChatModel._shared_client = CopilotClient(options or None)

                    # Suppress AssertionErrors from Copilot SDK event deserialization
                    def custom_exception_handler(loop, context):
                        exception = context.get("exception")
                        if isinstance(exception, AssertionError):
                            return
                        loop.default_exception_handler(context)

                    current_loop.set_exception_handler(custom_exception_handler)

                    await CopilotChatModel._shared_client.start()
                    CopilotChatModel._shared_loop = current_loop

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
            # Allow callers to override the model at runtime via a "model" kwarg
            # (e.g. model.invoke("Hello", model="gpt-5-mini")).
            "model": kwargs.get("model", self.model_name),
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
                if msg.tool_calls:
                    # Represent tool call intents so the model has context on turn 2+
                    tc_strs = [
                        f"{tc['name']}({json.dumps(tc['args'])})"
                        for tc in msg.tool_calls
                    ]
                    parts.append(f"Assistant: [Called tools: {', '.join(tc_strs)}]")
                elif msg.content:
                    parts.append(f"Assistant: {msg.content}")
                # Skip empty AIMessage with no tool calls (no useful content)
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "") or ""
                if tool_name:
                    parts.append(f"Tool result ({tool_name}): {msg.content}")
                else:
                    parts.append(f"Tool result: {msg.content}")
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
        try:
            asyncio.get_running_loop()
            is_running = True
        except RuntimeError:
            is_running = False

        if is_running:
            # If we're already in an async context, run in a thread with its own loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self._agenerate(messages, stop, run_manager, **kwargs)
                )
                return future.result()
        else:
            return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

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
        effective_model = kwargs.get("model", self.model_name)
        session_config = self._create_session_config(messages, **kwargs)

        # Collect the names of tools we registered so we can filter events
        registered_tools: list = session_config.get("tools") or []
        registered_tool_names: set[str] = {t.name for t in registered_tools}

        # Create a session
        session = await client.create_session(**session_config)

        try:
            full_prompt = self._messages_to_prompt(messages)

            # Capture tool-call intents emitted during the Copilot agentic loop.
            # The Copilot SDK executes tools automatically inside the loop, but
            # LangChain expects the *caller* to execute tools.  We intercept the
            # tool.execution_start events to collect the model's tool-call intents
            # and surface them as AIMessage.tool_calls.
            captured_tool_calls: list[dict] = []

            def _capture_tool_calls(evt: Any) -> None:
                try:
                    evt_type = (
                        evt.type.value if hasattr(evt.type, "value") else str(evt.type)
                    )
                    if (
                        evt_type == "tool.execution_start"
                        and evt.data.tool_name in registered_tool_names
                    ):
                        captured_tool_calls.append(
                            {
                                "name": evt.data.tool_name,
                                "args": evt.data.arguments or {},
                                "id": evt.data.tool_call_id,
                                "type": "tool_call",
                            }
                        )
                except (AttributeError, KeyError):
                    pass

            if registered_tool_names:
                session.on(_capture_tool_calls)

            last_event = await session.send_and_wait(full_prompt)

            if captured_tool_calls:
                # Return tool-call intents so the LangChain caller can execute them
                message = AIMessage(
                    content="",
                    tool_calls=captured_tool_calls,
                    response_metadata={"model_name": effective_model},
                )
            else:
                response_content = ""
                if last_event is not None and last_event.data is not None:
                    response_content = last_event.data.content or ""
                message = AIMessage(
                    content=response_content,
                    tool_calls=[],
                    response_metadata={"model_name": effective_model},
                )

            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        finally:
            # Disconnect session only; the shared client stays alive for reuse
            await session.disconnect()

    def batch(
        self,
        inputs: list[LanguageModelInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[AIMessage]:
        """Run batch requests sequentially.

        LangChain's default batch implementation uses a thread pool. The Copilot SDK
        client is tied to an asyncio event loop, so cross-thread reuse of the shared
        client can deadlock integration tests such as test_batch.
        """
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))
        results = []

        for input_, input_config in zip(inputs, configs, strict=False):
            if return_exceptions:
                try:
                    results.append(self.invoke(input_, config=input_config, **kwargs))
                except Exception as exc:
                    results.append(exc)
            else:
                results.append(self.invoke(input_, config=input_config, **kwargs))

        return results

    async def abatch(
        self,
        inputs: list[LanguageModelInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[AIMessage]:
        """Run async batch requests sequentially on the active event loop."""
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))
        results = []

        for input_, input_config in zip(inputs, configs, strict=False):
            if return_exceptions:
                try:
                    results.append(
                        await self.ainvoke(input_, config=input_config, **kwargs)
                    )
                except Exception as exc:
                    results.append(exc)
            else:
                results.append(
                    await self.ainvoke(input_, config=input_config, **kwargs)
                )

        return results

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
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot use sync streaming from an async context. "
                "Use astream() instead."
            )
        except RuntimeError as exc:
            if "async context" in str(exc):
                raise

        # Drain the async generator using asyncio.run() per chunk so each
        # iteration gets a fresh event loop, avoiding the stale-loop issue.
        async def _collect_all() -> list:
            chunks = []
            async for chunk in self._astream(messages, stop, run_manager, **kwargs):
                chunks.append(chunk)
            return chunks

        for chunk in asyncio.run(_collect_all()):
            yield chunk

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
        effective_model = kwargs.get("model", self.model_name)
        session_config = self._create_session_config(messages, **kwargs)
        session_config["streaming"] = True  # Force streaming mode

        # Collect the names of tools we registered so we can filter events
        registered_tools_stream: list = session_config.get("tools") or []
        registered_tool_names_stream: set[str] = {
            t.name for t in registered_tools_stream
        }

        # Create a session
        session = await client.create_session(**session_config)

        try:
            full_prompt = self._messages_to_prompt(messages)

            # Queue to collect text chunks from streaming.
            chunk_queue: asyncio.Queue = asyncio.Queue()

            # text_stream_done: set when the first assistant.message fires.
            # (The SDK fires assistant.message with empty content before executing tools,
            # so this signals "no more text deltas are coming for this turn".)
            text_stream_done: asyncio.Event = asyncio.Event()

            # complete: set when we have all the information we need.
            # - Without tools: set on assistant.message (same as text_stream_done).
            # - With tools: set on tool.execution_start (has what we need) or session.idle.
            complete: asyncio.Event = asyncio.Event()

            captured_tool_calls_stream: list[dict] = []

            def on_event(event):
                try:
                    evt_type = (
                        event.type.value
                        if hasattr(event.type, "value")
                        else str(event.type)
                    )
                    if evt_type == "assistant.message_delta":
                        # Yield text deltas only while no tool call has been captured yet
                        if not captured_tool_calls_stream:
                            content = event.data.delta_content or ""
                            asyncio.create_task(chunk_queue.put(content))
                    elif evt_type in (
                        "assistant.reasoning_delta",
                        "assistant.reasoning",
                    ):
                        pass
                    elif evt_type == "assistant.message":
                        # First turn ends: signal text streaming is done.
                        text_stream_done.set()
                        asyncio.create_task(chunk_queue.put(None))  # sentinel
                        # For sessions without tools, this is also the "complete" signal.
                        if not registered_tool_names_stream:
                            complete.set()
                    elif evt_type == "session.idle":
                        # Fallback: always complete on session.idle
                        complete.set()
                    elif (
                        evt_type == "tool.execution_start"
                        and event.data.tool_name in registered_tool_names_stream
                    ):
                        # Capture tool-call intent for LangChain tool_calls
                        captured_tool_calls_stream.append(
                            {
                                "name": event.data.tool_name,
                                "args": event.data.arguments or {},
                                "id": event.data.tool_call_id,
                                "type": "tool_call",
                            }
                        )
                        # We have what we need; signal completion.
                        complete.set()
                except (AttributeError, KeyError):
                    # Ignore malformed events
                    pass

            # Register event listener and send the prompt
            session.on(on_event)
            await session.send(full_prompt)

            # Phase 1: stream text chunks until the first turn ends.
            while not text_stream_done.is_set() or not chunk_queue.empty():
                try:
                    chunk_content = await asyncio.wait_for(
                        chunk_queue.get(), timeout=0.1
                    )

                    if chunk_content is None:
                        # Sentinel: end of text stream
                        break

                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=chunk_content)
                    )

                    if run_manager:
                        await run_manager.on_llm_new_token(chunk_content)

                    yield chunk

                except asyncio.TimeoutError:
                    continue

            # Phase 2: if tools are registered, wait for tool.execution_start
            # (or session.idle) to fire.  This must happen AFTER the text loop
            # because tool.execution_start arrives slightly after assistant.message.
            if registered_tool_names_stream and not complete.is_set():
                try:
                    await asyncio.wait_for(complete.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    pass  # Give up; no tool calls detected

            # Phase 3: yield the appropriate final chunk.
            if captured_tool_calls_stream:
                # Emit tool-call intents as tool_call_chunks so they accumulate
                # into AIMessage.tool_calls when the caller merges all chunks.
                tool_call_chunks = [
                    {
                        "name": tc["name"],
                        "args": json.dumps(tc["args"]),
                        "id": tc["id"],
                        "index": i,
                        "type": "tool_call_chunk",
                    }
                    for i, tc in enumerate(captured_tool_calls_stream)
                ]
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=tool_call_chunks,
                        response_metadata={"model_name": effective_model},
                    )
                )
            else:
                # Yield a final empty chunk with response_metadata so the merged
                # result has model_name available (e.g. test_stream_with_model_override)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={"model_name": effective_model},
                    )
                )

        finally:
            # Disconnect session only; the shared client stays alive for reuse
            await session.disconnect()

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

            model = CopilotChatModel(model="gpt-5-mini")
            model_with_tools = model.bind_tools([get_weather])
            ```
        """
        # Convert LangChain tools to Copilot SDK Tool format
        copilot_tools = []

        for tool in tools:
            if isinstance(tool, Tool):
                # Already a Copilot SDK Tool
                copilot_tools.append(tool)
            elif isinstance(tool, BaseTool):
                # LangChain BaseTool - convert to Copilot format
                # Must be checked before callable() since BaseTool is also callable
                tool_schema = convert_to_openai_tool(tool)

                # Create an async handler that wraps the BaseTool
                # We need to capture the tool instance in the closure
                def create_handler(base_tool: BaseTool):
                    async def handler(invocation):
                        args = invocation.arguments or {}
                        try:
                            # Try async invoke first
                            if hasattr(base_tool, "ainvoke"):
                                result = await base_tool.ainvoke(args)
                            else:
                                # Fall back to sync invoke
                                result = base_tool.invoke(args)

                            return ToolResult(
                                text_result_for_llm=str(result),
                                result_type="success",
                            )
                        except Exception as e:
                            return ToolResult(
                                text_result_for_llm=f"Error: {str(e)}",
                                result_type="failure",
                                error=str(e),
                            )

                    return handler

                copilot_tool = Tool(
                    name=tool_schema["function"]["name"],
                    description=tool_schema["function"]["description"],
                    parameters=tool_schema["function"]["parameters"],
                    handler=create_handler(tool),
                    overrides_built_in_tool=True,
                )
                copilot_tools.append(copilot_tool)
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
                                args = invocation.arguments or {}
                                try:
                                    # Check if it's async
                                    if asyncio.iscoroutinefunction(func):
                                        result = await func(**args)
                                    else:
                                        result = func(**args)

                                    return ToolResult(
                                        text_result_for_llm=str(result),
                                        result_type="success",
                                    )
                                except Exception as e:
                                    return ToolResult(
                                        text_result_for_llm=f"Error: {str(e)}",
                                        result_type="failure",
                                        error=str(e),
                                    )

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
