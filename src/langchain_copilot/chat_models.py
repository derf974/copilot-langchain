"""LangChain ChatModel implementation using GitHub Copilot SDK."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, ClassVar, Iterator, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, model_validator

from copilot import CopilotClient
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
                    options = {}
                    if self.cli_path:
                        options["cli_path"] = self.cli_path
                    if self.cli_url:
                        options["cli_url"] = self.cli_url

                    CopilotChatModel._shared_client = CopilotClient(**options)

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

    def _create_session_config(self) -> dict[str, Any]:
        """Create session configuration for Copilot SDK.

        Returns:
            Configuration dictionary for creating a Copilot session
        """
        config = {
            "model": self.model_name,
            "streaming": self.streaming,
        }

        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.max_tokens is not None:
            config["max_tokens"] = self.max_tokens

        return config

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
        session_config = self._create_session_config()

        # Create a session
        session = await client.create_session(session_config)

        try:
            # Convert messages
            copilot_messages = self._convert_messages(messages)

            # Send the last message and collect response
            response_content = ""
            complete = asyncio.Event()

            def on_event(event):
                nonlocal response_content
                try:
                    if event.type.value == "assistant.message":
                        response_content = event.data.content
                        complete.set()
                    elif event.type.value == "assistant.message_delta":
                        response_content += event.data.content
                except (AttributeError, KeyError):
                    # Ignore malformed events
                    pass

            # Register event listener
            session.on(on_event)

            # Send message with proper format
            if len(copilot_messages) > 0:
                # Send the last message with prompt format
                await session.send({"prompt": copilot_messages[-1]["content"]})

            # Wait for response
            await complete.wait()

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
        session_config = self._create_session_config()
        session_config["streaming"] = True  # Force streaming mode

        # Create a session
        session = await client.create_session(session_config)

        try:
            # Convert messages
            copilot_messages = self._convert_messages(messages)

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

            # Send message with proper format
            if len(copilot_messages) > 0:
                await session.send({"prompt": copilot_messages[-1]["content"]})

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
