"""Tests for CopilotChatModel."""

import pytest
from langchain_core.messages import HumanMessage
from langchain_copilot import CopilotChatModel


class TestCopilotChatModelIntegration:
    """Integration tests (require actual Copilot CLI setup)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_invocation(self):
        """Test real invocation (requires Copilot CLI)."""
        model = CopilotChatModel(model_name="gpt-5-mini")
        messages = [HumanMessage(content="Say 'test passed' and nothing else.")]

        result = await model._agenerate(messages)

        assert len(result.generations) == 1
        assert "test passed" in result.generations[0].message.content.lower()

    @pytest.mark.integration
    def test_real_invoke_sync(self):
        """Test real synchronous invocation (requires Copilot CLI)."""
        model = CopilotChatModel(model_name="gpt-5-mini")
        messages = [HumanMessage(content="Say 'test passed' and nothing else.")]

        result = model.invoke(messages)

        assert "test passed" in result.content.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_streaming(self):
        """Test real streaming (requires Copilot CLI)."""
        model = CopilotChatModel(model_name="gpt-5-mini", streaming=True)
        messages = [HumanMessage(content="Count from 1 to 3.")]

        chunks = []
        async for chunk in model._astream(messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Concatenate all chunks
        full_content = "".join(c.message.content for c in chunks)
        assert len(full_content) > 0

    @pytest.mark.integration
    def test_real_chain_with_system_message(self):
        """Test real LangChain chain with system messages (requires Copilot CLI)."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        model = CopilotChatModel(model_name="gpt-5-mini")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that translates {input_language} to {output_language}.",
                ),
                ("human", "{text}"),
            ]
        )

        chain = prompt | model | StrOutputParser()

        result = chain.invoke(
            {
                "input_language": "English",
                "output_language": "French",
                "text": "Hello, how are you?",
            }
        )

        # Check that the result is in French (should contain typical French words)
        # Not exact match since LLM output may vary
        result_lower = result.lower()
        assert any(
            word in result_lower
            for word in ["bonjour", "salut", "comment", "ça", "va", "allez"]
        ), f"Expected French translation but got: {result}"
