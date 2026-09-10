"""Example 3: Asynchronous invocation."""

import asyncio

from langchain_core.messages import HumanMessage

from langchain_copilot import CopilotChatModel


async def main():
    """Asynchronous invocation example."""
    print("=" * 60)
    print("Example: Async Invoke")
    print("=" * 60)

    model = CopilotChatModel(model="gpt-5-mini")

    messages = [HumanMessage(content="Explain async programming in one sentence.")]

    response = await model.ainvoke(messages)
    print(f"Response: {response.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
