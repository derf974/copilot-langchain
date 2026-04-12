"""Example 4: Asynchronous streaming."""

import asyncio
from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage


async def main():
    """Asynchronous streaming example."""
    print("=" * 60)
    print("Example: Async Streaming")
    print("=" * 60)

    model = CopilotChatModel(model="gpt-5-mini", streaming=True)

    messages = [HumanMessage(content="Count from 1 to 5.")]

    print("Response: ", end="", flush=True)
    async for chunk in model.astream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
