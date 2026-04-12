"""Example 2: Streaming response."""

from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage


def main():
    """Streaming response example."""
    print("=" * 60)
    print("Example: Streaming")
    print("=" * 60)

    model = CopilotChatModel(model="gpt-5-mini", streaming=True)

    messages = [HumanMessage(content="Write a haiku about coding.")]

    print("Response: ", end="", flush=True)
    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
