"""Example 1: Simple synchronous invocation."""

from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage, SystemMessage


def main():
    """Simple synchronous invocation example."""
    print("=" * 60)
    print("Example: Simple Invoke")
    print("=" * 60)

    model = CopilotChatModel(model="gpt-5-mini")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is LangChain?"),
    ]

    response = model.invoke(messages)
    print(f"Response: {response.content}\n")


if __name__ == "__main__":
    main()
