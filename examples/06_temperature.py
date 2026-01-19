"""Example 6: Using temperature parameter."""

from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage


def main():
    """Temperature control example."""
    print("=" * 60)
    print("Example: Temperature Control")
    print("=" * 60)

    # Lower temperature = more focused and deterministic
    model_focused = CopilotChatModel(model_name="gpt-4o", temperature=0.1)

    # Higher temperature = more creative and random
    model_creative = CopilotChatModel(model_name="gpt-4o", temperature=0.9)

    messages = [HumanMessage(content="Tell me a creative name for a coffee shop.")]

    print("Focused response (temp=0.1):")
    response1 = model_focused.invoke(messages)
    print(f"  {response1.content}\n")

    print("Creative response (temp=0.9):")
    response2 = model_creative.invoke(messages)
    print(f"  {response2.content}\n")


if __name__ == "__main__":
    main()
