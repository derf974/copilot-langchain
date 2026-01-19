"""Example 5: Using in a LangChain chain."""

from langchain_copilot import CopilotChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    """LangChain chain example."""
    print("=" * 60)
    print("Example: LangChain Chain")
    print("=" * 60)

    model = CopilotChatModel(model_name="gpt-4o")

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

    print(f"Translation: {result}\n")


if __name__ == "__main__":
    main()
