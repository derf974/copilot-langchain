"""Basic usage examples for langchain-copilot."""

import asyncio
from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage, SystemMessage


def example_simple_invoke():
    """Example 1: Simple synchronous invocation."""
    print("=" * 60)
    print("Example 1: Simple Invoke")
    print("=" * 60)
    
    model = CopilotChatModel(model_name="gpt-4o")
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is LangChain?")
    ]
    
    response = model.invoke(messages)
    print(f"Response: {response.content}\n")


def example_streaming():
    """Example 2: Streaming response."""
    print("=" * 60)
    print("Example 2: Streaming")
    print("=" * 60)
    
    model = CopilotChatModel(model_name="gpt-4o", streaming=True)
    
    messages = [
        HumanMessage(content="Write a haiku about coding.")
    ]
    
    print("Response: ", end="", flush=True)
    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")


async def example_async_invoke():
    """Example 3: Asynchronous invocation."""
    print("=" * 60)
    print("Example 3: Async Invoke")
    print("=" * 60)
    
    model = CopilotChatModel(model_name="gpt-4o")
    
    messages = [
        HumanMessage(content="Explain async programming in one sentence.")
    ]
    
    response = await model.ainvoke(messages)
    print(f"Response: {response.content}\n")


async def example_async_streaming():
    """Example 4: Asynchronous streaming."""
    print("=" * 60)
    print("Example 4: Async Streaming")
    print("=" * 60)
    
    model = CopilotChatModel(model_name="gpt-4o", streaming=True)
    
    messages = [
        HumanMessage(content="Count from 1 to 5.")
    ]
    
    print("Response: ", end="", flush=True)
    async for chunk in model.astream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")


def example_with_chain():
    """Example 5: Using in a LangChain chain."""
    print("=" * 60)
    print("Example 5: LangChain Chain")
    print("=" * 60)
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    model = CopilotChatModel(model_name="gpt-4o")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{text}")
    ])
    
    chain = prompt | model | StrOutputParser()
    
    result = chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "Hello, how are you?"
    })
    
    print(f"Translation: {result}\n")


def example_with_temperature():
    """Example 6: Using temperature parameter."""
    print("=" * 60)
    print("Example 6: Temperature Control")
    print("=" * 60)
    
    # Lower temperature = more focused and deterministic
    model_focused = CopilotChatModel(model_name="gpt-4o", temperature=0.1)
    
    # Higher temperature = more creative and random
    model_creative = CopilotChatModel(model_name="gpt-4o", temperature=0.9)
    
    messages = [
        HumanMessage(content="Tell me a creative name for a coffee shop.")
    ]
    
    print("Focused response (temp=0.1):")
    response1 = model_focused.invoke(messages)
    print(f"  {response1.content}\n")
    
    print("Creative response (temp=0.9):")
    response2 = model_creative.invoke(messages)
    print(f"  {response2.content}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LangChain Copilot - Usage Examples")
    print("=" * 60 + "\n")
    
    # Synchronous examples
    example_simple_invoke()
    example_streaming()
    example_with_chain()
    example_with_temperature()
    
    # Asynchronous examples
    print("\nRunning async examples...\n")
    asyncio.run(example_async_invoke())
    asyncio.run(example_async_streaming())
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
