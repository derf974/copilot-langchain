"""Example 8: Using bind_tools for LangChain integration."""

from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from copilot import define_tool
from pydantic import BaseModel, Field


# Method 1: Using Copilot SDK's @define_tool decorator
class WeatherParams(BaseModel):
    """Parameters for weather lookup."""

    location: str = Field(description="City name")


@define_tool(description="Get current weather for a location")
async def get_weather(params: WeatherParams) -> str:
    """Get weather information."""
    weather_data = {
        "paris": "15°C, cloudy",
        "new york": "10°C, sunny",
        "tokyo": "18°C, rainy",
    }
    location = params.location.lower()
    return weather_data.get(
        location, f"Weather data not available for {params.location}"
    )


# Method 2: Using LangChain's @tool decorator
@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query string
    """
    # Simulated search results
    results = {
        "python": "Python is a high-level programming language...",
        "langchain": "LangChain is a framework for developing applications with LLMs...",
        "copilot": "GitHub Copilot is an AI pair programmer...",
    }
    for key in results:
        if key in query.lower():
            return results[key]
    return f"No results found for: {query}"


def main():
    """Demonstrate bind_tools usage."""
    print("=" * 60)
    print("Example: Using bind_tools")
    print("=" * 60)

    # Create base model
    model = CopilotChatModel(model="gpt-4o")

    # Bind tools using bind_tools (LangChain standard pattern)
    # This returns a RunnableBinding that includes the tools
    model_with_tools = model.bind_tools([get_weather, search_web])

    print("\n[Query 1] Using bound tools:")
    messages = [HumanMessage(content="What's the weather in Paris?")]
    response = model_with_tools.invoke(messages)
    print(f"Response: {response.content}\n")

    print("\n[Query 2] Using search tool:")
    messages = [HumanMessage(content="Search for information about LangChain")]
    response = model_with_tools.invoke(messages)
    print(f"Response: {response.content}\n")

    # You can also bind tools and use in chains
    print("\n[Query 3] Using in a chain:")
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant with access to tools."),
            ("human", "{question}"),
        ]
    )

    chain = prompt | model_with_tools | StrOutputParser()

    result = chain.invoke({"question": "What's the weather in Tokyo?"})
    print(f"Chain result: {result}\n")


if __name__ == "__main__":
    main()
