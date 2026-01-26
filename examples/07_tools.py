"""Example 7: Using tools with Copilot."""

from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage
from copilot import define_tool
from pydantic import BaseModel, Field


# Define a tool parameter schema using Pydantic
class WeatherParams(BaseModel):
    """Parameters for weather lookup."""

    location: str = Field(description="City name (e.g., 'Paris', 'New York')")
    unit: str = Field(
        default="celsius", description="Temperature unit: 'celsius' or 'fahrenheit'"
    )


# Define a tool using the @define_tool decorator
@define_tool(description="Get current weather for a location")
async def get_weather(params: WeatherParams) -> str:
    """Simulate weather lookup (in reality, you'd call a weather API)."""
    # Simulate weather data
    weather_data = {
        "paris": {"celsius": "15°C", "fahrenheit": "59°F", "condition": "cloudy"},
        "new york": {"celsius": "10°C", "fahrenheit": "50°F", "condition": "sunny"},
        "tokyo": {"celsius": "18°C", "fahrenheit": "64°F", "condition": "rainy"},
    }

    location = params.location.lower()
    if location in weather_data:
        temp = weather_data[location].get(params.unit.lower(), "N/A")
        condition = weather_data[location]["condition"]
        return f"The weather in {params.location} is {condition} with a temperature of {temp}."
    else:
        return f"Weather data not available for {params.location}."


class CalculatorParams(BaseModel):
    """Parameters for calculator."""

    operation: str = Field(
        description="Operation to perform: 'add', 'subtract', 'multiply', or 'divide'"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


@define_tool(description="Perform basic arithmetic operations")
async def calculator(params: CalculatorParams) -> str:
    """Perform a calculation."""
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else "Error: Division by zero",
    }

    op = params.operation.lower()
    if op in operations:
        result = operations[op](params.a, params.b)
        return f"{params.a} {op} {params.b} = {result}"
    else:
        return f"Unknown operation: {params.operation}"


def main():
    """Tools usage example."""
    print("=" * 60)
    print("Example: Using Tools with Copilot")
    print("=" * 60)

    # Create a model with tools
    model = CopilotChatModel(model="gpt-4o", tools=[get_weather, calculator])

    # Example 1: Weather query
    print("\n[Query 1] Weather information:")
    messages = [
        HumanMessage(content="What's the weather like in Paris? Also in Tokyo?")
    ]
    response = model.invoke(messages)
    print(f"Response: {response.content}\n")

    # Example 2: Calculation query
    print("\n[Query 2] Calculation:")
    messages = [HumanMessage(content="What is 123 multiplied by 456?")]
    response = model.invoke(messages)
    print(f"Response: {response.content}\n")

    # Example 3: Combined query
    print("\n[Query 3] Combined tools:")
    messages = [
        HumanMessage(
            content="What's the weather in New York? If it's above 15°C, "
            "calculate what 72 divided by 8 is."
        )
    ]
    response = model.invoke(messages)
    print(f"Response: {response.content}\n")


if __name__ == "__main__":
    main()
