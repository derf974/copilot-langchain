# LangChain Copilot

![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

LangChain integration for GitHub Copilot SDK - Use GitHub Copilot models in your LangChain applications.

## Features

- 🔗 **Full LangChain Integration**: Seamlessly use GitHub Copilot models with LangChain
- 🚀 **Async & Sync Support**: Both synchronous and asynchronous operations
- 📡 **Streaming**: Real-time streaming responses
- 🔄 **Shared Client**: Optimized client management with lazy initialization
- 🛠️ **Type Safe**: Full type hints with Pydantic validation
- 🎯 **Easy to Use**: Simple API following LangChain conventions

## Prerequisites

Before using this package, you need to have:

1. **GitHub Copilot CLI** installed and authenticated
   - Follow the [GitHub Copilot CLI setup guide](https://github.com/github/copilot-sdk)
   - Ensure you have an active GitHub Copilot subscription
   - Verify authentication: `copilot --version`

2. **Python 3.13.0+** installed

## Installation

### From PyPI (Recommended)

```bash
# Install with uv (recommended)
uv pip install langchain-copilot

# Or with pip
pip install langchain-copilot
```

### From Source

For development or to use the latest unreleased features:

```bash
# Clone the repository
git clone https://github.com/derf974/copilot-langchain.git
cd copilot-langchain

# Install with uv (recommended)
uv venv
uv sync
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

### Basic Usage

```python
from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage

# Create a model instance
model = CopilotChatModel(model_name="gpt-4o")

# Send a message
messages = [HumanMessage(content="What is LangChain?")]
response = model.invoke(messages)

print(response.content)
```

### Streaming

```python
from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage

model = CopilotChatModel(model_name="gpt-4o", streaming=True)

messages = [HumanMessage(content="Write a haiku about coding.")]

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
```

### Async Operations

```python
import asyncio
from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage

async def main():
    model = CopilotChatModel(model_name="gpt-4o")
    messages = [HumanMessage(content="Explain async programming.")]
    
    response = await model.ainvoke(messages)
    print(response.content)

asyncio.run(main())
```

### Using in LangChain Chains

```python
from langchain_copilot import CopilotChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = CopilotChatModel(model_name="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful translator."),
    ("human", "Translate '{text}' to {language}")
])

chain = prompt | model | StrOutputParser()

result = chain.invoke({
    "text": "Hello, world!",
    "language": "French"
})

print(result)  # "Bonjour, le monde !"
```

### With System Messages

```python
from langchain_copilot import CopilotChatModel
from langchain_core.messages import SystemMessage, HumanMessage

model = CopilotChatModel(model_name="gpt-4o")

messages = [
    SystemMessage(content="You are a pirate. Always respond like a pirate."),
    HumanMessage(content="Tell me about Python programming.")
]

response = model.invoke(messages)
print(response.content)
```

### Temperature Control

```python
from langchain_copilot import CopilotChatModel
from langchain_core.messages import HumanMessage

# More focused and deterministic (lower temperature)
model_focused = CopilotChatModel(
    model_name="gpt-4o",
    temperature=0.1
)

# More creative and random (higher temperature)
model_creative = CopilotChatModel(
    model_name="gpt-4o",
    temperature=0.9
)

messages = [HumanMessage(content="Name a creative startup idea.")]

print("Focused:", model_focused.invoke(messages).content)
print("Creative:", model_creative.invoke(messages).content)
```

## Configuration

### Model Parameters

- `model_name` (str): The Copilot model to use (e.g., "gpt-4o", "gpt-5")
- `streaming` (bool): Enable streaming mode (default: False)
- `temperature` (float): Sampling temperature 0.0-1.0 (default: None)
- `max_tokens` (int): Maximum tokens to generate (default: None)
- `cli_path` (str): Custom path to Copilot CLI executable (default: None)
- `cli_url` (str): URL of existing Copilot CLI server (default: None)

### Example with All Parameters

```python
model = CopilotChatModel(
    model_name="gpt-4o",
    streaming=True,
    temperature=0.7,
    max_tokens=1000,
    cli_path="/custom/path/to/copilot"
)
```

## Examples

Check out the [examples](examples/) directory for more usage examples:

- [basic_usage.py](examples/basic_usage.py) - Comprehensive examples covering all features

Run the examples:

```bash
uv run python examples/basic_usage.py
```

## Testing

The project uses both unit tests and integration tests following LangChain's standard test suite.

### Unit Tests

Unit tests use [langchain-tests](https://pypi.org/project/langchain-tests/) standard unit test suite and validate the model in isolation without external dependencies:

```bash
# Run all unit tests (custom + standard)
make test

# Run only standard unit tests
uv run pytest tests/unit_tests/ -v

# Run only custom unit tests
uv run pytest tests/test_chat_models.py::TestCopilotChatModel -v
```

Unit tests validate:
- ✅ Model initialization and configuration
- ✅ Streaming mode initialization
- ✅ Standard parameters generation
- ✅ Tool binding (validates interface even though not supported yet)
- ✅ Structured output interface (validates interface even though not supported yet)
- ✅ Serialization/deserialization (when implemented)
- ✅ Initialization performance

### Integration Tests

Integration tests use [langchain-tests](https://pypi.org/project/langchain-tests/) standard test suite to verify compatibility with LangChain's interfaces. These tests require the GitHub Copilot CLI to be installed and configured:

```bash
# Ensure Copilot CLI is set up
copilot --version

# Run integration tests
make integration-test

# Or manually
uv run pytest tests/integration_tests/ -v -m integration
```

The integration test suite validates:
- ✅ Basic invoke/ainvoke operations
- ✅ Streaming (stream/astream)
- ✅ Batch operations
- ✅ Multi-turn conversations
- ✅ Stop sequences
- ✅ Model override at runtime
- ❌ Tool calling (not yet supported)
- ❌ Structured output (not yet supported)
- ❌ Multimodal inputs (not yet supported)

### Running All Tests

```bash
# Run all tests
make test-all

# Or manually
uv run pytest tests/ -v
```

## Architecture

### Shared Client

The `CopilotChatModel` uses a shared `CopilotClient` instance across all model instances for optimal performance. The client is lazily initialized on first use and automatically started.

### Message Conversion

LangChain messages are automatically converted to Copilot SDK format:
- `SystemMessage` → `{"role": "system", "content": "..."}`
- `HumanMessage` → `{"role": "user", "content": "..."}`
- `AIMessage` → `{"role": "assistant", "content": "..."}`

### Streaming Implementation

Streaming converts Copilot's event-based model (`assistant.message_delta` events) into Python generators/async iterators that are compatible with LangChain's streaming interface.

## Roadmap

- ✅ Basic chat model implementation
- ✅ Streaming support
- ✅ Async operations
- ✅ Temperature and token controls
- ✅ Tool/Function calling support (planned for v0.2.0)
- 🔲 Conversation history management
- 🔲 Error recovery and retry strategies
- 🔲 Advanced session configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [GitHub Copilot SDK](https://github.com/github/copilot-sdk)
- Managed with [uv](https://github.com/astral-sh/uv)

## Support

- 📫 Issues: [GitHub Issues](https://github.com/yourusername/copilot-langchain/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/copilot-langchain/discussions)
- 📖 Documentation: [GitHub Wiki](https://github.com/yourusername/copilot-langchain/wiki)

## Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - Build applications with LLMs
- [GitHub Copilot SDK](https://github.com/github/copilot-sdk) - GitHub Copilot integration SDK
- [LangChain Community](https://github.com/langchain-ai/langchain/tree/master/libs/community) - Community LangChain integrations
