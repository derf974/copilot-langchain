# LangChain Copilot - Quick Start Guide

## Installation

```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

## Running Examples

```bash
# Make sure Copilot CLI is installed and authenticated
copilot --version

# Run the basic examples
uv run python examples/basic_usage.py
```

## Running Tests

```bash
# Unit tests only (no Copilot CLI needed)
uv run pytest tests/ -v -m "not integration"

# All tests (requires Copilot CLI)
uv run pytest tests/ -v
```

## Project Structure

```
copilot-langchain/
├── src/
│   └── langchain_copilot/
│       ├── __init__.py           # Package exports
│       └── chat_models.py        # Main CopilotChatModel class
├── tests/
│   ├── __init__.py
│   └── test_chat_models.py       # Test suite
├── examples/
│   └── basic_usage.py            # Usage examples
├── pyproject.toml                # Project configuration
├── pytest.ini                    # Pytest configuration
└── README.md                     # Documentation
```

## Next Steps

1. Check out [examples/basic_usage.py](examples/basic_usage.py) for detailed examples
2. Read the [README.md](README.md) for full documentation
3. Explore the API in [src/langchain_copilot/chat_models.py](src/langchain_copilot/chat_models.py)
