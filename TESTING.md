# Testing Guide

This document provides detailed information about testing in the langchain-copilot project.

## Overview

The project uses two types of tests:

1. **Unit Tests**: Fast tests that mock external dependencies
2. **Integration Tests**: Tests that verify real interaction with the Copilot CLI

## Test Structure

```
tests/
├── __init__.py
├── test_chat_models.py          # Custom unit tests
├── unit_tests/                  # LangChain standard unit tests
│   ├── __init__.py
│   └── test_standard.py
└── integration_tests/           # LangChain standard integration tests
    ├── __init__.py
    └── test_standard.py
```

## Unit Tests

Unit tests validate the model in isolation without requiring external dependencies. There are two types:

1. **Standard Unit Tests** (`tests/unit_tests/test_standard.py`): LangChain standard test suite
2. **Custom Unit Tests** (`tests/test_chat_models.py`): Project-specific tests

### Running Unit Tests

```bash
# Using make (runs all unit tests)
make test

# Run all unit tests manually
uv run pytest tests/unit_tests/ -v

# Run only custom unit tests
uv run pytest tests/test_chat_models.py -v

# Run a specific test class
uv run pytest tests/test_chat_models.py::TestCopilotChatModel -v

# Run a specific test function
uv run pytest tests/test_chat_models.py::TestCopilotChatModel::test_initialization -v
```

### What Unit Tests Cover

**Standard Unit Tests** (from langchain-tests):
- ✅ Model initialization (`test_init`)
- ✅ Streaming initialization (`test_init_streaming`)
- ✅ Standard parameters generation (`test_standard_params`)
- ✅ Tool binding interface (`test_bind_tool_pydantic`)
- ✅ Structured output interface (`test_with_structured_output`)
- ✅ Initialization performance (`test_init_time`)
- ❌ Environment variable initialization (not supported)
- ❌ Serialization/deserialization (not yet implemented)

**Custom Unit Tests**:
- Model initialization and configuration
- Message conversion logic
- Session configuration creation
- Client lifecycle management
- Mock-based async operations
- System message handling

## Integration Tests

Integration tests use LangChain's standard test suite to ensure compatibility with the LangChain ecosystem.

### Prerequisites

Before running integration tests:

1. **Install the Copilot CLI**
   ```bash
   # Follow GitHub's installation guide
   # https://github.com/github/copilot-sdk
   ```

2. **Authenticate**
   ```bash
   # Verify authentication
   copilot --version
   ```

3. **Have an active Copilot subscription**

### Running Integration Tests

```bash
# Using make
make integration-test

# Using pytest directly
uv run pytest tests/integration_tests/ -v -m integration

# Run all tests including integration
make test-all
```

### What Integration Tests Cover

The integration test suite (`test_standard.py`) inherits from `ChatModelIntegrationTests` and validates:

#### ✅ Supported Features

- **Basic Operations**
  - `test_invoke`: Synchronous message invocation
  - `test_ainvoke`: Asynchronous message invocation
  - `test_stream`: Synchronous streaming
  - `test_astream`: Asynchronous streaming

- **Batch Operations**
  - `test_batch`: Process multiple messages synchronously
  - `test_abatch`: Process multiple messages asynchronously

- **Conversations**
  - `test_conversation`: Multi-turn conversation handling
  - `test_double_messages_conversation`: Handle multiple consecutive messages of the same role

- **Model Configuration**
  - `test_stop_sequence`: Stop sequences parameter
  - `test_invoke_with_model_override`: Runtime model override
  - `test_ainvoke_with_model_override`: Async runtime model override
  - `test_stream_with_model_override`: Streaming with model override
  - `test_astream_with_model_override`: Async streaming with model override

- **Message Handling**
  - `test_message_with_name`: Messages with name field

#### ❌ Unsupported Features (Skipped)

These tests are automatically skipped because CopilotChatModel doesn't yet support these features:

- Tool/function calling
- Structured output
- JSON mode
- Multimodal inputs (images, PDFs, audio, video)
- Usage metadata
- Anthropic-style inputs

### Configuring Test Features

Feature support is configured via properties in `test_standard.py`:

```python
@property
def has_tool_calling(self) -> bool:
    """CopilotChatModel does not support tool calling (yet)."""
    return False

@property
def supports_image_inputs(self) -> bool:
    """CopilotChatModel does not support image inputs (yet)."""
    return False
```

When implementing new features, update these properties to enable the corresponding tests.

## LangChain Standard Tests

The integration tests use the [`langchain-tests`](https://pypi.org/project/langchain-tests/) package, which provides:

- Standardized test suites for LangChain components
- Ensures compatibility with the LangChain ecosystem
- Validates that the implementation follows LangChain's contracts
- Tests common edge cases and error conditions

### Test Class Structure

```python
class TestCopilotChatModelIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[CopilotChatModel]:
        """Return the chat model class to test."""
        return CopilotChatModel

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the model."""
        return {
            "model_name": "gpt-5-mini",
            "temperature": 0,
        }
```

### Version Pinning

The project pins `langchain-tests>=1.0.0` to ensure test stability. When updating:

```bash
# Update langchain-tests
uv add --dev langchain-tests@latest

# Run tests to verify compatibility
make test-all
```

## Writing New Tests

### Adding Unit Tests

1. Add tests to `tests/test_chat_models.py`
2. Use mocks for external dependencies
3. Follow existing patterns for async tests

Example:
```python
@pytest.mark.asyncio
async def test_my_feature(self):
    """Test description."""
    model = CopilotChatModel()
    # Test implementation
    assert result == expected
```

### Enabling Integration Tests for New Features

When implementing a new feature:

1. Update the corresponding property in `tests/integration_tests/test_standard.py`
2. Run integration tests to verify
3. Update this documentation

Example:
```python
@property
def has_tool_calling(self) -> bool:
    """CopilotChatModel now supports tool calling."""
    return True  # Changed from False
```

## Continuous Integration

The CI pipeline (`.github/workflows/ci.yml`) runs:

1. Linting with ruff
2. Format checking with black  
3. Unit tests only (integration tests require CLI setup)

Integration tests should be run locally before submitting PRs.

## Troubleshooting

### Integration Tests Fail

If integration tests fail:

1. **Check Copilot CLI**
   ```bash
   copilot --version
   ```

2. **Verify Authentication**
   ```bash
   # Re-authenticate if needed
   copilot auth login
   ```

3. **Check Subscription**
   - Ensure your GitHub Copilot subscription is active

4. **Event Loop Issues**
   - Some tests may fail with "Event loop is closed" errors
   - This usually indicates the Copilot SDK client needs to be restarted
   - Try running tests individually instead of in batch

### Unit Tests Fail

If unit tests fail:

1. **Check Dependencies**
   ```bash
   uv sync --all-extras --dev
   ```

2. **Clear Cache**
   ```bash
   make clean
   ```

3. **Verify Python Version**
   ```bash
   python --version  # Should be 3.13+
   ```

## Best Practices

1. **Run unit tests frequently** during development
2. **Run integration tests** before submitting PRs
3. **Keep unit tests fast** by using mocks
4. **Document new test requirements** in this guide
5. **Pin langchain-tests version** to avoid CI breakage

## References

- [LangChain Standard Tests Documentation](https://docs.langchain.com/oss/python/contributing/standard-tests-langchain)
- [langchain-tests PyPI](https://pypi.org/project/langchain-tests/)
- [ChatModelIntegrationTests API](https://reference.langchain.com/python/langchain_tests/integration_tests/chat_models/)
- [pytest Documentation](https://docs.pytest.org/)
