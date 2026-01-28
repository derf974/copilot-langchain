# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-28

### Added
- LangChain standard interface compliance via `langchain-tests`
- Comprehensive test suite with unit and integration tests
- Support for tool binding and function calling
- Additional examples showcasing temperature control, async operations, and tool usage
- CI/CD automation with GitHub Actions for testing and publishing
- Automated PyPI publishing via GitHub Releases with Trusted Publishing
- TestPyPI manual publishing workflow for pre-release testing

### Changed
- Improved error handling and SDK event suppression
- Enhanced documentation with QUICKSTART.md, TESTING.md, and RELEASE.md
- Synchronized versioning between pyproject.toml and __init__.py

### Fixed
- AsyncIO event loop handling in synchronous contexts
- Version mismatch between package metadata and source code

## [0.1.0] - 2026-01-15

### Added
- Initial release of langchain-copilot
- CopilotChatModel implementing LangChain BaseChatModel interface
- Shared client pattern with lazy initialization
- Full async/sync support for generate and stream operations
- Event-based SDK integration with GitHub Copilot CLI
- Support for system, human, and AI messages
- Basic examples demonstrating invoke, streaming, and async operations
- Development tooling: pytest, black, ruff, uv integration
- MIT License

[unreleased]: https://github.com/derf974/copilot-langchain/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/derf974/copilot-langchain/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/derf974/copilot-langchain/releases/tag/v0.1.0
