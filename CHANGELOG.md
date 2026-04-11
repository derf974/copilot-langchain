# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.8](https://github.com/derf974/copilot-langchain/compare/v0.2.7...v0.2.8) (2026-04-11)


### Bug Fixes

* should work with python 3.11 ([#29](https://github.com/derf974/copilot-langchain/issues/29)) ([186f1fe](https://github.com/derf974/copilot-langchain/commit/186f1fea5bbc35524fa1950b51d9a06684aa757f))

## [0.2.7](https://github.com/derf974/copilot-langchain/compare/v0.2.6...v0.2.7) (2026-04-07)


### Bug Fixes

* update dependencies and refactor CopilotChatModel configuration handling ([#27](https://github.com/derf974/copilot-langchain/issues/27)) ([2a0fabe](https://github.com/derf974/copilot-langchain/commit/2a0fabeeaa67dc6389b2ba6c8a11b748b4777aa2))

## [0.2.6](https://github.com/derf974/copilot-langchain/compare/v0.2.5...v0.2.6) (2026-03-08)


### Bug Fixes

* update release workflow to include build and publish steps ([#23](https://github.com/derf974/copilot-langchain/issues/23)) ([5ce637a](https://github.com/derf974/copilot-langchain/commit/5ce637aab3b7f52faaa50af81b7424fd3a32cadb))

## [0.2.5](https://github.com/derf974/copilot-langchain/compare/v0.2.4...v0.2.5) (2026-03-08)


### Bug Fixes

* trigger PyPI publish on release event instead of tag push ([#21](https://github.com/derf974/copilot-langchain/issues/21)) ([a9f4ebb](https://github.com/derf974/copilot-langchain/commit/a9f4ebb8950faedd204518a6b907ce62bdd9cc28))

## [0.2.4](https://github.com/derf974/copilot-langchain/compare/v0.2.3...v0.2.4) (2026-03-08)


### Bug Fixes

* add on permission request to  create session config ([#17](https://github.com/derf974/copilot-langchain/issues/17)) ([505d30d](https://github.com/derf974/copilot-langchain/commit/505d30dca3a9399459c156dbe4fd69fa5ac36dba))

## [0.2.3](https://github.com/derf974/copilot-langchain/compare/v0.2.2...v0.2.3) (2026-03-07)


### Bug Fixes

* update installation instructions for clarity and consistency ([ab59512](https://github.com/derf974/copilot-langchain/commit/ab59512505249b07222817f78483f5c7f8363284))

## [Unreleased]

## [0.2.3] - 2026-03-08

### Changed
- Updated `github-copilot-sdk` minimum version from `0.1.23` to `0.1.32`
- Updated `langchain-core` minimum version from `1.2.7` to `1.2.17`

## [0.2.2] - 2026-02-26
### Added
- Memory example demonstrating conversation history with `RunnableWithMessageHistory` (example 09)

### Fixed
- Fixed `cli_url` TypeError by passing options as dictionary to `CopilotClient`
- Fixed indefinite hang when only system messages are provided
- Improved message validation to raise clear errors for empty prompts

### Changed
- Refactored message handling and prompt construction logic
- Improved error messages for invalid message configurations
- Enhanced test coverage for edge cases

### Removed
- Removed temperature example (06_temperature.py) as it was redundant

## [0.2.1] - 2026-01-29

### Added
- GitHub Actions workflow for automated PyPI releases
- TestPyPI manual publishing workflow for pre-release testing
- Comprehensive release documentation (RELEASE.md)

### Fixed
- Fixed author name in pyproject.toml metadata
- Fixed project URLs configuration
- Fixed dependency declarations in pyproject.toml

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

[unreleased]: https://github.com/derf974/copilot-langchain/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/derf974/copilot-langchain/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/derf974/copilot-langchain/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/derf974/copilot-langchain/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/derf974/copilot-langchain/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/derf974/copilot-langchain/releases/tag/v0.1.0
