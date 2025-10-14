# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-08-16

### Fixed

- Resolved import issues with phi Assistant class
- Fixed type compatibility issues with numpy floating point values
- Corrected LLM method implementations to properly support phi framework methods (invoke, response)
- Fixed tokenizer usage and error handling throughout the codebase
- Addressed variable scoping issues in main.py
- Resolved PyMuPDF method calls for PDF text extraction
- Fixed attribute access issues with AutoTokenizer and AutoModel classes

### Changed

- Updated pre-commit hooks to ensure code quality
- Improved error handling and fallback mechanisms for better robustness
- Enhanced documentation with network mirror endpoint information

### Added

- Proper method implementations for HuggingFaceLLM class to support phi framework
- Better logging for debugging and monitoring agent responses
- Support for network mirror endpoints for users with connectivity issues

## [0.1.0] - 2025-08-15

### Added

- Initial release of MAS Consensus
- Multi-agent system for text processing
- Support for question answering and summarization tasks
- PDF and TXT file processing capabilities
- Chain of agents architecture with worker and manager agents
- Integration with Hugging Face models
