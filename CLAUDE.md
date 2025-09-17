# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

novelWriter is a PyQt6-based desktop application for writing and organizing novels. It's written in Python 3.10+ and uses a modular architecture with separated core logic, GUI components, and text processing functionality.

## Core Architecture

The codebase follows MVC pattern with clear separation:

- **`novelwriter/core/`** - Project data management, document handling, and business logic
- **`novelwriter/gui/`** - Main interface, editor widgets, and UI components  
- **`novelwriter/dialogs/`** - Modal dialogs and preference windows
- **`novelwriter/formats/`** - Export formats (HTML, LaTeX, PDF, etc.)
- **`novelwriter/text/`** - Text processing, tokenization, and document analysis
- **`novelwriter/tools/`** - Utilities for project statistics and management
- **`novelwriter/extensions/`** - Custom PyQt widgets and UI extensions
- **`novelwriter/ai/`** - AI integration and API handling

Entry points are `novelWriter.py` (script) or `python -m novelwriter` command.

## Development Commands

### Testing
```bash
# Run full test suite (offscreen mode for CI/headless)
python run_tests.py -o

# Run tests with coverage reports
python run_tests.py -r -t

# Run specific test modules or filters
python run_tests.py -m "core" -k "test_project"

# Direct pytest (requires display)
pytest -vv
```

### Linting and Type Checking
```bash
# Run ruff linting (must pass for PR acceptance)
ruff check novelwriter tests

# Auto-fix linting issues where possible
ruff check --fix novelwriter tests

# Run import sorting (configured in pyproject.toml)
isort .

# Type checking with pyright (must pass)
pyright
```

### Running the Application
```bash
# Launch desktop application
python novelWriter.py

# Or using installed package
novelwriter
```

## Code Style Guidelines

- Line length: 99 characters maximum
- Type annotations required on all functions/parameters
- Docstrings follow pydocstyle conventions
- Use f-strings for formatting, avoid `%` except in logging
- Function names in camelCase (Qt consistency)
- No trailing whitespace, 4-space indentation only
- Test coverage required for all new code

## Key Configuration

- **pyproject.toml** - Contains all tool configurations (ruff, pytest, pyright, etc.)
- **setup/requirements** - Runtime and development dependencies
- **i18n/** - Translation files (use Crowdin for contributions)

## Important Notes

- GUI components require PyQt6 environment 
- Tests run on Linux, Windows, macOS - must pass on all platforms
- New features go to `main` branch, fixes to `release` branch
- Do not auto-format with black/ruff - manual formatting preferred
- All user-facing text must be UK English and wrapped in Qt translation calls