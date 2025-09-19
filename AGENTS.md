# First Rule
- Always respond in Chinese.
- Whenever you are confused about the specific use of the technology, always be sure to call the MCP tool `context7` to determine the latest technical details.
- Whenever you need to get the current time, be sure to call the MCP tool `time-mcp`.

# Repository Guidelines

## Project Structure & Module Organization
novelWriter adopts a layered MVC architecture. Source code lives in `novelwriter/`, with `core/` handling project logic, `gui/` providing the PyQt6 interface, `formats/` powering exporters, `extensions/` supplying custom widgets, and `dialogs/`, `text/`, and `tools/` covering supporting features. Tests reside in `tests/`, mirroring the module tree. Documentation is in `docs/`, while packaging scripts and installers live in `setup/`.

## Build, Test & Development Commands
- `python -m pip install -e .[dev]` — install the editable package with development extras.
- `python novelWriter.py` — launch the desktop app against local sources.
- `python run_tests.py` or `pytest` — execute the automated test suite before submitting changes.
- `ruff check novelwriter tests` — run linting configured in `pyproject.toml`.
- `pyright` — enforce static typing rules aligned with CI.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and a 99-character line ceiling (wrap at 79 when practical). Classes use PascalCase, methods favour camelCase to match Qt, and constants remain UPPER_CASE. Provide type hints for every function and non-trivial attribute; avoid `Any`. Prefer f-strings, reserve `%` formatting for logging, and wrap user-facing strings in Qt `tr()` with UK English spelling. Skip global auto-formatting; limited `isort` usage is acceptable.

## Testing Guidelines
Pytest suites live under `tests/` (`test_core`, `test_gui`, `test_formats`, etc.) and should mirror module names. Name files `test_<module>.py` and functions `test_<behavior>`. Use fixtures from `tests/fixtures` and add new ones when needed. Cover all new control paths, including platform-dependent branches, and run `pytest --maxfail=1 --disable-warnings` locally before opening a pull request.

## Commit & Pull Request Guidelines
Author commits in English, present tense (e.g., `Fix project tree duplication`) and reference issues with `#<id>`. Target `main` for features and `release` for hotfixes. Pull requests must summarise motivation, detail testing, and attach screenshots or GIFs for UI changes. Keep diffs focused: avoid drive-by formatting, documentation rewrites, or version bumps unless agreed.

## Security & Configuration Tips
Do not commit API keys or personal project data. Respect file paths stored in project XML and JSON artefacts. Store secrets outside the repo (e.g., `.env`, OS keychain) and verify packaging scripts before distribution builds.
