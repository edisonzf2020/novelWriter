# Gemini Code Assistant Context

## Project Overview

This project is `novelWriter`, a plain text editor for writing novels. It is written in Python and uses the PyQt6 framework for its graphical user interface. The application is designed to help authors organize their work, with a focus on novels assembled from many smaller text documents. It uses a Markdown-like syntax for formatting and supports metadata for comments, synopses, and cross-referencing.

The project is structured as a Python package. The main application logic is located in the `novelwriter` directory. The user interface is built with PyQt6, and the main window is defined in `novelwriter/guimain.py`. The application's entry point is `novelWriter.py`, which in turn calls the `main` function in the `novelwriter` package.

## Building and Running

This is a Python project, so there is no explicit build step required. To run the application, you need to have Python and the required dependencies installed.

**Dependencies:**

The main dependencies are listed in `pyproject.toml` and `requirements.txt`. The core dependencies are:

*   `PyQt6`
*   `pyenchant`

You can install the dependencies using pip:

```bash
pip install -r requirements.txt
```

**Running the application:**

To run the application, execute the `novelWriter.py` script:

```bash
python novelWriter.py
```

**Running tests:**

The project has a `tests` directory and a `run_tests.py` script. To run the tests, execute the script:

```bash
python run_tests.py
```

## Development Conventions

*   **Coding Style:** The project uses `ruff` and `isort` for code formatting and linting. The configuration for these tools can be found in `pyproject.toml`.
*   **Testing:** The project uses `pytest` for testing. Tests are located in the `tests` directory.
*   **Contributions:** The `CONTRIBUTING.md` file provides guidelines for contributing to the project. It is recommended to discuss new features with the maintainer before submitting a pull request.
