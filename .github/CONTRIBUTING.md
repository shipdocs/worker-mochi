# Contributing to worker-mochi

Thank you for your interest in contributing to worker-mochi! This document provides guidelines and instructions for contributing.

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU
- Docker (for container builds)

## Development Setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/runpod-workers/worker-mochi.git
   cd worker-mochi
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

## Video Output

The handler returns the generated video base64-encoded in the job output (`src/video_output.py`); it is not uploaded anywhere.

## Code Quality

We use ruff for code quality. Before submitting a PR, ensure your code:

1. Passes all ruff checks:
   ```bash
   ruff check .
   ```
2. Is properly formatted:
   ```bash
   ruff format .
   ```

### VSCode Integration

Install the ruff extension for VSCode for real-time linting and formatting.

## Testing

### Running Tests

Run all tests:

```bash
pytest
```

Run specific test categories:

```bash
# Run unit tests only
pytest tests/ -m "not integration"
```

### Writing Tests

1. Place tests in the `tests/` directory
2. Name test files with `test_` prefix
3. Name test functions with `test_` prefix
4. Use descriptive test names
5. Include both success and failure cases
6. Mock external services when appropriate

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Update tests and documentation
4. Run all tests and quality checks
5. Submit a PR with a clear description of changes

## Release Process

1. Update version in relevant files
2. Update CHANGELOG.md
3. Create a release PR
4. After merge, tag the release

## Questions?

Feel free to open an issue for any questions about contributing.
