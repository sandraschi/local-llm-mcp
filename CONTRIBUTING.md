# Contributing to LLM MCP Server

First off, thank you for considering contributing to LLM MCP Server! It's people like you that make LLM MCP Server such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/yourusername/llm-mcp/issues).
- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/yourusername/llm-mcp/issues/new). Be sure to include:
  - A clear title and description
  - Steps to reproduce the issue
  - Expected vs. actual behavior
  - Any relevant screenshots or logs
  - Your environment (OS, Python version, etc.)

### Suggesting Enhancements

- Open an issue with a clear description of the enhancement
- Include any relevant use cases or examples
- If possible, include a mockup or example of the proposed change

### Your First Code Contribution

1. Fork the repository
2. Create a new branch: `git checkout -b my-feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-feature-branch`
5. Submit a pull request

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for all function signatures
- Include docstrings for all public functions and classes
- Keep lines under 88 characters (Black's default line length)

### Testing

- Write tests for new features and bug fixes
- Ensure all tests pass before submitting a pull request
- Run the full test suite with `pytest`

### Pull Request Process

1. Update the README.md with details of changes if needed
2. Ensure your code passes all tests
3. Reference any related issues in your PR description
4. Request a review from one of the maintainers

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
