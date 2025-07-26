# LLM MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A FastMCP 2.10-compliant server for managing local and cloud LLMs with support for video generation and advanced chat features.

## 🌟 Features

- **Unified API** for multiple LLM providers (Ollama, LM Studio, vLLM, OpenAI, Anthropic, Gemini, etc.)
- **Model Management**: List, load, unload, and download models
- **Inference API**: Standardized interface for text generation
- **Video Generation**: Integration with Gemini Veo 3 for text/image to video
- **Failover & Fallback**: Automatic fallback to alternative models
- **Chat Terminal**: Interactive terminal with personas and rulebooks
- **FastMCP 2.10 Compliance**: Full compatibility with the MCP protocol

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- (Optional) [Ollama](https://ollama.ai/) or [LM Studio](https://lmstudio.ai/) for local models

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-mcp.git
   cd llm-mcp
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On Unix/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Configuration

Edit the `.env` file with your settings:

```env
# Server configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Authentication (comma-separated list of API keys)
API_KEYS=your-api-key-here

# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434

# LM Studio configuration
LMSTUDIO_BASE_URL=http://localhost:1234

# OpenAI configuration
OPENAI_API_KEY=your-openai-key

# Anthropic configuration
ANTHROPIC_API_KEY=your-anthropic-key
```

## 🛠️ Usage

### Start the server

```bash
python -m llm_mcp.server
```

### Using the Chat Terminal

```bash
# Start the chat terminal
python tools/chat_terminal.py

# With specific provider and model
python tools/chat_terminal.py --provider anthropic --model claude-3-opus-20240229

# With persona and rulebook
python tools/chat_terminal.py --persona code_expert --rulebook coding_rules
```

### API Documentation

Once the server is running, visit:
- API Docs: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

## 🤖 Supported Providers

- [x] Ollama
- [x] LM Studio
- [x] vLLM
- [x] OpenAI
- [x] Anthropic
- [x] Google Gemini
- [ ] More coming soon...

## 🧩 Extending with Custom Providers

1. Create a new provider in `src/llm_mcp/services/providers/`
2. Implement the required methods from `BaseProvider`
3. Add your provider to the `ProviderFactory`
4. Update configuration as needed

## 📦 Project Structure

```
llm-mcp/
├── src/
│   └── llm_mcp/
│       ├── api/              # API endpoints
│       ├── models/           # Data models
│       ├── services/         # Business logic
│       │   └── providers/    # LLM provider implementations
│       ├── utils/            # Utility functions
│       ├── config.py         # Configuration management
│       └── server.py         # FastAPI application
├── tests/                   # Test suite
├── tools/                   # Utility scripts
│   └── chat_terminal.py     # Interactive chat terminal
├── .env.example            # Example environment variables
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # This file
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📚 Resources

- [FastMCP Documentation](https://fastmcp.readthedocs.io/)
- [Ollama](https://ollama.ai/)
- [LM Studio](https://lmstudio.ai/)
- [vLLM](https://vllm.readthedocs.io/)

## 📬 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - your.email@example.com

Project Link: [https://github.com/yourusername/llm-mcp](https://github.com/yourusername/llm-mcp)

## Usage

### Starting the server

```bash
uvicorn llm_mcp.main:app --reload
```

### API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
llm-mcp/
├── src/
│   └── llm_mcp/
│       ├── api/
│       │   └── v1/              # API version 1 endpoints
│       ├── core/                # Core application logic
│       ├── models/              # Pydantic models
│       ├── services/
│       │   └── providers/       # LLM provider implementations
│       ├── utils/               # Utility functions
│       ├── __init__.py
│       └── main.py              # Application entry point
├── tests/                      # Test files
├── .env.example               # Example environment variables
├── pyproject.toml             # Project metadata and dependencies
└── README.md                  # This file
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

### Type Checking

```bash
mypy .
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
