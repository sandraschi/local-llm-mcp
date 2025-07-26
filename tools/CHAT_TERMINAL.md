# LLM MCP Chat Terminal

An advanced chat terminal for interacting with various LLM providers, featuring personas, rulebooks, and powerful conversation management.

## Features

- **Multiple LLM Providers**: Switch between different LLM providers (OpenAI, Anthropic, Ollama, etc.)
- **Personas**: Use pre-defined or custom personas to guide the assistant's behavior
- **Rulebooks**: Apply rulebooks to ensure consistent style and formatting
- **Conversation History**: Search and navigate through past conversations
- **Model Parameter Tuning**: Adjust temperature, max tokens, and other parameters
- **Streaming Responses**: Get responses in real-time as they're generated
- **Command Line Interface**: Intuitive commands for all functionality

## Installation

1. Make sure you have Python 3.8+ installed

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in `.env` (see `.env.example` for reference)

## Quick Start

1. Run the chat terminal:

   ```bash
   python tools/run_chat.py
   ```

2. Set a provider and model:

   ```bash
   /provider openai
   /model gpt-4
   ```

3. Start chatting! Type your message and press Enter to send.

## Available Commands

### General Commands

- `/help` - Show help message
- `/exit` - Exit the chat terminal
- `/status` - Show current status and configuration

### Conversation Management

- `/clear` - Clear the conversation history
- `/search <query>` - Search through conversation history
- `/history` - Show recent conversation history

### Configuration Management

- `/config` - View or modify configuration
- `/provider` - Set or show the current LLM provider
- `/model` - Set or show the current model

### Persona Management

- `/persona` - Set or show the current persona
- `/personas` - List all available personas

### Rulebook Management

- `/rulebook` - Apply a rulebook to the conversation
- `/rulebooks` - List all available rulebooks

## Personas

Personas define the assistant's behavior and style. Some example personas are included:

- **Research Assistant**: For academic and technical research
- **Creative Writer**: For storytelling and content creation

To create a custom persona, add a YAML file to `~/.config/llm-mcp/personas/`.

## Rulebooks

Rulebooks define style and formatting rules. Example rulebooks include:

- **Technical Writing**: For documentation and technical content
- **Creative Writing**: For fiction and creative content

To create a custom rulebook, add a YAML file to `~/.config/llm-mcp/rulebooks/`.

## Configuration

The chat terminal can be configured via environment variables or the config file at `~/.config/llm-mcp/terminal.yaml`.

### Environment Variables

- `LLM_MCP_PROVIDER`: Default provider
- `LLM_MCP_MODEL`: Default model
- `LLM_MCP_TEMPERATURE`: Default temperature
- `LLM_MCP_MAX_TOKENS`: Default max tokens
- `LLM_MCP_DEBUG`: Enable debug mode

## Development

To contribute to the chat terminal:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
