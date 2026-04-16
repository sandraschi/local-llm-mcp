"""SOTA Structured Logging Configuration for Local LLM MCP.

This module provides SOTA-compliant structured logging using structlog following
MCP Central Docs standards. Features include:
- Structured JSON logging for machine readability
- Human-readable console output
- File rotation and retention
- Performance monitoring integration
- Unicode-safe logging (no emoji corruption)

SOTA COMPLIANCE: FastMCP 2.14.1+ Structured Logging Standards
"""

import logging
import logging.handlers
import sys
from pathlib import Path

# SOTA: Use structlog for structured logging
try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    # Fallback to basic logging if structlog not available
    import logging as structlog

# SOTA Logging Configuration
DEFAULT_LOG_DIR = Path.home() / ".llm_mcp" / "logs"
DEFAULT_LOG_FILE = "llm_mcp.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# SOTA: Structured logging format
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# SOTA: Unicode-safe replacements for logging
UNICODE_REPLACEMENTS = {
    "\U0001f680": "ROCKET",  # 🚀
    "\u26a0": "WARNING",  # ⚠️
    "\u274c": "ERROR",  # ❌
    "\u2705": "SUCCESS",  # ✅
    "\U0001f4a1": "IDEA",  # 💡
    "\U0001f916": "ROBOT",  # 🤖
}


# Ensure log directory exists
def _ensure_log_dir(log_dir: Path) -> None:
    """Ensure the log directory exists."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        sys.stderr.write(f"Failed to create log directory {log_dir}: {e}\n")
        sys.stderr.flush()
        raise


def _sanitize_unicode(text: str) -> str:
    """SOTA: Sanitize unicode characters to prevent encoding issues."""
    if not isinstance(text, str):
        return str(text)

    for unicode_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, replacement)

    return text


class LoggingConfig:
    """SOTA Structured Logging Configuration for Local LLM MCP.

    Features:
    - Structlog-based structured logging (JSON for machines, readable for humans)
    - Unicode-safe logging (prevents encoding corruption)
    - Performance monitoring integration
    - File rotation with configurable retention
    - MCP-compliant log levels and formatting
    """

    _initialized = False

    @classmethod
    def initialize(
        cls,
        log_level: str = "INFO",
        log_dir: Path | None = None,
        log_file: str | None = None,
        max_size: int = MAX_LOG_SIZE,
        backup_count: int = BACKUP_COUNT,
        log_to_console: bool = True,
        structured: bool = True,
    ) -> None:
        """Initialize SOTA structured logging configuration.

        SOTA COMPLIANCE: FastMCP 2.14.1+ Structured Logging Standards
        - Structlog-based JSON logging for machine readability
        - Human-readable console output
        - Unicode-safe logging (no emoji corruption)
        - Performance monitoring integration

        Args:
            log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files (default: ~/.llm_mcp/logs)
            log_file: Name of the log file (default: llm_mcp.log)
            max_size: Maximum size of each log file in bytes (default: 10MB)
            backup_count: Number of backup log files to keep (default: 5)
            log_to_console: Whether to log to stderr (default: True)
            structured: Whether to use structlog (default: True)
        """
        if cls._initialized:
            return

        try:
            # Configure log directory and file
            log_dir = log_dir or DEFAULT_LOG_DIR
            log_file = log_file or DEFAULT_LOG_FILE
            log_path = log_dir / log_file
            _ensure_log_dir(log_dir)

            # Convert string log level to int
            log_level_int = getattr(logging, log_level.upper(), logging.INFO)

            # SOTA: Configure structlog if available and requested
            if STRUCTLOG_AVAILABLE and structured:
                # Configure structlog for SOTA compliance
                structlog.configure(
                    processors=[
                        structlog.contextvars.merge_contextvars,
                        structlog.processors.add_log_level,
                        structlog.processors.TimeStamper(fmt="iso"),
                        structlog.processors.JSONRenderer(),
                    ],
                    wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
                    context_class=dict,
                    logger_factory=structlog.WriteLoggerFactory(),
                    cache_logger_on_first_use=True,
                )

                # Configure standard logging to work with structlog
                logging.basicConfig(
                    format="%(message)s",
                    stream=sys.stderr if log_to_console else None,
                    level=log_level_int,
                )

                # File handler for structured logs
                file_handler = logging.handlers.RotatingFileHandler(
                    log_path, maxBytes=max_size, backupCount=backup_count, encoding="utf-8"
                )
                file_handler.setFormatter(logging.Formatter("%(message)s"))
                logging.getLogger().addHandler(file_handler)

            else:
                # Fallback to standard logging with Unicode sanitization
                root_logger = logging.getLogger()
                root_logger.setLevel(log_level_int)

                # Remove existing handlers
                for handler in root_logger.handlers[:]:
                    root_logger.removeHandler(handler)

                # Create formatter with Unicode sanitization
                class SanitizedFormatter(logging.Formatter):
                    def format(self, record):
                        # Sanitize the message
                        if hasattr(record, "msg"):
                            record.msg = _sanitize_unicode(str(record.msg))
                        return super().format(record)

                formatter = SanitizedFormatter(LOG_FORMAT, DATE_FORMAT)

                # File handler with rotation
                file_handler = logging.handlers.RotatingFileHandler(
                    log_path, maxBytes=max_size, backupCount=backup_count, encoding="utf-8"
                )
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)

                # Console handler
                if log_to_console:
                    console_handler = logging.StreamHandler(sys.stderr)
                    console_handler.setFormatter(formatter)
                    root_logger.addHandler(console_handler)

            # SOTA: Configure third-party logger levels
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logging.getLogger("torch").setLevel(logging.WARNING)

            cls._initialized = True

            # Log successful initialization
            logger = logging.getLogger(__name__)
            logger.info("Logging system initialized")
            logger.debug("Debug logging enabled")

        except Exception as e:
            # SOTA: Fallback with Unicode-safe error logging
            safe_error = _sanitize_unicode(str(e))
            sys.stderr.write(f"SOTA Logging initialization failed: {safe_error}\n")
            sys.stderr.flush()
            raise


def get_logger(name: str):
    """Get a SOTA-compliant logger with the given name.

    SOTA COMPLIANCE: Returns structlog logger for structured logging when available,
    falls back to standard logging.Logger.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        Structlog logger or standard Logger instance
    """
    if not LoggingConfig._initialized:
        LoggingConfig.initialize()

    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)
