"""Centralized logging configuration for the LLM MCP application.

This module provides a centralized logging configuration with file rotation and stderr output.
Logs are written to both a file and stderr, with rotation based on file size.
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# Default log directory (in user's home directory)
DEFAULT_LOG_DIR = Path.home() / ".llm_mcp" / "logs"
DEFAULT_LOG_FILE = "llm_mcp.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Ensure log directory exists
def _ensure_log_dir(log_dir: Path) -> None:
    """Ensure the log directory exists."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        sys.stderr.write(f"Failed to create log directory {log_dir}: {e}\n")
        sys.stderr.flush()
        raise

class LoggingConfig:
    """Centralized logging configuration."""
    _initialized = False
    
    @classmethod
    def initialize(
        cls,
        log_level: int = logging.INFO,
        log_dir: Optional[Path] = None,
        log_file: Optional[str] = None,
        max_size: int = MAX_LOG_SIZE,
        backup_count: int = BACKUP_COUNT,
        log_to_console: bool = True
    ) -> None:
        """Initialize the logging configuration.
        
        Args:
            log_level: Logging level (default: logging.INFO)
            log_dir: Directory to store log files (default: ~/.llm_mcp/logs)
            log_file: Name of the log file (default: llm_mcp.log)
            max_size: Maximum size of each log file in bytes (default: 10MB)
            backup_count: Number of backup log files to keep (default: 5)
            log_to_console: Whether to log to stderr (default: True)
        """
        if cls._initialized:
            return
            
        try:
            # Configure log directory and file
            log_dir = log_dir or DEFAULT_LOG_DIR
            log_file = log_file or DEFAULT_LOG_FILE
            log_path = log_dir / log_file
            
            # Ensure log directory exists
            _ensure_log_dir(log_dir)
            
            # Create formatter
            formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            
            # Remove any existing handlers to avoid duplicate logs
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # Add file handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
            
            # Add stderr handler if enabled
            if log_to_console:
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setFormatter(formatter)
                console_handler.setLevel(log_level)
                root_logger.addHandler(console_handler)
            
            # Disable propagation for third-party loggers
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
            
            cls._initialized = True
            
            # Log successful initialization
            logger = logging.getLogger(__name__)
            logger.info("Logging system initialized")
            logger.debug("Debug logging enabled")
            
        except Exception as e:
            sys.stderr.write(f"Failed to initialize logging: {e}\n")
            sys.stderr.flush()
            raise

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        Configured logger instance
    """
    if not LoggingConfig._initialized:
        LoggingConfig.initialize()
    return logging.getLogger(name)
