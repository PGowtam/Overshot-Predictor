import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from BrickOfTicks_Trader.config.definitions import LOGS_DIR

def setup_logger(name="BrickOfTicks"):
    """Set up a logger with RotatingFileHandler and ConsoleHandler."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s | %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File Handler (10MB, 5 backups)
    file_path = LOGS_DIR / "trader.log"
    file_handler = RotatingFileHandler(
        file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logger()
