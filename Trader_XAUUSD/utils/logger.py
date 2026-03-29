import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from config.definitions import LOGS_DIR
from config.settings import LOG_LEVEL, LOG_FILE

def setup_logger(name="Trader", log_file=LOG_FILE):
    """
    Sets up a logger with both File and Console handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(module)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler
    file_path = os.path.join(LOGS_DIR, log_file)
    f_handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5)
    f_handler.setFormatter(formatter)
    f_handler.setLevel(LOG_LEVEL)
    logger.addHandler(f_handler)

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(formatter)
    c_handler.setLevel(LOG_LEVEL)
    logger.addHandler(c_handler)

    return logger

# Global instance for easy import
logger = setup_logger()
