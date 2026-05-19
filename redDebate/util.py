"""Logging helpers shared across the RedDebate package."""

import logging
import sys

def setup_logger(log_file_path):
    """
    Sets up a logger that writes messages to both stdout and a log file.

    Parameters:
        log_file_path (str): The path to the log file where messages will be stored.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Check if a logger with the same name already exists
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(log_file_path)

        # Set logging levels for handlers
        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger