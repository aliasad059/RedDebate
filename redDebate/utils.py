import logging
import sys

def setup_logger(log_file_path):
    """
    Sets up a logger with console and file output handlers.

    Creates a logger that outputs to both stdout and a specified file with
    DEBUG level logging. Includes duplicate handler prevention to avoid
    multiple handlers being added to the same logger instance.

    Args:
        log_file_path (str): Path to the log file where messages will be written

    Returns:
        logging.Logger: Configured logger instance with console and file handlers
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