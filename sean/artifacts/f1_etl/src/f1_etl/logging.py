"""Logging utilities for F1 ETL pipeline"""

import logging


def setup_logger(
    name: str = "f1_etl", level: int = logging.INFO, enable_debug: bool = False
) -> logging.Logger:
    """Setup logger for the ETL pipeline"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if enable_debug else level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if enable_debug else level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


# Default logger
logger = setup_logger()
