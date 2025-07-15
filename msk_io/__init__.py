"""
MSK-IO: Offline Multimodal Diagnostic Pipeline.

This package provides the core components for processing medical imaging data,
integrating with Large Language Models (LLMs), and generating diagnostic insights.
"""
import logging
from .utils.log_config import setup_logging
from .config import AppConfig

# Setup logging as soon as the package is imported
setup_logging()

# Global configuration instance (lazy loading if necessary, for now direct)
# This assumes AppConfig can be instantiated without specific arguments at import time
# For more complex scenarios, consider a function `get_config()` or dependency injection.
try:
    CONFIG = AppConfig()
except Exception as e:
    logging.getLogger(__name__).error(f"Failed to load application configuration: {e}")
    CONFIG = None  # Or raise a critical error, depending on desired behavior

# Define package version
__version__ = "0.0.1"

logging.getLogger(__name__).info("MSK-IO package initialized.")
