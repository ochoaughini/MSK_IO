import logging
import functools
from typing import Any, Callable, TypeVar, ParamSpec

from msk_io.utils.log_config import get_logger
from msk_io.errors import MSKIOError, ConfigurationError

logger = get_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

def log_method_entry_exit(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        func_name = func.__qualname__
        logger.debug(f"Entering {func_name} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func_name}. Return value: {result}")
            return result
        except Exception as e:
            logger.error(f"Exception in {func_name}: {e}", exc_info=True)
            raise
    return wrapper

def handle_errors(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return func(*args, **kwargs)
        except MSKIOError as e:
            logger.error(f"Caught MSKIOError in {func.__qualname__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__qualname__}: {e}", exc_info=True)
            raise MSKIOError(f"An unexpected error occurred in {func.__qualname__}: {e}") from e
    return wrapper

def requires_config(setting_key: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to ensure a (possibly nested) config value exists before call."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            from msk_io import CONFIG

            if CONFIG is None:
                raise ConfigurationError("Application configuration not loaded.")

            parts = setting_key.split('.')
            current_config = CONFIG
            for part in parts:
                if not hasattr(current_config, part):
                    raise ConfigurationError(
                        f"Required configuration setting '{setting_key}' is missing or None."
                    )
                current_config = getattr(current_config, part)

            if current_config is None:
                raise ConfigurationError(
                    f"Required configuration setting '{setting_key}' is missing or None."
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
