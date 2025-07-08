from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Dict

from prometheus_client import Summary


def instrument_stage(name: str, summary: Summary) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Measure execution time of a pipeline stage."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with summary.time():
                return func(*args, **kwargs)

        return wrapper

    return decorator


def map_exceptions(to_exc: Exception) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Map arbitrary exceptions to a unified error type."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - passthrough
                raise to_exc from exc

        return wrapper

    return decorator


def cache_result(ttl: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache pure function results for ``ttl`` seconds."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache: Dict[tuple, tuple] = {}

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                value, ts = cache[key]
                if now - ts < ttl:
                    return value
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        return wrapper

    return decorator
