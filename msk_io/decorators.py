from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Dict

from prometheus_client import Counter, Summary

STAGE_DURATION = Summary("pipeline_stage_seconds", "Time spent in stage", ["stage"])
STAGE_ERRORS = Counter("pipeline_stage_errors_total", "Errors in stage", ["stage"])


def instrument_stage(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Measure execution time and error count of a pipeline stage."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with STAGE_DURATION.labels(stage=name).time():
                try:
                    return func(*args, **kwargs)
                except Exception:
                    STAGE_ERRORS.labels(stage=name).inc()
                    raise

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
