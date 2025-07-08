"""Asynchronous worker that processes tasks."""
from __future__ import annotations

import asyncio
from typing import Callable, Any


class Worker:
    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    async def run(self, arg: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.func, arg)
