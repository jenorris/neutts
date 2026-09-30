"""Exclusive model ownership, bounded streaming, and cooperative cancellation."""
import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
import threading

import anyio


async def wait_worker(future):
    # A cancelled HTTP response must not release ownership while its GPU thread runs.
    with anyio.CancelScope(shield=True):
        await asyncio.shield(future)


async def stream_owned(lock, executor, produce):
    async with lock:
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=4)
        cancelled = threading.Event()

        def emit(item):
            if cancelled.is_set():
                return False
            pending = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            try:
                while not cancelled.is_set():
                    try:
                        pending.result(timeout=0.1)
                        return True
                    except FutureTimeout:
                        continue
                return False
            finally:
                if not pending.done():
                    pending.cancel()

        def run():
            try:
                produce(cancelled, emit)
            except Exception as exc:
                emit(exc)
            finally:
                emit(None)

        future = loop.run_in_executor(executor, run)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancelled.set()
            await wait_worker(future)


async def run_owned(lock, executor, produce):
    async with lock:
        cancelled = threading.Event()
        future = asyncio.get_running_loop().run_in_executor(executor, produce, cancelled)
        try:
            return await asyncio.shield(future)
        finally:
            cancelled.set()
            await wait_worker(future)
