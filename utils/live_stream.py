from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, List, Optional

from alpaca.data.live import StockDataStream


class RealtimeStream:
    """
    Stream Alpaca live quotes in a background thread for Streamlit.

    - Uses IEX feed by default (works best on free/paper)
    - Stores messages in a thread-safe queue
    """

    def __init__(self, api_key: str, secret_key: str, symbol: str, *, feed: str = "iex"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol.upper().strip()
        self.feed = feed

        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = threading.Event()

        self._stream: Optional[StockDataStream] = None

    # ----------------------------------------------------
    # Alpaca async handlers
    # ----------------------------------------------------
    async def _on_quote(self, q):
        """Handle incoming quote."""
        self._queue.put(q)

    async def _run_stream(self):
        self._loop = asyncio.get_running_loop()

        self._stream = StockDataStream(
            self.api_key,
            self.secret_key,
            feed=self.feed,  # IMPORTANT for free accounts
        )

        self._stream.subscribe_quotes(self._on_quote, self.symbol)

        self._running.set()
        try:
            await self._stream.run()
        finally:
            self._running.clear()

    # ----------------------------------------------------
    # Thread control
    # ----------------------------------------------------
    def start(self) -> None:
        """Start the websocket stream in a background thread."""
        if self._thread and self._thread.is_alive():
            return  # already running

        def runner():
            asyncio.run(self._run_stream())

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the websocket stream (best effort)."""
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass

        self._running.clear()

    # ----------------------------------------------------
    # Streamlit-facing helpers
    # ----------------------------------------------------
    def is_running(self) -> bool:
        return self._running.is_set()

    def get_latest(self, max_items: int = 100) -> List[Any]:
        """Drain up to `max_items` messages from the queue."""
        out: List[Any] = []
        while len(out) < max_items:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out
