# utils/live_stream.py
from __future__ import annotations

import asyncio
import threading
import queue
from typing import Any, Optional, List

from alpaca.data.live import StockDataStream


class RealtimeStream:
    """
    Manages an Alpaca StockDataStream in a background thread.

    Notes:
    - Streamlit reruns the script often. Keep this object in st.session_state.
    - stop() attempts a best-effort shutdown (alpaca-py can vary by version).
    """

    def __init__(self, api_key: str, secret_key: str, symbol: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol.upper().strip()

        self.msg_queue: "queue.Queue[Any]" = queue.Queue()
        self.stream: Optional[StockDataStream] = None
        self.thread: Optional[threading.Thread] = None

        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def _data_handler(self, data: Any):
        """Callback that puts new data into the queue."""
        self.msg_queue.put(data)

    def _run_stream(self):
        """Runs an asyncio loop inside a background thread and starts the stream."""
        self._stop_event.clear()

        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        self.stream = StockDataStream(self.api_key, self.secret_key)

        # subscribe to quotes (bid/ask updates)
        self.stream.subscribe_quotes(self._data_handler, self.symbol)

        try:
            # Alpaca-py commonly provides a synchronous .run() that blocks
            if hasattr(self.stream, "run") and callable(getattr(self.stream, "run")):
                self.stream.run()
            else:
                # fallback to internal coroutine if present (older patterns)
                coro = getattr(self.stream, "_run_forever", None)
                if coro is None:
                    raise RuntimeError("StockDataStream has no run() or _run_forever().")
                loop.run_until_complete(coro())
        except Exception:
            # Keep it quiet; Streamlit will show errors elsewhere if needed
            pass
        finally:
            try:
                if self.stream and hasattr(self.stream, "unsubscribe_quotes"):
                    try:
                        self.stream.unsubscribe_quotes(self.symbol)
                    except Exception:
                        pass
            finally:
                try:
                    loop.stop()
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass
                self._loop = None
                self.stream = None

    def start(self):
        """Launch the stream in a daemon thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run_stream, daemon=True)
            self.thread.start()

    def stop(self):
        """Best-effort stop."""
        self._stop_event.set()

        # Try calling stop/close if alpaca-py exposes it
        if self.stream is not None:
            for method_name in ("stop", "close", "disconnect"):
                m = getattr(self.stream, method_name, None)
                if callable(m):
                    try:
                        m()
                        break
                    except Exception:
                        pass

        # Try stopping the event loop
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def get_latest(self, max_items: int = 200) -> List[Any]:
        """Drain up to max_items messages from the queue."""
        msgs = []
        for _ in range(max_items):
            try:
                msgs.append(self.msg_queue.get_nowait())
            except queue.Empty:
                break
        return msgs
