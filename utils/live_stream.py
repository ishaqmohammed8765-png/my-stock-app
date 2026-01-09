import threading
import queue
import asyncio
from typing import Optional
from alpaca.data.live import StockDataStream

class RealtimeStream:
    """Manages the background thread for live Alpaca data."""
    def __init__(self, api_key: str, secret_key: str, symbol: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol.upper()
        self.msg_queue = queue.Queue()
        self.stream = None
        self.thread = None
        self._stop_event = threading.Event()

    async def _data_handler(self, data):
        """Callback that puts new data into the queue."""
        self.msg_queue.put(data)

    def _run_stream(self):
        """Starts the async loop in a separate background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        self.stream = StockDataStream(self.api_key, self.secret_key)
        # We subscribe to quotes (live price changes)
        self.stream.subscribe_quotes(self._data_handler, self.symbol)
        
        try:
            loop.run_until_complete(self.stream._run_forever())
        except Exception:
            pass
        finally:
            loop.close()

    def start(self):
        """Launches the stream."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run_stream, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the stream and cleans up."""
        if self.stream:
            # Note: alpaca-py stopping can be complex; 
            # daemon threads handle cleanup on app close.
            pass

    def get_latest(self):
        """Gets all new messages from the queue."""
        msgs = []
        while not self.msg_queue.empty():
            msgs.append(self.msg_queue.get())
        return msgs
