import queue
import threading
import time


class BoundedAsyncWriter:
    """Run a write callback off-thread with bounded, non-blocking input."""

    _STOP = object()

    def __init__(self, write_item, max_queue_size: int) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._write_item = write_item
        self._queue = queue.Queue()
        # Only samples consume capacity. Tiny finalization records share FIFO
        # order but cannot be dropped behind a full sample backlog.
        self._sample_slots = threading.BoundedSemaphore(max_queue_size)
        self._failure = None
        self._failure_lock = threading.Lock()
        self._closed = False
        self.dropped_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, item) -> bool:
        return self.submit_factory(lambda: item)

    def submit_factory(self, make_item) -> bool:
        if self._closed:
            raise RuntimeError("async writer is closed")
        self.raise_if_failed()
        if not self._sample_slots.acquire(blocking=False):
            self.dropped_count += 1
            return False
        try:
            item = make_item()
            self._queue.put_nowait((False, item))
        except BaseException:
            self._sample_slots.release()
            raise
        return True

    def submit_final(self, item) -> bool:
        if self._closed:
            raise RuntimeError("async writer is closed")
        self.raise_if_failed()
        self._queue.put_nowait((True, item))
        return True

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError(f"async writer failed: {failure}") from failure

    def close(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait((True, self._STOP))
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("timed out while closing async writer")
        self.raise_if_failed()

    def _run(self) -> None:
        while True:
            critical, item = self._queue.get()
            if not critical:
                self._sample_slots.release()
            try:
                if item is self._STOP:
                    return
                with self._failure_lock:
                    failed = self._failure is not None
                if not failed:
                    self._write_item(item)
            except Exception as exc:  # noqa: BLE001
                with self._failure_lock:
                    if self._failure is None:
                        self._failure = exc
            finally:
                self._queue.task_done()
