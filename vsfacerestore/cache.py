"""Thread-safe compute-once LRU cache.

Several VapourSynth worker threads ask for overlapping windows of frames at the same time. The first thread that
asks for a key claims it and computes it; the others compute their own unclaimed keys first and then wait for the
claimed ones (the `OnceCache` pattern of AutoAdjust). Nothing is computed twice, and a waiting thread holds no lock.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from threading import Event, Lock
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class OnceCache(Generic[K, V]):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._lock = Lock()
        self._done: OrderedDict[K, V] = OrderedDict()
        self._pending: dict[K, Event] = {}
        self.computed = 0  # keys computed so far (statistics)

    def get(self, key: K, compute: Callable[[K], V]) -> V:
        return self.get_many([key], compute)[key]

    def get_many(self, keys: Iterable[K], compute: Callable[[K], V]) -> dict[K, V]:
        """Values for all `keys`; missing ones are computed by whoever claims them first."""
        keys = list(keys)
        result: dict[K, V] = {}
        waits: list[tuple[K, Event]] = []
        mine: list[tuple[K, Event]] = []
        with self._lock:
            for key in keys:
                if key in self._done:
                    self._done.move_to_end(key)
                    result[key] = self._done[key]
                elif key in self._pending:
                    waits.append((key, self._pending[key]))
                else:
                    event = self._pending[key] = Event()
                    mine.append((key, event))
        for key, event in mine:
            try:
                value = compute(key)
            except BaseException:
                with self._lock:
                    del self._pending[key]
                event.set()  # waiters retry and compute it themselves
                raise
            with self._lock:
                self._done[key] = value
                del self._pending[key]
                self.computed += 1
                while len(self._done) > self.capacity:
                    self._done.popitem(last=False)
            event.set()
            result[key] = value
        for key, event in waits:
            event.wait()
            result[key] = self.get(key, compute)  # done now, or recomputed if the claimer failed
        return result

    def __len__(self) -> int:
        with self._lock:
            return len(self._done)
