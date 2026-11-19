"""Per-address flood control for the expensive routes.

A bucket per client address: `per_window` requests can burst, then tokens refill
evenly across `window_seconds`. No dependency, no shared store - which is the
main thing to know about it:

* state lives in this process, so `uvicorn --workers 4` means four independent
  limits and a restart forgives everyone. That is fine for one box; a
  multi-worker deployment should put a shared limiter in front instead.
* the address is `request.client.host` unless `BUTLER_TRUST_FORWARDED=1`, in
  which case the left-most `X-Forwarded-For` entry is used. Only set that behind
  a proxy you control: as a header, it is otherwise whatever the client typed.
* a limit of 0 disables that route's limiter.

The check runs before the shared-key guard, so a bad-key flood is refused
without touching librosa - the key is not a rate limit and never was.
"""

from __future__ import annotations

import math
import os
import threading
import time

from fastapi import HTTPException, Request

TRUST_FORWARDED = os.environ.get("BUTLER_TRUST_FORWARDED", "").strip().lower() in {"1", "true", "yes"}
MAX_KEYS = 10_000
STALE_SECONDS = 300.0


class Limiter:
    """One bucket per address, refilling continuously."""

    def __init__(self, name: str, per_window: int, window_seconds: float = 60.0) -> None:
        self.name = name
        self.per_window = max(0, int(per_window))
        self.window_seconds = max(0.1, float(window_seconds))
        self._capacity = float(self.per_window)
        self._refill_per_second = self.per_window / self.window_seconds
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def describe(self) -> str:
        if not self.per_window:
            return "off"
        return f"{self.per_window} per {self.window_seconds:g}s, per address, per worker"

    def take(self, key: str) -> tuple[bool, float]:
        """Spend one token. Returns (allowed, seconds until one is available)."""
        if not self.per_window:
            return True, 0.0

        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - last) * self._refill_per_second)

            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)

            self._calls += 1
            if self._calls % 256 == 0:
                self._sweep(now)

            if allowed:
                return True, 0.0
            return False, (1.0 - tokens) / self._refill_per_second

    def _sweep(self, now: float) -> None:
        """Keep the map from growing without bound on a wide scan."""
        stale = [key for key, (_, last) in self._buckets.items() if now - last > STALE_SECONDS]
        for key in stale:
            self._buckets.pop(key, None)
        if len(self._buckets) > MAX_KEYS:
            oldest = sorted(self._buckets.items(), key=lambda item: item[1][1])
            for key, _ in oldest[: len(oldest) // 2]:
                self._buckets.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


def client_ip(request: Request) -> str:
    if TRUST_FORWARDED:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def enforce(limiter: Limiter, request: Request) -> None:
    allowed, wait = limiter.take(client_ip(request))
    if allowed:
        return
    raise HTTPException(
        status_code=429,
        detail="Too many requests from this address. Wait a moment and try again.",
        headers={"Retry-After": str(max(1, math.ceil(wait)))},
    )
