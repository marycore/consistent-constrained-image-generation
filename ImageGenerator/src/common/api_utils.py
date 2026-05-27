from __future__ import annotations

import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"Environment variable {name} must be an integer, got: {raw}") from e


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise RuntimeError(f"Environment variable {name} must be a float, got: {raw}") from e


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Environment variable {name} must be boolean-like (1/0,true/false), got: {raw}")


def retry_call(fn: Callable[[], T], *, retries: int, backoff_seconds: float) -> T:
    """Run fn with retries and linear backoff."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception:
            if attempt > retries:
                raise
            time.sleep(backoff_seconds * attempt)
