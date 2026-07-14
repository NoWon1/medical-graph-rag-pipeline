from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from config import (
    CACHE_BACKEND,
    CACHE_ENABLED,
    REDIS_DB,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_SSL,
    REDIS_URL,
    REDIS_USERNAME,
)


class BaseCacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        raise NotImplementedError


class NullCacheBackend(BaseCacheBackend):
    def get(self, key: str) -> Optional[str]:
        return None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None


class InMemoryCacheBackend(BaseCacheBackend):
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if exp <= now]
        for k in expired:
            self._store.pop(k, None)

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._purge_expired()
            item = self._store.get(key)
            return None if item is None else item[1]

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._store[key] = (time.time() + ttl_seconds, value)


class RedisCacheBackend(BaseCacheBackend):
    def __init__(self) -> None:
        self._client = self._build_client()

    def _build_client(self):
        try:
            import redis  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Redis cache backend selected but `redis` package is not installed. "
                "Install dependencies from requirements.txt."
            ) from e

        if REDIS_URL:
            client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        else:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                username=REDIS_USERNAME or None,
                ****** or None,
                ssl=REDIS_SSL,
                decode_responses=True,
            )
        client.ping()
        return client

    def get(self, key: str) -> Optional[str]:
        try:
            return self._client.get(key)
        except Exception as e:
            print(f"   ⚠️  Redis get failed ({e}); continuing without cache")
            return None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        try:
            self._client.setex(key, ttl_seconds, value)
        except Exception as e:
            print(f"   ⚠️  Redis set failed ({e}); continuing without cache")


_CACHE_BACKEND: Optional[BaseCacheBackend] = None


def get_cache_backend() -> BaseCacheBackend:
    global _CACHE_BACKEND
    if _CACHE_BACKEND is not None:
        return _CACHE_BACKEND

    if not CACHE_ENABLED:
        _CACHE_BACKEND = NullCacheBackend()
        return _CACHE_BACKEND

    backend = (CACHE_BACKEND or "memory").lower().strip()
    if backend == "redis":
        try:
            _CACHE_BACKEND = RedisCacheBackend()
            print("   ✅ Cache backend ready: redis")
            return _CACHE_BACKEND
        except Exception as e:
            print(f"   ⚠️  Redis unavailable ({e}); falling back to in-memory cache")
            _CACHE_BACKEND = InMemoryCacheBackend()
            return _CACHE_BACKEND

    _CACHE_BACKEND = InMemoryCacheBackend()
    print("   ✅ Cache backend ready: in-memory")
    return _CACHE_BACKEND
