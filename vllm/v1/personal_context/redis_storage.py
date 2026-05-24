# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Redis-backed KV block storage.

Drop-in replacement for ``InMemoryStorage``: same public API
(``put`` / ``get`` / ``lookup`` / ``__contains__`` / ``__len__`` and
``config`` property) so a connector or test can swap one for the other
without touching call sites.

Schema:

    pc:config              → pickled ``StoreConfig`` snapshot, written
                             on first ``put`` and verified on every
                             ``__init__`` so cross-restart / cross-
                             client mismatches surface as ``ValueError``
                             rather than silently corrupting reuse.
    pc:kv:<hex_hash>       → pickled ``KVBlock``. Value is binary;
                             ``<hex_hash>`` is the chunk block hash
                             encoded as lowercase hex (printable in
                             ``redis-cli`` for debugging, ~2× longer
                             than raw bytes but the key is ~32 bytes
                             so the overhead is negligible vs the
                             megabyte-scale value).

Persistence is a Redis-server concern (RDB snapshot + AOF append-only
log are the standard knobs; ``redis/redis-stack`` enables both by
default). Nothing in this client cares about restart durability.

Eviction / TTL is also a server concern. Phase 11+ will likely set
``maxmemory-policy allkeys-lru`` on dedicated PC instances; this
client doesn't impose policy.

This file imports ``redis`` lazily — installing PC-specific extras
(``redis>=5.0.0``) is only required when the connector is actually
configured with ``store_backend=redis``.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Optional

from vllm.v1.personal_context.entry import KVBlock, StoreConfig
from vllm.v1.personal_context.storage import validate_block

if TYPE_CHECKING:
    import redis  # noqa: F401


_KEY_PREFIX = b"pc:kv:"
_CONFIG_KEY = b"pc:config"


def _encode_key(block_hash: bytes) -> bytes:
    """Build the Redis key for a block hash. ``redis-cli`` friendly."""
    return _KEY_PREFIX + block_hash.hex().encode("ascii")


class RedisKVStorage:
    """Redis-backed implementation of the PC storage interface.

    Args:
        config: Expected ``StoreConfig``. Verified against the
            ``pc:config`` value already in Redis (or written there on
            first use). Mismatch raises ``ValueError`` — the caller is
            either pointing at the wrong instance or has changed model.
        url: Redis URL, e.g. ``redis://localhost:6379``. Mutually
            exclusive with ``client``.
        client: A pre-built ``redis.Redis`` (or test double such as
            ``fakeredis.FakeRedis``). Lets tests skip the network.

    Raises:
        ImportError: ``redis`` package not installed.
        ValueError: schema mismatch against existing ``pc:config``.
        ConnectionError: server unreachable (propagated from redis-py).
    """

    def __init__(
        self,
        config: StoreConfig,
        url: str | None = None,
        *,
        client: "redis.Redis | None" = None,
    ) -> None:
        if client is None and url is None:
            raise ValueError("RedisKVStorage requires either `url` or `client`")
        if client is None:
            try:
                import redis as _redis
            except ImportError as e:
                raise ImportError(
                    "RedisKVStorage requires the `redis` package. "
                    "Install via `uv pip install \"redis>=5.0.0\"` or "
                    "`uv pip install -e \".[personal-context]\"`."
                ) from e
            # decode_responses=False keeps values as bytes (we pickle blobs).
            client = _redis.Redis.from_url(url, decode_responses=False)
        self._r = client
        self._config = config
        self._verify_or_write_config()

    @property
    def config(self) -> StoreConfig:
        return self._config

    def put(self, key: bytes, block: KVBlock) -> None:
        validate_block(self._config, block)
        self._r.set(_encode_key(key), pickle.dumps(block))

    def get(self, key: bytes) -> Optional[KVBlock]:
        data = self._r.get(_encode_key(key))
        if data is None:
            return None
        return pickle.loads(data)

    def lookup(self, keys: list[bytes]) -> list[Optional[KVBlock]]:
        """Batched ``get`` via ``MGET``: one Redis round-trip for many keys.

        Hot path for the connector — ``load_plan`` issues ``lookup``
        over every block in every chunk in the reuse plan (often 100+
        keys per request). Single-key ``get`` would multiply latency
        by ``len(keys)``; ``MGET`` collapses to one network RTT.
        """
        if not keys:
            return []
        raw_values = self._r.mget([_encode_key(k) for k in keys])
        return [pickle.loads(v) if v is not None else None for v in raw_values]

    def __contains__(self, key: bytes) -> bool:
        # ``EXISTS`` returns 0/1 for our single-key check.
        return bool(self._r.exists(_encode_key(key)))

    def __len__(self) -> int:
        """Count of stored KV blocks (excludes the ``pc:config`` key).

        Uses ``SCAN`` rather than ``KEYS`` so we don't block the server
        on a long pattern match; the result is approximate when blocks
        are being concurrently inserted but stable for our test +
        debug paths.
        """
        count = 0
        for _ in self._r.scan_iter(match=_KEY_PREFIX + b"*", count=1000):
            count += 1
        return count

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _verify_or_write_config(self) -> None:
        """Write ``pc:config`` on first use; verify on subsequent inits.

        Race: two clients connect to a fresh Redis simultaneously and
        both see no config. They both write — last-write-wins is fine
        IF they agree (same model + same StoreConfig). If they
        disagree the second one's writes will appear to succeed while
        actually targeting a server whose config disagrees; subsequent
        ``get`` from a third client would surface the mismatch on
        whichever config is currently stored. Production should use
        per-namespace key prefixes (Phase 11) to make this impossible.
        For demo / dev the race is negligible.
        """
        existing = self._r.get(_CONFIG_KEY)
        if existing is None:
            self._r.set(_CONFIG_KEY, pickle.dumps(self._config))
            return
        stored: StoreConfig = pickle.loads(existing)
        if stored != self._config:
            raise ValueError(
                "RedisKVStorage: schema mismatch against existing "
                f"`{_CONFIG_KEY.decode()}` in Redis.\n"
                f"  stored:   {stored}\n"
                f"  expected: {self._config}\n"
                "Either point at a different Redis instance / DB, or "
                "FLUSHDB to start fresh."
            )
