# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Memory-mapped KV block storage (same-host persistent + shared).

Drop-in replacement for ``InMemoryStorage`` / ``RedisKVStorage``: same
public API (``put`` / ``get`` / ``lookup`` / ``__contains__`` / ``__len__``
and the ``config`` property) so a connector or test can swap one for the
other without touching call sites.

Why this exists
---------------
``RedisKVStorage`` pays a large per-fetch cost that has nothing to do
with where Redis keeps its data: every ``get`` must serialise the block
into the RESP protocol, ship it over a socket, and ``pickle.loads`` a
megabyte-scale tensor back into fresh Python/torch objects. Benchmarks
put that at ~290 ms/request for an 8B model vs ~60 ms for the in-process
``InMemoryStorage``.

``MmapKVStorage`` removes both costs for the **same-host** deployment
(ingestion process and serving worker on one machine):

  - **no serialisation on read** — block tensors live as raw bytes
    inside a memory-mapped file; ``get`` reinterprets them with
    ``torch.frombuffer`` over the mapped bytes, so there is no
    ``pickle.loads`` and no socket read — just a couple of local memcpys
    (the slice out of the mapping + one ``.clone()`` so the returned
    tensor is writable and independent of the mapping, which may be
    remapped on a later grow). Cheap next to Redis's protocol decode +
    socket buffer + full unpickle of a megabyte-scale object.
  - **cross-process sharing** — multiple processes ``mmap`` the same
    file and share the OS page cache.
  - **persistence** — the mapping is backed by an on-disk file, so the
    store survives a process restart; reopening re-reads the index.

Trade-offs vs Redis: same-host only (no network sharing), and this class
owns its own tiny storage engine (slot array + JSON index + doubling
growth) instead of leaning on a server. For cross-host / distributed KV
sharing, use ``RedisKVStorage``.

Layout
------
Two files under ``root_dir``:

    blocks.dat   memory-mapped array of FIXED-SIZE records. Every block
                 has identical geometry (``StoreConfig`` is fixed), so a
                 block always occupies ``record_size`` bytes and slot
                 ``i`` lives at offset ``i * record_size``. One record:

                     [ old_pos_start : int64 little-endian        ]
                     [ K layer 0 ][ K layer 1 ] ... [ K layer L-1 ]
                     [ V layer 0 ][ V layer 1 ] ... [ V layer L-1 ]

                 each tensor segment being
                 ``block_size * num_kv_heads * head_dim * itemsize``
                 bytes of the raw (contiguous, CPU) tensor.

    index.json   ``{"config": <serialised StoreConfig>,
                     "blocks": {hash_hex: slot_index, ...},
                     "next_slot": N}``. Persisted on every ``put`` so a
                 reopened store recovers its hash->slot map and verifies
                 the geometry matches (mismatch raises, mirroring
                 ``RedisKVStorage``'s ``pc:config`` check).

Concurrency: single-writer assumed (the offline indexer). Multiple
readers mapping the same file are fine. There is no inter-process write
lock — concurrent writers would race the index, same caveat as the other
backends in this package.
"""

from __future__ import annotations

import json
import mmap
import os
from typing import Optional

import torch

from vllm.v1.personal_context.chunk import AlignmentError
from vllm.v1.personal_context.entry import KVBlock, StoreConfig
from vllm.v1.personal_context.storage import validate_block

_BLOCKS_FILE = "blocks.dat"
_INDEX_FILE = "index.json"
_OLD_POS_HDR = 8  # bytes for the int64 old_pos_start prefix per record
_INITIAL_SLOTS = 16  # starting capacity; grows by doubling


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype)


def _str_to_dtype(s: str) -> torch.dtype:
    # ``str(torch.float16)`` -> ``"torch.float16"``; getattr resolves it.
    return getattr(torch, s.split(".", 1)[1])


def _config_to_dict(c: StoreConfig) -> dict:
    return {
        "model_id": c.model_id,
        "dtype": _dtype_to_str(c.dtype),
        "layout": c.layout,
        "num_layers": c.num_layers,
        "num_kv_heads": c.num_kv_heads,
        "head_dim": c.head_dim,
        "block_size": c.block_size,
    }


def _config_from_dict(d: dict) -> StoreConfig:
    return StoreConfig(
        model_id=d["model_id"],
        dtype=_str_to_dtype(d["dtype"]),
        layout=d["layout"],
        num_layers=d["num_layers"],
        num_kv_heads=d["num_kv_heads"],
        head_dim=d["head_dim"],
        block_size=d["block_size"],
    )


class MmapKVStorage:
    """Memory-mapped implementation of the PC storage interface.

    Args:
        config: Expected ``StoreConfig``. On first use it is written to
            ``index.json``; on reopen it is verified against the stored
            geometry (mismatch raises ``ValueError``, mirroring
            ``RedisKVStorage``).
        root_dir: Directory holding ``blocks.dat`` + ``index.json``.
            Created if absent.

    Raises:
        ValueError: stored geometry disagrees with ``config``.
    """

    def __init__(self, config: StoreConfig, root_dir: str) -> None:
        self._config = config
        self._root = root_dir
        os.makedirs(root_dir, exist_ok=True)
        self._blocks_path = os.path.join(root_dir, _BLOCKS_FILE)
        self._index_path = os.path.join(root_dir, _INDEX_FILE)

        # Per-tensor and per-record byte sizes are fully determined by the
        # (fixed) geometry, which is what lets us use flat fixed-size slots.
        self._itemsize = torch.empty(0, dtype=config.dtype).element_size()
        self._elems_per_tensor = (
            config.block_size * config.num_kv_heads * config.head_dim
        )
        self._tensor_bytes = self._elems_per_tensor * self._itemsize
        self._tensors_per_block = 2 * config.num_layers  # K + V
        self._record_size = (
            _OLD_POS_HDR + self._tensors_per_block * self._tensor_bytes
        )
        self._tensor_shape = (
            config.block_size,
            config.num_kv_heads,
            config.head_dim,
        )

        # hash(bytes) -> slot index. Loaded from / persisted to index.json.
        self._slots: dict[bytes, int] = {}
        self._next_slot = 0

        self._load_or_init_index()
        self._open_mmap()

    @property
    def config(self) -> StoreConfig:
        return self._config

    # ------------------------------------------------------------------
    # public API (parity with InMemoryStorage / RedisKVStorage)
    # ------------------------------------------------------------------

    def put(self, key: bytes, block: KVBlock) -> None:
        validate_block(self._config, block)
        slot = self._slots.get(key)
        if slot is None:
            slot = self._next_slot
            self._next_slot += 1
        self._ensure_capacity(slot + 1)
        self._write_slot(slot, block)
        # Index update + flush happen only after the bytes are written, so
        # a crash mid-put never leaves the index pointing at a half-written
        # slot (the slot is simply re-used on the next put of that key).
        self._slots[key] = slot
        self._persist_index()

    def get(self, key: bytes) -> Optional[KVBlock]:
        slot = self._slots.get(key)
        if slot is None:
            return None
        return self._read_slot(slot)

    def lookup(self, keys: list[bytes]) -> list[Optional[KVBlock]]:
        """Batched ``get``; preserves order, returns ``None`` for misses.

        No round-trip cost to amortise here (it's all local memory), but
        we keep the method for interface parity with the other backends.
        """
        return [self.get(k) for k in keys]

    def __contains__(self, key: bytes) -> bool:
        return key in self._slots

    def __len__(self) -> int:
        return len(self._slots)

    def close(self) -> None:
        """Flush and unmap. Safe to call multiple times."""
        mm = getattr(self, "_mm", None)
        if mm is not None:
            try:
                mm.flush()
            finally:
                mm.close()
            self._mm = None
        f = getattr(self, "_file", None)
        if f is not None:
            f.close()
            self._file = None

    def __del__(self):  # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # internals: index
    # ------------------------------------------------------------------

    def _load_or_init_index(self) -> None:
        if not os.path.exists(self._index_path):
            self._persist_index()  # writes config + empty block map
            return
        with open(self._index_path, encoding="utf-8") as f:
            data = json.load(f)
        stored_cfg = _config_from_dict(data["config"])
        if stored_cfg != self._config:
            raise ValueError(
                "MmapKVStorage: schema mismatch against existing index at "
                f"{self._index_path}.\n"
                f"  stored:   {stored_cfg}\n"
                f"  expected: {self._config}\n"
                "Point at a different root_dir, or delete the directory to "
                "start fresh."
            )
        # JSON keys are strings; block hashes were stored as hex.
        self._slots = {
            bytes.fromhex(h): int(slot)
            for h, slot in data.get("blocks", {}).items()
        }
        self._next_slot = int(data.get("next_slot", len(self._slots)))

    def _persist_index(self) -> None:
        data = {
            "config": _config_to_dict(self._config),
            "blocks": {h.hex(): slot for h, slot in self._slots.items()},
            "next_slot": self._next_slot,
        }
        # Write-then-rename for atomicity: a reader never sees a partially
        # written index file.
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self._index_path)

    # ------------------------------------------------------------------
    # internals: mmap
    # ------------------------------------------------------------------

    def _open_mmap(self) -> None:
        # Capacity must cover whatever the index already references after a
        # reopen, with a floor of _INITIAL_SLOTS.
        min_slots = max(self._next_slot, _INITIAL_SLOTS)
        cap_bytes = min_slots * self._record_size
        # Create / size the backing file.
        if not os.path.exists(self._blocks_path):
            with open(self._blocks_path, "wb") as f:
                f.truncate(cap_bytes)
        else:
            cur = os.path.getsize(self._blocks_path)
            if cur < cap_bytes:
                with open(self._blocks_path, "r+b") as f:
                    f.truncate(cap_bytes)
            else:
                cap_bytes = cur
        self._file = open(self._blocks_path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), cap_bytes)
        self._cap_bytes = cap_bytes

    def _ensure_capacity(self, slots_needed: int) -> None:
        need = slots_needed * self._record_size
        if need <= self._cap_bytes:
            return
        new_cap = self._cap_bytes
        while new_cap < need:
            new_cap *= 2
        # Remap larger. Any tensors previously returned by ``get`` are
        # already ``.clone()``-d, so they do not alias the old mapping and
        # are unaffected by the unmap/remap here.
        self._mm.flush()
        self._mm.close()
        self._file.truncate(new_cap)
        self._mm = mmap.mmap(self._file.fileno(), new_cap)
        self._cap_bytes = new_cap

    def _slot_offset(self, slot: int) -> int:
        return slot * self._record_size

    def _write_slot(self, slot: int, block: KVBlock) -> None:
        off = self._slot_offset(slot)
        # int64 old_pos_start header.
        self._mm[off : off + _OLD_POS_HDR] = int(
            block.old_pos_start
        ).to_bytes(_OLD_POS_HDR, "little", signed=True)
        p = off + _OLD_POS_HDR
        # K tensors then V tensors, layer order, raw contiguous CPU bytes.
        # flatten-then-view(uint8): Tensor.view(dtype) only reinterprets the
        # last dim, so flatten to 1-D first to get a clean byte stream
        # regardless of the original [bs, nh, hd] shape.
        for t in (*block.keys, *block.values):
            raw = (
                t.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
            )
            mv = raw.numpy().tobytes()
            if len(mv) != self._tensor_bytes:
                raise ValueError(
                    f"MmapKVStorage: tensor byte length {len(mv)} != "
                    f"expected {self._tensor_bytes}; geometry/dtype mismatch."
                )
            self._mm[p : p + self._tensor_bytes] = mv
            p += self._tensor_bytes

    def _read_slot(self, slot: int) -> KVBlock:
        off = self._slot_offset(slot)
        old_pos_start = int.from_bytes(
            self._mm[off : off + _OLD_POS_HDR], "little", signed=True
        )
        p = off + _OLD_POS_HDR
        # Slice the mapping with mm[a:b], which returns an independent bytes
        # copy (NOT a memoryview): it leaves no exported pointer into the
        # mapping, so a later grow's mm.close()/remap can never raise
        # BufferError. The copy is a single ~2 MB memcpy — negligible next
        # to the pickle.loads + socket read it replaces on the Redis path.
        # frombuffer needs a writable buffer, so wrap in bytearray; clone()
        # then hands the caller an independent, writable tensor.
        tensors: list[torch.Tensor] = []
        for _ in range(self._tensors_per_block):
            seg = bytearray(self._mm[p : p + self._tensor_bytes])
            t = (
                torch.frombuffer(seg, dtype=torch.uint8)
                .view(self._config.dtype)
                .reshape(self._tensor_shape)
                .clone()
            )
            tensors.append(t)
            p += self._tensor_bytes
        n = self._config.num_layers
        return KVBlock(
            keys=tensors[:n],
            values=tensors[n:],
            old_pos_start=old_pos_start,
        )
