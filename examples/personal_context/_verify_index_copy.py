#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Throwaway verification that per-layer index_copy_ on a paged-KV view
writes back correctly and is faster than per-block copy_. Run on GPU:

    .venv/bin/python examples/personal_context/_verify_index_copy.py

Delete after the scatter optimization is validated.
"""
import time

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
# Mimic 2k-context scatter geometry: 32 layers, ~56 blocks total, NHD.
NUM_LAYERS = 32
NUM_BLOCKS_CACHE = 256
N_WRITE = 56          # blocks actually written this plan
PAGE, NH, HD = 16, 8, 128
DT = torch.float16

# ---- block_first layout: [num_blocks, 2, page, nh, hd] ----
caches_a = [
    torch.zeros(NUM_BLOCKS_CACHE, 2, PAGE, NH, HD, dtype=DT, device=dev)
    for _ in range(NUM_LAYERS)
]
caches_b = [c.clone() for c in caches_a]

# Source blocks (already on device, as load_plan would hand them).
block_ids = torch.randperm(NUM_BLOCKS_CACHE, device=dev)[:N_WRITE]
# per (block, layer) K and V
K = [[torch.randn(PAGE, NH, HD, dtype=DT, device=dev) for _ in range(NUM_LAYERS)]
     for _ in range(N_WRITE)]
V = [[torch.randn(PAGE, NH, HD, dtype=DT, device=dev) for _ in range(NUM_LAYERS)]
     for _ in range(N_WRITE)]

# ===== Method A: per-block copy_ (current scatter) =====
torch.cuda.synchronize() if dev == "cuda" else None
t0 = time.perf_counter()
for b in range(N_WRITE):
    bid = int(block_ids[b])
    for layer in range(NUM_LAYERS):
        caches_a[layer][bid, 0].copy_(K[b][layer])
        caches_a[layer][bid, 1].copy_(V[b][layer])
torch.cuda.synchronize() if dev == "cuda" else None
t_a = time.perf_counter() - t0

# ===== Method B: per-layer index_copy_ on K/V views =====
torch.cuda.synchronize() if dev == "cuda" else None
t0 = time.perf_counter()
for layer in range(NUM_LAYERS):
    k_stack = torch.stack([K[b][layer] for b in range(N_WRITE)], 0)
    v_stack = torch.stack([V[b][layer] for b in range(N_WRITE)], 0)
    caches_b[layer][:, 0].index_copy_(0, block_ids, k_stack)
    caches_b[layer][:, 1].index_copy_(0, block_ids, v_stack)
torch.cuda.synchronize() if dev == "cuda" else None
t_b = time.perf_counter() - t0

# ===== Correctness: B must equal A exactly =====
ok = all(torch.equal(caches_a[layer], caches_b[layer])
         for layer in range(NUM_LAYERS))
print(f"device={dev}")
print(f"per-block copy_   : {t_a*1000:.1f}ms")
print(f"index_copy_ batch : {t_b*1000:.1f}ms")
print(f"byte-exact equal  : {ok}")
print("SPEEDUP" if t_b < t_a else "NO SPEEDUP", f"({t_a/t_b:.1f}x)" if t_b > 0 else "")
