# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PersonalContext sparse-Q helpers on GPUModelRunner.

Tests the two helpers added to ``gpu_model_runner.py``
(``_pc_collect_selected_positions`` + ``_pc_build_sparse_q_arrays``)
in isolation, using a duck-typed mock for the runner / input batch.
The full ``_prepare_inputs`` path uses many other runner attributes
that require an actual GPUModelRunner instance; that integration is
covered by the GPU smoke test (run separately on a CUDA host).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.personal_context_connector import (
    PersonalContextConnectorMetadata,
    PersonalContextReqMeta,
)
from vllm.v1.personal_context import Chunk, ReusePlan
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


# ----------------------- helpers / fixtures -----------------------


def _mock_runner(
    req_id_to_index: dict[str, int],
    num_computed_tokens_cpu: np.ndarray,
):
    """A SimpleNamespace shaped enough to call the two PC helpers
    via their unbound methods (i.e. ``cls.method.__get__(mock)``)."""
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index=dict(req_id_to_index),
            num_computed_tokens_cpu=num_computed_tokens_cpu,
        ),
    )


def _meta_with_selections(
    *items: tuple[str, list[int], tuple[int, ...]],
) -> PersonalContextConnectorMetadata:
    """Build a PersonalContextConnectorMetadata from
    ``[(request_id, chunk_token_ids, selected_positions), ...]``.

    Block assignments / new_pos_starts are not relevant for the helpers
    under test; we just make them structurally valid."""
    reqs = []
    for req_id, token_ids, selected in items:
        plan = ReusePlan(
            chunks=(Chunk(token_ids=tuple(token_ids), old_pos_start=0),)
        )
        reqs.append(
            PersonalContextReqMeta(
                request_id=req_id,
                plan=plan,
                block_assignments=((0,),),
                new_pos_starts=(0,),
                selected_positions=tuple(selected),
            )
        )
    return PersonalContextConnectorMetadata(requests=tuple(reqs))


def _collect(mock, scheduler_output, num_reqs):
    return GPUModelRunner._pc_collect_selected_positions(
        mock, scheduler_output, num_reqs
    )


def _build(mock, num_scheduled_tokens, pc_overrides):
    return GPUModelRunner._pc_build_sparse_q_arrays(
        mock, num_scheduled_tokens, pc_overrides
    )


# ----------------------- _pc_collect_selected_positions -----------------------


def test_collect_returns_none_for_no_metadata():
    mock = _mock_runner({"r": 0}, np.array([4]))
    out = _collect(
        mock,
        SimpleNamespace(kv_connector_metadata=None),
        num_reqs=1,
    )
    assert out is None


def test_collect_returns_none_for_non_personal_context_metadata():
    mock = _mock_runner({"r": 0}, np.array([4]))

    class _OtherMeta:
        pass

    out = _collect(
        mock,
        SimpleNamespace(kv_connector_metadata=_OtherMeta()),
        num_reqs=1,
    )
    assert out is None


def test_collect_returns_none_when_all_selections_empty():
    mock = _mock_runner({"r": 0}, np.array([4]))
    meta = _meta_with_selections(("r", [0, 1, 2, 3], ()))
    out = _collect(
        mock,
        SimpleNamespace(kv_connector_metadata=meta),
        num_reqs=1,
    )
    assert out is None


def test_collect_maps_req_id_to_idx():
    mock = _mock_runner({"a": 0, "b": 1}, np.array([4, 4]))
    meta = _meta_with_selections(
        ("a", [0, 1, 2, 3], (0, 2)),
        ("b", [10, 11, 12, 13], (1,)),
    )
    out = _collect(
        mock,
        SimpleNamespace(kv_connector_metadata=meta),
        num_reqs=2,
    )
    assert out == {0: (0, 2), 1: (1,)}


def test_collect_skips_requests_not_in_input_batch():
    """A request in connector meta but not in the active InputBatch
    (e.g. dropped between scheduler and runner) is silently skipped."""
    mock = _mock_runner({"present": 0}, np.array([4]))
    meta = _meta_with_selections(
        ("missing", [0, 1, 2, 3], (1,)),
        ("present", [10, 11, 12, 13], (0,)),
    )
    out = _collect(
        mock,
        SimpleNamespace(kv_connector_metadata=meta),
        num_reqs=1,
    )
    assert out == {0: (0,)}


def test_collect_clamps_to_num_reqs():
    """req_idx >= num_reqs is treated as out-of-batch and skipped."""
    mock = _mock_runner({"r": 5}, np.zeros(10))
    meta = _meta_with_selections(("r", [0, 1, 2, 3], (1,)))
    out = _collect(
        mock,
        SimpleNamespace(kv_connector_metadata=meta),
        num_reqs=3,
    )
    assert out is None


# ----------------------- _pc_build_sparse_q_arrays -----------------------


def test_build_vanilla_for_non_pc_request():
    """Without any pc_overrides entry for a request, that request's
    output mirrors the vanilla ``num_computed + range(n_query)``."""
    mock = _mock_runner({}, np.array([8, 12]))
    num_scheduled = np.array([2, 3], dtype=np.int64)
    eff, total, positions = _build(mock, num_scheduled, pc_overrides={})
    # No PC overrides: arrays unchanged, positions are contiguous from
    # num_computed_tokens_cpu.
    assert eff.tolist() == [2, 3]
    assert total == 5
    assert positions.tolist() == [8, 9, 12, 13, 14]


def test_build_expands_pc_request_with_selected_outside_query():
    """Selected positions inside the cached range get added to the Q range."""
    # 1 request, num_computed=8, n_query=2 → query positions [8, 9]
    # selected=(2, 5) (inside cached range, before num_computed)
    # combined sorted = [2, 5, 8, 9], effective n = 4
    mock = _mock_runner({}, np.array([8]))
    eff, total, positions = _build(
        mock, np.array([2], dtype=np.int64), pc_overrides={0: (2, 5)}
    )
    assert eff.tolist() == [4]
    assert total == 4
    assert positions.tolist() == [2, 5, 8, 9]


def test_build_dedupes_selected_in_query_range():
    """A selected position that overlaps the query range counts once."""
    mock = _mock_runner({}, np.array([8]))
    # query=[8, 9, 10], selected={2, 5, 8, 9} — 8 and 9 dedupe
    eff, total, positions = _build(
        mock, np.array([3], dtype=np.int64), pc_overrides={0: (2, 5, 8, 9)}
    )
    assert eff.tolist() == [5]  # {2,5,8,9,10}
    assert total == 5
    assert positions.tolist() == [2, 5, 8, 9, 10]


def test_build_per_request_blocks_are_contiguous_in_concat():
    """Multi-request: each req's positions form a contiguous block in
    the output, in req-idx order."""
    mock = _mock_runner({}, np.array([4, 8]))
    eff, total, positions = _build(
        mock,
        np.array([2, 3], dtype=np.int64),
        pc_overrides={0: (0, 1)},  # only req 0 has selection
    )
    # Req 0: query=[4,5], sel={0,1} → combined sorted [0,1,4,5], eff=4
    # Req 1 (no PC): query=[8,9,10], eff=3
    assert eff.tolist() == [4, 3]
    assert total == 7
    assert positions.tolist() == [0, 1, 4, 5, 8, 9, 10]


def test_build_positions_are_sorted_within_each_request():
    """Selected positions provided in any order come out sorted."""
    mock = _mock_runner({}, np.array([10]))
    # Provide selected in non-sorted order; expect sorted output.
    eff, total, positions = _build(
        mock,
        np.array([2], dtype=np.int64),
        pc_overrides={0: (7, 1, 5, 3)},
    )
    # query=[10,11], sel={1,3,5,7} → combined sorted [1,3,5,7,10,11]
    assert eff.tolist() == [6]
    assert total == 6
    assert positions.tolist() == [1, 3, 5, 7, 10, 11]


def test_build_empty_input_returns_empty():
    """Defensive: num_reqs==0 — no input at all."""
    mock = _mock_runner({}, np.array([]))
    eff, total, positions = _build(
        mock, np.array([], dtype=np.int64), pc_overrides={}
    )
    assert eff.tolist() == []
    assert total == 0
    assert positions.tolist() == []


def test_build_does_not_mutate_input_num_scheduled():
    """``num_scheduled_tokens`` argument is the caller's array; must
    not be mutated by the helper (we ``.copy()`` internally)."""
    mock = _mock_runner({}, np.array([5]))
    original = np.array([2], dtype=np.int64)
    snapshot = original.copy()
    _build(mock, original, pc_overrides={0: (0, 1)})
    np.testing.assert_array_equal(original, snapshot)


def test_build_outputs_int64_dtype():
    """Downstream tensor ops assume int64; verify dtype contract."""
    mock = _mock_runner({}, np.array([5]))
    eff, _, positions = _build(
        mock, np.array([2], dtype=np.int32), pc_overrides={0: (0, 1)}
    )
    assert eff.dtype == np.int64
    assert positions.dtype == np.int64
