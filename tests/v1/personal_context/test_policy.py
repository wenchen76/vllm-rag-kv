# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import pytest

from vllm.v1.personal_context import (
    Chunk,
    PrefixCachePollutionError,
    ReusePlan,
    assert_can_write_prefix_cache,
)


def _chunk(start: int = 0) -> Chunk:
    return Chunk(token_ids=(1, 2, 3, 4), old_pos_start=start)


def test_empty_plan_is_not_mixed_kv():
    plan = ReusePlan()
    assert plan.is_mixed_kv() is False
    assert_can_write_prefix_cache(plan)  # no raise


def test_none_plan_treated_as_vanilla():
    assert_can_write_prefix_cache(None)  # no raise


def test_plan_with_chunk_is_mixed_kv():
    plan = ReusePlan(chunks=(_chunk(),))
    assert plan.is_mixed_kv() is True


def test_mixed_kv_plan_blocks_prefix_cache_write():
    plan = ReusePlan(chunks=(_chunk(),))
    with pytest.raises(PrefixCachePollutionError):
        assert_can_write_prefix_cache(plan)


def test_multi_chunk_plan_blocks_prefix_cache_write():
    plan = ReusePlan(chunks=(_chunk(0), _chunk(4), _chunk(8)))
    assert plan.is_mixed_kv() is True
    with pytest.raises(PrefixCachePollutionError):
        assert_can_write_prefix_cache(plan)


def test_pollution_error_is_value_error():
    assert issubclass(PrefixCachePollutionError, ValueError)


def test_reuse_plan_is_frozen():
    plan = ReusePlan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.chunks = (_chunk(),)  # type: ignore[misc]


def test_reuse_plan_default_is_empty_tuple():
    plan = ReusePlan()
    assert plan.chunks == ()
