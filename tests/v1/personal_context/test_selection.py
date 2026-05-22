# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the stale-KV selection abstractions (Phase 7 Step 6.1/6.2)."""

import math

import pytest

from vllm.v1.personal_context import (
    Chunk,
    NoSelection,
    ReusePlan,
    SelectFirstR,
    Selector,
)


BLOCK_SIZE = 4


def _plan(chunks: list[Chunk]) -> ReusePlan:
    return ReusePlan(chunks=tuple(chunks))


# ----------------------- Protocol conformance -----------------------


def test_noselection_satisfies_protocol():
    assert isinstance(NoSelection(), Selector)


def test_selectfirstr_satisfies_protocol():
    assert isinstance(SelectFirstR(0.5), Selector)


# ----------------------- NoSelection -----------------------


def test_noselection_returns_empty_for_any_plan():
    sel = NoSelection()
    plan = _plan([Chunk(token_ids=tuple(range(8)), old_pos_start=0)])
    assert sel.select(plan, (0,)) == ()


def test_noselection_returns_empty_for_empty_plan():
    sel = NoSelection()
    assert sel.select(_plan([]), ()) == ()


# ----------------------- SelectFirstR construction -----------------------


@pytest.mark.parametrize("r", [-0.0001, -1.0, 1.0001, 2.0])
def test_selectfirstr_rejects_out_of_range_r(r):
    with pytest.raises(ValueError, match="r must be in"):
        SelectFirstR(r)


def test_selectfirstr_accepts_boundaries():
    SelectFirstR(0.0)
    SelectFirstR(1.0)


def test_selectfirstr_exposes_r():
    assert SelectFirstR(0.25).r == 0.25


# ----------------------- SelectFirstR.select -----------------------


def test_selectfirstr_zero_is_empty():
    sel = SelectFirstR(0.0)
    plan = _plan([Chunk(token_ids=tuple(range(16)), old_pos_start=0)])
    assert sel.select(plan, (0,)) == ()


def test_selectfirstr_one_selects_all_positions():
    sel = SelectFirstR(1.0)
    plan = _plan([Chunk(token_ids=tuple(range(8)), old_pos_start=0)])
    assert sel.select(plan, (0,)) == tuple(range(8))


def test_selectfirstr_half_on_clean_division():
    """L=8, r=0.5 → ceil(4) = 4 positions [0,1,2,3]."""
    sel = SelectFirstR(0.5)
    plan = _plan([Chunk(token_ids=tuple(range(8)), old_pos_start=0)])
    assert sel.select(plan, (0,)) == (0, 1, 2, 3)


def test_selectfirstr_quarter_on_clean_division():
    """L=16, r=0.25 → ceil(4) = 4 positions [0..3]."""
    sel = SelectFirstR(0.25)
    plan = _plan([Chunk(token_ids=tuple(range(16)), old_pos_start=0)])
    assert sel.select(plan, (0,)) == (0, 1, 2, 3)


def test_selectfirstr_ceil_rounds_up_non_integer_budget():
    """L=10, r=0.05 → ceil(0.5) = 1 position [0]."""
    sel = SelectFirstR(0.05)
    plan = _plan([Chunk(token_ids=tuple(range(BLOCK_SIZE * 4)), old_pos_start=0)])
    # length = 16, r=0.05 → ceil(0.8) = 1
    assert sel.select(plan, (0,)) == (0,)


def test_selectfirstr_uses_new_pos_start_as_base():
    """new_pos_start != 0 → returned positions are offset accordingly."""
    sel = SelectFirstR(0.5)
    plan = _plan([Chunk(token_ids=tuple(range(8)), old_pos_start=0)])
    assert sel.select(plan, (100,)) == (100, 101, 102, 103)


def test_selectfirstr_multi_chunk_concatenates_per_chunk_first_r():
    """Each chunk independently emits its own first-r% prefix."""
    sel = SelectFirstR(0.5)
    plan = _plan(
        [
            Chunk(token_ids=tuple(range(4)), old_pos_start=0),
            Chunk(token_ids=tuple(range(8)), old_pos_start=BLOCK_SIZE),
        ]
    )
    # Chunk A: L=4, ceil(2)=2 positions starting at 50 → [50, 51]
    # Chunk B: L=8, ceil(4)=4 positions starting at 100 → [100..103]
    assert sel.select(plan, (50, 100)) == (50, 51, 100, 101, 102, 103)


def test_selectfirstr_empty_plan_returns_empty():
    sel = SelectFirstR(0.5)
    assert sel.select(_plan([]), ()) == ()


def test_selectfirstr_rejects_length_mismatch():
    sel = SelectFirstR(0.5)
    plan = _plan([Chunk(token_ids=tuple(range(4)), old_pos_start=0)])
    with pytest.raises(ValueError, match="length"):
        sel.select(plan, (0, 100))


def test_selectfirstr_handles_short_chunk_with_small_r():
    """Edge: very small r against very small chunk still yields ceil(...) >= 0."""
    sel = SelectFirstR(0.01)
    plan = _plan([Chunk(token_ids=(1, 2, 3, 4), old_pos_start=0)])
    # ceil(0.04) = 1
    assert sel.select(plan, (0,)) == (0,)


def test_selectfirstr_outputs_are_consistent_with_math_ceil():
    """Property: |output| per chunk == ceil(r * L). Spot-check several r/L."""
    cases = [
        (0.1, 10),
        (0.33, 9),
        (0.5, 17),  # odd L
        (0.7, 21),
        (0.99, 100),
    ]
    for r, L in cases:
        plan = _plan([Chunk(token_ids=tuple(range(L)), old_pos_start=0)])
        sel = SelectFirstR(r)
        out = sel.select(plan, (0,))
        assert len(out) == math.ceil(r * L), f"r={r}, L={L}"
        assert out == tuple(range(math.ceil(r * L)))
