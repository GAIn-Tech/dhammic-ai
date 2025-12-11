"""
Auto Tactic - Automatic Proof Search

Implements depth-limited proof search with backtracking.
Tries safe tactics (assumption, intro, split) repeatedly.
"""
from __future__ import annotations
from typing import Callable
from loguru import logger

from hol.kernel.types import bool_ty
from hol.kernel.terms import Term, is_eq, dest_eq, alpha_equiv
from hol.kernel.theorems import Theorem, hyps, concl
from hol.tactics.goals import Goal, mk_goal
from hol.tactics.tactics import (
    Tactic, TacticResult, TacticError,
    assumption, intro, split, exact,
)
from hol.tactics.tacticals import ORELSE, TRY, FIRST


def auto(depth: int = 5, hints: list[Theorem] | None = None) -> Tactic:
    """
    Automatic proof search tactic.

    Tries to solve the goal automatically using:
    - assumption: if target is in hypotheses
    - intro: for implications
    - split: for conjunctions
    - exact with hints: if a hint theorem matches

    Args:
        depth: Maximum search depth (default 5)
        hints: List of theorems to try as hints

    Returns:
        A tactic that attempts automatic proof

    Raises:
        TacticError: If goal cannot be solved within depth limit
    """
    if hints is None:
        hints = []

    def auto_tactic(goal: Goal) -> TacticResult:
        return _auto_search(goal, depth, hints)

    return auto_tactic


def _auto_search(goal: Goal, depth: int, hints: list[Theorem]) -> TacticResult:
    """
    Depth-limited proof search with backtracking.
    """
    # Try assumption first (always safe, no depth cost)
    try:
        return assumption(goal)
    except TacticError:
        pass

    # Try hints (exact match)
    for hint in hints:
        try:
            return exact(hint)(goal)
        except TacticError:
            pass

    # Check depth limit
    if depth <= 0:
        raise TacticError(f"auto: depth limit reached, cannot solve {goal.target}")

    # Try intro for implications
    try:
        result = intro("h")(goal)
        # Recursively solve subgoals
        return _solve_subgoals(result, depth - 1, hints)
    except TacticError:
        pass

    # Try split for conjunctions
    try:
        result = split(goal)
        # Recursively solve subgoals
        return _solve_subgoals(result, depth - 1, hints)
    except TacticError:
        pass

    raise TacticError(f"auto: no applicable tactic for {goal.target}")


def _solve_subgoals(result: TacticResult, depth: int, hints: list[Theorem]) -> TacticResult:
    """
    Recursively solve all subgoals.
    """
    if not result.subgoals:
        return result

    # Solve each subgoal
    all_solved = True
    for subgoal in result.subgoals:
        sub_result = _auto_search(subgoal, depth, hints)
        if sub_result.subgoals:
            all_solved = False
            break

    if all_solved:
        # All subgoals solved - return empty subgoals
        def justify(thms: list[Theorem]) -> Theorem:
            return result.justification(thms)
        return TacticResult([], justify)
    else:
        raise TacticError("auto: could not solve all subgoals")
