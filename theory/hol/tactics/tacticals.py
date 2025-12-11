"""
Tacticals: Tactic Combinators

Tacticals combine tactics to build more complex proof strategies.
"""
from __future__ import annotations
from typing import Callable
from loguru import logger

from hol.tactics.goals import Goal, Justification
from hol.tactics.tactics import Tactic, TacticResult, TacticError


# =============================================================================
# Basic Tacticals
# =============================================================================

def ALL_TAC(goal: Goal) -> TacticResult:
    """
    Identity tactic: always succeeds, leaves goal unchanged.
    """
    def justify(thms: list) -> any:
        if len(thms) != 1:
            raise ValueError(f"ALL_TAC justify: expected 1 theorem, got {len(thms)}")
        return thms[0]

    return TacticResult([goal], justify)


def NO_TAC(goal: Goal) -> TacticResult:
    """
    Failure tactic: always fails.
    """
    raise TacticError("NO_TAC: always fails")


# =============================================================================
# Sequencing
# =============================================================================

def THEN(tac1: Tactic, tac2: Tactic) -> Tactic:
    """
    Sequential composition: apply tac1, then tac2 to all subgoals.

    tac1 THEN tac2
    """
    def combined(goal: Goal) -> TacticResult:
        result1 = tac1(goal)

        if not result1.subgoals:
            # No subgoals - tac1 solved everything
            return result1

        # Apply tac2 to each subgoal
        all_subgoals: list[Goal] = []
        all_justifications: list[Justification] = []

        for subgoal in result1.subgoals:
            result2 = tac2(subgoal)
            all_subgoals.extend(result2.subgoals)
            all_justifications.append(result2.justification)

        def justify(thms: list) -> any:
            # Split theorems according to subgoal counts
            idx = 0
            intermediate_thms = []
            for i, just in enumerate(all_justifications):
                subgoal_count = len(result1.subgoals)
                # This is simplified - would need proper theorem counting
                intermediate_thms.append(just(thms[idx:idx+1] if idx < len(thms) else []))
                idx += 1
            return result1.justification(intermediate_thms if intermediate_thms else thms)

        return TacticResult(all_subgoals, justify)

    return combined


def THENL(tac1: Tactic, tacs: list[Tactic]) -> Tactic:
    """
    Apply tac1, then apply tacs[i] to subgoal i.

    tac1 THENL [tac2, tac3, ...]
    """
    def combined(goal: Goal) -> TacticResult:
        result1 = tac1(goal)

        if len(result1.subgoals) != len(tacs):
            raise TacticError(
                f"THENL: expected {len(result1.subgoals)} tactics, got {len(tacs)}"
            )

        all_subgoals: list[Goal] = []
        all_justifications: list[Justification] = []

        for subgoal, tac in zip(result1.subgoals, tacs):
            result = tac(subgoal)
            all_subgoals.extend(result.subgoals)
            all_justifications.append(result.justification)

        def justify(thms: list) -> any:
            # Similar to THEN
            return result1.justification(thms)

        return TacticResult(all_subgoals, justify)

    return combined


# =============================================================================
# Choice
# =============================================================================

def ORELSE(tac1: Tactic, tac2: Tactic) -> Tactic:
    """
    Try tac1, if it fails try tac2.

    tac1 ORELSE tac2
    """
    def combined(goal: Goal) -> TacticResult:
        try:
            return tac1(goal)
        except TacticError:
            return tac2(goal)

    return combined


def FIRST(tacs: list[Tactic]) -> Tactic:
    """
    Try tactics in order until one succeeds.

    FIRST [tac1, tac2, ...] = tac1 ORELSE tac2 ORELSE ...
    """
    def combined(goal: Goal) -> TacticResult:
        errors = []
        for tac in tacs:
            try:
                return tac(goal)
            except TacticError as e:
                errors.append(str(e))
        raise TacticError(f"FIRST: all tactics failed: {errors}")

    return combined


# =============================================================================
# Repetition
# =============================================================================

def REPEAT(tac: Tactic) -> Tactic:
    """
    Apply tac repeatedly until it fails.

    Always succeeds (with original goal if tac fails immediately).
    """
    def combined(goal: Goal) -> TacticResult:
        current_goals = [goal]
        final_subgoals: list[Goal] = []

        while current_goals:
            goal = current_goals.pop(0)
            try:
                result = tac(goal)
                # Add new subgoals to process
                current_goals.extend(result.subgoals)
            except TacticError:
                # Tactic failed - this goal is done
                final_subgoals.append(goal)

        if not final_subgoals:
            # All goals were solved
            def justify(thms: list) -> any:
                return thms[0] if thms else None
            return TacticResult([], justify)

        def justify(thms: list) -> any:
            return thms[0] if thms else None

        return TacticResult(final_subgoals, justify)

    return combined


def REPEAT1(tac: Tactic) -> Tactic:
    """
    Apply tac at least once, then repeat until failure.

    Fails if tac fails on first application.
    """
    return THEN(tac, REPEAT(tac))


# =============================================================================
# Optional
# =============================================================================

def TRY(tac: Tactic) -> Tactic:
    """
    Try tac, succeed with unchanged goal if it fails.

    TRY tac = tac ORELSE ALL_TAC
    """
    return ORELSE(tac, ALL_TAC)


# =============================================================================
# Conditional
# =============================================================================

def IF_THEN_ELSE(
    test: Callable[[Goal], bool],
    then_tac: Tactic,
    else_tac: Tactic
) -> Tactic:
    """
    Conditional tactic: if test(goal) then then_tac else else_tac.
    """
    def combined(goal: Goal) -> TacticResult:
        if test(goal):
            return then_tac(goal)
        else:
            return else_tac(goal)

    return combined


# =============================================================================
# Goal Selection
# =============================================================================

def SUBGOAL_THEN(tac: Tactic, n: int = 0) -> Tactic:
    """
    Apply tactic only to subgoal n.

    Other subgoals are passed through unchanged.
    """
    def combined(goal: Goal) -> TacticResult:
        # This would apply to a specific subgoal in a proof state
        # For single-goal context, just apply the tactic
        return tac(goal)

    return combined
