"""
Core HOL Tactics

Tactics transform goals into subgoals with justifications.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from loguru import logger

from hol.kernel.types import bool_ty, mk_fun_ty, dest_fun_ty, is_fun_ty
from hol.kernel.terms import (
    Term, Var, Const, App, Abs,
    mk_var, mk_const, mk_app, mk_eq,
    dest_app, is_app, type_of, free_vars, alpha_equiv,
)
from hol.kernel.theorems import Theorem, hyps, concl
from hol.kernel.inference import ASSUME, EQ_MP, DEDUCT_ANTISYM, REFL
from hol.tactics.goals import Goal, mk_goal, Justification


# =============================================================================
# Tactic Types
# =============================================================================

class TacticError(Exception):
    """Raised when a tactic fails."""
    pass


@dataclass
class TacticResult:
    """
    Result of applying a tactic.

    Contains:
    - subgoals: New goals to prove
    - justification: Function to build theorem from subgoal theorems
    """
    subgoals: list[Goal]
    justification: Justification


# A tactic is a function from Goal to TacticResult
Tactic = Callable[[Goal], TacticResult]


# =============================================================================
# Basic Tactics
# =============================================================================

def assumption(goal: Goal) -> TacticResult:
    """
    Solve goal if target is in assumptions.

    If target p is among assumptions, goal is solved.
    """
    for asm in goal.assumptions:
        if alpha_equiv(asm, goal.target):
            # Goal is solved - justification returns ASSUME
            def justify(thms: list[Theorem]) -> Theorem:
                return ASSUME(goal.target)
            return TacticResult([], justify)

    raise TacticError(f"Target {goal.target} not in assumptions")


def intro(name: str) -> Tactic:
    """
    Introduce an implication hypothesis.

    Transforms:  ⊢ A → B
    Into:        {A} ⊢ B

    Args:
        name: Name for the new hypothesis

    Returns:
        A tactic
    """
    def tactic(goal: Goal) -> TacticResult:
        target = goal.target

        # Check if target is an implication: (==> A B)
        if not is_app(target):
            raise TacticError(f"intro: target is not an implication: {target}")

        func, b = dest_app(target)
        if not is_app(func):
            raise TacticError(f"intro: target is not an implication: {target}")

        imp_const, a = dest_app(func)
        if not isinstance(imp_const, Const) or imp_const.name != "==>":
            raise TacticError(f"intro: target is not an implication: {target}")

        # Create new goal with A as assumption
        new_asms = goal.assumptions + (a,)
        new_goal = mk_goal(new_asms, b)

        def justify(thms: list[Theorem]) -> Theorem:
            # From {A} ⊢ B, construct ⊢ A → B
            # Use DEDUCT_ANTISYM or build implication theorem
            if len(thms) != 1:
                raise ValueError(f"intro justify: expected 1 theorem, got {len(thms)}")
            # This is simplified - real implementation needs proper implication handling
            return thms[0]

        return TacticResult([new_goal], justify)

    return tactic


def exact(thm: Theorem) -> Tactic:
    """
    Solve goal exactly with given theorem.

    The theorem's conclusion must match the goal target.

    Args:
        thm: Theorem to use

    Returns:
        A tactic
    """
    def tactic(goal: Goal) -> TacticResult:
        if not alpha_equiv(concl(thm), goal.target):
            raise TacticError(
                f"exact: theorem conclusion {concl(thm)} "
                f"does not match goal {goal.target}"
            )

        # Check hypotheses are available
        for h in hyps(thm):
            found = False
            for asm in goal.assumptions:
                if alpha_equiv(h, asm):
                    found = True
                    break
            if not found:
                raise TacticError(f"exact: hypothesis {h} not available")

        def justify(thms: list[Theorem]) -> Theorem:
            return thm

        return TacticResult([], justify)

    return tactic


def apply_thm(thm: Theorem) -> Tactic:
    """
    Apply a theorem to the goal.

    If theorem is ⊢ A → B and goal is B, creates subgoal A.

    Args:
        thm: Theorem to apply

    Returns:
        A tactic
    """
    def tactic(goal: Goal) -> TacticResult:
        # Try to match theorem conclusion with goal
        # This is a simplified version
        thm_concl = concl(thm)

        # If theorem conclusion matches goal exactly
        if alpha_equiv(thm_concl, goal.target):
            return exact(thm)(goal)

        # If theorem is implication A → B where B matches goal
        if is_app(thm_concl):
            func, b = dest_app(thm_concl)
            if is_app(func):
                imp_const, a = dest_app(func)
                if isinstance(imp_const, Const) and imp_const.name == "==>":
                    if alpha_equiv(b, goal.target):
                        # Create subgoal for A
                        new_goal = mk_goal(goal.assumptions, a)

                        def justify(thms: list[Theorem]) -> Theorem:
                            # Would use EQ_MP or similar
                            return thms[0]

                        return TacticResult([new_goal], justify)

        raise TacticError(f"apply: cannot apply {thm} to goal {goal.target}")

    return tactic


def rewrite(thm: Theorem) -> Tactic:
    """
    Rewrite goal using an equality theorem.

    If theorem is ⊢ l = r, replaces l with r in goal.

    Args:
        thm: Equality theorem

    Returns:
        A tactic
    """
    def tactic(goal: Goal) -> TacticResult:
        from hol.kernel.terms import dest_eq, is_eq, subst

        thm_concl = concl(thm)
        if not is_eq(thm_concl):
            raise TacticError(f"rewrite: theorem is not an equality: {thm}")

        lhs, rhs = dest_eq(thm_concl)

        # Simple rewriting - replace lhs with rhs in goal target
        # A full implementation would do congruence closure
        def rewrite_term(t: Term) -> Term:
            if alpha_equiv(t, lhs):
                return rhs
            elif isinstance(t, App):
                new_func = rewrite_term(t.func)
                new_arg = rewrite_term(t.arg)
                if new_func is t.func and new_arg is t.arg:
                    return t
                return App(new_func, new_arg)
            elif isinstance(t, Abs):
                new_body = rewrite_term(t.body)
                if new_body is t.body:
                    return t
                return Abs(t.var, new_body)
            else:
                return t

        new_target = rewrite_term(goal.target)

        if alpha_equiv(new_target, goal.target):
            raise TacticError(f"rewrite: no rewrite occurred")

        new_goal = mk_goal(goal.assumptions, new_target)

        def justify(thms: list[Theorem]) -> Theorem:
            # Would construct equality chain
            return thms[0]

        return TacticResult([new_goal], justify)

    return tactic


def split(goal: Goal) -> TacticResult:
    """
    Split a conjunction goal into two subgoals.

    Transforms:  ⊢ A ∧ B
    Into:        ⊢ A  and  ⊢ B
    """
    target = goal.target

    # Check if target is a conjunction: (/\ A B)
    if not is_app(target):
        raise TacticError(f"split: target is not a conjunction: {target}")

    func, b = dest_app(target)
    if not is_app(func):
        raise TacticError(f"split: target is not a conjunction: {target}")

    conj_const, a = dest_app(func)
    if not isinstance(conj_const, Const) or conj_const.name != "/\\":
        raise TacticError(f"split: target is not a conjunction: {target}")

    # Create two subgoals
    goal_a = mk_goal(goal.assumptions, a)
    goal_b = mk_goal(goal.assumptions, b)

    def justify(thms: list[Theorem]) -> Theorem:
        # Would combine the two theorems into A ∧ B
        if len(thms) != 2:
            raise ValueError(f"split justify: expected 2 theorems, got {len(thms)}")
        # Simplified - real implementation builds conjunction
        return thms[0]

    return TacticResult([goal_a, goal_b], justify)


def left(goal: Goal) -> TacticResult:
    """
    Choose left disjunct.

    Transforms:  ⊢ A ∨ B
    Into:        ⊢ A
    """
    target = goal.target

    # Check if target is a disjunction: (\/ A B)
    if not is_app(target):
        raise TacticError(f"left: target is not a disjunction: {target}")

    func, b = dest_app(target)
    if not is_app(func):
        raise TacticError(f"left: target is not a disjunction: {target}")

    disj_const, a = dest_app(func)
    if not isinstance(disj_const, Const) or disj_const.name != "\\/":
        raise TacticError(f"left: target is not a disjunction: {target}")

    new_goal = mk_goal(goal.assumptions, a)

    def justify(thms: list[Theorem]) -> Theorem:
        if len(thms) != 1:
            raise ValueError(f"left justify: expected 1 theorem, got {len(thms)}")
        return thms[0]

    return TacticResult([new_goal], justify)


def right(goal: Goal) -> TacticResult:
    """
    Choose right disjunct.

    Transforms:  ⊢ A ∨ B
    Into:        ⊢ B
    """
    target = goal.target

    # Check if target is a disjunction: (\/ A B)
    if not is_app(target):
        raise TacticError(f"right: target is not a disjunction: {target}")

    func, b = dest_app(target)
    if not is_app(func):
        raise TacticError(f"right: target is not a disjunction: {target}")

    disj_const, a = dest_app(func)
    if not isinstance(disj_const, Const) or disj_const.name != "\\/":
        raise TacticError(f"right: target is not a disjunction: {target}")

    new_goal = mk_goal(goal.assumptions, b)

    def justify(thms: list[Theorem]) -> Theorem:
        if len(thms) != 1:
            raise ValueError(f"right justify: expected 1 theorem, got {len(thms)}")
        return thms[0]

    return TacticResult([new_goal], justify)
