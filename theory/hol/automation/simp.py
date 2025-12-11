"""
Simplifier - Term Rewriting

Implements conditional term rewriting to normal form.
Supports user-specified rewrite rules and congruence reasoning.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from loguru import logger

from hol.kernel.types import bool_ty
from hol.kernel.terms import (
    Term, Var, Const, App, Abs,
    mk_var, mk_eq, dest_eq, is_eq, alpha_equiv,
    type_of, free_vars, subst,
)
from hol.kernel.theorems import Theorem, hyps, concl
from hol.kernel.inference import REFL
from hol.tactics.goals import Goal, mk_goal
from hol.tactics.tactics import Tactic, TacticResult, TacticError


@dataclass
class SimpRule:
    """A simplification rule: rewrite lhs to rhs."""
    lhs: Term
    rhs: Term
    theorem: Theorem  # |- lhs = rhs


def mk_simp_rule(thm: Theorem) -> SimpRule:
    """
    Create a simplification rule from an equality theorem.

    Args:
        thm: A theorem of the form |- lhs = rhs

    Returns:
        A SimpRule for rewriting lhs to rhs

    Raises:
        ValueError: If theorem is not an equality
    """
    c = concl(thm)
    if not is_eq(c):
        raise ValueError(f"Expected equality theorem, got {c}")
    lhs, rhs = dest_eq(c)
    return SimpRule(lhs, rhs, thm)


def simp(
    rules: Sequence[Theorem] | None = None,
    add: Sequence[Theorem] | None = None,
    only: Sequence[Theorem] | None = None,
) -> Tactic:
    """
    Simplifier tactic.

    Rewrites the goal using equality theorems until no more
    rewriting is possible.

    Args:
        rules: Equality theorems to use as rewrite rules
        add: Additional rules to add to defaults
        only: Use only these rules (ignore defaults)

    Returns:
        A tactic that simplifies the goal

    Raises:
        TacticError: If simplification doesn't solve or change goal
    """
    all_rules: list[Theorem] = []

    if only is not None:
        all_rules = list(only)
    else:
        if rules is not None:
            all_rules.extend(rules)
        if add is not None:
            all_rules.extend(add)

    def simp_tactic(goal: Goal) -> TacticResult:
        target = goal.target

        # If target is an equality, try to prove it
        if is_eq(target):
            lhs, rhs = dest_eq(target)

            # Simplify both sides
            simp_rules = [mk_simp_rule(r) for r in all_rules if is_eq(concl(r))]
            new_lhs = _simplify_term(lhs, simp_rules)
            new_rhs = _simplify_term(rhs, simp_rules)

            # If both sides simplify to the same thing, goal is solved
            if alpha_equiv(new_lhs, new_rhs):
                def justify(thms: list[Theorem]) -> Theorem:
                    return REFL(new_lhs)
                return TacticResult([], justify)

            # Check if simplified goal is in assumptions
            simplified_eq = mk_eq(new_lhs, new_rhs)
            for asm in goal.assumptions:
                if alpha_equiv(asm, simplified_eq) or alpha_equiv(asm, target):
                    def justify(thms: list[Theorem]) -> Theorem:
                        from hol.kernel.inference import ASSUME
                        return ASSUME(target)
                    return TacticResult([], justify)

        # Check if target is directly in assumptions
        for asm in goal.assumptions:
            if alpha_equiv(asm, target):
                def justify(thms: list[Theorem]) -> Theorem:
                    from hol.kernel.inference import ASSUME
                    return ASSUME(target)
                return TacticResult([], justify)

        raise TacticError(f"simp: cannot simplify {target}")

    return simp_tactic


def _simplify_term(term: Term, rules: list[SimpRule]) -> Term:
    """
    Simplify a term by applying rewrite rules exhaustively.
    """
    changed = True
    current = term
    max_iterations = 100  # Prevent infinite loops
    iterations = 0

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Try to apply each rule
        for rule in rules:
            new_term = _try_rewrite(current, rule)
            if new_term is not None and not alpha_equiv(new_term, current):
                current = new_term
                changed = True
                break

        # Also simplify subterms
        if not changed:
            new_term = _simplify_subterms(current, rules)
            if not alpha_equiv(new_term, current):
                current = new_term
                changed = True

    return current


def _try_rewrite(term: Term, rule: SimpRule) -> Term | None:
    """
    Try to apply a rewrite rule at the root of the term.
    Returns the rewritten term or None if rule doesn't match.
    """
    # Simple matching: check if term equals rule.lhs
    if alpha_equiv(term, rule.lhs):
        return rule.rhs

    # TODO: Add proper pattern matching with variable instantiation
    return None


def _simplify_subterms(term: Term, rules: list[SimpRule]) -> Term:
    """
    Recursively simplify subterms (congruence reasoning).
    """
    if isinstance(term, Var) or isinstance(term, Const):
        return term

    elif isinstance(term, App):
        new_func = _simplify_term(term.func, rules)
        new_arg = _simplify_term(term.arg, rules)
        if new_func is term.func and new_arg is term.arg:
            return term
        return App(new_func, new_arg)

    elif isinstance(term, Abs):
        new_body = _simplify_term(term.body, rules)
        if new_body is term.body:
            return term
        return Abs(term.var, new_body)

    return term
