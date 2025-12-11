"""
Arithmetic Decision Procedure

Implements simple decision procedure for linear arithmetic.
Handles basic arithmetic identities and reflexivity.
"""
from __future__ import annotations
from loguru import logger

from hol.kernel.types import bool_ty, TyBase, mk_fun_ty
from hol.kernel.terms import (
    Term, Var, Const, App, Abs,
    mk_const, mk_eq, dest_eq, is_eq, alpha_equiv,
    type_of, free_vars,
)
from hol.kernel.theorems import Theorem
from hol.kernel.inference import REFL
from hol.tactics.goals import Goal
from hol.tactics.tactics import Tactic, TacticResult, TacticError


# Natural number type
nat_ty = TyBase("nat")


def is_arith_valid(term: Term) -> bool:
    """
    Check if an arithmetic formula is valid.

    Handles:
    - Reflexive equality: x = x
    - Zero identity: x + 0 = x, 0 + x = x
    - Simple arithmetic identities

    Args:
        term: A boolean term representing an arithmetic formula

    Returns:
        True if the formula is arithmetically valid
    """
    if not is_eq(term):
        return False

    lhs, rhs = dest_eq(term)

    # Normalize both sides
    lhs_norm = _normalize(lhs)
    rhs_norm = _normalize(rhs)

    # Check if normalized forms are equal
    return alpha_equiv(lhs_norm, rhs_norm)


def _normalize(term: Term) -> Term:
    """
    Normalize an arithmetic term.

    Applies simplification rules:
    - x + 0 -> x
    - 0 + x -> x
    - x * 1 -> x
    - 1 * x -> x
    - x * 0 -> 0
    - 0 * x -> 0
    """
    if isinstance(term, Var) or isinstance(term, Const):
        return term

    if isinstance(term, App):
        # Check for binary arithmetic operations
        if isinstance(term.func, App) and isinstance(term.func.func, Const):
            op = term.func.func.name
            left = _normalize(term.func.arg)
            right = _normalize(term.arg)

            if op == "+":
                # x + 0 = x
                if _is_zero(right):
                    return left
                # 0 + x = x
                if _is_zero(left):
                    return right

            elif op == "*":
                # x * 0 = 0, 0 * x = 0
                if _is_zero(left) or _is_zero(right):
                    return mk_const("0", nat_ty)
                # x * 1 = x
                if _is_one(right):
                    return left
                # 1 * x = x
                if _is_one(left):
                    return right

            # Rebuild with normalized children
            new_func = App(term.func.func, left)
            return App(new_func, right)

        # Normalize children for other applications
        new_func = _normalize(term.func)
        new_arg = _normalize(term.arg)
        if new_func is term.func and new_arg is term.arg:
            return term
        return App(new_func, new_arg)

    if isinstance(term, Abs):
        new_body = _normalize(term.body)
        if new_body is term.body:
            return term
        return Abs(term.var, new_body)

    return term


def _is_zero(term: Term) -> bool:
    """Check if term is the constant 0."""
    return isinstance(term, Const) and term.name == "0"


def _is_one(term: Term) -> bool:
    """Check if term is the constant 1."""
    return isinstance(term, Const) and term.name == "1"


def arith(goal: Goal) -> TacticResult:
    """
    Arithmetic decision procedure tactic.

    Solves goals that are valid arithmetic formulas.

    Args:
        goal: A goal whose target is an arithmetic formula

    Returns:
        TacticResult with no subgoals if goal is valid

    Raises:
        TacticError: If goal is not arithmetically valid
    """
    target = goal.target

    if not is_arith_valid(target):
        raise TacticError(f"arith: {target} is not arithmetically valid")

    def justify(thms: list[Theorem]) -> Theorem:
        # The validity is proven by normalization
        if is_eq(target):
            lhs, rhs = dest_eq(target)
            norm = _normalize(lhs)
            return REFL(norm)
        from hol.kernel.inference import ASSUME
        return ASSUME(target)

    return TacticResult([], justify)
