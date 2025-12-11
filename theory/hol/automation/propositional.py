"""
Propositional Solver

Implements tautology checking for propositional formulas.
Uses truth table evaluation.
"""
from __future__ import annotations
from itertools import product
from loguru import logger

from hol.kernel.types import bool_ty
from hol.kernel.terms import (
    Term, Var, Const, App, Abs,
    type_of, free_vars,
)
from hol.kernel.theorems import Theorem
from hol.tactics.goals import Goal
from hol.tactics.tactics import Tactic, TacticResult, TacticError


def is_tautology(term: Term) -> bool:
    """
    Check if a propositional term is a tautology.

    Uses truth table evaluation: tries all possible assignments
    of truth values to variables and checks if the formula is
    always true.

    Args:
        term: A boolean term

    Returns:
        True if term is a tautology, False otherwise
    """
    if type_of(term) != bool_ty:
        return False

    # Get all boolean variables
    vars_set = free_vars(term)
    bool_vars = [v for v in vars_set if v.ty == bool_ty]

    if not bool_vars:
        # No variables - evaluate directly
        return _evaluate(term, {})

    # Try all truth assignments
    for assignment in product([True, False], repeat=len(bool_vars)):
        env = dict(zip(bool_vars, assignment))
        if not _evaluate(term, env):
            return False

    return True


def _evaluate(term: Term, env: dict[Var, bool]) -> bool:
    """
    Evaluate a propositional term under a truth assignment.

    Args:
        term: Boolean term to evaluate
        env: Mapping from variables to truth values

    Returns:
        Boolean result of evaluation
    """
    if isinstance(term, Var):
        if term in env:
            return env[term]
        # Unknown variable - treat as False (shouldn't happen)
        return False

    elif isinstance(term, Const):
        if term.name == "T":
            return True
        elif term.name == "F":
            return False
        else:
            # Unknown constant - treat as False
            return False

    elif isinstance(term, App):
        # Check for binary operators: /\, \/, ==>, =
        if isinstance(term.func, App) and isinstance(term.func.func, Const):
            op = term.func.func.name
            left = _evaluate(term.func.arg, env)
            right = _evaluate(term.arg, env)

            if op == "/\\":  # Conjunction
                return left and right
            elif op == "\\/":  # Disjunction
                return left or right
            elif op == "==>":  # Implication
                return (not left) or right
            elif op == "=":  # Equality (iff)
                return left == right

        # Check for unary operator: ~
        if isinstance(term.func, Const) and term.func.name == "~":
            return not _evaluate(term.arg, env)

        # Unknown application - treat as False
        return False

    elif isinstance(term, Abs):
        # Lambda not supported in propositional logic
        return False

    return False


def tautology(goal: Goal) -> TacticResult:
    """
    Tactic that solves propositional tautologies.

    Args:
        goal: A goal whose target is a propositional tautology

    Returns:
        TacticResult with no subgoals if target is a tautology

    Raises:
        TacticError: If target is not a tautology
    """
    target = goal.target

    if not is_tautology(target):
        raise TacticError(f"tautology: {target} is not a tautology")

    def justify(thms: list[Theorem]) -> Theorem:
        # Would construct the proof using axioms
        # For now, return a placeholder
        from hol.kernel.inference import ASSUME
        return ASSUME(target)

    return TacticResult([], justify)
