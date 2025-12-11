"""
Blast - First-Order Hammer

Aggressive proof search combining multiple strategies.
"""
from __future__ import annotations
import time
from loguru import logger

from hol.kernel.types import bool_ty
from hol.kernel.terms import Term, is_eq, type_of
from hol.kernel.theorems import Theorem, concl
from hol.tactics.goals import Goal
from hol.tactics.tactics import Tactic, TacticResult, TacticError
from hol.automation.auto import auto
from hol.automation.simp import simp
from hol.automation.propositional import tautology, is_tautology
from hol.automation.arith import arith, is_arith_valid


def blast(
    lemmas: list[Theorem] | None = None,
    timeout: float = 5.0,
) -> Tactic:
    """
    Aggressive proof search tactic.

    Combines multiple proof strategies:
    1. Direct assumption/reflexivity
    2. Propositional tautology checking
    3. Arithmetic decision procedure
    4. Simplification with lemmas
    5. Auto tactic with lemmas

    Args:
        lemmas: Theorems to use as hints
        timeout: Maximum time in seconds (default 5.0)

    Returns:
        A tactic that aggressively tries to solve the goal

    Raises:
        TacticError: If goal cannot be solved within timeout
    """
    if lemmas is None:
        lemmas = []

    def blast_tactic(goal: Goal) -> TacticResult:
        start_time = time.time()
        target = goal.target

        # Strategy 1: Check if target is in assumptions
        for asm in goal.assumptions:
            from hol.kernel.terms import alpha_equiv
            if alpha_equiv(asm, target):
                from hol.tactics.tactics import assumption
                return assumption(goal)

        # Strategy 2: Check for reflexive equality
        if is_eq(target):
            from hol.kernel.terms import dest_eq, alpha_equiv
            lhs, rhs = dest_eq(target)
            if alpha_equiv(lhs, rhs):
                from hol.kernel.inference import REFL
                def justify(thms: list[Theorem]) -> Theorem:
                    return REFL(lhs)
                return TacticResult([], justify)

        # Check timeout
        if time.time() - start_time > timeout:
            raise TacticError(f"blast: timeout after {timeout}s")

        # Strategy 3: Try propositional tautology
        if type_of(target) == bool_ty:
            try:
                if is_tautology(target):
                    return tautology(goal)
            except TacticError:
                pass

        # Check timeout
        if time.time() - start_time > timeout:
            raise TacticError(f"blast: timeout after {timeout}s")

        # Strategy 4: Try arithmetic
        if is_eq(target):
            try:
                if is_arith_valid(target):
                    return arith(goal)
            except TacticError:
                pass

        # Check timeout
        if time.time() - start_time > timeout:
            raise TacticError(f"blast: timeout after {timeout}s")

        # Strategy 5: Try simp with lemmas
        if lemmas:
            try:
                eq_lemmas = [l for l in lemmas if is_eq(concl(l))]
                if eq_lemmas:
                    return simp(rules=eq_lemmas)(goal)
            except TacticError:
                pass

        # Check timeout
        if time.time() - start_time > timeout:
            raise TacticError(f"blast: timeout after {timeout}s")

        # Strategy 6: Try auto with lemmas
        try:
            return auto(depth=10, hints=lemmas)(goal)
        except TacticError:
            pass

        raise TacticError(f"blast: cannot solve {target}")

    return blast_tactic
