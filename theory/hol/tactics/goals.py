"""
Goal-Directed Proof Management

Goals represent what we're trying to prove. A goal has:
- assumptions: context of assumed propositions
- target: the proposition to prove

ProofState tracks the current goals and how to combine
their proofs into a final theorem.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence
from loguru import logger

from hol.kernel.types import bool_ty
from hol.kernel.terms import Term, type_of
from hol.kernel.theorems import Theorem


# =============================================================================
# Goal Type
# =============================================================================

@dataclass(frozen=True)
class Goal:
    """
    A proof goal.

    Represents the sequent: assumptions ⊢ target
    """
    assumptions: tuple[Term, ...]
    target: Term

    def __repr__(self) -> str:
        if self.assumptions:
            asms = ", ".join(str(a) for a in self.assumptions)
            return f"{asms} ⊢ {self.target}"
        return f"⊢ {self.target}"


def mk_goal(assumptions: Sequence[Term], target: Term) -> Goal:
    """
    Create a goal.

    Args:
        assumptions: List of assumed propositions
        target: Proposition to prove

    Returns:
        A new Goal

    Raises:
        TypeError: If target is not boolean
    """
    if type_of(target) != bool_ty:
        raise TypeError(f"Goal target must be boolean, got {type_of(target)}")

    for asm in assumptions:
        if type_of(asm) != bool_ty:
            raise TypeError(f"Assumption must be boolean, got {type_of(asm)}")

    return Goal(tuple(assumptions), target)


# =============================================================================
# Justification Type
# =============================================================================

# A justification builds a theorem for the original goal
# from theorems for the subgoals
Justification = Callable[[list[Theorem]], Theorem]


def trivial_justification(thms: list[Theorem]) -> Theorem:
    """Justification that requires exactly one theorem and returns it."""
    if len(thms) != 1:
        raise ValueError(f"Expected 1 theorem, got {len(thms)}")
    return thms[0]


# =============================================================================
# Proof State
# =============================================================================

@dataclass
class ProofState:
    """
    State of an ongoing proof.

    Tracks:
    - Current goals to prove
    - Justification to build final theorem
    - History for undo
    """
    goals: list[Goal]
    justification: Justification
    history: list["ProofState"] = field(default_factory=list)

    @classmethod
    def initial(cls, target: Term, assumptions: Sequence[Term] = ()) -> "ProofState":
        """
        Create initial proof state for a goal.

        Args:
            target: Proposition to prove
            assumptions: Initial assumptions

        Returns:
            ProofState with single goal
        """
        goal = mk_goal(assumptions, target)
        return cls(
            goals=[goal],
            justification=trivial_justification,
            history=[],
        )

    def is_complete(self) -> bool:
        """Check if proof is complete (no remaining goals)."""
        return len(self.goals) == 0

    def current_goal(self) -> Goal:
        """
        Get the current goal.

        Returns:
            The first goal in the list

        Raises:
            ValueError: If no goals remain
        """
        if not self.goals:
            raise ValueError("No goals remaining")
        return self.goals[0]

    def __repr__(self) -> str:
        if self.is_complete():
            return "Proof complete!"
        lines = [f"{len(self.goals)} subgoal(s):"]
        for i, g in enumerate(self.goals):
            prefix = ">" if i == 0 else " "
            lines.append(f"{prefix} {i+1}. {g}")
        return "\n".join(lines)
