"""Tests for core tactics."""
import pytest
from hol.kernel.types import bool_ty, mk_fun_ty
from hol.kernel.terms import mk_var, mk_const, mk_app, mk_abs, mk_eq
from hol.tactics.goals import Goal, ProofState, mk_goal
from hol.tactics.tactics import (
    TacticError, TacticResult,
    assumption, intro, exact, apply_thm, rewrite,
    split, left, right,
)
from hol.kernel.inference import REFL, ASSUME


class TestAssumption:
    """Test assumption tactic."""

    def test_assumption_solves_goal(self):
        """assumption solves goal when target is in assumptions."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([p], p)
        result = assumption(goal)
        assert len(result.subgoals) == 0

    def test_assumption_fails_when_not_present(self):
        """assumption fails when target not in assumptions."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        goal = mk_goal([p], q)
        with pytest.raises(TacticError):
            assumption(goal)


class TestIntro:
    """Test intro tactic."""

    def test_intro_implication(self):
        """intro on A → B gives {A} ⊢ B."""
        a = mk_var("A", bool_ty)
        b = mk_var("B", bool_ty)
        # Implication: A → B represented as ∀p. (A → p) → (B → p) → p
        # For simplicity, test with a mock implication constant
        imp_ty = mk_fun_ty(bool_ty, mk_fun_ty(bool_ty, bool_ty))
        imp = mk_const("==>", imp_ty)
        target = mk_app(mk_app(imp, a), b)  # A ==> B
        goal = mk_goal([], target)
        result = intro("h")(goal)
        assert len(result.subgoals) == 1
        new_goal = result.subgoals[0]
        # The assumption should contain A (named h)
        assert new_goal.target == b


class TestExact:
    """Test exact tactic."""

    def test_exact_solves_matching_goal(self):
        """exact solves goal when theorem matches."""
        p = mk_var("p", bool_ty)
        thm = ASSUME(p)  # {p} ⊢ p
        goal = mk_goal([p], p)
        result = exact(thm)(goal)
        assert len(result.subgoals) == 0


class TestSplit:
    """Test split tactic for conjunction."""

    def test_split_conjunction(self):
        """split on A ∧ B gives two subgoals."""
        a = mk_var("A", bool_ty)
        b = mk_var("B", bool_ty)
        # Conjunction constant
        conj_ty = mk_fun_ty(bool_ty, mk_fun_ty(bool_ty, bool_ty))
        conj = mk_const("/\\", conj_ty)
        target = mk_app(mk_app(conj, a), b)  # A ∧ B
        goal = mk_goal([], target)
        result = split(goal)
        assert len(result.subgoals) == 2
        assert result.subgoals[0].target == a
        assert result.subgoals[1].target == b
