"""Tests for blast tactic."""
import pytest
from hol.kernel.types import bool_ty, TyBase, mk_fun_ty
from hol.kernel.terms import mk_var, mk_const, mk_app, mk_eq
from hol.kernel.inference import REFL
from hol.tactics.goals import mk_goal
from hol.tactics.tactics import TacticError
from hol.automation.blast import blast


# Define nat type for tests
nat_ty = TyBase("nat")


class TestBlast:
    """Test blast tactic."""

    def test_blast_solves_assumption(self):
        """blast solves goal when target is in assumptions."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([p], p)
        result = blast()(goal)
        assert len(result.subgoals) == 0

    def test_blast_solves_reflexive_equality(self):
        """blast solves x = x."""
        x = mk_var("x", bool_ty)
        goal = mk_goal([], mk_eq(x, x))
        result = blast()(goal)
        assert len(result.subgoals) == 0

    def test_blast_solves_tautology(self):
        """blast solves propositional tautologies."""
        p = mk_var("p", bool_ty)
        imp_ty = mk_fun_ty(bool_ty, mk_fun_ty(bool_ty, bool_ty))
        imp = mk_const("==>", imp_ty)
        p_implies_p = mk_app(mk_app(imp, p), p)
        goal = mk_goal([], p_implies_p)
        result = blast()(goal)
        assert len(result.subgoals) == 0

    def test_blast_solves_arithmetic(self):
        """blast solves simple arithmetic."""
        x = mk_var("x", nat_ty)
        zero = mk_const("0", nat_ty)
        plus_ty = mk_fun_ty(nat_ty, mk_fun_ty(nat_ty, nat_ty))
        plus = mk_const("+", plus_ty)
        x_plus_0 = mk_app(mk_app(plus, x), zero)
        goal = mk_goal([], mk_eq(x_plus_0, x))
        result = blast()(goal)
        assert len(result.subgoals) == 0

    def test_blast_with_lemmas(self):
        """blast can use lemma hints."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        eq_thm = REFL(x)  # |- x = x
        goal = mk_goal([], mk_eq(x, x))
        result = blast(lemmas=[eq_thm])(goal)
        assert len(result.subgoals) == 0

    def test_blast_fails_on_unprovable(self):
        """blast fails on goals it cannot solve."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        goal = mk_goal([], mk_eq(p, q))
        with pytest.raises(TacticError):
            blast(timeout=1.0)(goal)
