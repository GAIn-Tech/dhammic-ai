"""Tests for propositional solver."""
import pytest
from hol.kernel.types import bool_ty, mk_fun_ty
from hol.kernel.terms import mk_var, mk_const, mk_app
from hol.tactics.goals import mk_goal
from hol.tactics.tactics import TacticError
from hol.automation.propositional import tautology, is_tautology


class TestIsTautology:
    """Test tautology detection."""

    def test_var_not_tautology(self):
        """A single variable is not a tautology."""
        p = mk_var("p", bool_ty)
        assert not is_tautology(p)

    def test_true_is_tautology(self):
        """T (true) is a tautology."""
        true_const = mk_const("T", bool_ty)
        assert is_tautology(true_const)

    def test_p_or_not_p_is_tautology(self):
        """p \\/ ~p (excluded middle) is a tautology."""
        p = mk_var("p", bool_ty)
        # Build: p \/ ~p
        disj_ty = mk_fun_ty(bool_ty, mk_fun_ty(bool_ty, bool_ty))
        disj = mk_const("\\/", disj_ty)
        neg_ty = mk_fun_ty(bool_ty, bool_ty)
        neg = mk_const("~", neg_ty)
        not_p = mk_app(neg, p)
        p_or_not_p = mk_app(mk_app(disj, p), not_p)
        assert is_tautology(p_or_not_p)

    def test_p_implies_p_is_tautology(self):
        """p ==> p is a tautology."""
        p = mk_var("p", bool_ty)
        imp_ty = mk_fun_ty(bool_ty, mk_fun_ty(bool_ty, bool_ty))
        imp = mk_const("==>", imp_ty)
        p_implies_p = mk_app(mk_app(imp, p), p)
        assert is_tautology(p_implies_p)


class TestTautologyTactic:
    """Test tautology tactic."""

    def test_tautology_solves_excluded_middle(self):
        """tautology solves p \\/ ~p."""
        p = mk_var("p", bool_ty)
        disj_ty = mk_fun_ty(bool_ty, mk_fun_ty(bool_ty, bool_ty))
        disj = mk_const("\\/", disj_ty)
        neg_ty = mk_fun_ty(bool_ty, bool_ty)
        neg = mk_const("~", neg_ty)
        not_p = mk_app(neg, p)
        target = mk_app(mk_app(disj, p), not_p)
        goal = mk_goal([], target)
        result = tautology(goal)
        assert len(result.subgoals) == 0

    def test_tautology_fails_on_non_tautology(self):
        """tautology fails on non-tautologies."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([], p)
        with pytest.raises(TacticError):
            tautology(goal)
