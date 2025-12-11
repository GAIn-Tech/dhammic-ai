"""Tests for simplifier tactic."""
import pytest
from hol.kernel.types import bool_ty, mk_fun_ty
from hol.kernel.terms import mk_var, mk_const, mk_app, mk_eq
from hol.kernel.inference import REFL, ASSUME
from hol.tactics.goals import mk_goal
from hol.tactics.tactics import TacticError
from hol.automation.simp import simp, SimpRule, mk_simp_rule


class TestSimpRule:
    """Test simplification rule construction."""

    def test_mk_simp_rule_from_equality(self):
        """Create simp rule from equality theorem."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        eq_thm = ASSUME(mk_eq(x, y))  # {x=y} |- x = y
        rule = mk_simp_rule(eq_thm)
        assert rule.lhs == x
        assert rule.rhs == y


class TestSimp:
    """Test simp tactic."""

    def test_simp_with_refl(self):
        """simp can solve x = x goals."""
        x = mk_var("x", bool_ty)
        goal = mk_goal([], mk_eq(x, x))
        result = simp()(goal)
        assert len(result.subgoals) == 0

    def test_simp_with_rule(self):
        """simp applies rewrite rules."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        # Rule: x -> y
        eq_thm = ASSUME(mk_eq(x, y))
        # Goal: x = y (after rewriting x to y, becomes y = y)
        goal = mk_goal([mk_eq(x, y)], mk_eq(x, y))
        result = simp(rules=[eq_thm])(goal)
        # Should succeed (x = y in assumptions, goal is x = y)
        assert len(result.subgoals) == 0

    def test_simp_no_change_fails(self):
        """simp fails if no rewriting possible and goal not solved."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        goal = mk_goal([], mk_eq(p, q))  # Cannot simplify p = q
        with pytest.raises(TacticError):
            simp()(goal)

    def test_simp_in_subterm(self):
        """simp rewrites in subterms."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        f = mk_var("f", mk_fun_ty(bool_ty, bool_ty))
        # Rule: x -> y
        eq_thm = ASSUME(mk_eq(x, y))
        # Goal: f x = f y (rewrite x to y in f x)
        lhs = mk_app(f, x)
        rhs = mk_app(f, y)
        goal = mk_goal([mk_eq(x, y)], mk_eq(lhs, rhs))
        result = simp(rules=[eq_thm])(goal)
        assert len(result.subgoals) == 0
