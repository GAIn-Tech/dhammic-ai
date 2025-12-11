"""Tests for auto tactic."""
import pytest
from hol.kernel.types import bool_ty, mk_fun_ty
from hol.kernel.terms import mk_var, mk_const, mk_app
from hol.tactics.goals import mk_goal
from hol.tactics.tactics import TacticError
from hol.automation.auto import auto


class TestAuto:
    """Test auto tactic."""

    def test_auto_solves_assumption(self):
        """auto solves goal when target is in assumptions."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([p], p)
        result = auto()(goal)
        assert len(result.subgoals) == 0

    def test_auto_solves_intro_then_assumption(self):
        """auto solves A -> A by intro then assumption."""
        a = mk_var("A", bool_ty)
        imp_ty = mk_fun_ty(bool_ty, mk_fun_ty(bool_ty, bool_ty))
        imp = mk_const("==>", imp_ty)
        target = mk_app(mk_app(imp, a), a)  # A ==> A
        goal = mk_goal([], target)
        result = auto()(goal)
        assert len(result.subgoals) == 0

    def test_auto_fails_on_unprovable(self):
        """auto fails on goals it cannot solve."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        goal = mk_goal([], q)  # No assumptions, cannot prove q
        with pytest.raises(TacticError):
            auto(depth=3)(goal)

    def test_auto_respects_depth_limit(self):
        """auto respects depth limit."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([p], p)
        # Even depth=0 should solve this (assumption is depth 0)
        result = auto(depth=0)(goal)
        assert len(result.subgoals) == 0

    def test_auto_with_hints(self):
        """auto can use hint theorems."""
        from hol.kernel.inference import REFL
        from hol.kernel.terms import mk_eq

        x = mk_var("x", bool_ty)
        # Goal: x = x (can be solved with REFL hint)
        goal = mk_goal([], mk_eq(x, x))
        refl_thm = REFL(x)
        result = auto(hints=[refl_thm])(goal)
        assert len(result.subgoals) == 0
