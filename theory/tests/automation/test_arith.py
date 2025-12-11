"""Tests for arithmetic decision procedure."""
import pytest
from hol.kernel.types import bool_ty, TyBase, mk_fun_ty
from hol.kernel.terms import mk_var, mk_const, mk_app, mk_eq
from hol.tactics.goals import mk_goal
from hol.tactics.tactics import TacticError
from hol.automation.arith import arith, is_arith_valid


# Define nat type for tests
nat_ty = TyBase("nat")


class TestArithValid:
    """Test arithmetic validity checking."""

    def test_zero_equals_zero(self):
        """0 = 0 is valid."""
        zero = mk_const("0", nat_ty)
        eq = mk_eq(zero, zero)
        assert is_arith_valid(eq)

    def test_x_equals_x(self):
        """x = x is valid for any x."""
        x = mk_var("x", nat_ty)
        eq = mk_eq(x, x)
        assert is_arith_valid(eq)

    def test_x_plus_zero_equals_x(self):
        """x + 0 = x is valid."""
        x = mk_var("x", nat_ty)
        zero = mk_const("0", nat_ty)
        plus_ty = mk_fun_ty(nat_ty, mk_fun_ty(nat_ty, nat_ty))
        plus = mk_const("+", plus_ty)
        x_plus_0 = mk_app(mk_app(plus, x), zero)
        eq = mk_eq(x_plus_0, x)
        assert is_arith_valid(eq)

    def test_x_not_equals_y(self):
        """x = y is not generally valid."""
        x = mk_var("x", nat_ty)
        y = mk_var("y", nat_ty)
        eq = mk_eq(x, y)
        assert not is_arith_valid(eq)


class TestArithTactic:
    """Test arith tactic."""

    def test_arith_solves_reflexive(self):
        """arith solves x = x."""
        x = mk_var("x", nat_ty)
        eq = mk_eq(x, x)
        goal = mk_goal([], eq)
        result = arith(goal)
        assert len(result.subgoals) == 0

    def test_arith_solves_zero_identity(self):
        """arith solves x + 0 = x."""
        x = mk_var("x", nat_ty)
        zero = mk_const("0", nat_ty)
        plus_ty = mk_fun_ty(nat_ty, mk_fun_ty(nat_ty, nat_ty))
        plus = mk_const("+", plus_ty)
        x_plus_0 = mk_app(mk_app(plus, x), zero)
        eq = mk_eq(x_plus_0, x)
        goal = mk_goal([], eq)
        result = arith(goal)
        assert len(result.subgoals) == 0

    def test_arith_fails_on_invalid(self):
        """arith fails on invalid equalities."""
        x = mk_var("x", nat_ty)
        y = mk_var("y", nat_ty)
        eq = mk_eq(x, y)
        goal = mk_goal([], eq)
        with pytest.raises(TacticError):
            arith(goal)
