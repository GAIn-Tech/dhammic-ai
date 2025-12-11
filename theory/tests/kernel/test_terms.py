"""Tests for HOL terms."""
import pytest
from hol.kernel.types import TyVar, TyBase, bool_ty, mk_fun_ty
from hol.kernel.terms import (
    Term, Var, Const, App, Abs,
    mk_var, mk_const, mk_app, mk_abs,
    dest_var, dest_const, dest_app, dest_abs,
    is_var, is_const, is_app, is_abs,
    type_of, free_vars, bound_vars,
    subst, alpha_equiv,
    mk_eq, dest_eq, is_eq,
)


class TestTermConstruction:
    """Test term constructors."""

    def test_var_creation(self):
        """Variables have name and type."""
        x = mk_var("x", bool_ty)
        assert is_var(x)
        name, ty = dest_var(x)
        assert name == "x"
        assert ty == bool_ty

    def test_const_creation(self):
        """Constants have name and type."""
        true = mk_const("T", bool_ty)
        assert is_const(true)
        name, ty = dest_const(true)
        assert name == "T"
        assert ty == bool_ty

    def test_app_creation(self):
        """Applications combine function and argument."""
        a = TyVar("a")
        f_ty = mk_fun_ty(a, bool_ty)
        f = mk_var("f", f_ty)
        x = mk_var("x", a)
        app = mk_app(f, x)
        assert is_app(app)
        fn, arg = dest_app(app)
        assert fn == f
        assert arg == x

    def test_abs_creation(self):
        """Abstractions bind a variable in a body."""
        x = mk_var("x", bool_ty)
        body = x
        lam = mk_abs(x, body)
        assert is_abs(lam)
        var, bd = dest_abs(lam)
        assert var == x
        assert bd == body


class TestTypeOf:
    """Test type inference."""

    def test_type_of_var(self):
        """Variables have their declared type."""
        x = mk_var("x", bool_ty)
        assert type_of(x) == bool_ty

    def test_type_of_const(self):
        """Constants have their declared type."""
        c = mk_const("c", bool_ty)
        assert type_of(c) == bool_ty

    def test_type_of_app(self):
        """Application has the range type of the function."""
        a = TyVar("a")
        f = mk_var("f", mk_fun_ty(a, bool_ty))
        x = mk_var("x", a)
        app = mk_app(f, x)
        assert type_of(app) == bool_ty

    def test_type_of_abs(self):
        """Abstraction has function type."""
        x = mk_var("x", bool_ty)
        lam = mk_abs(x, x)  # λx:bool. x
        expected = mk_fun_ty(bool_ty, bool_ty)
        assert type_of(lam) == expected

    def test_app_type_mismatch_raises(self):
        """Application fails if argument type doesn't match."""
        f = mk_var("f", mk_fun_ty(bool_ty, bool_ty))
        y = mk_var("y", TyBase("nat"))
        with pytest.raises(TypeError, match="type mismatch"):
            mk_app(f, y)


class TestFreeVars:
    """Test free variable extraction."""

    def test_free_vars_of_var(self):
        """A variable is free in itself."""
        x = mk_var("x", bool_ty)
        assert free_vars(x) == {x}

    def test_free_vars_of_const(self):
        """Constants have no free variables."""
        c = mk_const("c", bool_ty)
        assert free_vars(c) == set()

    def test_free_vars_of_app(self):
        """Application collects free vars from both parts."""
        f = mk_var("f", mk_fun_ty(bool_ty, bool_ty))
        x = mk_var("x", bool_ty)
        app = mk_app(f, x)
        assert free_vars(app) == {f, x}

    def test_free_vars_of_abs(self):
        """Abstraction removes bound variable from free vars."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        # λx. y (x is bound, y is free)
        lam = mk_abs(x, y)
        assert free_vars(lam) == {y}

    def test_bound_var_not_free(self):
        """Bound variable is not free."""
        x = mk_var("x", bool_ty)
        lam = mk_abs(x, x)  # λx. x
        assert free_vars(lam) == set()


class TestSubstitution:
    """Test term substitution."""

    def test_subst_var(self):
        """Substitution replaces matching variable."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        result = subst({x: y}, x)
        assert result == y

    def test_subst_var_no_match(self):
        """Non-matching variables unchanged."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        z = mk_var("z", bool_ty)
        result = subst({x: y}, z)
        assert result == z

    def test_subst_const_unchanged(self):
        """Constants are unchanged by substitution."""
        x = mk_var("x", bool_ty)
        c = mk_const("c", bool_ty)
        result = subst({x: c}, c)
        assert result == c

    def test_subst_in_app(self):
        """Substitution applies through applications."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        f = mk_var("f", mk_fun_ty(bool_ty, bool_ty))
        app = mk_app(f, x)
        result = subst({x: y}, app)
        assert result == mk_app(f, y)

    def test_subst_avoids_capture(self):
        """Substitution avoids variable capture."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        # λy. x  -- substitute x -> y
        # Should NOT become λy. y (capture!)
        # Should become λy'. y for fresh y'
        lam = mk_abs(y, x)
        result = subst({x: y}, lam)
        # The bound variable should be renamed
        var, body = dest_abs(result)
        assert body == y  # The substituted y
        assert var != y   # Bound var renamed to avoid capture


class TestAlphaEquivalence:
    """Test alpha equivalence."""

    def test_same_term_alpha_equiv(self):
        """Identical terms are alpha equivalent."""
        x = mk_var("x", bool_ty)
        assert alpha_equiv(x, x)

    def test_renamed_bound_var_alpha_equiv(self):
        """λx.x and λy.y are alpha equivalent."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        lam_x = mk_abs(x, x)
        lam_y = mk_abs(y, y)
        assert alpha_equiv(lam_x, lam_y)

    def test_different_body_not_alpha_equiv(self):
        """λx.x and λx.y are not alpha equivalent."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        lam1 = mk_abs(x, x)
        lam2 = mk_abs(x, y)
        assert not alpha_equiv(lam1, lam2)


class TestEquality:
    """Test equality term construction."""

    def test_mk_eq(self):
        """Equality terms are constructed."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        eq = mk_eq(x, y)
        assert is_eq(eq)
        assert type_of(eq) == bool_ty

    def test_dest_eq(self):
        """Equality terms can be destructured."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        eq = mk_eq(x, y)
        lhs, rhs = dest_eq(eq)
        assert lhs == x
        assert rhs == y

    def test_eq_type_mismatch_raises(self):
        """Equality requires same types on both sides."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", TyBase("nat"))
        with pytest.raises(TypeError, match="type mismatch"):
            mk_eq(x, y)
