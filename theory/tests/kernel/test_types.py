"""Tests for the HOL type system."""
import pytest
from hol.kernel.types import (
    Type, TyVar, TyBase, TyApp,
    mk_fun_ty, dest_fun_ty, is_fun_ty,
    bool_ty, ind_ty,
    type_subst, type_vars, types_match,
)


class TestTypeConstruction:
    """Test type construction and basic operations."""

    def test_tyvar_creation(self):
        """Type variables are created with names."""
        a = TyVar("a")
        assert a.name == "a"
        assert isinstance(a, Type)

    def test_tybase_creation(self):
        """Base types are created with names."""
        nat = TyBase("nat")
        assert nat.name == "nat"
        assert isinstance(nat, Type)

    def test_tyapp_creation(self):
        """Type applications combine operator and arguments."""
        list_ty = TyBase("list")
        a = TyVar("a")
        list_a = TyApp(list_ty, [a])
        assert list_a.op == list_ty
        assert list_a.args == (a,)

    def test_bool_ty_is_base(self):
        """bool is a base type."""
        assert isinstance(bool_ty, TyBase)
        assert bool_ty.name == "bool"

    def test_ind_ty_is_base(self):
        """ind (individuals) is a base type."""
        assert isinstance(ind_ty, TyBase)
        assert ind_ty.name == "ind"


class TestFunctionTypes:
    """Test function type operations."""

    def test_mk_fun_ty(self):
        """Function types are constructed from domain and range."""
        a = TyVar("a")
        b = TyVar("b")
        fn = mk_fun_ty(a, b)
        assert is_fun_ty(fn)

    def test_dest_fun_ty(self):
        """Function types can be destructured."""
        a = TyVar("a")
        b = TyVar("b")
        fn = mk_fun_ty(a, b)
        dom, rng = dest_fun_ty(fn)
        assert dom == a
        assert rng == b

    def test_dest_fun_ty_fails_on_non_function(self):
        """dest_fun_ty raises on non-function types."""
        with pytest.raises(ValueError, match="not a function type"):
            dest_fun_ty(bool_ty)

    def test_nested_function_type(self):
        """Function types can be nested: a -> b -> c."""
        a, b, c = TyVar("a"), TyVar("b"), TyVar("c")
        # a -> (b -> c)
        fn = mk_fun_ty(a, mk_fun_ty(b, c))
        dom, rng = dest_fun_ty(fn)
        assert dom == a
        assert is_fun_ty(rng)


class TestTypeVariables:
    """Test type variable extraction."""

    def test_type_vars_of_tyvar(self):
        """Type variables contain themselves."""
        a = TyVar("a")
        assert type_vars(a) == {a}

    def test_type_vars_of_base(self):
        """Base types have no type variables."""
        assert type_vars(bool_ty) == set()

    def test_type_vars_of_function(self):
        """Function types collect vars from domain and range."""
        a, b = TyVar("a"), TyVar("b")
        fn = mk_fun_ty(a, b)
        assert type_vars(fn) == {a, b}

    def test_type_vars_in_operator(self):
        """Type variables in operators are collected."""
        f = TyVar("f")
        a = TyVar("a")
        app = TyApp(f, [a])
        assert type_vars(app) == {f, a}


class TestTypeSubstitution:
    """Test type substitution."""

    def test_subst_tyvar(self):
        """Substitution replaces type variables."""
        a = TyVar("a")
        result = type_subst({a: bool_ty}, a)
        assert result == bool_ty

    def test_subst_base_unchanged(self):
        """Base types are unchanged by substitution."""
        a = TyVar("a")
        result = type_subst({a: bool_ty}, ind_ty)
        assert result == ind_ty

    def test_subst_in_function(self):
        """Substitution applies through function types."""
        a, b = TyVar("a"), TyVar("b")
        fn = mk_fun_ty(a, b)
        result = type_subst({a: bool_ty}, fn)
        dom, rng = dest_fun_ty(result)
        assert dom == bool_ty
        assert rng == b

    def test_subst_in_operator(self):
        """Substitution applies to operators."""
        f = TyVar("f")
        a = TyVar("a")
        app = TyApp(f, [a])
        list_ty = TyBase("list")
        result = type_subst({f: list_ty}, app)
        assert result.op == list_ty
        assert result.args == (a,)


class TestTypeMatching:
    """Test type matching/unification."""

    def test_match_identical(self):
        """Identical types match with empty substitution."""
        match = types_match(bool_ty, bool_ty)
        assert match == {}

    def test_match_tyvar_to_concrete(self):
        """Type variable matches concrete type."""
        a = TyVar("a")
        match = types_match(a, bool_ty)
        assert match == {a: bool_ty}

    def test_match_function_types(self):
        """Function types match structurally."""
        a = TyVar("a")
        fn1 = mk_fun_ty(a, a)
        fn2 = mk_fun_ty(bool_ty, bool_ty)
        match = types_match(fn1, fn2)
        assert match == {a: bool_ty}

    def test_match_fails_on_conflict(self):
        """Matching fails if variable must equal two different types."""
        a = TyVar("a")
        fn1 = mk_fun_ty(a, a)
        fn2 = mk_fun_ty(bool_ty, ind_ty)
        match = types_match(fn1, fn2)
        assert match is None

    def test_match_fails_on_different_bases(self):
        """Different base types don't match."""
        match = types_match(bool_ty, ind_ty)
        assert match is None
