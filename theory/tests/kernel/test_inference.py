"""Tests for HOL primitive inference rules."""
import pytest
from hol.kernel.types import TyVar, bool_ty, mk_fun_ty
from hol.kernel.terms import (
    mk_var, mk_const, mk_app, mk_abs, mk_eq,
    dest_eq, type_of,
)
from hol.kernel.theorems import Theorem, hyps, concl
from hol.kernel.inference import (
    REFL, TRANS, MK_COMB, ABS, BETA,
    ASSUME, EQ_MP, DEDUCT_ANTISYM,
    INST, INST_TYPE,
)


class TestReflexivity:
    """Test REFL rule: ⊢ t = t"""

    def test_refl_bool_var(self):
        """REFL on boolean variable."""
        x = mk_var("x", bool_ty)
        thm = REFL(x)
        assert hyps(thm) == frozenset()
        lhs, rhs = dest_eq(concl(thm))
        assert lhs == x
        assert rhs == x

    def test_refl_complex_term(self):
        """REFL works on complex terms."""
        f = mk_var("f", mk_fun_ty(bool_ty, bool_ty))
        x = mk_var("x", bool_ty)
        t = mk_app(f, x)
        thm = REFL(t)
        lhs, rhs = dest_eq(concl(thm))
        assert lhs == t
        assert rhs == t


class TestTransitivity:
    """Test TRANS rule: ⊢ s=t, ⊢ t=u --> ⊢ s=u"""

    def test_trans_simple(self):
        """Transitivity on variables."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        z = mk_var("z", bool_ty)
        # We need theorems s=t and t=u
        # For testing, use REFL and ASSUME creatively
        # Actually need proper way to get these - use ASSUME for now
        xy_eq = mk_eq(x, y)
        yz_eq = mk_eq(y, z)
        thm1 = ASSUME(xy_eq)  # {x=y} ⊢ x=y
        thm2 = ASSUME(yz_eq)  # {y=z} ⊢ y=z
        thm = TRANS(thm1, thm2)
        lhs, rhs = dest_eq(concl(thm))
        assert lhs == x
        assert rhs == z
        assert xy_eq in hyps(thm)
        assert yz_eq in hyps(thm)

    def test_trans_mismatch_fails(self):
        """TRANS fails if middle terms don't match."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        z = mk_var("z", bool_ty)
        w = mk_var("w", bool_ty)
        thm1 = ASSUME(mk_eq(x, y))
        thm2 = ASSUME(mk_eq(z, w))  # z != y
        with pytest.raises(ValueError, match="middle terms"):
            TRANS(thm1, thm2)


class TestMkComb:
    """Test MK_COMB rule: ⊢ f=g, ⊢ x=y --> ⊢ f x = g y"""

    def test_mk_comb_simple(self):
        """MK_COMB on function and argument equalities."""
        f = mk_var("f", mk_fun_ty(bool_ty, bool_ty))
        g = mk_var("g", mk_fun_ty(bool_ty, bool_ty))
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        thm1 = ASSUME(mk_eq(f, g))
        thm2 = ASSUME(mk_eq(x, y))
        thm = MK_COMB(thm1, thm2)
        lhs, rhs = dest_eq(concl(thm))
        assert lhs == mk_app(f, x)
        assert rhs == mk_app(g, y)


class TestAbs:
    """Test ABS rule: ⊢ s=t --> ⊢ (λx.s) = (λx.t) [x not free in hyps]"""

    def test_abs_simple(self):
        """ABS on equality without x in hypotheses."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        # ⊢ y = y (from REFL, no hypotheses)
        thm = REFL(y)
        result = ABS(x, thm)
        lhs, rhs = dest_eq(concl(result))
        # Should be λx.y = λx.y
        assert lhs == mk_abs(x, y)
        assert rhs == mk_abs(x, y)

    def test_abs_free_var_in_hyps_fails(self):
        """ABS fails if x is free in hypotheses."""
        x = mk_var("x", bool_ty)
        # {x=x} ⊢ x=x
        thm = ASSUME(mk_eq(x, x))
        with pytest.raises(ValueError, match="free in hypotheses"):
            ABS(x, thm)


class TestBeta:
    """Test BETA rule: ⊢ (λx.b) t = b[t/x]"""

    def test_beta_simple(self):
        """Beta reduction on identity."""
        x = mk_var("x", bool_ty)
        t = mk_var("t", bool_ty)
        # (λx.x) t = t
        lam = mk_abs(x, x)
        app = mk_app(lam, t)
        thm = BETA(app)
        lhs, rhs = dest_eq(concl(thm))
        assert lhs == app
        assert rhs == t
        assert hyps(thm) == frozenset()


class TestAssume:
    """Test ASSUME rule: {p} ⊢ p"""

    def test_assume_simple(self):
        """ASSUME creates theorem with p as hypothesis and conclusion."""
        p = mk_var("p", bool_ty)
        thm = ASSUME(p)
        assert concl(thm) == p
        assert p in hyps(thm)


class TestEqMp:
    """Test EQ_MP rule: ⊢ p=q, ⊢ p --> ⊢ q"""

    def test_eq_mp_simple(self):
        """Modus ponens on equality."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        thm1 = ASSUME(mk_eq(p, q))  # {p=q} ⊢ p=q
        thm2 = ASSUME(p)            # {p} ⊢ p
        thm = EQ_MP(thm1, thm2)
        assert concl(thm) == q
        assert mk_eq(p, q) in hyps(thm)
        assert p in hyps(thm)


class TestDeductAntisym:
    """Test DEDUCT_ANTISYM: Γ⊢p, Δ⊢q --> (Γ-{q})∪(Δ-{p}) ⊢ p=q"""

    def test_deduct_antisym(self):
        """Deduction antisymmetry."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        # {q} ⊢ p means we assume q to prove p
        # {p} ⊢ q means we assume p to prove q
        # If both hold, then p = q
        thm1 = ASSUME(p)  # {p} ⊢ p
        thm2 = ASSUME(q)  # {q} ⊢ q
        # For a real test, we'd need actual derivations
        # Here we test structure
        thm = DEDUCT_ANTISYM(thm1, thm2)
        lhs, rhs = dest_eq(concl(thm))
        assert lhs == p
        assert rhs == q


class TestInst:
    """Test INST rule: term instantiation."""

    def test_inst_simple(self):
        """Instantiate variable in theorem."""
        x = mk_var("x", bool_ty)
        y = mk_var("y", bool_ty)
        thm = REFL(x)  # ⊢ x = x
        result = INST([(x, y)], thm)  # ⊢ y = y
        lhs, rhs = dest_eq(concl(result))
        assert lhs == y
        assert rhs == y


class TestInstType:
    """Test INST_TYPE rule: type instantiation."""

    def test_inst_type_simple(self):
        """Instantiate type variable in theorem."""
        a = TyVar("a")
        x = mk_var("x", a)
        thm = REFL(x)  # ⊢ x:'a = x:'a
        result = INST_TYPE([(a, bool_ty)], thm)
        # Conclusion should now have bool type
        eq_term = concl(result)
        assert type_of(eq_term) == bool_ty
