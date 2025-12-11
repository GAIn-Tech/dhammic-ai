"""
HOL Primitive Inference Rules

The 10 primitive rules that form the trusted kernel:
1. REFL       - Reflexivity of equality
2. TRANS      - Transitivity of equality
3. MK_COMB    - Congruence for application
4. ABS        - Congruence for abstraction
5. BETA       - Beta reduction
6. ASSUME     - Assumption introduction
7. EQ_MP      - Equality modus ponens
8. DEDUCT_ANTISYM - Deduction antisymmetry
9. INST       - Term instantiation
10. INST_TYPE - Type instantiation
"""
from __future__ import annotations
from loguru import logger

from hol.kernel.types import Type, TyVar, type_subst, type_vars
from hol.kernel.terms import (
    Term, Var, Const, App, Abs,
    mk_var, mk_const, mk_app, mk_abs, mk_eq,
    dest_eq, is_eq, type_of, free_vars, subst, alpha_equiv,
)
from hol.kernel.theorems import (
    Theorem, hyps, concl,
    PrfRefl, PrfTrans, PrfMkComb, PrfAbs, PrfBeta,
    PrfAssume, PrfEqMp, PrfDeductAntisym, PrfInst, PrfInstType,
)


# =============================================================================
# Inference Rules
# =============================================================================

def REFL(t: Term) -> Theorem:
    """
    REFL: Reflexivity of equality.

        ⊢ t = t

    Args:
        t: Any term

    Returns:
        Theorem stating t = t
    """
    eq = mk_eq(t, t)
    return Theorem(frozenset(), eq, PrfRefl(t))


def TRANS(thm1: Theorem, thm2: Theorem) -> Theorem:
    """
    TRANS: Transitivity of equality.

        ⊢ s = t    ⊢ t = u
        -------------------
             ⊢ s = u

    Args:
        thm1: Theorem with conclusion s = t
        thm2: Theorem with conclusion t = u

    Returns:
        Theorem with conclusion s = u

    Raises:
        ValueError: If conclusions are not equalities or middle terms don't match
    """
    if not is_eq(concl(thm1)):
        raise ValueError(f"First theorem not an equality: {thm1}")
    if not is_eq(concl(thm2)):
        raise ValueError(f"Second theorem not an equality: {thm2}")

    s, t1 = dest_eq(concl(thm1))
    t2, u = dest_eq(concl(thm2))

    if not alpha_equiv(t1, t2):
        raise ValueError(f"Transitivity: middle terms don't match: {t1} vs {t2}")

    new_hyps = hyps(thm1) | hyps(thm2)
    new_concl = mk_eq(s, u)
    return Theorem(new_hyps, new_concl, PrfTrans(0, 0))  # TODO: proper indexing


def MK_COMB(thm1: Theorem, thm2: Theorem) -> Theorem:
    """
    MK_COMB: Congruence for application.

        ⊢ f = g    ⊢ x = y
        -------------------
           ⊢ f x = g y

    Args:
        thm1: Theorem with conclusion f = g (functions)
        thm2: Theorem with conclusion x = y (arguments)

    Returns:
        Theorem with conclusion f x = g y
    """
    if not is_eq(concl(thm1)):
        raise ValueError(f"First theorem not an equality: {thm1}")
    if not is_eq(concl(thm2)):
        raise ValueError(f"Second theorem not an equality: {thm2}")

    f, g = dest_eq(concl(thm1))
    x, y = dest_eq(concl(thm2))

    # Type check: f must be a function applicable to x
    fx = mk_app(f, x)
    gy = mk_app(g, y)

    new_hyps = hyps(thm1) | hyps(thm2)
    new_concl = mk_eq(fx, gy)
    return Theorem(new_hyps, new_concl, PrfMkComb(0, 0))


def ABS(x: Var, thm: Theorem) -> Theorem:
    """
    ABS: Congruence for abstraction.

           ⊢ s = t
        ---------------  (x not free in hypotheses)
        ⊢ (λx.s) = (λx.t)

    Args:
        x: Variable to abstract over
        thm: Theorem with conclusion s = t

    Returns:
        Theorem with conclusion (λx.s) = (λx.t)

    Raises:
        ValueError: If x is free in hypotheses
    """
    if not is_eq(concl(thm)):
        raise ValueError(f"Theorem not an equality: {thm}")

    # Check x is not free in hypotheses
    for h in hyps(thm):
        if x in free_vars(h):
            raise ValueError(f"Variable {x} is free in hypotheses")

    s, t = dest_eq(concl(thm))
    lam_s = mk_abs(x, s)
    lam_t = mk_abs(x, t)

    new_concl = mk_eq(lam_s, lam_t)
    return Theorem(hyps(thm), new_concl, PrfAbs(x, 0))


def BETA(t: Term) -> Theorem:
    """
    BETA: Beta reduction.

        ⊢ (λx.b) t = b[t/x]

    Args:
        t: An application of a lambda abstraction

    Returns:
        Theorem stating the beta reduction equality

    Raises:
        ValueError: If t is not a beta redex
    """
    if not isinstance(t, App):
        raise ValueError(f"Not an application: {t}")
    if not isinstance(t.func, Abs):
        raise ValueError(f"Not a beta redex: {t}")

    lam = t.func
    arg = t.arg

    # Compute b[t/x]
    reduced = subst({lam.var: arg}, lam.body)

    eq = mk_eq(t, reduced)
    return Theorem(frozenset(), eq, PrfBeta(t))


def ASSUME(p: Term) -> Theorem:
    """
    ASSUME: Assumption introduction.

        {p} ⊢ p

    Args:
        p: A proposition (term of type bool)

    Returns:
        Theorem with p as both hypothesis and conclusion

    Raises:
        TypeError: If p is not of type bool
    """
    from hol.kernel.types import bool_ty

    if type_of(p) != bool_ty:
        raise TypeError(f"ASSUME requires boolean term, got {type_of(p)}")

    return Theorem(frozenset({p}), p, PrfAssume(p))


def EQ_MP(thm1: Theorem, thm2: Theorem) -> Theorem:
    """
    EQ_MP: Equality modus ponens.

        ⊢ p = q    ⊢ p
        ---------------
             ⊢ q

    Args:
        thm1: Theorem with conclusion p = q
        thm2: Theorem with conclusion p

    Returns:
        Theorem with conclusion q
    """
    if not is_eq(concl(thm1)):
        raise ValueError(f"First theorem not an equality: {thm1}")

    p, q = dest_eq(concl(thm1))

    if not alpha_equiv(p, concl(thm2)):
        raise ValueError(f"EQ_MP: {p} does not match {concl(thm2)}")

    new_hyps = hyps(thm1) | hyps(thm2)
    return Theorem(new_hyps, q, PrfEqMp(0, 0))


def DEDUCT_ANTISYM(thm1: Theorem, thm2: Theorem) -> Theorem:
    """
    DEDUCT_ANTISYM: Deduction antisymmetry rule.

        Γ ⊢ p    Δ ⊢ q
        ------------------------
        (Γ - {q}) ∪ (Δ - {p}) ⊢ p = q

    This is a powerful rule that allows proving equalities between
    propositions by showing they imply each other.

    Args:
        thm1: Theorem with conclusion p
        thm2: Theorem with conclusion q

    Returns:
        Theorem with conclusion p = q
    """
    p = concl(thm1)
    q = concl(thm2)

    # Remove q from Γ and p from Δ
    new_hyps = (hyps(thm1) - {q}) | (hyps(thm2) - {p})
    new_concl = mk_eq(p, q)

    return Theorem(new_hyps, new_concl, PrfDeductAntisym(0, 0))


def INST(subst_list: list[tuple[Var, Term]], thm: Theorem) -> Theorem:
    """
    INST: Term instantiation.

        ⊢ p
        -------- [x₁ := t₁, ..., xₙ := tₙ]
        ⊢ p'

    where p' is p with each xᵢ replaced by tᵢ.

    Args:
        subst_list: List of (variable, replacement) pairs
        thm: Theorem to instantiate

    Returns:
        Theorem with variables replaced
    """
    subst_map = {v: t for v, t in subst_list}

    # Apply substitution to hypotheses and conclusion
    new_hyps = frozenset(subst(subst_map, h) for h in hyps(thm))
    new_concl = subst(subst_map, concl(thm))

    return Theorem(new_hyps, new_concl, PrfInst(tuple(subst_list), 0))


def INST_TYPE(subst_list: list[tuple[TyVar, Type]], thm: Theorem) -> Theorem:
    """
    INST_TYPE: Type instantiation.

        ⊢ p
        -------- ['a := τ₁, ..., 'b := τₙ]
        ⊢ p'

    where p' is p with each type variable replaced.

    Args:
        subst_list: List of (type variable, type) pairs
        thm: Theorem to instantiate

    Returns:
        Theorem with type variables replaced
    """
    ty_subst = {tv: ty for tv, ty in subst_list}

    def inst_term(t: Term) -> Term:
        """Apply type substitution to a term."""
        if isinstance(t, Var):
            new_ty = type_subst(ty_subst, t.ty)
            if new_ty == t.ty:
                return t
            return mk_var(t.name, new_ty)
        elif isinstance(t, Const):
            new_ty = type_subst(ty_subst, t.ty)
            if new_ty == t.ty:
                return t
            return mk_const(t.name, new_ty)
        elif isinstance(t, App):
            new_func = inst_term(t.func)
            new_arg = inst_term(t.arg)
            if new_func is t.func and new_arg is t.arg:
                return t
            return App(new_func, new_arg)
        elif isinstance(t, Abs):
            new_var = inst_term(t.var)
            new_body = inst_term(t.body)
            if new_var is t.var and new_body is t.body:
                return t
            return Abs(new_var, new_body)
        else:
            raise TypeError(f"Unknown term: {t}")

    new_hyps = frozenset(inst_term(h) for h in hyps(thm))
    new_concl = inst_term(concl(thm))

    return Theorem(new_hyps, new_concl, PrfInstType(tuple(subst_list), 0))
