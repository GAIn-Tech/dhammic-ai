"""
HOL Theorems and Proof Recording

A theorem is a sequent: hypotheses ⊢ conclusion
with an attached proof trace for recording/replay.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from loguru import logger

from hol.kernel.terms import Term


# =============================================================================
# Proof Steps (for proof recording)
# =============================================================================

@dataclass(frozen=True)
class ProofStep:
    """Base class for proof steps."""
    pass


@dataclass(frozen=True)
class PrfRefl(ProofStep):
    """REFL(t): ⊢ t = t"""
    term: Term


@dataclass(frozen=True)
class PrfTrans(ProofStep):
    """TRANS(thm1, thm2): ⊢ s=t, ⊢ t=u --> ⊢ s=u"""
    thm1_idx: int
    thm2_idx: int


@dataclass(frozen=True)
class PrfMkComb(ProofStep):
    """MK_COMB(thm1, thm2): ⊢ f=g, ⊢ x=y --> ⊢ f x = g y"""
    thm1_idx: int
    thm2_idx: int


@dataclass(frozen=True)
class PrfAbs(ProofStep):
    """ABS(x, thm): ⊢ s=t --> ⊢ (λx.s) = (λx.t)"""
    var: Term
    thm_idx: int


@dataclass(frozen=True)
class PrfBeta(ProofStep):
    """BETA(t): ⊢ (λx.b) t = b[t/x]"""
    term: Term


@dataclass(frozen=True)
class PrfAssume(ProofStep):
    """ASSUME(p): {p} ⊢ p"""
    prop: Term


@dataclass(frozen=True)
class PrfEqMp(ProofStep):
    """EQ_MP(thm1, thm2): ⊢ p=q, ⊢ p --> ⊢ q"""
    thm1_idx: int
    thm2_idx: int


@dataclass(frozen=True)
class PrfDeductAntisym(ProofStep):
    """DEDUCT_ANTISYM(thm1, thm2): Γ⊢p, Δ⊢q --> (Γ-{q})∪(Δ-{p}) ⊢ p=q"""
    thm1_idx: int
    thm2_idx: int


@dataclass(frozen=True)
class PrfInst(ProofStep):
    """INST(subst, thm): term instantiation"""
    subst: tuple[tuple[Term, Term], ...]
    thm_idx: int


@dataclass(frozen=True)
class PrfInstType(ProofStep):
    """INST_TYPE(subst, thm): type instantiation"""
    subst: tuple[tuple[Any, Any], ...]  # TyVar -> Type
    thm_idx: int


# =============================================================================
# Theorem Type
# =============================================================================

@dataclass(frozen=True)
class Theorem:
    """
    A proven theorem: hypotheses ⊢ conclusion.

    Theorems can only be created through inference rules.
    The proof trace records how this theorem was derived.
    """
    _hyps: frozenset[Term]
    _concl: Term
    _trace: ProofStep

    def __repr__(self) -> str:
        if self._hyps:
            hyps_str = ", ".join(str(h) for h in self._hyps)
            return f"{{{hyps_str}}} ⊢ {self._concl}"
        else:
            return f"⊢ {self._concl}"


# =============================================================================
# Theorem Accessors
# =============================================================================

def hyps(thm: Theorem) -> frozenset[Term]:
    """Get the hypotheses of a theorem."""
    return thm._hyps


def concl(thm: Theorem) -> Term:
    """Get the conclusion of a theorem."""
    return thm._concl


def trace(thm: Theorem) -> ProofStep:
    """Get the proof trace of a theorem."""
    return thm._trace
