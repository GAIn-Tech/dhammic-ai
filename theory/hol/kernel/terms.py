"""
HOL Term Language

Terms in Higher-Order Logic:
- Variables: x, y, z with types
- Constants: T, F, =, ∀, etc.
- Applications: f x
- Abstractions: λx. t
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from loguru import logger

from hol.kernel.types import (
    Type, TyVar, TyBase, TyApp,
    bool_ty, mk_fun_ty, dest_fun_ty, is_fun_ty, type_subst,
)


# =============================================================================
# Term Definitions
# =============================================================================

@dataclass(frozen=True)
class Term:
    """Base class for all terms."""
    pass


@dataclass(frozen=True)
class Var(Term):
    """
    Variable term.

    Variables are identified by name AND type.
    x:bool and x:nat are different variables.
    """
    name: str
    ty: Type

    def __repr__(self) -> str:
        return f"{self.name}"


@dataclass(frozen=True)
class Const(Term):
    """
    Constant term.

    Constants are defined globally with a name and polymorphic type.
    Instances may have type variables instantiated.
    """
    name: str
    ty: Type

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class App(Term):
    """
    Application term: f x

    Applies a function to an argument.
    """
    func: Term
    arg: Term

    def __repr__(self) -> str:
        # Special case for binary operators
        if isinstance(self.func, App) and isinstance(self.func.func, Const):
            op_name = self.func.func.name
            if op_name in {"=", "+", "-", "*", "/", "∧", "∨", "→"}:
                return f"({self.func.arg} {op_name} {self.arg})"
        return f"({self.func} {self.arg})"


@dataclass(frozen=True)
class Abs(Term):
    """
    Lambda abstraction: λx. body

    Binds a variable in the body term.
    """
    var: Var
    body: Term

    def __repr__(self) -> str:
        return f"(λ{self.var.name}:{self.var.ty}. {self.body})"


# =============================================================================
# Smart Constructors
# =============================================================================

def mk_var(name: str, ty: Type) -> Var:
    """Create a variable with given name and type."""
    return Var(name, ty)


def mk_const(name: str, ty: Type) -> Const:
    """Create a constant with given name and type."""
    return Const(name, ty)


def mk_app(func: Term, arg: Term) -> App:
    """
    Create an application term.

    Type checks that func has function type and arg matches domain.

    Raises:
        TypeError: If types don't match
    """
    func_ty = type_of(func)
    if not is_fun_ty(func_ty):
        raise TypeError(f"Cannot apply non-function {func} of type {func_ty}")

    dom, rng = dest_fun_ty(func_ty)
    arg_ty = type_of(arg)

    if dom != arg_ty:
        raise TypeError(
            f"Application type mismatch: function expects {dom}, got {arg_ty}"
        )

    return App(func, arg)


def mk_abs(var: Var, body: Term) -> Abs:
    """Create a lambda abstraction."""
    if not isinstance(var, Var):
        raise TypeError(f"Abstraction variable must be Var, got {type(var)}")
    return Abs(var, body)


# =============================================================================
# Destructors
# =============================================================================

def dest_var(t: Term) -> tuple[str, Type]:
    """Destructure a variable into name and type."""
    if not isinstance(t, Var):
        raise ValueError(f"Not a variable: {t}")
    return t.name, t.ty


def dest_const(t: Term) -> tuple[str, Type]:
    """Destructure a constant into name and type."""
    if not isinstance(t, Const):
        raise ValueError(f"Not a constant: {t}")
    return t.name, t.ty


def dest_app(t: Term) -> tuple[Term, Term]:
    """Destructure an application into function and argument."""
    if not isinstance(t, App):
        raise ValueError(f"Not an application: {t}")
    return t.func, t.arg


def dest_abs(t: Term) -> tuple[Var, Term]:
    """Destructure an abstraction into variable and body."""
    if not isinstance(t, Abs):
        raise ValueError(f"Not an abstraction: {t}")
    return t.var, t.body


# =============================================================================
# Predicates
# =============================================================================

def is_var(t: Term) -> bool:
    """Check if term is a variable."""
    return isinstance(t, Var)


def is_const(t: Term) -> bool:
    """Check if term is a constant."""
    return isinstance(t, Const)


def is_app(t: Term) -> bool:
    """Check if term is an application."""
    return isinstance(t, App)


def is_abs(t: Term) -> bool:
    """Check if term is an abstraction."""
    return isinstance(t, Abs)


# =============================================================================
# Type Operations
# =============================================================================

def type_of(t: Term) -> Type:
    """
    Get the type of a term.

    Returns:
        The type of the term
    """
    if isinstance(t, Var):
        return t.ty
    elif isinstance(t, Const):
        return t.ty
    elif isinstance(t, App):
        func_ty = type_of(t.func)
        _, rng = dest_fun_ty(func_ty)
        return rng
    elif isinstance(t, Abs):
        var_ty = t.var.ty
        body_ty = type_of(t.body)
        return mk_fun_ty(var_ty, body_ty)
    else:
        raise TypeError(f"Unknown term: {t}")


# =============================================================================
# Variable Operations
# =============================================================================

def free_vars(t: Term) -> set[Var]:
    """
    Get all free variables in a term.

    A variable is free if it's not bound by an enclosing lambda.
    """
    if isinstance(t, Var):
        return {t}
    elif isinstance(t, Const):
        return set()
    elif isinstance(t, App):
        return free_vars(t.func) | free_vars(t.arg)
    elif isinstance(t, Abs):
        return free_vars(t.body) - {t.var}
    else:
        raise TypeError(f"Unknown term: {t}")


def bound_vars(t: Term) -> set[Var]:
    """Get all bound variables in a term."""
    if isinstance(t, Var):
        return set()
    elif isinstance(t, Const):
        return set()
    elif isinstance(t, App):
        return bound_vars(t.func) | bound_vars(t.arg)
    elif isinstance(t, Abs):
        return {t.var} | bound_vars(t.body)
    else:
        raise TypeError(f"Unknown term: {t}")


# =============================================================================
# Fresh Variable Generation
# =============================================================================

_var_counter = 0

def fresh_var(name: str, ty: Type, avoid: set[Var] | None = None) -> Var:
    """
    Generate a fresh variable not in the avoid set.

    Args:
        name: Base name for the variable
        ty: Type of the variable
        avoid: Set of variables to avoid

    Returns:
        A fresh variable
    """
    global _var_counter
    if avoid is None:
        avoid = set()

    avoid_names = {v.name for v in avoid}

    if name not in avoid_names:
        return mk_var(name, ty)

    while True:
        _var_counter += 1
        new_name = f"{name}{_var_counter}"
        if new_name not in avoid_names:
            return mk_var(new_name, ty)


# =============================================================================
# Substitution
# =============================================================================

def subst(mapping: Mapping[Var, Term], t: Term) -> Term:
    """
    Substitute terms for variables.

    Performs capture-avoiding substitution.

    Args:
        mapping: Map from variables to replacement terms
        t: Term to substitute in

    Returns:
        Term with substitutions applied
    """
    if isinstance(t, Var):
        return mapping.get(t, t)

    elif isinstance(t, Const):
        return t

    elif isinstance(t, App):
        new_func = subst(mapping, t.func)
        new_arg = subst(mapping, t.arg)
        if new_func is t.func and new_arg is t.arg:
            return t
        return App(new_func, new_arg)

    elif isinstance(t, Abs):
        # Check if bound variable shadows any substitution
        if t.var in mapping:
            # Variable is shadowed, don't substitute for it in body
            new_mapping = {k: v for k, v in mapping.items() if k != t.var}
            if not new_mapping:
                return t
            mapping = new_mapping

        # Check for variable capture
        # If any replacement term contains the bound variable free,
        # we need to rename the bound variable
        subst_free_vars: set[Var] = set()
        for v, replacement in mapping.items():
            if v in free_vars(t.body):
                subst_free_vars |= free_vars(replacement)

        if t.var in subst_free_vars:
            # Capture would occur - rename bound variable
            all_vars = free_vars(t.body) | subst_free_vars | bound_vars(t.body)
            new_var = fresh_var(t.var.name, t.var.ty, all_vars)
            # First rename the bound variable
            renamed_body = subst({t.var: new_var}, t.body)
            # Then apply the original substitution
            new_body = subst(mapping, renamed_body)
            return Abs(new_var, new_body)
        else:
            new_body = subst(mapping, t.body)
            if new_body is t.body:
                return t
            return Abs(t.var, new_body)

    else:
        raise TypeError(f"Unknown term: {t}")


# =============================================================================
# Alpha Equivalence
# =============================================================================

def alpha_equiv(t1: Term, t2: Term, var_map: dict[Var, Var] | None = None) -> bool:
    """
    Check if two terms are alpha-equivalent.

    Two terms are alpha-equivalent if they differ only in the names
    of bound variables.

    Args:
        t1, t2: Terms to compare
        var_map: Mapping of bound variables from t1 to t2

    Returns:
        True if terms are alpha-equivalent
    """
    if var_map is None:
        var_map = {}

    if isinstance(t1, Var) and isinstance(t2, Var):
        # Check if this is a bound variable
        if t1 in var_map:
            return var_map[t1] == t2
        else:
            # Free variable - must be identical
            return t1 == t2

    elif isinstance(t1, Const) and isinstance(t2, Const):
        return t1 == t2

    elif isinstance(t1, App) and isinstance(t2, App):
        return (alpha_equiv(t1.func, t2.func, var_map) and
                alpha_equiv(t1.arg, t2.arg, var_map))

    elif isinstance(t1, Abs) and isinstance(t2, Abs):
        # Types must match
        if t1.var.ty != t2.var.ty:
            return False
        # Extend variable mapping
        new_map = var_map.copy()
        new_map[t1.var] = t2.var
        return alpha_equiv(t1.body, t2.body, new_map)

    else:
        return False


# =============================================================================
# Equality Primitive
# =============================================================================

# The equality constant has polymorphic type: 'a -> 'a -> bool
_eq_ty_var = TyVar("a")
_eq_ty = mk_fun_ty(_eq_ty_var, mk_fun_ty(_eq_ty_var, bool_ty))


def mk_eq(lhs: Term, rhs: Term) -> Term:
    """
    Create an equality term: lhs = rhs

    Args:
        lhs: Left-hand side
        rhs: Right-hand side

    Returns:
        Equality term

    Raises:
        TypeError: If lhs and rhs have different types
    """
    lhs_ty = type_of(lhs)
    rhs_ty = type_of(rhs)

    if lhs_ty != rhs_ty:
        raise TypeError(
            f"Equality type mismatch: {lhs_ty} vs {rhs_ty}"
        )

    # Instantiate equality constant at the appropriate type
    eq_ty = mk_fun_ty(lhs_ty, mk_fun_ty(lhs_ty, bool_ty))
    eq_const = mk_const("=", eq_ty)

    return mk_app(mk_app(eq_const, lhs), rhs)


def is_eq(t: Term) -> bool:
    """Check if a term is an equality."""
    if not isinstance(t, App):
        return False
    if not isinstance(t.func, App):
        return False
    if not isinstance(t.func.func, Const):
        return False
    return t.func.func.name == "="


def dest_eq(t: Term) -> tuple[Term, Term]:
    """
    Destructure an equality into lhs and rhs.

    Raises:
        ValueError: If t is not an equality
    """
    if not is_eq(t):
        raise ValueError(f"Not an equality: {t}")
    return t.func.arg, t.arg
