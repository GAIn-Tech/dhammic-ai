"""
HOL Type System

Implements Simple Type Theory (Church's foundation):
- Type variables: 'a, 'b, etc.
- Base types: bool, ind, user-defined
- Type operators: function types (->), list, option, etc.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from loguru import logger


# =============================================================================
# Type Definitions
# =============================================================================

@dataclass(frozen=True)
class Type:
    """Base class for all types."""
    pass


@dataclass(frozen=True)
class TyVar(Type):
    """
    Type variable, e.g., 'a, 'b.

    Used for polymorphism: ∀'a. 'a -> 'a
    """
    name: str

    def __repr__(self) -> str:
        return f"'{self.name}"


@dataclass(frozen=True)
class TyBase(Type):
    """
    Base type (nullary type constant), e.g., bool, nat, int.
    """
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TyApp(Type):
    """
    Type application: operator applied to arguments.

    Examples:
    - list 'a  ->  TyApp(TyBase("list"), [TyVar("a")])
    - 'a -> 'b ->  TyApp(TyBase("fun"), [TyVar("a"), TyVar("b")])
    """
    op: Type
    args: tuple[Type, ...]

    def __init__(self, op: Type, args: Sequence[Type]):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "args", tuple(args))

    def __repr__(self) -> str:
        if self.op == _fun_ty_op and len(self.args) == 2:
            dom, rng = self.args
            # Right-associative: a -> b -> c = a -> (b -> c)
            dom_str = f"({dom})" if isinstance(dom, TyApp) and dom.op == _fun_ty_op else str(dom)
            return f"{dom_str} -> {rng}"
        elif self.args:
            args_str = ", ".join(str(a) for a in self.args)
            return f"{self.op}[{args_str}]"
        else:
            return str(self.op)


# =============================================================================
# Built-in Types
# =============================================================================

# The function type operator (internal)
_fun_ty_op = TyBase("fun")

# Primitive base types
bool_ty = TyBase("bool")
ind_ty = TyBase("ind")


# =============================================================================
# Type Constructors
# =============================================================================

def mk_fun_ty(domain: Type, range_: Type) -> TyApp:
    """
    Construct a function type: domain -> range.

    Args:
        domain: The argument type
        range_: The return type

    Returns:
        Function type domain -> range_
    """
    return TyApp(_fun_ty_op, [domain, range_])


def is_fun_ty(ty: Type) -> bool:
    """Check if a type is a function type."""
    return isinstance(ty, TyApp) and ty.op == _fun_ty_op and len(ty.args) == 2


def dest_fun_ty(ty: Type) -> tuple[Type, Type]:
    """
    Destructure a function type into domain and range.

    Args:
        ty: A function type

    Returns:
        Tuple of (domain, range)

    Raises:
        ValueError: If ty is not a function type
    """
    if not is_fun_ty(ty):
        raise ValueError(f"{ty} is not a function type")
    return ty.args[0], ty.args[1]


# =============================================================================
# Type Operations
# =============================================================================

def type_vars(ty: Type) -> set[TyVar]:
    """
    Extract all type variables from a type.

    Args:
        ty: Any type

    Returns:
        Set of type variables occurring in ty
    """
    if isinstance(ty, TyVar):
        return {ty}
    elif isinstance(ty, TyBase):
        return set()
    elif isinstance(ty, TyApp):
        result: set[TyVar] = set()
        for arg in ty.args:
            result |= type_vars(arg)
        return result
    else:
        raise TypeError(f"Unknown type: {ty}")


def type_subst(subst: dict[TyVar, Type], ty: Type) -> Type:
    """
    Apply a type substitution.

    Args:
        subst: Mapping from type variables to types
        ty: Type to substitute in

    Returns:
        Type with substitutions applied
    """
    if isinstance(ty, TyVar):
        return subst.get(ty, ty)
    elif isinstance(ty, TyBase):
        return ty
    elif isinstance(ty, TyApp):
        new_args = tuple(type_subst(subst, arg) for arg in ty.args)
        if new_args == ty.args:
            return ty
        return TyApp(ty.op, new_args)
    else:
        raise TypeError(f"Unknown type: {ty}")


def types_match(pattern: Type, target: Type, subst: dict[TyVar, Type] | None = None) -> dict[TyVar, Type] | None:
    """
    Match a pattern type against a target type.

    Finds a substitution such that pattern[subst] == target.
    Type variables in pattern can match any type in target.
    Type variables in target are treated as constants (not matched).

    Args:
        pattern: Type pattern (may contain type variables to match)
        target: Concrete type to match against
        subst: Existing substitution to extend (default: empty)

    Returns:
        Substitution dict if match succeeds, None if no match
    """
    if subst is None:
        subst = {}

    if isinstance(pattern, TyVar):
        if pattern in subst:
            # Already bound - check consistency
            if subst[pattern] == target:
                return subst
            else:
                return None
        else:
            # Bind the variable
            new_subst = subst.copy()
            new_subst[pattern] = target
            return new_subst

    elif isinstance(pattern, TyBase):
        if pattern == target:
            return subst
        else:
            return None

    elif isinstance(pattern, TyApp):
        if not isinstance(target, TyApp):
            return None
        if pattern.op != target.op:
            # Check if operators match (recursively for complex operators)
            op_match = types_match(pattern.op, target.op, subst)
            if op_match is None:
                return None
            subst = op_match
        if len(pattern.args) != len(target.args):
            return None
        for p_arg, t_arg in zip(pattern.args, target.args):
            result = types_match(p_arg, t_arg, subst)
            if result is None:
                return None
            subst = result
        return subst

    else:
        raise TypeError(f"Unknown type: {pattern}")
