"""Tests for goal management."""
import pytest
from hol.kernel.types import bool_ty
from hol.kernel.terms import mk_var, mk_eq
from hol.tactics.goals import Goal, ProofState, mk_goal


class TestGoal:
    """Test Goal type."""

    def test_goal_creation(self):
        """Goals have assumptions and target."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        goal = mk_goal([p], q)
        assert p in goal.assumptions
        assert goal.target == q

    def test_goal_no_assumptions(self):
        """Goals can have empty assumptions."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([], p)
        assert len(goal.assumptions) == 0
        assert goal.target == p


class TestProofState:
    """Test ProofState management."""

    def test_initial_state(self):
        """Initial state has one goal."""
        p = mk_var("p", bool_ty)
        state = ProofState.initial(p)
        assert len(state.goals) == 1
        assert state.goals[0].target == p

    def test_is_complete(self):
        """State is complete when no goals remain."""
        p = mk_var("p", bool_ty)
        state = ProofState.initial(p)
        assert not state.is_complete()

    def test_current_goal(self):
        """Current goal is first in list."""
        p = mk_var("p", bool_ty)
        state = ProofState.initial(p)
        assert state.current_goal().target == p
