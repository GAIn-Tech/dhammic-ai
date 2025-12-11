"""Tests for tacticals (tactic combinators)."""
import pytest
from hol.kernel.types import bool_ty
from hol.kernel.terms import mk_var
from hol.tactics.goals import mk_goal
from hol.tactics.tactics import TacticError, assumption
from hol.tactics.tacticals import (
    THEN, ORELSE, REPEAT, TRY, ALL_TAC, NO_TAC, FIRST,
)


class TestThen:
    """Test THEN combinator."""

    def test_then_sequences_tactics(self):
        """THEN applies second tactic to all subgoals of first."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        goal = mk_goal([p, q], p)
        # assumption should solve this in one step
        # THEN with ALL_TAC should still work
        tac = THEN(assumption, ALL_TAC)
        result = tac(goal)
        assert len(result.subgoals) == 0


class TestOrelse:
    """Test ORELSE combinator."""

    def test_orelse_first_succeeds(self):
        """ORELSE uses first tactic if it succeeds."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([p], p)
        tac = ORELSE(assumption, NO_TAC)
        result = tac(goal)
        assert len(result.subgoals) == 0

    def test_orelse_falls_back(self):
        """ORELSE uses second tactic if first fails."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        goal = mk_goal([p], p)
        # NO_TAC fails, then assumption succeeds
        tac = ORELSE(NO_TAC, assumption)
        result = tac(goal)
        assert len(result.subgoals) == 0


class TestRepeat:
    """Test REPEAT combinator."""

    def test_repeat_applies_until_failure(self):
        """REPEAT applies tactic until it fails."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([p], p)
        # REPEAT assumption should solve and stop
        tac = REPEAT(assumption)
        result = tac(goal)
        assert len(result.subgoals) == 0


class TestTry:
    """Test TRY combinator."""

    def test_try_succeeds_on_success(self):
        """TRY returns result if tactic succeeds."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([p], p)
        tac = TRY(assumption)
        result = tac(goal)
        assert len(result.subgoals) == 0

    def test_try_succeeds_on_failure(self):
        """TRY succeeds with unchanged goal if tactic fails."""
        p = mk_var("p", bool_ty)
        q = mk_var("q", bool_ty)
        goal = mk_goal([p], q)  # assumption will fail
        tac = TRY(assumption)
        result = tac(goal)
        assert len(result.subgoals) == 1
        assert result.subgoals[0].target == q


class TestNoTac:
    """Test NO_TAC."""

    def test_no_tac_fails(self):
        """NO_TAC always fails."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([], p)
        with pytest.raises(TacticError):
            NO_TAC(goal)


class TestAllTac:
    """Test ALL_TAC."""

    def test_all_tac_succeeds(self):
        """ALL_TAC always succeeds without changing goal."""
        p = mk_var("p", bool_ty)
        goal = mk_goal([], p)
        result = ALL_TAC(goal)
        assert len(result.subgoals) == 1
        assert result.subgoals[0] == goal
