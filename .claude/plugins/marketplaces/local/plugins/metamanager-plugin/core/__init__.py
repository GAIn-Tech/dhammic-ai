"""
Core Metamanager engine module.

Exports main classes and functions for task graph management and execution planning.
"""

from .models import (
    Task,
    Graph,
    TaskStatus,
    Conflict,
    ExecutionPlan,
    Phase,
    TaskLineage,
    MetamanagerState,
    ConflictType,
    ConflictSeverity,
    TaskType,
    TaskPriority,
)

from .dag_engine import DAGEngine

from .conflict_rules import ConflictRules

from .planning_engine import PlanningEngine

__all__ = [
    "Task",
    "Graph",
    "TaskStatus",
    "Conflict",
    "ExecutionPlan",
    "Phase",
    "TaskLineage",
    "MetamanagerState",
    "ConflictType",
    "ConflictSeverity",
    "TaskType",
    "TaskPriority",
    "DAGEngine",
    "ConflictRules",
    "PlanningEngine",
]
