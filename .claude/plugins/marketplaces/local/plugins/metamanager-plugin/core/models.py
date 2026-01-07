"""
Core data models for the Metamanager task orchestration system.

Defines Task, Graph, Conflict, ExecutionPlan, and related structures
used throughout the plugin for representing task dependencies and execution flow.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Optional, Any


class TaskStatus(str, Enum):
    """Current status of a task in the workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"
    TIMED_OUT = "timed_out"


class ConflictType(str, Enum):
    """Types of conflicts that can occur in task graphs."""
    INCOMPLETE_SUBTASKS = "incomplete_subtasks"
    INCOMPLETE_DEPENDENCIES = "incomplete_dependencies"
    ORPHANED_DEPENDENTS = "orphaned_dependents"
    ORPHANED_TASK = "orphaned_task"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    ABANDONED_TASK = "abandoned_task"


class ConflictSeverity(str, Enum):
    """Severity level of conflicts."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class TaskType(str, Enum):
    """Type/category of task."""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    REVIEW = "review"
    RESEARCH = "research"
    OTHER = "other"


class TaskPriority(str, Enum):
    """Priority level of a task."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Task:
    """Represents a single task in the workflow."""

    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING

    # Dependencies and structure
    subtasks: List[str] = field(default_factory=list)  # Child task IDs
    dependencies: List[str] = field(default_factory=list)  # Blocking task IDs

    # Metadata
    task_type: TaskType = TaskType.OTHER
    priority: TaskPriority = TaskPriority.MEDIUM
    completion_percentage: float = 0.0

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    estimated_duration: Optional[timedelta] = None

    # Ownership and tracking
    owner: Optional[str] = None
    agent_assigned: Optional[str] = None

    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_blocking(self, other_task_id: str) -> bool:
        """Check if this task blocks another task."""
        return other_task_id in self.dependencies

    def is_blocked_by(self, other_task_id: str) -> bool:
        """Check if this task is blocked by another task."""
        return self.is_blocking(other_task_id)

    def has_pending_subtasks(self) -> bool:
        """Check if any subtasks are not completed."""
        # This will be verified against graph
        return len(self.subtasks) > 0

    def is_abandoned(self, days: int = 7) -> bool:
        """Check if task hasn't been updated for specified days."""
        age = datetime.now() - self.updated_at
        return age > timedelta(days=days)


@dataclass
class Graph:
    """Directed Acyclic Graph (DAG) representing task dependencies."""

    task_map: Dict[str, Task] = field(default_factory=dict)

    # Forward and reverse dependency tracking
    dependency_matrix: Dict[str, Set[str]] = field(default_factory=dict)  # task_id -> set of blocking tasks
    reverse_deps: Dict[str, Set[str]] = field(default_factory=dict)  # task_id -> set of dependent tasks

    # Metadata
    graph_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def root_tasks(self) -> List[str]:
        """Get all tasks with no dependencies."""
        return [tid for tid, task in self.task_map.items() if not task.dependencies]

    @property
    def leaf_tasks(self) -> List[str]:
        """Get all tasks with no dependents."""
        return [tid for tid in self.task_map.keys() if tid not in self.reverse_deps or not self.reverse_deps[tid]]

    @property
    def completed_tasks(self) -> List[str]:
        """Get all completed tasks."""
        return [tid for tid, task in self.task_map.items() if task.status == TaskStatus.COMPLETED]

    @property
    def pending_tasks(self) -> List[str]:
        """Get all pending or in-progress tasks."""
        return [tid for tid, task in self.task_map.items()
                if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)]

    @property
    def blocked_tasks(self) -> List[str]:
        """Get all blocked tasks."""
        return [tid for tid, task in self.task_map.items() if task.status == TaskStatus.BLOCKED]

    def size(self) -> int:
        """Total number of tasks in graph."""
        return len(self.task_map)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve task by ID."""
        return self.task_map.get(task_id)

    def get_dependencies(self, task_id: str) -> List[Task]:
        """Get all tasks that block the given task."""
        blocking_ids = self.dependency_matrix.get(task_id, set())
        return [self.task_map[tid] for tid in blocking_ids if tid in self.task_map]

    def get_dependents(self, task_id: str) -> List[Task]:
        """Get all tasks that depend on the given task."""
        dependent_ids = self.reverse_deps.get(task_id, set())
        return [self.task_map[tid] for tid in dependent_ids if tid in self.task_map]


@dataclass
class Conflict:
    """Represents a detected conflict in the task graph."""

    conflict_id: str
    conflict_type: ConflictType
    task_id: str

    description: str
    severity: ConflictSeverity = ConflictSeverity.WARNING

    # Additional context
    blocking_tasks: List[str] = field(default_factory=list)
    dependent_tasks: List[str] = field(default_factory=list)

    suggested_resolution: str = ""
    auto_resolvable: bool = False

    detected_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase:
    """Represents an execution phase with parallelizable tasks."""

    phase_number: int
    tasks: List[str]

    # Execution constraints
    dependencies_resolved: bool = False
    can_parallelize: bool = True
    blocked_by_phases: List[int] = field(default_factory=list)

    # Estimation
    estimated_duration: Optional[timedelta] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Represents an execution plan for a task graph."""

    plan_id: str
    graph_id: str

    phases: List[Phase] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    parallelizable_groups: List[List[str]] = field(default_factory=list)

    # Estimation
    estimated_duration: Optional[timedelta] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""  # Agent name that created this
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_phases(self) -> int:
        """Total number of execution phases."""
        return len(self.phases)


@dataclass
class ConflictResolution:
    """Represents a resolution to a conflict."""

    conflict_id: str
    resolution_type: str  # reorder, delete, split, merge, etc.

    actions: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""

    applied_at: Optional[datetime] = None
    applied_by: str = ""  # User or agent name

    result: Optional[bool] = None  # True if successful, False if failed
    error_message: Optional[str] = None


@dataclass
class TaskLineageEntry:
    """Represents a single event in a task's history."""

    timestamp: datetime
    action: str  # created, updated, completed, status_changed, etc.

    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None

    actor: str = ""  # user, orchestrator-agent, planner-agent, etc.
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskLineage:
    """Complete history of changes to a task."""

    task_id: str
    entries: List[TaskLineageEntry] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)

    def add_entry(self, entry: TaskLineageEntry) -> None:
        """Add an event to the task's history."""
        self.entries.append(entry)

    def get_last_update(self) -> Optional[TaskLineageEntry]:
        """Get most recent change."""
        return self.entries[-1] if self.entries else None


@dataclass
class MetamanagerState:
    """Overall state of the metamanager system."""

    current_graph: Graph
    last_plan: Optional[ExecutionPlan] = None
    conflicts: List[Conflict] = field(default_factory=list)

    active_agents: Dict[str, str] = field(default_factory=dict)  # agent_name -> task_id

    last_orchestration: Optional[datetime] = None
    last_planning: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_completion_percentage(self) -> float:
        """Calculate overall completion percentage."""
        if not self.current_graph.task_map:
            return 0.0

        total = len(self.current_graph.task_map)
        completed = len(self.current_graph.completed_tasks)
        return (completed / total) * 100.0 if total > 0 else 0.0
