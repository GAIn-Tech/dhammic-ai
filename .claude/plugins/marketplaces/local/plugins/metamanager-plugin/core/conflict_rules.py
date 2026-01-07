"""
Conflict detection rules for task graphs.

Implements rules to detect various types of conflicts such as:
- Incomplete subtasks
- Circular dependencies
- Orphaned tasks
- Abandoned tasks
"""

from datetime import datetime, timedelta
from typing import List, Optional

from .models import (
    Task, Graph, Conflict, ConflictType, ConflictSeverity, TaskStatus
)
from .dag_engine import DAGEngine


class ConflictRules:
    """Implements conflict detection rules for task graphs."""

    @staticmethod
    def detect_all_conflicts(graph: Graph, days_abandoned: int = 7) -> List[Conflict]:
        """
        Run all conflict detection rules on a graph.

        Args:
            graph: Graph to check
            days_abandoned: Threshold for abandoned task detection

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check each task
        for task_id, task in graph.task_map.items():
            # Rule: Incomplete subtasks
            conflict = ConflictRules.rule_complete_with_pending_subtasks(task_id, task, graph)
            if conflict:
                conflicts.append(conflict)

            # Rule: Incomplete dependencies
            conflict = ConflictRules.rule_blocking_dependencies_incomplete(task_id, task, graph)
            if conflict:
                conflicts.append(conflict)

            # Rule: Abandoned task
            conflict = ConflictRules.rule_abandoned_task(task_id, task, days_abandoned)
            if conflict:
                conflicts.append(conflict)

        # Check graph-level conflicts
        # Rule: Circular dependencies
        conflicts.extend(ConflictRules.rule_circular_dependency(graph))

        # Rule: Orphaned dependents (when deleting)
        # This is handled separately in deletion context

        # Rule: Orphaned tasks
        conflicts.extend(ConflictRules.rule_orphaned_tasks(graph))

        return conflicts

    @staticmethod
    def rule_complete_with_pending_subtasks(task_id: str, task: Task, graph: Graph) -> Optional[Conflict]:
        """
        Detect: Attempting to complete task with incomplete subtasks.

        Args:
            task_id: ID of task being checked
            task: Task object
            graph: Graph containing tasks

        Returns:
            Conflict if detected, None otherwise
        """
        if task.status != TaskStatus.COMPLETED:
            return None

        pending_subtasks = []
        for subtask_id in task.subtasks:
            subtask = graph.get_task(subtask_id)
            if subtask and subtask.status != TaskStatus.COMPLETED:
                pending_subtasks.append(subtask_id)

        if pending_subtasks:
            return Conflict(
                conflict_id=f"incomplete_subtasks_{task_id}",
                conflict_type=ConflictType.INCOMPLETE_SUBTASKS,
                task_id=task_id,
                description=f"Cannot complete task '{task.name}': {len(pending_subtasks)} subtask(s) still pending",
                severity=ConflictSeverity.CRITICAL,
                dependent_tasks=pending_subtasks,
                suggested_resolution=f"Complete all {len(pending_subtasks)} subtask(s) first, or convert them to independent tasks",
                auto_resolvable=False,
            )

        return None

    @staticmethod
    def rule_delete_task_with_dependents(task_id: str, task: Task, graph: Graph) -> Optional[Conflict]:
        """
        Detect: Deleting task that has dependent tasks.

        Args:
            task_id: ID of task being deleted
            task: Task object
            graph: Graph containing tasks

        Returns:
            Conflict if detected, None otherwise
        """
        dependents = graph.reverse_deps.get(task_id, set())

        if dependents:
            dependent_tasks = [graph.get_task(did) for did in dependents]
            dependent_names = [t.name for t in dependent_tasks if t]

            return Conflict(
                conflict_id=f"orphaned_dependents_{task_id}",
                conflict_type=ConflictType.ORPHANED_DEPENDENTS,
                task_id=task_id,
                description=f"Cannot delete '{task.name}': {len(dependents)} task(s) depend on it",
                severity=ConflictSeverity.CRITICAL,
                dependent_tasks=list(dependents),
                suggested_resolution=f"Reassign or delete dependent tasks: {', '.join(dependent_names[:3])}{'...' if len(dependent_names) > 3 else ''}",
                auto_resolvable=False,
            )

        return None

    @staticmethod
    def rule_circular_dependency(graph: Graph) -> List[Conflict]:
        """
        Detect: Circular dependencies in the graph.

        Args:
            graph: Graph to check

        Returns:
            List of conflicts (one per cycle detected)
        """
        cycles = DAGEngine.find_cycles(graph)
        conflicts = []

        for cycle in cycles:
            cycle_str = " -> ".join(cycle)
            task_names = [graph.get_task(tid).name if graph.get_task(tid) else tid for tid in cycle]
            task_names_str = " -> ".join(task_names[:3])
            if len(task_names) > 3:
                task_names_str += "..."

            conflicts.append(Conflict(
                conflict_id=f"circular_dependency_{cycle[0]}",
                conflict_type=ConflictType.CIRCULAR_DEPENDENCY,
                task_id=cycle[0],
                description=f"Circular dependency detected: {task_names_str}",
                severity=ConflictSeverity.CRITICAL,
                suggested_resolution=f"Break cycle by removing one dependency: {cycle_str}",
                auto_resolvable=False,
            ))

        return conflicts

    @staticmethod
    def rule_blocking_dependencies_incomplete(task_id: str, task: Task, graph: Graph) -> Optional[Conflict]:
        """
        Detect: Task with incomplete blocking dependencies.

        Args:
            task_id: ID of task being checked
            task: Task object
            graph: Graph containing tasks

        Returns:
            Conflict if detected, None otherwise
        """
        incomplete_blockers = []
        for blocker_id in task.dependencies:
            blocker = graph.get_task(blocker_id)
            if blocker and blocker.status != TaskStatus.COMPLETED:
                incomplete_blockers.append(blocker_id)

        if incomplete_blockers and task.status == TaskStatus.IN_PROGRESS:
            blocker_names = [graph.get_task(bid).name for bid in incomplete_blockers if graph.get_task(bid)]

            return Conflict(
                conflict_id=f"incomplete_dependencies_{task_id}",
                conflict_type=ConflictType.INCOMPLETE_DEPENDENCIES,
                task_id=task_id,
                description=f"Task '{task.name}' in progress but blocked by {len(incomplete_blockers)} incomplete task(s)",
                severity=ConflictSeverity.WARNING,
                blocking_tasks=incomplete_blockers,
                suggested_resolution=f"Wait for: {', '.join(blocker_names[:3])}{'...' if len(blocker_names) > 3 else ''}",
                auto_resolvable=False,
            )

        return None

    @staticmethod
    def rule_orphaned_tasks(graph: Graph) -> List[Conflict]:
        """
        Detect: Tasks disconnected from the task hierarchy.

        Args:
            graph: Graph to check

        Returns:
            List of conflicts (one per orphaned task)
        """
        orphans = DAGEngine.find_orphaned_tasks(graph)
        conflicts = []

        for orphan_id in orphans:
            orphan = graph.get_task(orphan_id)
            if orphan:
                conflicts.append(Conflict(
                    conflict_id=f"orphaned_task_{orphan_id}",
                    conflict_type=ConflictType.ORPHANED_TASK,
                    task_id=orphan_id,
                    description=f"Task '{orphan.name}' is disconnected from task hierarchy",
                    severity=ConflictSeverity.INFO,
                    suggested_resolution="Connect to a parent task or mark for deletion",
                    auto_resolvable=False,
                ))

        return conflicts

    @staticmethod
    def rule_abandoned_task(task_id: str, task: Task, days: int = 7) -> Optional[Conflict]:
        """
        Detect: Task not updated for extended period.

        Args:
            task_id: ID of task being checked
            task: Task object
            days: Number of days before considering abandoned

        Returns:
            Conflict if detected, None otherwise
        """
        if task.status == TaskStatus.COMPLETED:
            return None

        age = datetime.now() - task.updated_at
        if age > timedelta(days=days):
            days_ago = age.days

            return Conflict(
                conflict_id=f"abandoned_task_{task_id}",
                conflict_type=ConflictType.ABANDONED_TASK,
                task_id=task_id,
                description=f"Task '{task.name}' not updated for {days_ago} days",
                severity=ConflictSeverity.INFO,
                suggested_resolution=f"Review task status or close if no longer needed",
                auto_resolvable=False,
            )

        return None

    @staticmethod
    def validate_task_completion(task_id: str, graph: Graph) -> tuple[bool, Optional[str]]:
        """
        Validate whether a task can be marked complete.

        Args:
            task_id: ID of task to check
            graph: Graph containing tasks

        Returns:
            Tuple of (can_complete, error_message)
        """
        task = graph.get_task(task_id)
        if not task:
            return False, f"Task {task_id} not found"

        # Check for pending subtasks
        for subtask_id in task.subtasks:
            subtask = graph.get_task(subtask_id)
            if subtask and subtask.status != TaskStatus.COMPLETED:
                subtask_name = subtask.name
                return False, f"Cannot complete: subtask '{subtask_name}' is not complete"

        return True, None

    @staticmethod
    def validate_task_deletion(task_id: str, graph: Graph) -> tuple[bool, Optional[str]]:
        """
        Validate whether a task can be deleted.

        Args:
            task_id: ID of task to check
            graph: Graph containing tasks

        Returns:
            Tuple of (can_delete, error_message)
        """
        task = graph.get_task(task_id)
        if not task:
            return False, f"Task {task_id} not found"

        # Check for dependent tasks
        dependents = graph.reverse_deps.get(task_id, set())
        if dependents:
            dependent_count = len(dependents)
            return False, f"Cannot delete: {dependent_count} task(s) depend on this task"

        return True, None

    @staticmethod
    def check_circular_dependencies(graph: Graph) -> tuple[bool, List[List[str]]]:
        """
        Check for circular dependencies in graph.

        Args:
            graph: Graph to check

        Returns:
            Tuple of (has_cycles, list_of_cycles)
        """
        cycles = DAGEngine.find_cycles(graph)
        return (len(cycles) > 0, cycles)

    @staticmethod
    def get_blocked_tasks(graph: Graph) -> List[str]:
        """
        Get all tasks that are currently blocked.

        A task is blocked if it has incomplete dependencies.

        Args:
            graph: Graph to check

        Returns:
            List of blocked task IDs
        """
        blocked = []

        for task_id, task in graph.task_map.items():
            if task.status == TaskStatus.IN_PROGRESS:
                for blocker_id in task.dependencies:
                    blocker = graph.get_task(blocker_id)
                    if blocker and blocker.status != TaskStatus.COMPLETED:
                        blocked.append(task_id)
                        break

        return blocked

    @staticmethod
    def get_ready_tasks(graph: Graph) -> List[str]:
        """
        Get all tasks that are ready to execute.

        A task is ready if all its dependencies are complete.

        Args:
            graph: Graph to check

        Returns:
            List of ready task IDs (not yet started)
        """
        ready = []

        for task_id, task in graph.task_map.items():
            if task.status == TaskStatus.PENDING:
                all_deps_complete = all(
                    graph.get_task(dep_id).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                    if graph.get_task(dep_id)
                )

                if all_deps_complete:
                    ready.append(task_id)

        return ready
