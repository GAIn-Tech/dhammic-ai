"""
DAG (Directed Acyclic Graph) Engine for task dependency management.

Implements graph construction, validation, cycle detection, topological sorting,
critical path analysis, and query operations on task graphs.
"""

from collections import defaultdict, deque
from typing import List, Set, Dict, Optional, Tuple
from datetime import datetime, timedelta

from .models import Task, Graph, TaskStatus


class DAGEngine:
    """Engine for operating on task dependency graphs."""

    @staticmethod
    def build_graph(tasks: List[Task], graph_id: str = "") -> Graph:
        """
        Build a DAG from a list of tasks.

        Args:
            tasks: List of Task objects
            graph_id: Optional identifier for the graph

        Returns:
            Constructed Graph object
        """
        graph = Graph(graph_id=graph_id)

        # Add all tasks to the graph
        for task in tasks:
            graph.task_map[task.id] = task

        # Build dependency matrices
        for task in tasks:
            if task.dependencies:
                graph.dependency_matrix[task.id] = set(task.dependencies)
            else:
                graph.dependency_matrix[task.id] = set()

            # Build reverse dependencies
            for dep_id in task.dependencies:
                if dep_id not in graph.reverse_deps:
                    graph.reverse_deps[dep_id] = set()
                graph.reverse_deps[dep_id].add(task.id)

        return graph

    @staticmethod
    def validate_graph(graph: Graph) -> Tuple[bool, List[str]]:
        """
        Validate graph integrity.

        Checks for:
        - Circular dependencies
        - Missing task references
        - Orphaned tasks

        Args:
            graph: Graph to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for missing task references
        for task_id, task in graph.task_map.items():
            for dep_id in task.dependencies:
                if dep_id not in graph.task_map:
                    errors.append(f"Task {task_id} depends on non-existent task {dep_id}")

            for sub_id in task.subtasks:
                if sub_id not in graph.task_map:
                    errors.append(f"Task {task_id} has non-existent subtask {sub_id}")

        # Check for circular dependencies
        cycles = DAGEngine.find_cycles(graph)
        if cycles:
            for cycle in cycles:
                errors.append(f"Circular dependency detected: {' -> '.join(cycle)} -> {cycle[0]}")

        # Check for orphaned tasks (not connected to any root or leaf)
        orphans = DAGEngine.find_orphaned_tasks(graph)
        if orphans:
            for orphan_id in orphans:
                errors.append(f"Orphaned task detected: {orphan_id}")

        return (len(errors) == 0, errors)

    @staticmethod
    def find_cycles(graph: Graph) -> List[List[str]]:
        """
        Detect circular dependencies using DFS.

        Args:
            graph: Graph to check

        Returns:
            List of cycles, where each cycle is a list of task IDs
        """
        cycles = []
        visited = set()
        rec_stack = set()
        parent = {}

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Check all tasks that depend on this node
            for dependent in graph.reverse_deps.get(node, set()):
                if dependent not in visited:
                    dfs(dependent, path[:])
                elif dependent in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(dependent)
                    cycle = path[cycle_start:] + [dependent]
                    cycles.append(cycle)

            rec_stack.remove(node)

        for task_id in graph.task_map:
            if task_id not in visited:
                dfs(task_id, [])

        return cycles

    @staticmethod
    def topological_sort(graph: Graph) -> List[str]:
        """
        Topologically sort tasks respecting dependencies.

        Uses Kahn's algorithm. Tasks with no dependencies come first.

        Args:
            graph: Graph to sort

        Returns:
            Sorted list of task IDs (dependencies before dependents)
        """
        in_degree = {task_id: len(graph.dependency_matrix.get(task_id, set()))
                     for task_id in graph.task_map}

        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        sorted_tasks = []

        while queue:
            task_id = queue.popleft()
            sorted_tasks.append(task_id)

            # Process all tasks that depend on this one
            for dependent_id in graph.reverse_deps.get(task_id, set()):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        return sorted_tasks

    @staticmethod
    def get_critical_path(graph: Graph) -> List[str]:
        """
        Find the longest dependency chain (critical path).

        This is the minimum time needed to complete all tasks.

        Args:
            graph: Graph to analyze

        Returns:
            List of task IDs representing critical path
        """
        # Use topological sort to process in order
        sorted_tasks = DAGEngine.topological_sort(graph)

        # Calculate longest path to each task
        longest_path = {}
        longest_path_to = {}

        for task_id in sorted_tasks:
            # Find longest path of predecessors
            predecessors = graph.dependency_matrix.get(task_id, set())

            if not predecessors:
                longest_path[task_id] = 0
                longest_path_to[task_id] = [task_id]
            else:
                max_pred_length = max((longest_path.get(pred_id, 0) for pred_id in predecessors), default=0)
                longest_path[task_id] = max_pred_length + 1
                max_pred = max(predecessors, key=lambda p: longest_path.get(p, 0))
                longest_path_to[task_id] = longest_path_to.get(max_pred, []) + [task_id]

        # Find task with longest path
        if longest_path:
            critical_task = max(longest_path, key=longest_path.get)
            return longest_path_to.get(critical_task, [critical_task])

        return []

    @staticmethod
    def get_blocking_tasks(task_id: str, graph: Graph) -> List[Task]:
        """
        Get all tasks that block the given task.

        Args:
            task_id: ID of task to check
            graph: Graph containing tasks

        Returns:
            List of Task objects that must complete first
        """
        blocking_ids = graph.dependency_matrix.get(task_id, set())
        return [graph.task_map[bid] for bid in blocking_ids if bid in graph.task_map]

    @staticmethod
    def get_dependent_tasks(task_id: str, graph: Graph) -> List[Task]:
        """
        Get all tasks that depend on the given task.

        Args:
            task_id: ID of task to check
            graph: Graph containing tasks

        Returns:
            List of Task objects that depend on this one
        """
        dependent_ids = graph.reverse_deps.get(task_id, set())
        return [graph.task_map[did] for did in dependent_ids if did in graph.task_map]

    @staticmethod
    def get_all_blockers_for_task(task_id: str, graph: Graph, recursive: bool = True) -> Set[str]:
        """
        Get all tasks that block the given task (transitively).

        Args:
            task_id: ID of task to check
            graph: Graph containing tasks
            recursive: If True, include indirect blockers

        Returns:
            Set of all blocking task IDs
        """
        if not recursive:
            return graph.dependency_matrix.get(task_id, set())

        blockers = set()
        visited = set()
        queue = deque([task_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            direct_blockers = graph.dependency_matrix.get(current, set())
            blockers.update(direct_blockers)

            for blocker_id in direct_blockers:
                if blocker_id not in visited:
                    queue.append(blocker_id)

        return blockers

    @staticmethod
    def get_all_dependents_for_task(task_id: str, graph: Graph, recursive: bool = True) -> Set[str]:
        """
        Get all tasks that depend on the given task (transitively).

        Args:
            task_id: ID of task to check
            graph: Graph containing tasks
            recursive: If True, include indirect dependents

        Returns:
            Set of all dependent task IDs
        """
        if not recursive:
            return graph.reverse_deps.get(task_id, set())

        dependents = set()
        visited = set()
        queue = deque([task_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            direct_dependents = graph.reverse_deps.get(current, set())
            dependents.update(direct_dependents)

            for dependent_id in direct_dependents:
                if dependent_id not in visited:
                    queue.append(dependent_id)

        return dependents

    @staticmethod
    def find_orphaned_tasks(graph: Graph) -> List[str]:
        """
        Find tasks that are disconnected from the task hierarchy.

        A task is orphaned if it has no path to any root task
        and no path to any leaf task.

        Args:
            graph: Graph to analyze

        Returns:
            List of orphaned task IDs
        """
        orphans = []
        roots = set(graph.root_tasks)
        leaves = set(graph.leaf_tasks)

        for task_id in graph.task_map:
            # Check if reachable from any root
            reachable_from_root = DAGEngine._is_reachable_from_roots(task_id, graph, roots)

            # Check if can reach any leaf
            can_reach_leaf = DAGEngine._can_reach_leaves(task_id, graph, leaves)

            if not reachable_from_root and not can_reach_leaf:
                orphans.append(task_id)

        return orphans

    @staticmethod
    def _is_reachable_from_roots(task_id: str, graph: Graph, roots: Set[str]) -> bool:
        """Check if task is reachable from any root task."""
        visited = set()
        queue = deque(roots)

        while queue:
            current = queue.popleft()
            if current == task_id:
                return True
            if current in visited:
                continue
            visited.add(current)

            for dependent in graph.reverse_deps.get(current, set()):
                queue.append(dependent)

        return False

    @staticmethod
    def _can_reach_leaves(task_id: str, graph: Graph, leaves: Set[str]) -> bool:
        """Check if task can reach any leaf task."""
        visited = set()
        queue = deque([task_id])

        while queue:
            current = queue.popleft()
            if current in leaves:
                return True
            if current in visited:
                continue
            visited.add(current)

            for dependent in graph.reverse_deps.get(current, set()):
                queue.append(dependent)

        return False

    @staticmethod
    def calculate_completion_percentage(task_id: str, graph: Graph) -> float:
        """
        Calculate completion percentage of a task including subtasks.

        Args:
            task_id: ID of task to check
            graph: Graph containing tasks

        Returns:
            Completion percentage (0.0 to 100.0)
        """
        task = graph.get_task(task_id)
        if not task:
            return 0.0

        if not task.subtasks:
            return 100.0 if task.status == TaskStatus.COMPLETED else task.completion_percentage

        subtasks = [graph.get_task(st) for st in task.subtasks]
        subtasks = [t for t in subtasks if t is not None]

        if not subtasks:
            return task.completion_percentage

        completed = sum(1 for st in subtasks if st.status == TaskStatus.COMPLETED)
        return (completed / len(subtasks)) * 100.0

    @staticmethod
    def get_task_depth(task_id: str, graph: Graph) -> int:
        """
        Get depth of a task in the dependency hierarchy.

        Root tasks have depth 0. Each level of dependency adds 1.

        Args:
            task_id: ID of task to check
            graph: Graph containing tasks

        Returns:
            Depth of task (0 for roots)
        """
        blockers = graph.dependency_matrix.get(task_id, set())
        if not blockers:
            return 0

        return 1 + max((DAGEngine.get_task_depth(bid, graph) for bid in blockers), default=0)

    @staticmethod
    def partition_into_phases(graph: Graph, max_tasks_per_phase: Optional[int] = None) -> List[List[str]]:
        """
        Partition tasks into execution phases.

        Each phase contains tasks that can execute in parallel (no inter-dependencies).

        Args:
            graph: Graph to partition
            max_tasks_per_phase: Optional limit on tasks per phase

        Returns:
            List of phases, where each phase is a list of task IDs
        """
        sorted_tasks = DAGEngine.topological_sort(graph)
        phases = []
        assigned = set()

        for task_id in sorted_tasks:
            # Find a phase where all blockers are already assigned
            task_blockers = graph.dependency_matrix.get(task_id, set())

            if not task_blockers or task_blockers.issubset(assigned):
                # Can be added to a new or existing phase
                # Try to add to last phase if no inter-dependencies within phase
                if phases:
                    last_phase = phases[-1]
                    if max_tasks_per_phase is None or len(last_phase) < max_tasks_per_phase:
                        # Check if task conflicts with others in phase
                        conflicts = False
                        for other_id in last_phase:
                            if (other_id in graph.dependency_matrix.get(task_id, set()) or
                                task_id in graph.dependency_matrix.get(other_id, set())):
                                conflicts = True
                                break

                        if not conflicts:
                            last_phase.append(task_id)
                            assigned.add(task_id)
                            continue

                # Create new phase
                phases.append([task_id])
                assigned.add(task_id)

        return phases
