"""
Execution planning engine for task graphs.

Generates optimal execution plans considering:
- Dependency constraints
- Parallelizable groups
- Critical path analysis
- Phase sequencing
"""

from datetime import timedelta
from typing import List, Set, Optional, Dict

from .models import Graph, ExecutionPlan, Phase, Task
from .dag_engine import DAGEngine


class PlanningEngine:
    """Engine for creating execution plans from task graphs."""

    @staticmethod
    def create_execution_plan(
        graph: Graph,
        plan_id: str = "",
        created_by: str = "planner-agent"
    ) -> ExecutionPlan:
        """
        Create an optimal execution plan for a task graph.

        Args:
            graph: Task graph to plan
            plan_id: Optional identifier for the plan
            created_by: Name of agent/user creating plan

        Returns:
            ExecutionPlan with phases, critical path, and parallelizable groups
        """
        plan = ExecutionPlan(
            plan_id=plan_id or f"plan-{id(graph)}",
            graph_id=graph.graph_id,
            created_by=created_by
        )

        # Calculate critical path
        plan.critical_path = DAGEngine.get_critical_path(graph)

        # Partition into phases
        phase_lists = PlanningEngine.calculate_phases(graph)
        plan.phases = [
            Phase(
                phase_number=i + 1,
                tasks=phase_tasks,
                can_parallelize=True,
                blocked_by_phases=list(range(1, i + 1)) if i > 0 else []
            )
            for i, phase_tasks in enumerate(phase_lists)
        ]

        # Identify parallelizable groups
        plan.parallelizable_groups = PlanningEngine.identify_parallel_groups(graph)

        # Estimate duration
        plan.estimated_duration = PlanningEngine.estimate_duration(graph, plan.phases)

        return plan

    @staticmethod
    def calculate_phases(graph: Graph) -> List[List[str]]:
        """
        Partition tasks into execution phases.

        Each phase contains tasks that can execute in parallel
        (no direct inter-dependencies within phase).

        Args:
            graph: Graph to partition

        Returns:
            List of phases, each phase is a list of task IDs
        """
        sorted_tasks = DAGEngine.topological_sort(graph)
        phases = []
        assigned = set()

        for task_id in sorted_tasks:
            task_blockers = graph.dependency_matrix.get(task_id, set())

            # Can add to existing or new phase if all blockers assigned
            if task_blockers.issubset(assigned):
                # Try to add to last phase
                if phases:
                    last_phase = phases[-1]

                    # Check for conflicts with others in same phase
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
            else:
                # Create new phase when blockers not yet assigned
                phases.append([task_id])
                assigned.add(task_id)

        return phases

    @staticmethod
    def identify_parallel_groups(graph: Graph) -> List[List[str]]:
        """
        Identify groups of tasks that can execute in parallel.

        Two tasks can run in parallel if neither blocks the other
        (directly or transitively).

        Args:
            graph: Graph to analyze

        Returns:
            List of parallel groups (each group is a list of task IDs)
        """
        parallel_groups = []
        assigned = set()

        for task_id in graph.task_map:
            if task_id in assigned:
                continue

            # Start a new group with this task
            group = [task_id]
            assigned.add(task_id)

            # Try to add other unassigned tasks to this group
            for other_id in graph.task_map:
                if other_id in assigned or other_id == task_id:
                    continue

                # Check if can run in parallel
                can_parallel = True

                # Check direct and transitive blocking relationships
                task_blockers = DAGEngine.get_all_blockers_for_task(task_id, graph)
                other_blockers = DAGEngine.get_all_blockers_for_task(other_id, graph)
                task_dependents = DAGEngine.get_all_dependents_for_task(task_id, graph)
                other_dependents = DAGEngine.get_all_dependents_for_task(other_id, graph)

                # They conflict if either blocks the other
                if other_id in task_blockers or other_id in task_dependents:
                    can_parallel = False
                elif task_id in other_blockers or task_id in other_dependents:
                    can_parallel = False

                if can_parallel:
                    # Also check against others in group
                    for group_task in group:
                        if (group_task in other_blockers or group_task in other_dependents or
                            other_id in DAGEngine.get_all_blockers_for_task(group_task, graph) or
                            other_id in DAGEngine.get_all_dependents_for_task(group_task, graph)):
                            can_parallel = False
                            break

                if can_parallel:
                    group.append(other_id)
                    assigned.add(other_id)

            if group:
                parallel_groups.append(group)

        return parallel_groups

    @staticmethod
    def estimate_duration(graph: Graph, phases: Optional[List[Phase]] = None) -> Optional[timedelta]:
        """
        Estimate total execution time for the task graph.

        Sums phase durations (accounting for parallelization) and adds overhead.

        Args:
            graph: Graph to estimate
            phases: Pre-calculated phases (optional)

        Returns:
            Estimated total duration
        """
        if not phases:
            return None

        total_duration = timedelta()

        for phase in phases:
            # Find max duration in phase (parallel tasks)
            phase_max_duration = timedelta()

            for task_id in phase.tasks:
                task = graph.get_task(task_id)
                if task and task.estimated_duration:
                    phase_max_duration = max(phase_max_duration, task.estimated_duration)

            total_duration += phase_max_duration

        # Add 10% overhead for synchronization and coordination
        overhead = total_duration * 0.1
        return total_duration + overhead

    @staticmethod
    def optimize_plan(plan: ExecutionPlan, graph: Graph) -> ExecutionPlan:
        """
        Optimize an execution plan.

        Applies optimizations such as:
        - Reordering within phases
        - Merging compatible phases
        - Identifying bottlenecks

        Args:
            plan: Plan to optimize
            graph: Graph being executed

        Returns:
            Optimized plan
        """
        # Remove empty phases
        plan.phases = [p for p in plan.phases if p.tasks]

        # Reorder tasks within phases by priority
        for phase in plan.phases:
            phase.tasks.sort(
                key=lambda tid: (
                    # Sort by: priority (higher first), then depth (deeper first)
                    -graph.get_task(tid).priority.value if graph.get_task(tid) else 0,
                    -DAGEngine.get_task_depth(tid, graph)
                )
            )

        # Mark phases that can be merged
        PlanningEngine._mark_mergeable_phases(plan, graph)

        return plan

    @staticmethod
    def _mark_mergeable_phases(plan: ExecutionPlan, graph: Graph) -> None:
        """
        Identify phases that could be merged without conflicts.

        Phases can be merged if no task in one blocks any in the other.
        """
        for i in range(len(plan.phases) - 1):
            phase1 = plan.phases[i]
            phase2 = plan.phases[i + 1]

            can_merge = True
            for task1_id in phase1.tasks:
                for task2_id in phase2.tasks:
                    if (task2_id in graph.dependency_matrix.get(task1_id, set()) or
                        task1_id in graph.dependency_matrix.get(task2_id, set())):
                        can_merge = False
                        break
                if not can_merge:
                    break

            # Could add metadata flag if mergeable
            if can_merge:
                phase2.metadata["can_merge_with_previous"] = True

    @staticmethod
    def get_critical_tasks(plan: ExecutionPlan) -> List[str]:
        """
        Get tasks on the critical path.

        These tasks have the highest impact on total execution time.

        Args:
            plan: Execution plan

        Returns:
            List of critical task IDs
        """
        return plan.critical_path

    @staticmethod
    def identify_bottlenecks(plan: ExecutionPlan, graph: Graph) -> List[Dict[str, any]]:
        """
        Identify bottleneck tasks that limit parallelization.

        Args:
            plan: Execution plan
            graph: Graph being executed

        Returns:
            List of bottleneck analysis objects
        """
        bottlenecks = []

        for task_id in plan.critical_path:
            task = graph.get_task(task_id)
            if not task:
                continue

            dependents = DAGEngine.get_all_dependents_for_task(task_id, graph)

            if len(dependents) > 2:  # Multiple downstream tasks
                bottlenecks.append({
                    "task_id": task_id,
                    "task_name": task.name,
                    "affected_tasks": list(dependents),
                    "impact": len(dependents),
                    "recommendation": "Consider breaking this task into smaller parallel subtasks"
                })

        return bottlenecks

    @staticmethod
    def get_next_phase(plan: ExecutionPlan, completed_phase: int) -> Optional[Phase]:
        """
        Get the next executable phase after a phase completes.

        Args:
            plan: Execution plan
            completed_phase: Phase number that just completed (1-indexed)

        Returns:
            Next phase or None if all complete
        """
        next_phase_num = completed_phase + 1
        if next_phase_num <= len(plan.phases):
            return plan.phases[next_phase_num - 1]
        return None

    @staticmethod
    def calculate_completion_progress(
        plan: ExecutionPlan,
        completed_tasks: Set[str],
        graph: Graph
    ) -> float:
        """
        Calculate overall completion progress for execution plan.

        Args:
            plan: Execution plan
            completed_tasks: Set of completed task IDs
            graph: Graph being executed

        Returns:
            Completion percentage (0.0 to 100.0)
        """
        if not graph.task_map:
            return 0.0

        total_tasks = len(graph.task_map)
        completed_count = len(completed_tasks)

        return (completed_count / total_tasks) * 100.0 if total_tasks > 0 else 0.0

    @staticmethod
    def get_phase_completion(phase: Phase, graph: Graph) -> float:
        """
        Calculate completion percentage for a single phase.

        Args:
            phase: Phase to check
            graph: Graph containing tasks

        Returns:
            Completion percentage (0.0 to 100.0)
        """
        if not phase.tasks:
            return 0.0

        completed = sum(
            1 for task_id in phase.tasks
            if graph.get_task(task_id) and graph.get_task(task_id).status.value == "completed"
        )

        return (completed / len(phase.tasks)) * 100.0 if phase.tasks else 0.0
