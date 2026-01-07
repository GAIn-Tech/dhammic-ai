"""
Supermemory integration for persistent task graph and plan storage.

Handles saving and retrieving task graphs, execution plans, and task history
from Supermemory for cross-session persistence.
"""

import json
from datetime import datetime
from typing import Dict, Optional, List, Any

from core.models import Graph, Task, ExecutionPlan, TaskLineage, MetamanagerState


class SupermemoryClient:
    """Interface to Supermemory for persisting metamanager state."""

    CONTAINER_TAG = "metamanager-plugin"

    @staticmethod
    def save_graph(graph: Graph, project_id: str = "") -> bool:
        """
        Save a task graph to Supermemory.

        Args:
            graph: Graph to save
            project_id: Optional project identifier

        Returns:
            True if successful
        """
        try:
            graph_data = SupermemoryClient._serialize_graph(graph)

            memory_content = json.dumps({
                "type": "task-graph",
                "graph_id": graph.graph_id,
                "project_id": project_id,
                "data": graph_data,
                "timestamp": datetime.now().isoformat(),
                "size": len(graph.task_map)
            })

            # In actual implementation, would call supermemory_memory() function
            # For now, this is a template showing the structure
            SupermemoryClient._persist_memory(memory_content, f"graph:{graph.graph_id}")

            return True
        except Exception as e:
            print(f"Error saving graph to Supermemory: {e}")
            return False

    @staticmethod
    def load_graph(graph_id: str) -> Optional[Graph]:
        """
        Load a task graph from Supermemory.

        Args:
            graph_id: ID of graph to load

        Returns:
            Deserialized Graph or None if not found
        """
        try:
            memory_content = SupermemoryClient._retrieve_memory(f"graph:{graph_id}")
            if not memory_content:
                return None

            data = json.loads(memory_content)
            return SupermemoryClient._deserialize_graph(data.get("data", {}))
        except Exception as e:
            print(f"Error loading graph from Supermemory: {e}")
            return None

    @staticmethod
    def save_execution_plan(plan: ExecutionPlan) -> bool:
        """
        Save an execution plan to Supermemory.

        Args:
            plan: Plan to save

        Returns:
            True if successful
        """
        try:
            plan_data = SupermemoryClient._serialize_plan(plan)

            memory_content = json.dumps({
                "type": "execution-plan",
                "plan_id": plan.plan_id,
                "graph_id": plan.graph_id,
                "data": plan_data,
                "timestamp": datetime.now().isoformat(),
                "phases": len(plan.phases)
            })

            SupermemoryClient._persist_memory(memory_content, f"plan:{plan.plan_id}")
            return True
        except Exception as e:
            print(f"Error saving plan to Supermemory: {e}")
            return False

    @staticmethod
    def load_execution_plan(plan_id: str) -> Optional[ExecutionPlan]:
        """
        Load an execution plan from Supermemory.

        Args:
            plan_id: ID of plan to load

        Returns:
            Deserialized ExecutionPlan or None if not found
        """
        try:
            memory_content = SupermemoryClient._retrieve_memory(f"plan:{plan_id}")
            if not memory_content:
                return None

            data = json.loads(memory_content)
            return SupermemoryClient._deserialize_plan(data.get("data", {}))
        except Exception as e:
            print(f"Error loading plan from Supermemory: {e}")
            return None

    @staticmethod
    def save_task_lineage(lineage: TaskLineage) -> bool:
        """
        Save task history/lineage to Supermemory.

        Args:
            lineage: Task lineage to save

        Returns:
            True if successful
        """
        try:
            lineage_data = SupermemoryClient._serialize_lineage(lineage)

            memory_content = json.dumps({
                "type": "task-lineage",
                "task_id": lineage.task_id,
                "data": lineage_data,
                "timestamp": datetime.now().isoformat(),
                "entries": len(lineage.entries)
            })

            SupermemoryClient._persist_memory(memory_content, f"lineage:{lineage.task_id}")
            return True
        except Exception as e:
            print(f"Error saving lineage to Supermemory: {e}")
            return False

    @staticmethod
    def get_task_history(task_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve complete history of a task.

        Args:
            task_id: ID of task

        Returns:
            List of history entries
        """
        try:
            memory_content = SupermemoryClient._retrieve_memory(f"lineage:{task_id}")
            if not memory_content:
                return []

            data = json.loads(memory_content)
            return data.get("data", {}).get("entries", [])
        except Exception as e:
            print(f"Error retrieving task history: {e}")
            return []

    @staticmethod
    def save_conflicts(graph_id: str, conflicts: List[Dict[str, Any]]) -> bool:
        """
        Save detected conflicts to Supermemory.

        Args:
            graph_id: Graph ID associated with conflicts
            conflicts: List of conflict objects

        Returns:
            True if successful
        """
        try:
            memory_content = json.dumps({
                "type": "conflict-report",
                "graph_id": graph_id,
                "data": {
                    "conflicts": conflicts,
                    "timestamp": datetime.now().isoformat(),
                    "count": len(conflicts)
                }
            })

            SupermemoryClient._persist_memory(memory_content, f"conflicts:{graph_id}")
            return True
        except Exception as e:
            print(f"Error saving conflicts to Supermemory: {e}")
            return False

    @staticmethod
    def _serialize_graph(graph: Graph) -> Dict[str, Any]:
        """Serialize a Graph object to JSON-compatible dictionary."""
        return {
            "graph_id": graph.graph_id,
            "created_at": graph.created_at.isoformat(),
            "updated_at": graph.updated_at.isoformat(),
            "tasks": {
                task_id: {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status.value,
                    "task_type": task.task_type.value,
                    "priority": task.priority.value,
                    "completion_percentage": task.completion_percentage,
                    "subtasks": task.subtasks,
                    "dependencies": task.dependencies,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                    "owner": task.owner,
                    "estimated_duration": task.estimated_duration.total_seconds() if task.estimated_duration else None,
                    "metadata": task.metadata
                }
                for task_id, task in graph.task_map.items()
            },
            "metadata": graph.metadata
        }

    @staticmethod
    def _deserialize_graph(data: Dict[str, Any]) -> Graph:
        """Deserialize a Graph object from JSON-compatible dictionary."""
        graph = Graph(graph_id=data.get("graph_id", ""))

        for task_id, task_data in data.get("tasks", {}).items():
            task = Task(
                id=task_data["id"],
                name=task_data["name"],
                status=task_data["status"],
                task_type=task_data.get("task_type", "other"),
                priority=task_data.get("priority", "medium"),
                completion_percentage=task_data.get("completion_percentage", 0.0),
                subtasks=task_data.get("subtasks", []),
                dependencies=task_data.get("dependencies", []),
                owner=task_data.get("owner"),
                metadata=task_data.get("metadata", {})
            )
            graph.task_map[task_id] = task

        # Rebuild dependency matrices
        for task_id, task in graph.task_map.items():
            if task.dependencies:
                graph.dependency_matrix[task_id] = set(task.dependencies)
            else:
                graph.dependency_matrix[task_id] = set()

            for dep_id in task.dependencies:
                if dep_id not in graph.reverse_deps:
                    graph.reverse_deps[dep_id] = set()
                graph.reverse_deps[dep_id].add(task_id)

        return graph

    @staticmethod
    def _serialize_plan(plan: ExecutionPlan) -> Dict[str, Any]:
        """Serialize an ExecutionPlan to JSON-compatible dictionary."""
        return {
            "plan_id": plan.plan_id,
            "graph_id": plan.graph_id,
            "created_at": plan.created_at.isoformat(),
            "created_by": plan.created_by,
            "version": plan.version,
            "phases": [
                {
                    "phase_number": phase.phase_number,
                    "tasks": phase.tasks,
                    "can_parallelize": phase.can_parallelize,
                    "blocked_by_phases": phase.blocked_by_phases,
                    "estimated_duration": phase.estimated_duration.total_seconds() if phase.estimated_duration else None
                }
                for phase in plan.phases
            ],
            "critical_path": plan.critical_path,
            "parallelizable_groups": plan.parallelizable_groups,
            "estimated_duration": plan.estimated_duration.total_seconds() if plan.estimated_duration else None,
            "metadata": plan.metadata
        }

    @staticmethod
    def _deserialize_plan(data: Dict[str, Any]) -> ExecutionPlan:
        """Deserialize an ExecutionPlan from JSON-compatible dictionary."""
        plan = ExecutionPlan(
            plan_id=data.get("plan_id", ""),
            graph_id=data.get("graph_id", ""),
            created_by=data.get("created_by", "")
        )

        plan.phases = [
            Phase(
                phase_number=phase_data["phase_number"],
                tasks=phase_data.get("tasks", []),
                can_parallelize=phase_data.get("can_parallelize", True),
                blocked_by_phases=phase_data.get("blocked_by_phases", [])
            )
            for phase_data in data.get("phases", [])
        ]

        plan.critical_path = data.get("critical_path", [])
        plan.parallelizable_groups = data.get("parallelizable_groups", [])

        return plan

    @staticmethod
    def _serialize_lineage(lineage: TaskLineage) -> Dict[str, Any]:
        """Serialize a TaskLineage to JSON-compatible dictionary."""
        return {
            "task_id": lineage.task_id,
            "created_at": lineage.created_at.isoformat(),
            "entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "action": entry.action,
                    "old_value": entry.old_value,
                    "new_value": entry.new_value,
                    "actor": entry.actor,
                    "reason": entry.reason,
                    "metadata": entry.metadata
                }
                for entry in lineage.entries
            ]
        }

    @staticmethod
    def _persist_memory(content: str, memory_key: str) -> None:
        """
        Persist content to Supermemory.

        In actual implementation, this would call the supermemory_memory tool.
        For now, this is a template.
        """
        # This would be replaced with actual Supermemory API call:
        # supermemory_memory(
        #     content=content,
        #     action="save",
        #     containerTag=SupermemoryClient.CONTAINER_TAG
        # )
        pass

    @staticmethod
    def _retrieve_memory(memory_key: str) -> Optional[str]:
        """
        Retrieve content from Supermemory.

        In actual implementation, this would call the supermemory_recall tool.
        For now, this returns None (not found).
        """
        # This would be replaced with actual Supermemory API call:
        # return supermemory_recall(
        #     query=memory_key,
        #     containerTag=SupermemoryClient.CONTAINER_TAG
        # )
        return None
