#!/usr/bin/env python3
"""
Main TodoWrite hook handler for task conflict detection.

Intercepts TodoWrite operations before they execute, detects conflicts
with task dependencies, and returns allow/block/transform decisions.

This script reads JSON from stdin (hook input) and writes JSON to stdout (hook output).
"""

import sys
import json
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, '/home/mikeb/.claude/plugins/marketplaces/local/plugins/metamanager-plugin')

from core.models import Graph, Task, TaskStatus, Conflict
from core.dag_engine import DAGEngine
from core.conflict_rules import ConflictRules
from utils.supermemory_client import SupermemoryClient


class HookResponse:
    """Encapsulates hook response data."""

    def __init__(self):
        self.allow = True
        self.reason: Optional[str] = None
        self.conflicts: List[Dict[str, Any]] = []
        self.transform: Optional[Dict[str, Any]] = None
        self.system_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        output = {"allow": self.allow}
        if self.reason:
            output["reason"] = self.reason
        if self.conflicts:
            output["conflicts"] = self.conflicts
        if self.transform:
            output["transform"] = self.transform
        if self.system_message:
            output["systemMessage"] = self.system_message
        return output


def read_hook_input() -> Dict[str, Any]:
    """
    Read and parse hook input from stdin.

    Hook input contains:
    - tool_name: "TodoWrite"
    - todos: List of Todo objects with todos structure
    """
    try:
        input_data = json.load(sys.stdin)
        return input_data
    except json.JSONDecodeError as e:
        print(f"Error parsing hook input: {e}", file=sys.stderr)
        return {}


def write_hook_response(response: HookResponse) -> None:
    """Write hook response to stdout."""
    try:
        json.dump(response.to_dict(), sys.stdout)
        sys.stdout.flush()
    except Exception as e:
        print(f"Error writing hook response: {e}", file=sys.stderr)


def todos_to_graph(todos: List[Dict[str, Any]]) -> Graph:
    """
    Convert TodoWrite format to internal Graph format.

    Expects todos to be a list of todo objects with:
    - content: str
    - status: "pending" | "in_progress" | "completed"
    - activeForm: str (for display)
    """
    graph = Graph(graph_id="current")
    task_map = {}

    # First pass: create all tasks
    for i, todo in enumerate(todos):
        task_id = str(i)
        content = todo.get("content", "")

        # Try to infer parent-child relationships from naming
        # Simple heuristic: indented tasks are subtasks
        is_subtask = content.startswith("  ")
        clean_content = content.strip()

        task = Task(
            id=task_id,
            name=clean_content,
            status=TaskStatus(todo.get("status", "pending"))
        )

        task_map[task_id] = task
        graph.task_map[task_id] = task

    # Second pass: infer dependencies from hierarchical structure
    # This is a simplified approach - in real usage, TodoWrite would have explicit structure
    for task_id, task in task_map.items():
        graph.dependency_matrix[task_id] = set()

    return graph


def detect_conflicts_for_change(
    current_graph: Graph,
    new_todos: List[Dict[str, Any]],
    operation: str
) -> List[Conflict]:
    """
    Detect conflicts when applying TodoWrite changes.

    Args:
        current_graph: Current task graph
        new_todos: New todos being applied
        operation: Operation type ("add", "update", "complete", "delete")

    Returns:
        List of conflicts detected
    """
    conflicts = []

    if operation == "complete":
        # Check if completing task would leave subtasks incomplete
        for todo in new_todos:
            if todo.get("status") == "completed":
                content = todo.get("content", "")
                # This would require matching todos to graph tasks
                # In real implementation, would have explicit task IDs

    elif operation == "delete":
        # Check if deleting task would break dependencies
        pass

    elif operation == "update":
        # Check if updates would create circular dependencies
        pass

    return conflicts


def handle_todoupdate_hook(hook_input: Dict[str, Any]) -> HookResponse:
    """
    Main hook handler for TodoWrite operations.

    Args:
        hook_input: Input from hook system

    Returns:
        HookResponse indicating allow/block/transform decision
    """
    response = HookResponse()

    try:
        # Extract TodoWrite data
        tool_name = hook_input.get("tool", {}).get("name", "")
        if tool_name != "TodoWrite":
            # Not a TodoWrite call, allow it
            return response

        # Get the TodoWrite operation details
        tool_input = hook_input.get("tool_input", {})
        todos = tool_input.get("todos", [])

        if not todos:
            return response

        # Try to load current graph from Supermemory
        current_graph = SupermemoryClient.load_graph("current")
        if not current_graph:
            # No existing graph, this is first operation, allow
            return response

        # Validate new graph would be valid
        new_graph = todos_to_graph(todos)
        is_valid, errors = DAGEngine.validate_graph(new_graph)

        if not is_valid:
            response.allow = False
            response.reason = "Graph validation failed"
            response.conflicts = [
                {
                    "error": error,
                    "type": "validation_error"
                }
                for error in errors
            ]
            return response

        # Detect conflicts
        conflicts = ConflictRules.detect_all_conflicts(new_graph)

        if conflicts:
            # Convert conflicts to JSON
            conflict_dicts = []
            critical_conflicts = []

            for conflict in conflicts:
                conflict_dict = {
                    "id": conflict.conflict_id,
                    "type": conflict.conflict_type.value,
                    "task_id": conflict.task_id,
                    "description": conflict.description,
                    "severity": conflict.severity.value,
                    "suggested_resolution": conflict.suggested_resolution
                }
                conflict_dicts.append(conflict_dict)

                # Critical conflicts block the operation
                if conflict.severity.value == "critical":
                    critical_conflicts.append(conflict)

            response.conflicts = conflict_dicts

            # Block on critical conflicts
            if critical_conflicts:
                response.allow = False
                response.reason = f"Critical conflicts detected: {len(critical_conflicts)} issue(s)"
                return response

        # Allow the operation
        response.system_message = "TodoWrite operation validated successfully"
        return response

    except Exception as e:
        # Log error but don't block (fail safe)
        response.allow = True
        response.system_message = f"Hook validation skipped due to error: {str(e)}"
        return response


def main():
    """Main entry point for hook script."""
    try:
        # Read input from OpenCode hook system
        hook_input = read_hook_input()

        # Process the hook
        response = handle_todoupdate_hook(hook_input)

        # Write response back to OpenCode
        write_hook_response(response)

        # Exit with appropriate code
        if not response.allow:
            # Signal block with exit code 2
            sys.exit(2)
        else:
            # Allow operation
            sys.exit(0)

    except Exception as e:
        # Unhandled error - allow operation but log
        error_response = HookResponse()
        error_response.allow = True
        error_response.system_message = f"Hook error (allowing operation): {str(e)}"
        write_hook_response(error_response)
        sys.exit(0)


if __name__ == "__main__":
    main()
