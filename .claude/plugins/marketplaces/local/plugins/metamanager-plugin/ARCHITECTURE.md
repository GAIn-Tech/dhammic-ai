# Metamanager Architecture Guide

Complete technical documentation of the Metamanager plugin system architecture, design patterns, and implementation details.

## System Overview

Metamanager is a layered system with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│         OpenCode CLI (Claude Code)                  │
├─────────────────────────────────────────────────────┤
│  User Commands & Interactions                       │
│  ├─ /metamanager:analyze   ├─ /metamanager:plan    │
│  ├─ /metamanager:visualize ├─ /metamanager:resolve │
│  └─ /metamanager:delegate                           │
├─────────────────────────────────────────────────────┤
│  Agents (Orchestrator, Planner, Executor)          │
│  ├─ Orchestrator (coordination layer)               │
│  ├─ Planner (planning layer)                        │
│  └─ Executor (execution layer)                      │
├─────────────────────────────────────────────────────┤
│  Skills (Auto-invoked knowledge)                    │
│  ├─ dependency-analysis                             │
│  ├─ conflict-resolution                             │
│  └─ execution-planning                              │
├─────────────────────────────────────────────────────┤
│  Hook System (Event handlers)                       │
│  └─ PreToolUse: TodoWrite monitoring               │
├─────────────────────────────────────────────────────┤
│  Engine Layer (Core algorithms)                     │
│  ├─ DAG Engine (graph operations)                   │
│  ├─ Conflict Rules (detection)                      │
│  ├─ Planning Engine (optimization)                  │
│  └─ Models (data structures)                        │
├─────────────────────────────────────────────────────┤
│  Persistence Layer                                  │
│  └─ Supermemory Client (cross-session storage)     │
├─────────────────────────────────────────────────────┤
│  Supermemory (Persistent Memory)                    │
└─────────────────────────────────────────────────────┘
```

## Layer Descriptions

### 1. Command Interface Layer

Five user-facing commands provide different perspectives on the task graph:

**`/metamanager:analyze`**
- Provides status report
- Identifies conflicts
- Recommendations
- Export: text, JSON, Mermaid

**`/metamanager:plan`**
- Creates execution plan
- Analyzes critical path
- Identifies parallelization
- Export: text, JSON

**`/metamanager:visualize`**
- Task graph visualization
- Dependency structure
- Phase organization
- Export: text (ASCII), Mermaid, JSON

**`/metamanager:resolve`**
- Conflict detection
- Interactive resolution
- Auto-resolution when possible
- Impact analysis

**`/metamanager:delegate`**
- Task assignment to agents
- Readiness verification
- Parallel execution
- Progress tracking

### 2. Agent Layer

Three specialized autonomous agents handle different aspects:

**Orchestrator Agent**
- Responsibilities:
  - Overall coordination
  - Progress monitoring
  - Conflict escalation
  - Dynamic replanning
  - State management

- Invokes:
  - Planner (for new phase planning)
  - Executor (for task delegation)

- Triggers: After TodoWrite changes, periodically

**Planner Agent**
- Responsibilities:
  - Graph analysis
  - Phase generation
  - Critical path calculation
  - Parallelization identification
  - Plan optimization

- Produces: ExecutionPlan objects
- Stores: Plans in Supermemory
- Triggers: When new phase needed

**Executor Agent**
- Responsibilities:
  - Task type classification
  - Agent selection
  - Task delegation
  - Progress tracking
  - Completion validation

- Spawns: Specialized agents per task
- Updates: Task status
- Reports: Back to Orchestrator

### 3. Skills Layer

Auto-invoked knowledge patterns that enhance Claude's capability:

**dependency-analysis** Skill
- When invoked: When discussing dependencies, blockers, prerequisites
- What it teaches:
  - How to trace dependency chains
  - Analyze transitive relationships
  - Calculate impact of tasks
  - Identify critical paths

**conflict-resolution** Skill
- When invoked: When resolving dependency issues
- What it teaches:
  - Conflict types and symptoms
  - Analysis approaches
  - Resolution strategies
  - Verification methods

**execution-planning** Skill
- When invoked: When creating execution plans
- What it teaches:
  - Planning principles
  - Phase generation algorithms
  - Parallelization strategies
  - Bottleneck analysis

### 4. Hook System Layer

Intercepts and monitors key operations:

**PreToolUse Hook** (todoupdate.py)
- Triggered: Before TodoWrite executes
- Input: TodoWrite operation details
- Processing:
  1. Parse input
  2. Load current graph
  3. Validate new graph
  4. Run conflict detection
  5. Return decision
- Output: allow/block/transform decision
- Timeout: 10 seconds

Hook Response Codes:
- Exit 0: Allow (proceed normally)
- Exit 1: Allow with warning
- Exit 2: Block (prevent operation)

### 5. Engine Layer

Core computational engine with three main components:

**DAG Engine** (`core/dag_engine.py`)
- Graph construction from tasks
- Validation (cycle detection, orphan finding)
- Operations:
  - Topological sort
  - Critical path finding
  - Reachability analysis
  - Depth calculation
  - Phase partitioning
- Time Complexity:
  - Build graph: O(T + E) where T=tasks, E=edges
  - Topological sort: O(T + E)
  - Find cycles: O(T + E)

**Conflict Rules Engine** (`core/conflict_rules.py`)
- Rule-based conflict detection
- Six conflict types:
  1. Incomplete Subtasks
  2. Incomplete Dependencies
  3. Orphaned Dependents
  4. Orphaned Tasks
  5. Circular Dependencies
  6. Abandoned Tasks
- Validation functions
- Impact analysis

**Planning Engine** (`core/planning_engine.py`)
- Algorithms:
  - Phase generation (topological + grouping)
  - Parallel group identification
  - Critical path calculation
  - Duration estimation
  - Bottleneck detection
  - Plan optimization
- Time Complexity:
  - Phase generation: O(T + E)
  - Critical path: O(T + E)
  - Parallelization: O(T²) worst case

### 6. Persistence Layer

**Supermemory Client** (`utils/supermemory_client.py`)
- Serialization:
  - Task → JSON
  - Graph → JSON
  - Plan → JSON
  - Lineage → JSON
- Deserialization (reverse)
- Schema validation
- Container tagging for project isolation
- Methods:
  - `save_graph(graph)`: Store graph
  - `load_graph(graph_id)`: Retrieve graph
  - `save_execution_plan(plan)`: Store plan
  - `get_task_history(task_id)`: Retrieve lineage

## Data Structures

### Core Models (`core/models.py`)

```python
Task
├─ id: str
├─ name: str
├─ status: TaskStatus (pending, in_progress, completed, blocked, abandoned, timed_out)
├─ dependencies: List[str]      # Blocking task IDs
├─ subtasks: List[str]          # Child task IDs
├─ completion_percentage: float
├─ metadata: Dict[str, Any]
└─ timestamps: created_at, updated_at

Graph
├─ task_map: Dict[str, Task]
├─ dependency_matrix: Dict[str, Set[str]]
├─ reverse_deps: Dict[str, Set[str]]
├─ root_tasks: List[str]        # Computed
├─ leaf_tasks: List[str]        # Computed
└─ completed_tasks: List[str]   # Computed

ExecutionPlan
├─ plan_id: str
├─ graph_id: str
├─ phases: List[Phase]
├─ critical_path: List[str]
├─ parallelizable_groups: List[List[str]]
├─ estimated_duration: timedelta
└─ metadata: Dict[str, Any]

Conflict
├─ conflict_id: str
├─ conflict_type: ConflictType
├─ task_id: str
├─ description: str
├─ severity: ConflictSeverity
├─ suggested_resolution: str
└─ auto_resolvable: bool
```

## Algorithms

### 1. Topological Sorting (Kahn's Algorithm)

Ensures tasks ordered respecting dependencies:

```
1. Calculate in-degree (dependencies count) for each task
2. Queue all tasks with in-degree 0 (no dependencies)
3. While queue not empty:
   a. Dequeue task, add to result
   b. For each dependent task:
      - Decrement in-degree
      - If in-degree becomes 0, enqueue
4. Return result (or error if cycles detected)

Time: O(T + E)
Space: O(T)
```

### 2. Critical Path Calculation

Finds longest dependency chain:

```
1. Topologically sort tasks
2. For each task in order:
   a. Find longest path among predecessors
   b. This task's longest path = max(predecessor paths) + 1
3. Find task with longest path
4. Backtrack to reconstruct path

Time: O(T + E)
Space: O(T)
```

### 3. Cycle Detection (DFS)

Detects circular dependencies:

```
1. For each unvisited task:
   a. DFS from task
   b. Track recursion stack
   c. If revisit task in stack → cycle found
   d. Record cycle path
2. Return all cycles

Time: O(T + E)
Space: O(T)
```

### 4. Phase Generation

Partitions tasks into execution phases:

```
1. Topologically sort all tasks
2. For each task in order:
   a. Check if all dependencies assigned
   b. Try to add to last phase (if no conflicts)
   c. If conflicts, create new phase
3. Return phase list

Time: O(T² + E) worst case
Space: O(T)
```

### 5. Parallel Group Identification

Finds independent task groups:

```
1. Initialize: unassigned = all tasks
2. While unassigned tasks exist:
   a. Start new group with first unassigned
   b. For each other unassigned task:
      - Check no blocking relationships
      - Check no transitive dependencies
      - Add to group if compatible
   c. Mark group tasks as assigned
3. Return groups

Time: O(T² + E)
Space: O(T)
```

## State Management

### Graph State Lifecycle

```
┌─ Created (new graph, no tasks)
│
├─ Building (tasks being added)
│  ├─ Valid: no cycles, all refs valid
│  └─ Invalid: cycles, orphans, etc.
│
├─ Loaded (from Supermemory)
│  └─ Validated on load
│
├─ Modified (after TodoWrite)
│  ├─ Hook validation
│  └─ Conflict detection
│
└─ Saved (to Supermemory)
   └─ Serialized for persistence
```

### Plan State Lifecycle

```
┌─ Generated (created from graph)
│  └─ Validated
│
├─ Optimized (improvements applied)
│  ├─ Phase merging
│  ├─ Task reordering
│  └─ Parallelization suggested
│
├─ Active (being executed)
│  ├─ Phases executing
│  ├─ Tasks completing
│  └─ Progress tracked
│
└─ Saved (to Supermemory)
   └─ For recovery/reference
```

## Serialization Schema

### Graph Serialization

```json
{
  "graph_id": "current",
  "created_at": "2024-01-06T...",
  "updated_at": "2024-01-06T...",
  "tasks": {
    "task-1": {
      "id": "task-1",
      "name": "Task Name",
      "status": "pending",
      "task_type": "feature",
      "priority": "high",
      "completion_percentage": 0,
      "subtasks": [],
      "dependencies": [],
      "metadata": {...}
    },
    ...
  },
  "metadata": {...}
}
```

### Plan Serialization

```json
{
  "plan_id": "plan-xxxxx",
  "graph_id": "current",
  "created_at": "2024-01-06T...",
  "created_by": "planner-agent",
  "phases": [
    {
      "phase_number": 1,
      "tasks": ["task-1", "task-2"],
      "can_parallelize": true,
      "blocked_by_phases": [],
      "estimated_duration": 3600
    },
    ...
  ],
  "critical_path": ["task-1", "task-3"],
  "estimated_duration": 10800,
  "metadata": {...}
}
```

## Error Handling

### Hook Error Handling

```
If hook validation fails:
  ├─ LOG error details
  ├─ ALLOW operation (fail-safe)
  ├─ WARN user
  └─ CONTINUE normally

If hook times out:
  ├─ LOG timeout
  ├─ ALLOW operation (circuit break)
  └─ WARN user
```

### Graph Error Handling

```
If graph invalid:
  ├─ Detect issue (cycles, refs, orphans)
  ├─ Generate error report
  ├─ Don't proceed with planning
  └─ WARN user to fix issues

If parsing fails:
  ├─ LOG parse error
  ├─ Fall back to last valid
  └─ ALERT user
```

## Performance Considerations

### Time Complexity Analysis

| Operation | Complexity | Note |
|-----------|-----------|------|
| Build graph | O(T + E) | T=tasks, E=edges |
| Validate graph | O(T + E) | Cycles, refs, orphans |
| Topological sort | O(T + E) | Standard algorithm |
| Critical path | O(T + E) | Single pass after sort |
| Detect cycles | O(T + E) | DFS-based |
| Find orphans | O(T + E) | Reachability check |
| Phase generation | O(T² + E) | Worst case |
| Parallelization | O(T² + E) | Worst case O(T²) comparisons |

### Space Complexity

| Structure | Complexity | Note |
|-----------|-----------|------|
| Task map | O(T) | One entry per task |
| Dependency matrix | O(T + E) | Sets of edges |
| Reverse dependencies | O(T + E) | Reverse mapping |
| Phases | O(T) | Total tasks |
| Plans | O(P × T) | P plans, T tasks each |

### Optimization Strategies

1. **Lazy Evaluation**: Don't compute until needed
2. **Caching**: Store expensive computations
3. **Early Termination**: Stop when answer found
4. **Algorithm Selection**: Choose best for graph size
5. **Incremental Updates**: Update only what changed

## Testing Strategy

### Unit Tests

- `test_dag_engine.py`: Graph operations
- `test_conflict_rules.py`: Conflict detection
- `test_planning_engine.py`: Planning algorithms
- `test_models.py`: Data structure integrity

### Integration Tests

- Hook → Graph → Plan flow
- Conflict detection → Resolution flow
- Agent coordination
- Supermemory persistence

### Scenario Tests

- Simple linear chains
- Diamond dependencies
- Complex networks
- Large graphs (1000+ tasks)
- Pathological cases (cycles, orphans)

## Security Considerations

1. **Input Validation**: Validate all TodoWrite input
2. **Injection Prevention**: No code execution from tasks
3. **Resource Limits**: Guard against runaway graphs
4. **Access Control**: Supermemory containerTag isolation
5. **Audit Logging**: Track all modifications

## Future Enhancements

### Short Term (v1.1)
- Performance optimization for large graphs
- Better error messages
- Caching layer
- Batch operations

### Medium Term (v1.2)
- ML-based optimization
- Resource allocation
- Time tracking
- Team notifications

### Long Term (v2.0)
- Integration with external tools
- Advanced scheduling
- Predictive analytics
- Multi-project coordination

## References

- **Algorithms**: CLRS Introduction to Algorithms
- **DAGs**: Cormen's topological sort
- **Critical Path**: Project management theory
- **Dependency Analysis**: Software architecture patterns
