---
description: Generate execution plan for tasks, identifying phases, ordering, and parallelization opportunities
argument-hint: [--all|--json] [--optimize]
---

# Generate Execution Plan

Creates an optimal execution plan that:
- Respects all task dependencies
- Identifies parallelizable groups
- Finds critical path
- Sequences work into phases
- Estimates completion timeline

## Usage

```bash
/metamanager:plan              # Generate plan for all tasks
/metamanager:plan --json       # Output as JSON
/metamanager:plan --optimize   # Apply optimizations
```

## Output Format (Default)

Shows structured execution plan:

1. **Overview**
   - Total phases: N
   - Critical path length: N steps
   - Estimated duration: HH:MM format
   - Parallelization factor: X.XX

2. **Phase Breakdown**
   ```
   Phase 1: Preparation (0-2 days)
   ├─ Task A (parallel possible with B)
   └─ Task B (parallel possible with A)

   Phase 2: Foundation (2-5 days)
   ├─ Task C (blocked by A)
   └─ Task D (blocked by B)

   Phase 3: Main Work (5-12 days)
   ├─ Task E (parallel group 1)
   ├─ Task F (parallel group 1)
   └─ Task G (blocked by C, D)
   ```

3. **Critical Path**
   ```
   Critical Path: A → C → G
   Duration: 12 days
   Impact: 100% (any delay affects total time)

   Acceleration opportunities:
   - Parallelize D with C (gain ~3 days)
   - Split G into subtasks (gain ~2 days)
   ```

4. **Parallelizable Groups**
   ```
   Group 1: [E, F] (can run simultaneously)
   Group 2: [B, A] (can run simultaneously)
   Gain: ~5 days from parallelization
   ```

5. **Bottleneck Analysis**
   ```
   Task C: Blocks 2 downstream tasks (high impact)
   Recommendation: Prioritize, consider splitting
   ```

6. **Next Steps**
   ```
   Recommended execution order:
   1. Start Task A and B (parallel)
   2. When A complete: Start Task C
   3. When B complete: Start Task D
   4. When C, D complete: Start Task G
   5. In parallel: Start E, F early (low blocker)
   ```

## JSON Output

```json
{
  "plan": {
    "plan_id": "plan-xxxxx",
    "graph_id": "graph-xxxxx",
    "total_phases": 4,
    "critical_path": ["task-1", "task-3", "task-5"],
    "estimated_duration": "12 days",
    "phases": [
      {
        "phase": 1,
        "tasks": ["task-1", "task-2"],
        "can_parallelize": true,
        "duration_estimate": "2 days"
      },
      ...
    ],
    "parallelizable_groups": [
      ["task-1", "task-2"],
      ["task-4", "task-5"]
    ],
    "bottlenecks": [
      {
        "task": "task-3",
        "impact": "high",
        "recommendation": "Prioritize"
      }
    ]
  }
}
```

## Options

### --all
Plan all tasks (default).

### --json
Output as JSON for programmatic use.

### --optimize
Apply optimization algorithms:
- Reorder tasks within phases
- Suggest phase merging
- Identify early-startable tasks
- Calculate optimal sequencing

## Examples

### Generate Basic Plan
```
/metamanager:plan
```

Shows phases, ordering, and critical path.

### Get JSON Output
```
/metamanager:plan --json
```

Useful for integration with other tools.

### Optimized Plan
```
/metamanager:plan --optimize
```

Includes optimization suggestions:
- "Task B can start immediately (doesn't depend on A)"
- "Phases 2 and 3 can be merged"
- "Task D is on critical path, prioritize"

## Plan Interpretation

### Critical Path
- Tasks on the longest dependency chain
- Any delay here affects total duration
- Prioritize these tasks
- Acceleration here saves most time

### Parallelizable Groups
- Tasks with no inter-dependencies
- Can execute simultaneously
- Significant time savings
- Maximize group size for efficiency

### Bottleneck Tasks
- Block many downstream tasks
- Small delays have large impact
- Prioritize completion
- Consider splitting for parallelization

## Understanding Phases

Phases are sequential groups where:
- All tasks in a phase can start when previous phase completes
- Tasks within a phase can run in parallel
- Phase N cannot start until Phase N-1 complete

Example:
```
Phase 1: [A, B] → Can run together
Phase 2: [C, D] → Can run together, but only after Phase 1
Phase 3: [E] → Starts only after Phase 2
```

## Time Estimates

Plan provides three duration types:

1. **Sequential Duration**: If tasks ran one-by-one (worst case)
2. **Parallel Duration**: If maximum parallelization (best case with your dependencies)
3. **Realistic Duration**: Accounts for overhead, coordination (~10% buffer)

## Tips

- Run plan before starting work
- Review critical path first
- Parallelize where possible
- Update plan as tasks complete
- Reassess when new tasks added
- Use for status reporting and scheduling
