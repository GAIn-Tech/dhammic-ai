---
name: execution-planning
description: Use when creating execution plans, breaking work into phases, identifying critical paths, or optimizing task sequencing. Activates for "planning", "phases", "order", "critical path", "what's next", "how to execute", or execution strategy discussions.
version: 1.0.0
---

# Execution Planning Heuristics

Create optimal execution strategies for complex task graphs using proven planning principles.

## When This Skill Activates

Use when:
- Creating execution plan for task set
- Determining task order
- Identifying critical path
- Deciding what to work on next
- Optimizing execution timeline
- Planning work phases
- Analyzing dependencies for execution

## Planning Principles

### Principle 1: Respect Dependencies
- Never schedule task before its blockers complete
- Identify all constraint chains
- Verify blocking relationships
- Check for transitive dependencies
- Validate no circular dependencies

### Principle 2: Maximize Parallelization
- Group independent tasks together
- Execute parallel groups simultaneously
- Balance workload across agents
- Reduce unnecessary sequencing
- Combine unrelated work

### Principle 3: Minimize Critical Path
- Identify longest dependency chain
- Prioritize critical path tasks
- Address bottlenecks first
- Accelerate high-impact tasks
- Consider path compression techniques

### Principle 4: Optimize Sequencing
- Front-load preparatory tasks
- Batch similar work
- Reduce context switching
- Order tasks by dependency depth
- Consider skill/resource matching

### Principle 5: Track and Adapt
- Monitor actual vs estimated duration
- Adjust plan when reality diverges
- Accelerate bottlenecks
- Parallelize opportunities
- Learn from execution

## Phase Structure

Optimal phase organization:

```
Phase 1: Setup
├─ Preparatory tasks (no blockers)
├─ Infrastructure tasks
└─ Planning tasks

Phase 2: Foundation
├─ Core tasks that enable later work
├─ Critical infrastructure
└─ Data setup

Phase 3: Main Work
├─ Bulk of work (high parallelization)
├─ Independent feature work
└─ Parallel implementations

Phase 4: Integration
├─ Combining results
├─ Cross-component testing
└─ System assembly

Phase 5: Validation
├─ End-to-end testing
├─ Quality assurance
└─ Cleanup
```

## Planning Algorithm

### Step 1: Analyze Graph
```
Input: Task graph with dependencies
├─ Validate graph (no cycles)
├─ Calculate depths (steps from root)
├─ Find critical path
└─ Identify parallelizable groups
```

### Step 2: Create Phases
```
├─ Topological sort (respecting dependencies)
├─ Group independent tasks
├─ Calculate phase ordering
└─ Estimate phase duration
```

### Step 3: Identify Bottlenecks
```
├─ Find high-impact tasks
├─ Analyze blocking relationships
├─ Estimate impact on timeline
└─ Suggest optimization
```

### Step 4: Optimize Plan
```
├─ Reorder within phases
├─ Merge compatible phases
├─ Suggest parallelization
└─ Calculate total duration
```

### Step 5: Document Plan
```
├─ Task ordering with rationale
├─ Phase dependencies
├─ Critical path highlight
└─ Execution guidance
```

## Creating Execution Plan

### 1. Basic Phase Division
```
Phase 1: Tasks with no dependencies
├─ Task A (no blockers)
├─ Task B (no blockers)
└─ Task C (no blockers)
→ Can all run in parallel

Phase 2: Tasks blocked only by Phase 1
├─ Task D (blocked by A)
├─ Task E (blocked by B, C)
→ Can run in parallel if A, B, C done

Phase 3: Tasks blocked by Phase 2
├─ Task F (blocked by D, E)
→ Depends on D and E
```

### 2. Parallel Group Identification
```
Independent Groups (can run simultaneously):
├─ Group 1: [Task A, Task B] (no inter-dependencies)
├─ Group 2: [Task C, Task D] (no inter-dependencies)
└─ Group 3: [Task E] (depends on earlier phases)

Execution time:
├─ Sequential: A + B + C + D + E = Sum of all
└─ Parallel: max(A,B) + max(C,D) + E = Much less
```

### 3. Critical Path Calculation
```
Paths through graph:
├─ A → D → F (3 steps)
├─ B → E → F (3 steps)
└─ C → G → F (3 steps)

Critical Path: Any of above (3 steps)
Minimum duration: Sum of critical path steps
Optimization: Target critical path tasks
```

## Examples

### Example 1: Linear Dependencies
```
A → B → C → D → E

Graph: Linear chain
Phases:
├─ Phase 1: [A]
├─ Phase 2: [B]
├─ Phase 3: [C]
├─ Phase 4: [D]
└─ Phase 5: [E]

Issue: No parallelization
Duration: A + B + C + D + E (no improvement)

Optimization: Can any be done in parallel?
→ No, fully sequential
```

### Example 2: Diamond Pattern
```
    A
   / \
  B   C
   \ /
    D

Phases:
├─ Phase 1: [A]
├─ Phase 2: [B, C] ← Parallel!
└─ Phase 3: [D]

Duration: A + max(B, C) + D
Parallelization: B and C can run simultaneously
Savings: min(B, C) of time
```

### Example 3: Complex Network
```
A → C → E
B → D → E → F

Phases:
├─ Phase 1: [A, B] ← Parallel
├─ Phase 2: [C, D] ← Parallel
├─ Phase 3: [E]
└─ Phase 4: [F]

Duration: max(A,B) + max(C,D) + E + F
Critical Path: A → C → E → F or B → D → E → F
Optimization targets: C or D (on critical path)
```

## Decision Points

### "What should I work on next?"
```
1. Are all blockers complete?
   └─ NO: Wait (or work on other independent tasks)
   └─ YES: This task is ready

2. Is this on critical path?
   └─ YES: Prioritize (time-critical)
   └─ NO: Lower priority

3. Do I have required resources?
   └─ NO: Work on something else
   └─ YES: Start this task

→ Start critical path tasks first
```

### "Can these run in parallel?"
```
1. Does Task A block Task B?
   └─ YES: Must be sequential
   └─ NO: Continue...

2. Does Task B block Task A?
   └─ YES: Must be sequential
   └─ NO: Continue...

3. Do they share resources?
   └─ YES: Consider sequencing
   └─ NO: Can parallelize

→ If independent: Run in parallel
```

### "How long will this take?"
```
1. Calculate critical path: Sum of longest chain
2. Adjust for parallelization: Account for simultaneous work
3. Add overhead: Coordination, switching costs (~10%)

Total ≈ Critical Path Length + Overhead
```

## Output Format

Document execution plans with:

```json
{
  "plan": {
    "total_phases": 4,
    "critical_path_length": 3,
    "critical_path_tasks": ["A", "C", "E"],
    "estimated_duration": "3 weeks",
    "parallelization_factor": 1.8,
    "phases": [
      {
        "phase": 1,
        "tasks": ["A", "B"],
        "can_parallelize": true,
        "rationale": "A and B have no inter-dependencies"
      },
      ...
    ]
  }
}
```

## Success Criteria

A good execution plan:
- ✅ Respects all dependencies
- ✅ Maximizes parallelization
- ✅ Minimizes critical path
- ✅ Is clearly documented
- ✅ Is executable
- ✅ Provides clear next steps
- ✅ Identifies bottlenecks
- ✅ Is adaptable to changes

---

**This skill is automatically invoked when planning task execution, creating schedules, or optimizing work flows.**
