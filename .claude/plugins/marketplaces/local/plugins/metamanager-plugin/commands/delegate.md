---
description: Delegate executable tasks to specialized agents for parallel execution
argument-hint: <task-spec> [--parallel|--agent=type]
---

# Delegate Task Execution

Assigns executable tasks to appropriate specialized agents for execution.

## Usage

```bash
/metamanager:delegate phase-1       # Delegate all Phase 1 tasks
/metamanager:delegate task-5        # Delegate specific task
/metamanager:delegate --all          # Delegate all ready tasks
/metamanager:delegate task-3 --parallel  # Run subtasks in parallel
```

## How Delegation Works

1. **Task Selection**: Identify tasks to delegate
2. **Readiness Check**: Verify all blockers are complete
3. **Type Classification**: Determine task type (feature, fix, test, etc.)
4. **Agent Selection**: Choose appropriate agent
5. **Context Preparation**: Gather required information
6. **Agent Spawn**: Launch agent with full context
7. **Monitoring**: Track execution and report progress
8. **Completion**: Validate results and update status

## Agent Selection

System automatically selects agent based on task type:

| Task Type | Primary Agent | Secondary |
|-----------|---|---|
| `feature` | feature-dev:code-architect | feature-dev:code-explorer |
| `bugfix` | feature-dev:code-reviewer | General executor |
| `refactor` | Code simplicity reviewer | General executor |
| `test` | Test-focused agent | General executor |
| `docs` | Documentation agent | General executor |
| `review` | Code reviewer | code-reviewer |
| `research` | Research agent | General executor |
| `other` | General executor | - |

## Task Readiness

Tasks must be ready before delegation:

```
✓ All blockers complete
✓ Task description clear
✓ Success criteria defined
✓ Resources available
✓ No conflicts
```

If not ready, system will:
- Explain what's blocking
- Suggest wait time
- Offer alternatives

## Delegation Options

### Parallel Execution
```
/metamanager:delegate phase-2 --parallel
```

Spawns multiple agents simultaneously for:
- Multiple independent tasks
- Subtasks of one task
- Parallelizable groups

Monitor all in parallel and wait for completion.

### Specific Agent
```
/metamanager:delegate task-5 --agent=architect
```

Force specific agent type even if different from default.

### Dry Run
```
/metamanager:delegate task-3 --dry-run
```

Show what would happen without executing.

## Task Context

When delegating, agent receives:

```json
{
  "task_id": "task-123",
  "task_name": "Implement User Authentication",
  "description": "Add JWT-based auth system",
  "phase": 2,

  "acceptance_criteria": [
    "Users can log in with email/password",
    "Tokens expire after 24 hours",
    "80%+ test coverage"
  ],

  "dependencies": {
    "blockers": ["database-setup"],
    "dependents": ["user-authorization"],
    "subtasks": ["login-endpoint", "token-generation"]
  },

  "resources": {
    "files_to_modify": ["src/auth.ts", "middleware.ts"],
    "files_to_create": ["src/jwt.ts"],
    "references": ["RFC 7519", "project-spec.md"]
  },

  "constraints": {
    "timeline": "2 days",
    "technology": "jsonwebtoken library",
    "testing": "80% minimum coverage"
  }
}
```

## Execution Flow

```
1. Task Delegation Request
   └─ /metamanager:delegate phase-2

2. Readiness Check
   ├─ Verify blockers complete
   ├─ Validate task specification
   └─ Check resources available

3. Agent Selection
   ├─ Classify task type
   ├─ Select primary agent
   └─ Prepare context

4. Agent Spawn
   ├─ Launch agent process
   ├─ Pass task context
   └─ Start execution

5. Monitoring
   ├─ Track progress
   ├─ Report updates
   └─ Handle failures

6. Completion
   ├─ Validate output
   ├─ Update task status
   ├─ Save results
   └─ Report to orchestrator

7. Next Phase
   └─ Identify newly-ready tasks
```

## Monitoring Execution

While delegated tasks execute:

```
Delegated Task Monitoring:
├─ Task-5: "Feature A" (architect agent)
│  ├─ Status: In Progress (45%)
│  └─ ETA: 1.5 hours
├─ Task-6: "Feature B" (architect agent)
│  ├─ Status: In Progress (60%)
│  └─ ETA: 1 hour
└─ Task-8: "Testing" (test agent)
   ├─ Status: Waiting (Task-5, Task-6)
   └─ ETA: When others complete
```

## Task Status Updates

During execution, agent reports:
- % complete
- Current sub-task
- Issues encountered
- ETA for completion
- Any blockers

## Completion Handling

When agents report task complete:

1. **Validate Output**
   - Check acceptance criteria
   - Verify code quality
   - Test functionality

2. **Update Graph**
   - Mark task complete
   - Update progress %
   - Trigger dependents

3. **Save Results**
   - Store outputs
   - Log completion time
   - Update Supermemory

4. **Trigger Next Phase**
   - Identify newly-ready tasks
   - Prepare for delegation
   - Update status

## Examples

### Delegate Phase
```
/metamanager:delegate phase-1
```

Delegates all ready tasks in Phase 1 to appropriate agents.

### Delegate Task
```
/metamanager:delegate feature-user-auth
```

Delegates specific task "feature-user-auth" to best agent.

### Parallel Delegation
```
/metamanager:delegate phase-2 --parallel
```

Spawns agents for all Phase 2 tasks simultaneously.

### Specific Agent
```
/metamanager:delegate refactor-utils --agent=simplicity-reviewer
```

Forces specific agent type.

### Preview
```
/metamanager:delegate phase-1 --dry-run
```

Shows what would be delegated without executing.

## Success Criteria

Successful delegation when:
- ✅ Agent spawned successfully
- ✅ Task marked in_progress
- ✅ Progress tracked
- ✅ Task completed per acceptance criteria
- ✅ Status updated to completed
- ✅ Results saved
- ✅ Dependents notified

## Tips

- Delegate ready tasks early to parallelize
- Group related tasks for same agent
- Monitor progress regularly
- Intervene quickly if tasks stuck
- Use dry-run to preview
- Save detailed results for documentation
