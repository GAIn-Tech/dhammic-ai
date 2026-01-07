---
name: conflict-resolution
description: Use when detecting or resolving task conflicts, handling incomplete dependencies, orphaned tasks, circular dependencies, or broken task hierarchies. Activates for mentions of "conflict", "can't complete", "dependency violation", "broken", or similar issues.
version: 1.0.0
---

# Task Conflict Resolution Skill

Systematically identify, analyze, and resolve conflicts in task graphs.

## When This Skill Activates

Use when encountering:
- "Cannot complete task: subtasks pending"
- "Cannot delete task: other tasks depend on it"
- "Circular dependency detected"
- "Task is orphaned"
- "Stalled task"
- "Dependency violation"
- Any graph validation error

## Conflict Types

### 1. Incomplete Subtasks
**Problem**: Task marked complete but child tasks still pending
**Root Cause**: Parent task completed before children
**Solution**: Complete all subtasks first, or convert to independent tasks

```
Parent Task (completed)
├─ Child A (pending) ❌
├─ Child B (pending) ❌
└─ Child C (completed)
```

### 2. Incomplete Dependencies
**Problem**: Task in progress but blocking tasks not complete
**Root Cause**: Task started before dependencies ready
**Solution**: Wait for blockers to complete before progressing

```
Task (in_progress)
├─ Blocked by: Task X (pending) ❌
└─ Blocked by: Task Y (in_progress)
```

### 3. Orphaned Dependents
**Problem**: Deleting task breaks dependent tasks
**Root Cause**: Task has children with no other parents
**Solution**: Reassign dependents to another parent, or keep task

```
Task (to delete)
├─ Dependent A (pending) ❌
└─ Dependent B (pending) ❌
```

### 4. Orphaned Tasks
**Problem**: Task disconnected from hierarchy
**Root Cause**: No path to root, no path to completion
**Solution**: Reconnect to hierarchy or delete

```
Orphaned Task
└─ No incoming or outgoing edges
```

### 5. Circular Dependency
**Problem**: Tasks block each other (cycle)
**Root Cause**: Incorrect dependency specification
**Solution**: Break cycle by removing one dependency

```
Task A → Task B → Task C → Task A (cycle!) ❌
```

### 6. Abandoned Task
**Problem**: Task not updated for extended period
**Root Cause**: Task forgotten or stalled
**Solution**: Review and update, or mark complete/delete

```
Task A (pending)
└─ Last updated: 7 days ago ⚠️
```

## Resolution Process

### Step 1: Identify Conflict
- What type of conflict is this?
- Which tasks are involved?
- What is the root cause?
- How many tasks affected?

### Step 2: Assess Impact
- How many tasks blocked by this conflict?
- What's the critical path impact?
- What's the time cost of resolution?
- Can we defer or work around?

### Step 3: Generate Options
For each conflict type, generate resolution options:

| Conflict | Option 1 | Option 2 | Option 3 |
|----------|----------|----------|----------|
| Incomplete Subtasks | Complete subtasks | Convert to independent | Keep incomplete |
| Circular Dependency | Remove dependency A | Remove dependency B | Refactor structure |
| Orphaned Dependents | Reassign to parent | Keep task | Delete dependents |
| Abandoned Task | Reactivate task | Mark complete | Delete task |

### Step 4: Recommend Best Path
Consider:
- User intent
- Timeline impact
- Complexity of resolution
- Side effects
- Reversibility

### Step 5: Apply Resolution
- Execute recommended option
- Update task graph
- Verify no new conflicts
- Save to Supermemory
- Document changes

### Step 6: Verify
- Re-validate graph
- Check for new conflicts
- Confirm dependencies correct
- Test with execution plan

## Resolution Strategies

### For Incomplete Subtasks
```
Option A: Complete subtasks first
└─ Mark children complete → Parent can complete

Option B: Convert to independent tasks
└─ Remove parent-child relationship → Independent timeline

Option C: Create new parent
└─ Move subtasks under new parent → Organizational fix
```

### For Circular Dependencies
```
Option A: Remove dependency A→B
└─ Break cycle at weakest link

Option B: Restructure as separate chains
└─ A→C, B→D (no cycle)

Option C: Create intermediate task
└─ A→Bridge→B (explicit sequencing)
```

### For Orphaned Tasks
```
Option A: Connect to parent
└─ Create dependency from root task

Option B: Make independent
└─ Remove all dependencies

Option C: Delete task
└─ If no longer needed
```

## Output Criteria

When resolving conflicts, document:

1. **Conflict Identified**
   ```
   Type: Incomplete Subtasks
   Task: Feature Implementation
   Subtasks Pending: 3
   ```

2. **Root Cause**
   ```
   Task marked complete before children finished
   ```

3. **Impact Analysis**
   ```
   - Blocks: Testing phase (2 tasks)
   - Timeline impact: +2 days
   - Affected stakeholders: QA team
   ```

4. **Resolution Recommendation**
   ```
   Complete subtasks before marking parent done:
   1. Finish Database Schema (1 day)
   2. Finish API Implementation (2 days)
   3. Then mark Feature complete
   ```

5. **Verification**
   ```
   ✅ Graph validates after resolution
   ✅ No new conflicts detected
   ✅ Dependencies correct
   ✅ Timeline updated
   ```

## Conflict Prevention

### Guidelines for Avoiding Conflicts
1. **Complete subtasks before parent**: Always finish children first
2. **Verify dependencies before deleting**: Check dependents exist
3. **Avoid circular dependencies**: Design acyclic graphs
4. **Keep tasks connected**: No orphaned tasks in graph
5. **Monitor for abandonment**: Update stalled tasks

### Best Practices
- Review graph regularly for conflicts
- Use dependency analysis before major changes
- Document why dependencies exist
- Consider future dependencies when designing
- Test changes with execution planner

---

**This skill is automatically invoked when conflicts are detected or when resolving dependency issues.**
