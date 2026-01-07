---
description: Detect and interactively resolve task conflicts step by step
argument-hint: [--auto|--show-all]
---

# Resolve Task Conflicts

Guides you through detecting and resolving task graph conflicts:
- Incomplete subtasks
- Circular dependencies
- Orphaned tasks
- Abandoned tasks
- Other dependency violations

## Usage

```bash
/metamanager:resolve          # Show conflicts and walk through resolution
/metamanager:resolve --auto   # Auto-resolve when possible
/metamanager:resolve --show-all  # Show all conflicts including info level
```

## Conflict Types

### Critical (Block Execution)

**Incomplete Subtasks**
- Problem: Parent task marked complete with pending children
- Impact: Child tasks orphaned
- Fix: Complete children first or separate from parent

**Circular Dependency**
- Problem: Tasks block each other (A→B→A)
- Impact: Impossible to execute
- Fix: Remove one dependency to break cycle

**Orphaned Dependents**
- Problem: Deleting task that others depend on
- Impact: Dependent tasks broken
- Fix: Reassign or keep task

### Warning (May Block Progress)

**Incomplete Dependencies**
- Problem: Task in progress but blockers incomplete
- Impact: Task may be blocked
- Fix: Wait for blockers or reconsider order

**Abandoned Task**
- Problem: Task not updated for 7+ days
- Impact: May be forgotten or stalled
- Fix: Review and update or complete

### Info (No Blocking)

**Orphaned Tasks**
- Problem: Task disconnected from hierarchy
- Impact: Hard to understand structure
- Fix: Connect to parent or delete

## Interactive Resolution

### Step 1: View Conflicts
Shows all detected conflicts:
```
Found 3 conflicts:
1. [CRITICAL] Incomplete Subtasks - Task "Feature A"
2. [WARNING] Incomplete Dependencies - Task "Testing"
3. [INFO] Orphaned Task - "Old Prototype"
```

### Step 2: Select Conflict
Choose which to address:
```
Which conflict to resolve? (1-3): 1

Conflict: Incomplete Subtasks
Task: Feature A
Description: Cannot complete - 2 subtasks still pending
```

### Step 3: Analyze Impact
Understand what fixing costs:
```
Impact Analysis:
- Blocks: 3 downstream tasks
- Timeline impact: +1 day if resolved now
- Affected team: Backend team
```

### Step 4: View Options
See resolution options:
```
Resolution Options:

Option A: Complete Subtasks First (Recommended)
└─ Cost: 1-2 days
└─ Benefit: Maintains structure
└─ Action: Work on subtasks before parent

Option B: Convert to Independent Tasks
└─ Cost: Restructuring
└─ Benefit: Parallel work possible
└─ Action: Remove parent-child relationship

Option C: Force Complete Parent
└─ Cost: Unclear subtask status
└─ Benefit: Unblocks dependent tasks
└─ Action: Mark parent complete anyway
```

### Step 5: Apply Resolution
Execute chosen option:
```
Applying Option A...
✓ Identified subtasks
✓ Prepared action plan
✓ Ready to proceed

Next steps:
1. Complete: "Subtask 1: Database Schema"
2. Complete: "Subtask 2: Data Migration"
3. Then mark parent "Feature A" complete
```

## Auto-Resolution

With `--auto` flag, system resolves when possible:

```
/metamanager:resolve --auto

Auto-resolving conflicts...
✓ Abandoned Task "Old Work" → Marked for deletion
✓ Orphaned Task "Helper" → Connected to Phase 1
✗ Incomplete Subtasks "Feature A" → Requires manual action
✗ Circular Dependency A→B→A → Requires manual decision

2/4 conflicts auto-resolved
2/4 require manual review
```

## Conflict Details

Each conflict includes:

1. **Description**: What's wrong
2. **Severity**: CRITICAL / WARNING / INFO
3. **Affected Tasks**: Which tasks involved
4. **Root Cause**: Why this happened
5. **Impact**: What breaks if not fixed
6. **Resolution Options**: How to fix
7. **Recommendation**: Best option

## Common Scenarios

### Incomplete Subtasks
```
Scenario: You marked parent task complete but children aren't
Resolution:
A) Complete all children first
B) Split into independent tasks
C) Create new parent for children

Recommendation: Option A (maintains hierarchy)
```

### Circular Dependency
```
Scenario: Task A blocks B, B blocks C, C blocks A
Resolution:
A) Remove A→B dependency
B) Remove B→C dependency
C) Remove C→A dependency

Recommendation: Analyze to find weakest link, remove that
```

### Orphaned Task
```
Scenario: Task disconnected from hierarchy
Resolution:
A) Connect to parent task
B) Make independent (remove all dependencies)
C) Delete if no longer needed

Recommendation: Reconnect if useful, delete if not
```

### Abandoned Task
```
Scenario: Task not updated for 7+ days
Resolution:
A) Review and update status
B) Complete if done
C) Delete if no longer needed

Recommendation: Check with owner, update or remove
```

## Best Practices

### Prevention
1. **Complete subtasks before parent**: Always finish children first
2. **Review before deleting**: Check what depends on task
3. **Monitor for abandonment**: Update stalled tasks regularly
4. **Verify dependencies**: Ensure they're necessary

### Resolution
1. **Fix critical first**: CRITICAL conflicts block execution
2. **Understand impact**: Know what changes will affect
3. **Document decisions**: Record why you resolved this way
4. **Verify fix**: Rerun resolve to check no new conflicts

## Tips

- Run `/metamanager:analyze` to see current state
- Address critical conflicts first
- Use `--show-all` to see complete picture
- Document complex resolutions
- Review conflicts monthly
- Prevent is better than cure

## Examples

### Check Conflicts
```
/metamanager:resolve
```

Shows all conflicts needing attention.

### Auto-fix When Possible
```
/metamanager:resolve --auto
```

Automatically fixes simple issues, shows manual ones.

### Detailed View
```
/metamanager:resolve --show-all
```

Shows critical, warning, AND info level conflicts.

## Success Criteria

After resolution:
- ✅ No critical conflicts remain
- ✅ Graph validates without errors
- ✅ All dependencies respectedvalid
- ✅ No circular dependencies
- ✅ All tasks reachable
