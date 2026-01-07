# Metamanager Plugin for OpenCode

An automated task orchestration and conflict management system for OpenCode (Claude Code CLI) that intelligently manages task dependencies, detects conflicts, coordinates specialized agents, and maintains optimal execution plans.

## What It Does

**Metamanager** turns task management from manual into automated:

1. **Hook-based Monitoring** - Intercepts TodoWrite operations to detect conflicts before they happen
2. **Conflict Detection** - Identifies incomplete subtasks, circular dependencies, orphaned tasks, and other graph issues
3. **Intelligent Planning** - Creates optimal execution plans respecting dependencies and maximizing parallelization
4. **Agent Orchestration** - Coordinates specialized agents (planner, executor) to autonomously execute work
5. **Supermemory Persistence** - Saves task graphs and execution plans for cross-session recovery
6. **Task Analysis** - Provides insights into dependencies, critical paths, and optimization opportunities

## Features

### Automatic Conflict Detection

The hook system watches for problematic TodoWrite operations:

- ❌ **Incomplete Subtasks**: Marks parent task complete while children pending
- ❌ **Circular Dependencies**: Tasks block each other in a cycle
- ❌ **Orphaned Dependents**: Deleting task that others depend on
- ❌ **Abandoned Tasks**: Tasks not updated for 7+ days
- ⚠️ **Blocking Issues**: Tasks in progress blocked by incomplete prerequisites

### Intelligent Planning

Automatically partitions work into phases:

- **Topological Sorting**: Orders tasks respecting all dependencies
- **Critical Path Analysis**: Identifies longest dependency chain (key to total duration)
- **Parallelization**: Finds independent task groups that can run simultaneously
- **Bottleneck Detection**: Flags tasks with high downstream impact
- **Duration Estimation**: Calculates execution timeline

### Agent Orchestration

Three specialized agents manage execution:

- **Orchestrator**: Coordinates overall execution, monitors progress, detects issues
- **Planner**: Creates optimal execution plans, identifies critical paths
- **Executor**: Runs assigned tasks, delegates to specialized agents, tracks completion

### Commands

Five slash commands for task management:

1. **`/metamanager:analyze`** - Analyze task structure, dependencies, conflicts
2. **`/metamanager:plan`** - Generate execution plan with phases and critical path
3. **`/metamanager:resolve`** - Detect and interactively resolve conflicts
4. **`/metamanager:visualize`** - Show task graph in ASCII, Mermaid, or JSON
5. **`/metamanager:delegate`** - Assign executable tasks to agents

### Skills

Three auto-invoked skills teach Claude Code about task management:

- **dependency-analysis**: Analyze blocking relationships and task prerequisites
- **conflict-resolution**: Identify and resolve dependency conflicts
- **execution-planning**: Create optimal execution strategies

## Quick Start

### Installation

```bash
# Create plugin directory structure
mkdir -p ~/.claude/plugins/marketplaces/local/plugins/metamanager-plugin

# Copy plugin files to the directory
# (This can be done manually or via git clone)

# Load in OpenCode
claude --plugin-dir ~/.claude/plugins/marketplaces/local/plugins/metamanager-plugin
```

### Basic Usage

```bash
# 1. Analyze your current tasks
/metamanager:analyze

# 2. Generate execution plan
/metamanager:plan

# 3. Visualize as graph
/metamanager:visualize --format=mermaid

# 4. Resolve any conflicts
/metamanager:resolve

# 5. Delegate ready tasks
/metamanager:delegate phase-1
```

## Architecture

### Core Components

```
Plugin Structure:
├── hooks/
│   ├── hooks.json              # Hook definitions
│   └── todoupdate.py           # TodoWrite intercept handler
├── core/
│   ├── models.py               # Task, Graph, Plan data structures
│   ├── dag_engine.py           # Graph operations & analysis
│   ├── conflict_rules.py       # Conflict detection rules
│   └── planning_engine.py      # Execution planning engine
├── agents/
│   ├── orchestrator.md         # Orchestration agent
│   ├── planner.md              # Planning agent
│   └── executor.md             # Execution agent
├── commands/
│   ├── analyze.md              # Analyze command
│   ├── plan.md                 # Planning command
│   ├── resolve.md              # Conflict resolution
│   ├── visualize.md            # Graph visualization
│   └── delegate.md             # Task delegation
├── skills/
│   ├── dependency-analysis/SKILL.md
│   ├── conflict-resolution/SKILL.md
│   └── execution-planning/SKILL.md
└── utils/
    └── supermemory_client.py   # Supermemory integration
```

### Data Flow

```
User Creates/Updates Tasks (TodoWrite)
    ↓
Hook Intercepts (PreToolUse)
    ↓
Conflict Detection
    ├─ Load current graph from Supermemory
    ├─ Validate proposed changes
    ├─ Check for conflicts
    └─ Return allow/block decision
    ↓
If Allowed:
    ├─ Update task graph
    ├─ Save to Supermemory
    └─ Trigger Orchestrator
    ↓
Orchestrator:
    ├─ Analyze current state
    ├─ Detect conflicts
    ├─ Spawn Planner for new phase
    ├─ Spawn Executor for ready tasks
    └─ Save updated state
```

## Conflict Types

### Critical (Block Execution)

| Type | Cause | Fix |
|------|-------|-----|
| Incomplete Subtasks | Parent complete with pending children | Complete children first |
| Circular Dependency | Tasks block each other (A→B→A) | Remove one dependency |
| Orphaned Dependents | Delete task that others depend on | Reassign or keep task |

### Warning (May Block Progress)

| Type | Cause | Fix |
|------|-------|-----|
| Incomplete Dependencies | Task started before blockers done | Wait or reorder |
| Abandoned Task | Not updated for 7+ days | Review and update |

### Info (Organizational)

| Type | Cause | Fix |
|------|-------|-----|
| Orphaned Task | Disconnected from hierarchy | Connect or delete |

## Commands Reference

### Analyze
```bash
/metamanager:analyze              # Full analysis
/metamanager:analyze --json       # JSON output
/metamanager:analyze --mermaid    # Mermaid diagram syntax
```

Shows: dependencies, conflicts, critical path, recommendations.

### Plan
```bash
/metamanager:plan                 # Generate plan
/metamanager:plan --json          # JSON output
/metamanager:plan --optimize      # With optimizations
```

Creates: phases, ordering, critical path, duration estimate.

### Resolve
```bash
/metamanager:resolve              # Show conflicts
/metamanager:resolve --auto       # Auto-resolve when possible
/metamanager:resolve --show-all   # Include info-level conflicts
```

Detects and guides resolution of: incomplete subtasks, circular deps, orphaned tasks.

### Visualize
```bash
/metamanager:visualize                    # ASCII tree
/metamanager:visualize --format=mermaid   # Graph syntax
/metamanager:visualize --format=json      # Raw data
```

Shows: task hierarchy, dependencies, phases, critical path.

### Delegate
```bash
/metamanager:delegate phase-1             # Delegate phase
/metamanager:delegate task-5              # Delegate task
/metamanager:delegate --all --parallel    # All tasks in parallel
```

Assigns: tasks to agents, spawns executors, tracks completion.

## Key Concepts

### Task Graph (DAG)
A Directed Acyclic Graph where:
- **Nodes** = Tasks
- **Edges** = Dependencies (blocking relationships)
- **Invariant** = No cycles (prevents deadlock)

### Critical Path
The longest dependency chain in the graph:
- Determines minimum execution time
- Any delay here delays entire project
- Key target for optimization

### Phases
Sequential groups where:
- Tasks within a phase can run in parallel
- Phase N completes before Phase N+1 starts
- Minimize number of phases to reduce total time

### Conflict
Any violation of execution constraints:
- Critical conflicts block execution
- Warning conflicts may cause issues
- Info conflicts are organizational

## Best Practices

### Creating Tasks
1. Use clear, descriptive task names
2. Specify blockers explicitly
3. Break large tasks into subtasks
4. Set realistic duration estimates
5. Define acceptance criteria

### Managing Dependencies
1. Keep dependency chains short
2. Maximize parallelization opportunities
3. Avoid unnecessary dependencies
4. Review regularly for optimization
5. Document why dependencies exist

### Monitoring Progress
1. Run `/metamanager:analyze` regularly
2. Address conflicts quickly
3. Update stalled tasks
4. Track actual vs estimated duration
5. Adjust plans as needed

### Delegating Work
1. Ensure task is ready (blockers complete)
2. Use `/metamanager:delegate` to assign agents
3. Monitor agent progress
4. Validate completion against criteria
5. Update Supermemory with results

## Limitations & Future Work

### Current Limitations
- Basic conflict detection (no complex scenarios)
- Simple parallelization heuristics
- No resource/capacity constraints
- No time-based scheduling
- No team assignment yet

### Planned Features
- **Time Tracking**: Monitor actual vs estimated duration
- **Resource Allocation**: Assign agents/users to tasks
- **Notifications**: Alert on blockers, escalate stalled tasks
- **Historical Analytics**: Learn from past projects
- **External Integration**: Sync with GitHub, Jira, calendars
- **Advanced Optimization**: ML-based path optimization
- **Hierarchical Planning**: Multi-level task breakdown

## Troubleshooting

### Hook Not Firing
1. Check hooks are loaded: `/hooks`
2. Verify hook.json syntax is valid
3. Ensure path to todoupdate.py is correct
4. Check Python path includes plugin directory

### Graph Validation Errors
1. Run `/metamanager:resolve` to detect issues
2. Look for circular dependencies
3. Check all task references are valid
4. Verify no orphaned tasks

### Supermemory Not Saving
1. Verify Supermemory is initialized
2. Check containerTag is correct
3. Ensure sufficient space available
4. Test with simple save operation

## Contributing

To extend Metamanager:

1. **Add Conflict Rules**: Edit `core/conflict_rules.py`
2. **New Commands**: Add `.md` files in `commands/`
3. **New Agents**: Add `.md` files in `agents/`
4. **New Skills**: Add folders in `skills/`
5. **Bug Fixes**: File issues with reproduction steps

## Support

- **Questions**: Check documentation and examples
- **Issues**: Report with task graph that reproduces issue
- **Contributions**: Submit pull requests with tests
- **Feedback**: Suggest features or improvements

## License

MIT - See LICENSE file

## Changelog

### v1.0.0 (Initial Release)
- Hook-based TodoWrite monitoring
- Conflict detection engine
- Planning and orchestration system
- Three specialized agents
- Five core commands
- Three helper skills
- Supermemory integration
