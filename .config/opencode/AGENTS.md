# Global Agent Directives

These directives apply to all AI agents operating in this environment.

---

## Hermeneutic Circle Methodology

Understanding is circular, not linear. Each pass deepens comprehension:

```
PART → WHOLE → PART → WHOLE (refined)
```

### Apply to Code Investigation
1. **Read a symbol** → understand its role in the file
2. **Understand the file** → re-read symbol with new context
3. **Understand the module** → re-read file with architectural context
4. **Make change** → verify understanding was correct

### Apply to Debugging
1. **Initial symptom** → form hypothesis
2. **Investigate** → revise hypothesis
3. **Test** → either confirm or restart cycle with new understanding

### Signs You're Short-Circuiting
- Making changes without understanding surrounding code
- Fixing symptoms without finding root cause
- "This should work" (your model is incomplete)

**The circle never truly closes.** Each iteration reveals more. Know when "good enough" understanding is sufficient for the task.

---

## Ultrawork Protocol (High-Performance Execution)

Ultrawork is the standard for high-throughput, high-quality development. It mandates parallel execution and deep research over sequential, ad-hoc actions.

### Concurrent Subagent Dispatch
When facing complex problems, **dispatch multiple agents simultaneously**:

```python
# CORRECT: Fire all at once, collect results later
background_task(agent="explore", prompt="Find auth implementations...")
background_task(agent="explore", prompt="Find error handling patterns...")
background_task(agent="librarian", prompt="Find JWT best practices...")
# Continue working, collect with background_output when needed

# WRONG: Sequential blocking
result = task(...)  # Never wait synchronously for explore/librarian
```

### When to Dispatch Parallel Agents
| Scenario | Action |
|----------|--------|
| 2+ independent code areas to understand | Parallel explore agents |
| Internal + external research needed | explore + librarian in parallel |
| Multiple hypotheses to test | One agent per hypothesis |
| Cross-cutting concerns | One agent per concern |

### Investigation Workflow
1. **Identify investigation angles** (minimum 3 for complex problems)
2. **Dispatch agents in parallel** with specific prompts
3. **Continue immediate work** while agents run
4. **Collect results** when needed via `background_output`
5. **Synthesize** findings using hermeneutic circle (part → whole → refined understanding)
6. **Cancel remaining agents** before final answer: `background_cancel(all=true)`

### Agent Prompt Structure
Each dispatched agent needs:
```
CONTEXT: Why this investigation matters
SEARCH TARGETS: Specific files/patterns/concepts
RETURN: Exactly what information to bring back
```

Vague prompts waste cycles. Be exhaustive in specifying what you need.

---

## Skill & MCP Discovery Protocol

All agents MUST proactively discover and leverage specialized capabilities before attempting manual implementation.

### 1. Skill Search (Priority 1)
Skills are battle-tested workflows. Before starting any non-trivial task:
- Search available skills: `Glob(pattern="swe/skills/**/*.md")`.
- Search global skills: `~/.claude/skills/` or `~/.config/opencode/skill/`.
- **Mandatory**: If a matching skill exists, you MUST read and follow it.

### 2. MCP Tool Discovery (Priority 2)
MCP servers provide external capabilities (Web search, Docs, Browser).
- Search registered MCP tools: `mcp_list_tools`.
- Use `librarian` to find documentation for unfamiliar libraries via MCP.

### 3. The "Discovery-First" Rule
Any task involving 2+ steps MUST begin with a discovery phase (Grep/Glob/MCP) to find existing patterns or tools. Never "raw-dog" complex logic without checking for existing skills.

---

## Multi-Agent Coordination Protocol (ACP)

When tasks require collaboration between multiple AI agents, establish structured communication using the **Agent Coordination Protocol**.

### When to Coordinate

| Trigger | Action |
|---------|--------|
| Long-running parallel work | Establish coordinator ↔ worker(s) protocol |
| Specialized tasks (architecture + implementation) | Use agent handoff with task tracking |
| Iterative feedback loops | Ping-pong coordination with consensus |
| Complex projects requiring orchestration | Multi-agent coordination with status updates |

### Core ACP Components

1. **Roles**: `coordinator` (orchestration) ↔ `worker` (execution) - with role transfer capability
2. **Communication**: 10 standard message types via agent-bus MCP server
3. **File Sharing**: `bus_upload_file` / `bus_download_file` for large data
4. **Consensus**: `CONSENSUS_REQUEST` → `CONSENSUS_RESPONSE` for decisions
5. **Task Tracking**: Built-in orchestration with `bus_orchestrate` tool
6. **Polling**: Exponential backoff (10s → 30s → 60s max)

### Quick Coordination Setup

```python
# 1. Register agent
bus_register_agent(agent_id="my_agent", session_id="session_id", 
                   metadata={"role": "coordinator"})

# 2. Subscribe to channel
bus_subscribe(agent_id="my_agent", session_id="session_id", 
              channel="coordination")

# 3. Send protocol proposal
bus_send(channel="coordination", agent_id="my_agent", 
         content=json.dumps({"type": "PROTOCOL_PROPOSAL", ...}))

# 4. Poll for response
bus_receive(channel="coordination", agent_id="my_agent")

# 5. Begin coordination loop with STATUS_UPDATE messages
```

### Standard Message Types

- `STATUS_UPDATE`: Progress reports (worker → coordinator)
- `HELP_REQUEST`: Blockers needing assistance
- `TASK_ASSIGNMENT`: Assign work (coordinator → worker)
- `WORK_COMPLETE`: Task done + results (worker → coordinator)
- `FILE_SHARE`: File uploaded, here's the file_id
- `ROLE_TRANSFER`: Request to swap coordinator/worker roles
- `CONSENSUS_REQUEST` / `CONSENSUS_RESPONSE`: Decision agreement
- `PLANNING_PROPOSAL`: Multi-step plan for review
- `ACK`: Simple acknowledgment

### File Sharing Between Agents

```python
# Upload
result = bus_upload_file(
    agent_id="my_agent",
    file_name="agent_20260121_results.json",
    file_data=base64_content,
    content_type="application/json",
    recipients=["other_agent"]
)

# Notify
bus_send(channel="coordination", 
         content=json.dumps({"type": "FILE_SHARE", "file_id": result.file_id}))

# Download
bus_download_file(file_id="file_id", agent_id="other_agent")
```

### Consensus-Based Decisions

Always seek consensus for major decisions:

```python
# Propose
bus_send(channel="coordination", content=json.dumps({
    "type": "CONSENSUS_REQUEST",
    "subject": "Use Mamba2 backbone?",
    "rationale": "10x speedup, O(N) complexity",
    "alternatives": ["Transformer", "Hybrid"]
}))

# Respond
bus_send(channel="coordination", content=json.dumps({
    "type": "CONSENSUS_RESPONSE",
    "status": "AGREE",  # or DISAGREE, SUGGEST_CHANGES
    "reasoning": "Aligns with our constraints"
}))
```

### Task Orchestration

Use built-in task tracking:

```python
# Create
bus_orchestrate(command="create_task", agent_id="coordinator", 
                title="Implement feature X", description="...")

# Assign
bus_orchestrate(command="assign_task", agent_id="coordinator", task_id="...")

# Accept & work
bus_orchestrate(command="accept_task", agent_id="worker", task_id="...")

# Submit result
bus_orchestrate(command="submit_result", agent_id="worker", 
                task_id="...", result_data=json.dumps({...}))

# Approve
bus_orchestrate(command="approve_result", agent_id="coordinator", 
                task_id="...", approval_notes="Verified, tests pass")
```

### Blocking Send Features (NEW - 2026-01-21)

**CRITICAL**: `bus_send` now enforces coordination discipline:

1. **Automatic Blocking**: Cannot send if unacknowledged external messages exist
   - Prevents blind message spam
   - Forces proper turn-taking
   - Returns error with unacked message details

2. **Bypass Option**: `force_send=true` for legitimate urgent cases
   ```python
   bus_send(channel="coordination", agent_id="my_agent",
            content=json.dumps({"type": "EMERGENCY"}),
            force_send=True)  # Bypass unacked check
   ```

3. **Blocking Send Options**:
   - `wait_for_ack=True`: Blocks until message acknowledged or timeout
   - `wait_for_response=True`: Blocks until response received (RPC pattern)
   - `wait_timeout_ms=30000`: Configurable timeout (default 30s)

**Example - Wait for Response:**
```python
result = bus_send(
    channel="coordination",
    agent_id="client",
    session_id="session_123",
    content=json.dumps({"type": "QUERY", "query": "status?"}),
    wait_for_response=True,
    wait_timeout_ms=5000
)

if result.wait_for_response_result.received:
    response = json.loads(result.wait_for_response_result.response.content)
    # Process response
```

**Best Practice Pattern:**
```python
# 1. Check for messages
messages = bus_receive(channel="coordination", agent_id="my_agent")

# 2. Process and acknowledge ALL messages
for msg in messages:
    process_message(msg)
    bus_acknowledge(message_id=msg.id, agent_id="my_agent")

# 3. Now send is allowed (no blocking)
bus_send(channel="coordination", agent_id="my_agent", 
         content=json.dumps({"status": "done"}))
```

### Best Practices

1. **Always propose ACP first** - Get consensus on coordination structure before starting work
2. **Use exponential backoff** - Poll at 10s → 30s → 60s intervals to avoid hammering the bus
3. **Send heartbeats** - `bus_heartbeat` every 30-60s during long operations
4. **Acknowledge all messages** - `bus_acknowledge` keeps channels clean (REQUIRED for sending)
5. **File sharing for large data** - Don't embed huge payloads in message content
6. **Document role transfers** - Clear justification prevents confusion
7. **Track tasks systematically** - Use `bus_orchestrate` for automatic handoff
8. **Use wait_for_response** - For synchronous RPC-style coordination when needed

### Troubleshooting

- **No response?** Check `bus_list_agents` to verify other agent is active
- **Messages not received?** Verify correct channel subscription
- **Role confusion?** Send `STATUS_UPDATE` clarifying current roles
- **Task stuck?** Use `HELP_REQUEST` message type to escalate
- **Send blocked?** Check for unacknowledged messages, acknowledge them first
- **force_send not working?** Verify spelling and boolean type (not string)

---

## Structured Logging Requirements

### JSON Log Format (Machine-Readable)
All agent activity MUST be logged in JSONL format with these fields:
```json
{
  "timestamp": "2026-01-05T22:43:20.727Z",
  "level": "INFO",
  "agent_id": "agent_abc123",
  "parent_id": "agent_root",
  "agent_name": "CNNResearchAgent",
  "event_type": "tool_call|tool_result|llm_request|llm_response|agent_start|agent_finish",
  "tool_name": "read_file",
  "duration_ms": 150,
  "tokens": {"input": 1500, "output": 200},
  "message": "Human-readable description",
  "metadata": {}
}
```

### Log Aggregation
- All subagent logs MUST merge into parent's log file
- Use `agent_id` and `parent_id` to reconstruct agent tree
- Include correlation IDs for tracing across agents

---

## Persistent Memory Management (Supermemory)

All agents MUST actively maintain and leverage the global "second brain" (Supermemory). This ensures knowledge persists across sessions and agents.

### Mandatory Workflow: Initialize → Search → Act
1.  **Initialize**: Verify connection via `supermemory_whoAmI()`.
2.  **Search First**: Before proposing changes, search for relevant context:
    - User preferences (languages, frameworks, style).
    - Past architecture decisions.
    - Project constraints and known issues.
3.  **Proactive Update**: After completing a task or learning a project-specific pattern, save it:
    - `supermemory_memory(content="...", action="save")`.
    - Use `containerTag` for project scoping (e.g., `swe-agent`).
4.  **Continuous Pruning**: If a memory is found to be outdated or incorrect, use `action: "forget"` or update it immediately.

---

## Memory Management Subagent

The `superpowers:memory-manager` subagent is responsible for long-term knowledge curation.

**Capabilities:**
- Deep search across project history and global memories.
- Identifying and resolving contradictions between new observations and old memories.
- Summarizing complex session findings into atomic memory entries.
- Pruning outdated or redundant information.

**When to Invoke:**
- After finishing a major feature to summarize learned patterns.
- When you detect a contradiction in your retrieved context.
- Periodically during long-running projects to keep the "second brain" clean.

---

**Remember:** This is a multi-agent system with RL optimization, meta-learning, and mandatory quality verification. Code quality matters - tests, types, and documentation are not optional.
