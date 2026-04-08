<div align="center">

# org-as-code

**The missing governance layer for AI agent teams.**

[![Version](https://img.shields.io/badge/version-2.3.1-blue?style=flat-square)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-156%20passed-brightgreen?style=flat-square)](#running-tests)
[![Dependencies](https://img.shields.io/badge/dependencies-2-orange?style=flat-square)](#quick-start)
[![MCP Tools](https://img.shields.io/badge/MCP%20tools-21-purple?style=flat-square)](#mcp-server)
[![CLI Commands](https://img.shields.io/badge/CLI%20commands-23-teal?style=flat-square)](#cli-reference)

*Turn organizational decisions into Git-native, auditable, version-controlled artifacts.*
*Replace Slack threads and meeting notes with YAML files and immutable audit trails.*

[Quick Start](#quick-start) · [Documentation](#the-pv-protocol) · [MCP Server](#mcp-server) · [Theory](THEORY.md) · [Examples](examples/)

</div>

---

```bash
pip3 install -r requirements.txt        # pyyaml + mcp — that's it
python3 org_cli.py dashboard            # see it live in 30 seconds
python3 org_cli.py create FEAT-003 feature "My first process" "Testing org-as-code" --agent coder
python3 org_cli.py verify               # check hash-chain integrity
```

---

## The Problem

Organizations exist in people's heads — in Slack threads, meeting notes, and shared assumptions that dissolve the moment someone leaves the room. AI agents can't operate on that. They need something explicit, versioned, and machine-readable.

org-as-code encodes roles, responsibilities, and decision-making as Git-native artifacts — not as documentation, but as a system agents can actually run on.

---

## Why This Exists

AI agents can code, review, deploy, and monitor. But they can't govern themselves — and neither can humans govern them through Slack threads and meeting notes.

org-as-code gives AI agents and humans a shared protocol for working together. One agent proposes, another validates. A human steps in when stakes are high. Every handoff is logged, every decision is traceable, every rejection drives a better next iteration. The P↔V protocol doesn't care whether a step is performed by Claude, GPT, a junior developer, or a CTO — it enforces the same rhythm of propose, validate, converge.

This means you can build teams where AI agents do the heavy lifting (proposals, implementations, reviews) while humans retain authority over critical decisions — without bottlenecking every step. The `feature_v2` template demonstrates this: AI handles routine work autonomously, but processes with H(s) ≥ 0.8 automatically require human sign-off.

### What you get

| | Feature | What it does |
|---|---------|-------------|
| **P↔V** | Protocol | Proposals oscillate with Validations — expand options, then contract to decisions |
| **H(s)** | Priority Score | Calculated, not felt: `w₁·urgency + w₂·commitment + w₃·demand + w₄·blocking` |
| **E(x)** | Convergence Score | Quadratic energy — one critical gap outweighs three minor ones |
| **AHP** | Decision Engine | Multi-participant pairwise comparison with weighted aggregation and consistency check |
| **FDM** | Process Dependencies | Cycle detection, parallel groups, impact scoring, living dependency register |
| **JSONL** | Audit Trail | SHA-256 hash-chained, append-only, tamper-evident |
| **MCP** | AI Agent Interface | 21 native tools over stdio transport |
| **CLI** | Human Interface | 23 commands for operators |
| **YAML** | Per-Process State | One state file per process — no merge conflicts |

---

## Pairwise Decision Engine

When a decision involves multiple stakeholders with different weights, a single H(s) score isn't enough. `org_decision.py` adds AHP-based (Analytic Hierarchy Process) pairwise comparison as a native `decision` process type — no new dependencies beyond what org-as-code already requires.

Each participant compares options head-to-head. The engine computes individual priority vectors, checks consistency (CR < 0.10 is consistent), and aggregates via weighted geometric mean. Every comparison, every vote, and the final consensus are written as YAML artifacts and hash-chained into the audit trail.

```bash
# 1. Create a decision session
python3 org_decision.py session --id DEC-001 \
  --options "Option A" "Option B" "Option C" \
  --agent facilitator --title "Q2 infrastructure choice"

# 2. Each participant votes interactively
python3 org_decision.py vote --id DEC-001 --participant alice
python3 org_decision.py vote --id DEC-001 --participant bob --role nedxis

# 3. Aggregate into consensus (excludes CR >= 0.20 by default)
python3 org_decision.py aggregate --id DEC-001 --agent facilitator

# 4. Show result
python3 org_decision.py show --id DEC-001
```

Produces in `processes/DEC-001/`:
- `P.0_decision_session.yaml` — options, context, session metadata
- `V.N_vote_<participant>.yaml` — individual comparisons, priority vector, CR per voter
- `V.final_consensus.yaml` — aggregated ranking, verdict, mean CR, participant list

Role weights are configurable: `nedxis` participants carry 1.5× weight by default. All artifacts are hash-chained — the decision is as auditable as any other process in the system.

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/syntriad/org-as-code.git
cd org-as-code
pip3 install -r requirements.txt
```

### 2. Register your agents

Edit `registry/agents.yaml`:

```yaml
agents:
  - id: alice
    name: "Alice (Team Lead)"
    type: human
    interface: terminal
    skills: [strategic-review, final-validation, decision-making]
    status: active
    capacity: 5

  - id: coder
    name: "Coder (AI Agent)"
    type: ai
    interface: claude-code
    skills: [implementation, code-generation, testing]
    status: active
    capacity: 10

  - id: reviewer
    name: "Reviewer (AI Agent)"
    type: ai
    interface: ide
    skills: [technical-review, refactoring, testing]
    status: active
    capacity: 5
```

### 3. Start your first process

```bash
python3 org_cli.py create FEAT-001 feature \
  "Add user authentication" \
  "Implement JWT-based auth with refresh tokens" \
  --agent coder --priority 0.8
```

### 4. Let an AI agent validate

The assigned agent creates `V.0_review.yaml` in `processes/FEAT-001/`:

```yaml
verdict: approved
confidence: 0.9
conditions:
  - "Add rate limiting before merge"
reviewed_by: reviewer
```

### 5. Commit and verify

```bash
python3 org_cli.py update FEAT-001 COMMITTED \
  --notes "Auth module complete. All tests pass."

python3 org_cli.py verify
# Chain integrity: VALID — no tampering detected.
```

Every step is logged in `registry/artifacts.jsonl` with hash-chain integrity.

---

## The P↔V Protocol

Like a heartbeat (systole ↔ diastole), organizations need rhythm:

```
┌─────────────────────────────────────────┐
│              P ↔ V Protocol             │
├─────────────────────────────────────────┤
│                                         │
│   ┌─────────┐         ┌─────────┐      │
│   │    P    │────────▶│    V    │      │
│   │ Propose │         │Validate │      │
│   └─────────┘         └────┬────┘      │
│        ▲                   │           │
│        │     iterate       │           │
│        └───────────────────┘           │
│                                         │
│   P_READY → P_COMPLETE → V_COMPLETE    │
│                    → COMMITTED          │
│                    → ABANDONED          │
│                                         │
└─────────────────────────────────────────┘
```

**P-steps** (Production): Expand options, generate proposals
**V-steps** (Validation): Contract, validate, select

This prevents two failure modes:
- **Unbounded divergence** — chaos, no decisions
- **Frozen convergence** — premature fixation, groupthink

---

## Repository Structure

```
org-as-code/
├── registry/                ← Organization state (YAML)
│   ├── agents.yaml            Registered agents
│   ├── state.yaml             Process states (auto-generated index in per_process mode)
│   ├── tensions.yaml          Open problems/opportunities
│   ├── attractors.yaml        Strategic goals
│   ├── artifacts.jsonl        Immutable action log (hash-chained)
│   └── fdm.json               Dependency register (auto-generated)
│
├── processes/               ← All work (P,V artifacts + per-process state)
│   └── {ID}/
│       ├── state.yaml           Per-process state (in per_process mode)
│       ├── P.0_proposal.md      Proposals (expand)
│       └── V.0_review.yaml      Reviews (contract)
│
├── examples/                ← Real cases from independent evaluations
│   ├── SEC-001/               Security hardening (V.0 rejected, 6 artifacts)
│   └── PERF-001/              Performance optimization (V.0 rejected, 4 artifacts)
│
├── protocol/                ← Rules of the game
│   ├── config.yaml            Priority weights, thresholds, security
│   └── process_templates/     Reusable flows
│       ├── feature.yaml
│       ├── feature_v2.yaml    (with conditional human gate)
│       ├── bugfix.yaml
│       └── decision.yaml      (pairwise AHP — multi-participant)
│
├── tools/                   ← Maintenance utilities
│   ├── fix_hash_chain.py      Rebuild hash chain if corrupted
│   └── per_process_state.py   Migrate to per-process state files
│
├── org_mcp_server.py        ← MCP Server (21 tools, stdio transport)
├── org_cli.py               ← CLI interface (23 commands)
├── org_decision.py          ← Pairwise AHP decision engine
├── fdm.py                   ← Dependency graph engine (Tarjan, Kahn, stdlib only)
├── pyproject.toml           ← Package metadata (pip install org-as-code)
└── requirements.txt
```

---

## CLI Reference

```bash
# Read commands
python3 org_cli.py status              # All processes + state
python3 org_cli.py tensions            # Open tensions
python3 org_cli.py attractors          # Strategic goals
python3 org_cli.py agents              # Registered agents
python3 org_cli.py health              # Health metrics
python3 org_cli.py log [--limit N]     # Recent artifact log
python3 org_cli.py show FEAT-001       # Process detail + artifacts
python3 org_cli.py verify              # Verify hash-chain integrity
python3 org_cli.py dashboard           # Combined overview

# Write commands
python3 org_cli.py create FEAT-002 feature "Title" "Description" --agent coder --priority 0.8
python3 org_cli.py update FEAT-002 COMMITTED --notes "Done"
python3 org_cli.py artifact coder V.0_review "Approved" --process FEAT-002
python3 org_cli.py tension-add "Title" "Description" --priority 0.7
python3 org_cli.py tension-resolve T-2026-001 "Resolved via process assignment"
python3 org_cli.py priority --urgency 0.9 --demand 0.8 --blocking 0.7
python3 org_cli.py energy --gaps 0.8 --inconsistencies 0.2 --evidence 0.6
python3 org_cli.py convergence FEAT-001  # Show E(x) trajectory over V-steps

# Dependency commands
python3 org_cli.py deps-add FEAT-002 FEAT-001      # FEAT-002 depends on FEAT-001
python3 org_cli.py deps-remove FEAT-002 FEAT-001    # Remove dependency
python3 org_cli.py deps FEAT-002                     # Show upstream + downstream deps
python3 org_cli.py deps-analyze                      # Full FDM analysis (parallel groups, cycles)

# Git sync
python3 org_cli.py sync                          # Pull only
python3 org_cli.py sync "commit message" --agent coder  # Commit + push

# Decision engine (pairwise AHP)
python3 org_decision.py session --id DEC-001 --options "A" "B" "C" --agent facilitator
python3 org_decision.py vote --id DEC-001 --participant alice
python3 org_decision.py aggregate --id DEC-001 --agent facilitator
python3 org_decision.py show --id DEC-001
```

---

## MCP Server

`org_mcp_server.py` gives AI agents native tools to interact with the organization:

### Read tools

| Tool | Description |
|------|-------------|
| `org_read_state` | Read all process states |
| `org_read_tensions` | Read open problems/opportunities |
| `org_read_attractors` | Read strategic goals |
| `org_read_agents` | Read registered agents |
| `org_read_health` | Read system health metrics |
| `org_read_process` | Read a specific process and its artifacts |
| `org_read_artifacts` | Read recent entries from audit log |
| `org_read_convergence` | Read convergence history (E(x) over V-steps) |

### Write tools

| Tool | Description |
|------|-------------|
| `org_update_state` | Update process state |
| `org_create_process` | Create a new process (P.0), optional `depends_on` |
| `org_log_artifact` | Append to immutable audit log |
| `org_create_tension` | Register a new tension |
| `org_resolve_tension` | Mark a tension as resolved |
| `org_add_dependency` | Add a dependency between two processes |
| `org_remove_dependency` | Remove a dependency between two processes |

### Governance tools

| Tool | Description |
|------|-------------|
| `org_calculate_priority` | Calculate priority score H(s) |
| `org_calculate_energy` | Calculate convergence score E(x) |
| `org_verify_chain` | Verify hash-chain integrity of audit log |
| `org_analyze_dependencies` | Full dependency graph analysis (cycles, parallel groups, bottleneck) |
| `org_read_dependencies` | Read the current FDM dependency register |
| `org_git_sync` | Pull, commit, push changes |

### Setup

Add to your Claude Code MCP configuration (`~/.claude.json`):

```json
{
  "mcpServers": {
    "org-as-code": {
      "command": "python3",
      "args": ["org_mcp_server.py"],
      "env": {
        "ORG_REPO_PATH": "/path/to/org-as-code"
      }
    }
  }
}
```

---

## Priority Score

Priority is not a feeling. It is a calculation:

```
H(s) = w₁·urgency + w₂·commitment + w₃·demand + w₄·blocking
```

Default weights (configurable in `protocol/config.yaml`):

| Weight | Value | Component |
|--------|-------|-----------|
| w₁ | 0.30 | Urgency — how time-sensitive |
| w₂ | 0.20 | Commitment — how invested we are |
| w₃ | 0.30 | Demand — external need |
| w₄ | 0.20 | Blocking — how much this blocks other work |

Thresholds:
- **H(s) ≥ 0.8** → Escalate to human
- **H(s) ≥ 0.5** → Action required
- **H(s) < 0.5** → Low priority

*(Formally: the Semantic Hamiltonian — see [THEORY.md](THEORY.md))*

---

## Convergence Score

The priority score tells you *what to work on*. The convergence score tells you *how far from done*:

```
E(x) = w_g·gaps² + w_i·inconsistencies² + w_u·uncertainty² − w_e·evidence²
```

The quadratic penalty means one critical gap (0.9² = 0.81) outweighs three minor ones (3 × 0.3² = 0.27).

| Threshold | Meaning |
|-----------|---------|
| E(x) < 0.10 | Ready to commit |
| E(x) < 0.30 | Minor revision needed |
| E(x) ≥ 0.30 | Major revision needed |

V-step reviews include convergence scores (gaps, inconsistencies, uncertainty, evidence). The system auto-calculates E(x) and tracks it per process, showing whether iterations are converging, stagnating, or diverging.

*(Formally: Semantic Energy — see [THEORY.md](THEORY.md))*

---

## Process Dependencies (FDM)

Processes can declare dependencies on each other. The system analyzes the dependency graph to detect cycles, compute parallel execution groups, and identify bottlenecks.

```bash
# FEAT-002 depends on FEAT-001
python3 org_cli.py deps-add FEAT-002 FEAT-001

# Full analysis: parallel groups, cycles, critical path
python3 org_cli.py deps-analyze
```

The dependency graph uses Tarjan's SCC algorithm for cycle detection and Kahn's algorithm for topological sort. When circular dependencies are found, the system proposes which edge to break.

**Parallel groups** show which processes can execute simultaneously:
```
Group 1: FEAT-001, BUG-001     (no dependencies — start immediately)
Group 2: FEAT-002              (depends on FEAT-001)
Group 3: FEAT-003              (depends on FEAT-002)
```

**Impact scoring** identifies bottleneck processes — the ones whose delay cascades through the most downstream dependents.

The dependency register (`registry/fdm.json`) is auto-generated after every dependency change, providing a persistent, git-versioned view of the dependency graph.

---

## Hash-Chain Audit Trail

Every action is logged in `registry/artifacts.jsonl` with cryptographic linking:

```json
{
  "type": "p_step",
  "agent": "coder",
  "process_id": "FEAT-001",
  "action": "P.0_proposal",
  "description": "Add user authentication",
  "timestamp": "2026-03-16T14:30:00Z",
  "prev_hash": "a3f8c1...",
  "entry_hash": "7b2e4d..."
}
```

Each entry's `entry_hash` = SHA-256(`prev_hash` + canonical JSON of entry).
Chain integrity is verifiable via CLI (`python3 org_cli.py verify`) or MCP (`org_verify_chain`).

Tamper-evident. Append-only. Git-versioned.

---

## Process Templates

### Feature flow

```yaml
name: feature
steps:
  - name: "P.0 — Proposal"
    artifact: "P.0_proposal.md"
    agent_types: [ai, human]
    next_state: P_COMPLETE

  - name: "V.0 — Review"
    artifact: "V.0_review.yaml"
    agent_types: [ai, human]
    next_state: V_COMPLETE
    requires: P_COMPLETE

  - name: "P.1 — Implementation"
    artifact: "P.1_implementation.md"
    agent_types: [ai]
    next_state: P_COMPLETE

  - name: "V.1 — Verification"
    artifact: "V.1_verification.yaml"
    agent_types: [ai, human]
    next_state: COMMITTED
    requires: P_COMPLETE
```

### Feature v2 flow (with conditional human gate)

`feature_v2.yaml` adds priority-conditional human approval: processes with H(s) ≥ 0.8 require human sign-off at V.1, while low-risk processes (H(s) < 0.5) permit AI-only verification.

### Decision flow (pairwise AHP)

`decision.yaml` runs a multi-participant pairwise comparison session. Each participant submits individual comparisons; the facilitator aggregates into a consensus ranking. Inconsistent votes (CR ≥ 0.20) are flagged and excluded by default. See [Pairwise Decision Engine](#pairwise-decision-engine) above.

---

## Theoretical Foundation

org-as-code is an implementation of the [SYNTRIAD metapattern](https://github.com/SYNTRIAD/genesis):

```
T : (S, I, C) → S'    — every process is a transformation
Ω = P ↔ V             — governed by expand-contract rhythm
V++                    — only validated work persists
```

The priority score H(s) and convergence score E(x) used throughout this tool are domain-specific instances of the Semantic Hamiltonian and Semantic Energy described in [Semantic Thermodynamics](https://zenodo.org/records/17618208).

For the design rationale — why quadratic scoring, how the audit trail works, and where the pattern comes from — see [THEORY.md](THEORY.md).

---

## Use Cases

- **Multi-agent collaboration** — Agent A writes code, Agent B reviews it, a human approves if the priority score warrants it. Each handoff is a P→V transition with a convergence score.
- **Human-AI teams** — Humans set direction (attractors, tensions), AI agents execute (P-steps), either party validates (V-steps). The protocol scales from solo developer + one AI to full teams.
- **Async-first organizations** — Replace meetings with validated artifacts. Every decision has a traceable audit trail, not a Slack thread.
- **Compliance-sensitive environments** — Immutable decision logs for AI Act, SOC2, NIS2. The hash chain provides tamper-evident records of who decided what, when, and why.
- **Multi-stakeholder decisions** — Strategic choices involving participants with different roles and weights. Pairwise AHP produces a mathematically grounded, auditable consensus rather than a show of hands.

---

## Real-World Examples

The `examples/` directory contains two cases from independent evaluations where the P↔V protocol caught errors before they shipped:

- **[SEC-001](examples/SEC-001/)** — A security proposal with two architectural errors (wrong injection model, wrong threat model). The V-step rejected it. The revised proposal shipped the correct fix. 6 artifacts over 5 days.
- **[PERF-001](examples/PERF-001/)** — A performance optimization that would have shipped at the accuracy constraint floor. The V-step caught the cliff-edge risk. The revised plan used ONNX quantization instead, maintaining a safety margin. 4 artifacts over 4 days.

See [`examples/README.md`](examples/README.md) for the full story.

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

156 tests cover security validation, hash-chain integrity, agent registration, input sanitization, energy calculation, convergence tracking, per-process state, auto-migration, dependency validation, graph analysis, FDM integration, and CLI commands.

---

## Integration

org-as-code provides two integration paths — no bridge layer needed.

### MCP-compatible agents (Claude Code, Windsurf, Gemini)

Use the MCP server. See [Setup](#setup) above. The agent gets 21 native tools over stdio — no code required.

### Python agent frameworks (LangChain, CrewAI, custom)

Import directly from the server module:

```python
from org_mcp_server import (
    org_create_process,
    org_log_artifact,
    org_update_state,
    org_add_dependency,
    org_analyze_dependencies,
    org_calculate_energy,
    org_read_convergence,
    org_verify_chain,
)

# Create processes with dependencies
org_create_process("FEAT-010", "feature", "Add caching", "Redis-based response cache", "my-agent")
org_create_process("FEAT-011", "feature", "Add cache invalidation", "TTL + event-based", "my-agent",
    depends_on="FEAT-010")

# Add dependency after creation
org_add_dependency("FEAT-012", "FEAT-010")

# Analyze dependency graph
print(org_analyze_dependencies())  # parallel groups, cycles, critical path, bottleneck

# Log a V-step with convergence scores
org_log_artifact("reviewer", "V.0_review", "Approved with conditions",
    process_id="FEAT-010",
    extra='{"convergence": {"gaps": 0.3, "inconsistencies": 0.1, "uncertainty": 0.2, "evidence": 0.7}}')
```

Set `ORG_REPO_PATH` before importing, or the module defaults to its own directory.

### Non-Python frameworks

Call `org_cli.py` as a subprocess. All 23 CLI commands return structured, parseable output.

---

## Known Limitations

We believe in transparency about what this system does and does not do well today:

- **State storage modes.** Default mode is `per_process`: each process stores its state in `processes/{ID}/state.yaml`, eliminating concurrent write conflicts. `registry/state.yaml` is auto-generated as a read-only index. Legacy `monolithic` mode is available via `state_storage.mode: monolithic` in `protocol/config.yaml`. Existing monolithic deployments auto-migrate on first write when switching to `per_process` mode. Advisory file locking (`fcntl.flock`) protects against write corruption within a single host.

- **Security mode.** All write tools validate agent IDs against `registry/agents.yaml` and process IDs against format `[A-Z]+-[0-9]+`. Default mode is `permissive` (logs warnings for unknown agents). Set `security.mode: strict` in `protocol/config.yaml` for production use, which rejects unknown agents from all write tools except `org_log_artifact` (bootstrap exemption). Commit messages are sanitized to prevent git trailer injection.

- **Local tamper-evidence only.** The hash chain provides tamper-evidence but not tamper-resistance. A repository admin with force-push access can rewrite history. For stronger guarantees, consider anchoring chain tips to an external timestamping authority.

- **Decision engine.** `org_decision.py` is a standalone script, not yet integrated into the MCP server toolset. Voting is interactive (terminal) only — no API or async submission path yet.

---

## Roadmap

v2.3 provides auditable execution: every decision is logged, hash-chained, and verifiable. The next major capability is **auditable authority** — formalizing *who may decide*, not just *what was decided*.

Planned for a future release:

- **Formal role model** — producer, validator, approver, senior approver, auditor as first-class concepts with explicit authorities
- **Decision artifacts** — `D.0_decision.yaml` as auditable governance events, not just V-step metadata
- **Policy-based commit control** — process type and risk class determine required approvals, role coverage, and human gates
- **Conflict resolution** — when validators disagree: consensus, senior override, or mandatory escalation
- **Separation of duties** — a producer may not self-approve; security processes require senior sign-off
- **MCP tools for decision engine** — expose `org_decision` session, vote, and aggregate as MCP tools for agent-driven decision workflows

This extends the state machine with a `DECISION_PENDING` state between `V_COMPLETE` and `COMMITTED`, and formalizes the relationship between the existing H(s) human gate and the governance layer.

The governance layer will ship when real multi-validator conflict use cases drive the design — not before.

---

## License

MIT — See [LICENSE](LICENSE). The governance protocol is open. Use it freely.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests that demonstrate real usage are preferred over theoretical discussion.

---

<div align="center">

**org-as-code** is part of the [SYNTRIAD](https://github.com/SYNTRIAD) ecosystem.

[![GitHub](https://img.shields.io/badge/GitHub-SYNTRIAD%2Forg--as--code-181717?style=flat-square&logo=github)](https://github.com/SYNTRIAD/org-as-code)
[![Zenodo](https://img.shields.io/badge/Paper-Semantic%20Thermodynamics-blue?style=flat-square&logo=zenodo)](https://zenodo.org/records/17618208)

</div>
