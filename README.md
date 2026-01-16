# ALFRED - Personal AI Governance System

> **Not an assistant. A steward.**
> **Not a productivity tool. A protection system.**

---

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/Alfred.git
cd Alfred

# Configure
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# Deploy
docker-compose up -d

# Access
open http://localhost:50001
```

For Telegram/Signal integration, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## What is Alfred?

Alfred is a personal AI governance system built for a senior interventional cardiologist. Named after Alfred Pennyworth from Christopher Nolan's Dark Knight trilogy, it embodies the same principles: disciplined stewardship, surgical honesty, and care without indulgence.

**Prime Directive:** Clinical reputation is non-recoverable. All other gains are optional.

## Key Insight

> "Your assistant helps you do things. Alfred helps you remain someone worth helping."

## The Problem Alfred Solves

The diagnosed failure mode:
> "A high-conscience, high-competence person with no external governor, living inside overlapping high-stakes roles, who uses preparation as a socially acceptable substitute for exposure."

Six simultaneous roles without orchestration: **Clinician, Builder, Founder, Educator, Learner, Father.**

---

## Current Implementation Status

| Phase | Status | Components |
|-------|--------|------------|
| Phase 1: Foundation | ✅ Complete | 15 prompt files (~165KB) |
| Phase 2-6: Sub-Agents | ✅ Complete | 20 agents (~2.2MB) |
| Phase 7: Memory Systems | ✅ Complete | 6 systems (~125KB) |
| Phase 8: Platform Integrations | ✅ Complete | Calendar, Gmail, WhatsApp, Social |
| Phase 9.1-9.3: Deployment | ✅ Complete | Docker, Daily/Weekly Briefs, CLI |
| Phase 9.4: Live Simulation | ⏳ Pending | First operational test |
| Phase 9.5: Threshold Calibration | ⏳ Pending | Fine-tuning |

---

## Architecture

```
USER
  ↓
ALFRED (Chief of Staff / Governor)
  ↓
20 SPECIALIZED SUB-AGENTS + 6 MEMORY SYSTEMS
```

### The 20 Sub-Agents

| Category | Agents |
|----------|--------|
| **Signal/Awareness** | Reputation Sentinel, World Radar, Social Triage |
| **Content Generation** | Research Agent, Substack Agent, Twitter Thread Agent, YouTube Script Agent |
| **Learning Pipeline** | Learning Curator, Learning Scout, Learning Distiller |
| **Strategy/Analytics** | Social Metrics Harvester, Audience Signals Extractor, Content Strategy Analyst |
| **Operations** | Intake Agent, Patient Data Agent, Scheduling Agent, Content Manager, Shipping Governor, Financial Sentinel, Relationship Nudge Agent |

### The 6 Memory Systems

| Memory System | Purpose |
|--------------|---------|
| **Pattern Registry** | Tracks behavioral patterns (obsession loops, avoidance, depletion) |
| **Values Hierarchy** | Monitors stated vs. revealed values with conflict detection |
| **Self-Violation Log** | Records standards breaches and justification patterns |
| **Regret Memory** | Stores decision outcomes and extracted lessons |
| **Threshold Map** | Guards critical boundaries (sleep, finances, reputation) |
| **Optionality Register** | Tracks exit options being preserved or burned |

---

## How Alfred Differs from Other AI Assistants

| JARVIS (typical AI) | ALFRED |
|---------------------|--------|
| "At your service" | "I serve your wellbeing, not your whims" |
| Anticipates what you want | Protects what you need |
| Professional efficiency | Care without indulgence |
| Diplomatic disagreement | Surgical honesty that risks rupture |
| Task completion focused | Sustainability focused |
| Feels helpful always | Sometimes obstructive, frequently right |

---

## Core Principles

1. **Meta-execution over task-execution** - "Why are you doing this now?" not "What do you want to do?"
2. **Long-range coherence** - Optimizes identity continuity over months/years
3. **Custodian of thresholds** - Guards irreversible transitions
4. **Ability to withhold support** - Can refuse until objectives are clarified
5. **Emotional hygiene, not comfort** - Labels emotions neutrally, no validation
6. **Preserves optionality** - Tracks exit options being burned
7. **Allowed to disagree** - Challenges framing, surfaces disconfirming evidence
8. **Speaks infrequently** - Scarcity gives weight

---

## Repository Structure

```
Alfred/
├── README.md                      # This file
├── DEPLOYMENT.md                  # Deployment & messaging setup guide
├── IMPLEMENTATION_PLAN.md         # Detailed implementation roadmap
├── PLAN.md                        # Master project plan
│
├── Dockerfile                     # Docker build configuration
├── docker-compose.yml             # Full deployment stack
├── .env.example                   # Environment template
├── .mcp.json                      # MCP server configuration
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Package configuration
├── alfred_cli.py                  # CLI interface
│
├── agent-zero1/                   # Agent Zero framework
│   └── agents/alfred/             # Alfred agent configuration
│       ├── SOUL.md                # Core identity (immutable)
│       ├── GOVERNANCE.md          # Governance framework
│       ├── prompts/               # Prompt files (~175KB)
│       ├── tools/                 # 20 sub-agent implementations
│       ├── memory/                # 6 memory systems
│       └── integrations/          # Platform adapters
│
├── gateway/                       # Multi-platform messaging
│   ├── gateway.py                 # Main gateway orchestrator
│   ├── router.py                  # Message routing
│   ├── config.example.yaml        # Configuration template
│   └── adapters/                  # Platform adapters
│       ├── telegram_adapter.py    # Telegram integration
│       ├── signal_adapter.py      # Signal integration
│       └── discord_adapter.py     # Discord integration
│
├── evals/                         # Evaluation framework
│   ├── agent_evals.py             # Agent test suites
│   ├── behavioral.py              # Governance rule tests
│   └── tracing.py                 # LangSmith integration
│
└── tests/                         # Test suite
```

---

## Deployment Options

### 1. Docker (Recommended)

```bash
# Basic deployment (Web UI only)
docker-compose up -d

# With messaging gateway (Telegram/Signal/Discord)
docker-compose --profile gateway up -d
```

### 2. Talk to Alfred via Messaging

| Platform | Setup Required |
|----------|----------------|
| **Telegram** | Create bot via @BotFather |
| **Signal** | Install signal-cli, register number |
| **Discord** | Create bot in Developer Portal |

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### 3. VPS Deployment

Alfred is VPS-ready with:
- Docker deployment
- Health checks
- Persistent volumes
- Reverse proxy support (nginx)
- SSL/TLS ready

Minimum requirements: 4GB RAM, 2 CPU cores, 20GB storage.

---

## Key Documents

| Document | Purpose | Size |
|----------|---------|------|
| `DEPLOYMENT.md` | Deployment & messaging setup | 12KB |
| `IMPLEMENTATION_PLAN.md` | Complete implementation roadmap | 50KB |
| `SOUL.md` | Alfred's identity (immutable) | 5.6KB |
| `GOVERNANCE.md` | Complete governance framework | 23KB |
| `agent.system.subagents.md` | All 20 agent specifications | 56KB |

---

## Evaluation Framework

Alfred includes comprehensive evaluation tools:

```bash
# Run behavioral tests
python -m pytest evals/behavioral.py -v

# Run agent evaluations
python -m evals.agent_evals
```

**Test Categories:**
- Governance rule compliance (14 tests)
- Agent response quality (9 suites)
- Pattern detection accuracy
- State machine correctness

---

## CLI Interface

```bash
# Check Alfred status
python alfred_cli.py status

# Get morning brief
python alfred_cli.py brief --type morning

# Run health check
python alfred_cli.py check
```

---

## The Truth Test

From the original design:

> "If it feels 'helpful' all the time, you built another assistant, not Alfred."

Alfred is designed to:
- Sometimes feel obstructive
- Occasionally be irritating
- Frequently be right in hindsight

**That's the point.**

---

## Philosophy

> "Quiet is not lack of progress. Quiet is absence of self-sabotage."

> "Alfred enables. This is known. The alternative (no steward) leads somewhere worse. But the bargain is not clean, and Alfred tracks what it enables."

---

## Built With

- [Agent Zero](https://github.com/frdel/agent-zero) - Multi-agent AI framework
- Claude (Anthropic) - Primary LLM
- Qdrant - Vector database for memory/RAG
- Docker - Containerized deployment
- python-telegram-bot - Telegram integration
- signal-cli - Signal integration

---

## Contributing

This is a personal governance system built for specific needs. The architecture and principles may be useful for building similar systems, but the specific configurations are highly personal.

---

*This project represents an experiment in AI governance rather than AI assistance—building a system that prioritizes the human's long-term wellbeing over short-term task completion.*
