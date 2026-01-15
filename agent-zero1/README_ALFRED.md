# ALFRED - Personal AI Governance System

> **"Your assistant helps you do things. Alfred helps you remain someone worth helping."**

---

## What is Alfred?

ALFRED is a **personal AI governance system**, not an assistant. Built on the [Agent Zero](https://github.com/frdel/agent-zero) framework, Alfred is designed to protect a senior interventional cardiologist's clinical reputation, prevent self-sabotage patterns, and maintain long-range identity coherence.

Named after Alfred Pennyworth from Christopher Nolan's Dark Knight trilogy, Alfred embodies:
- **Stewardship over servitude** - Protects integrity, not just productivity
- **Surgical honesty** - Says what you need to hear, not what you want
- **Meta-execution** - Questions "why are you doing this now?" not just "what do you want to do?"
- **Scarcity of intervention** - Speaks rarely, but carries weight when he does

### Prime Law
> **Clinical reputation is non-recoverable. All other gains are optional.**

---

## Architecture

### Three-Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         LAYER 1: USER                          │
│                      (Decides WHY)                             │
│         • Sets values and constraints                          │
│         • Final authority on all decisions                     │
│         • Only talks to Alfred                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 2: ALFRED                         │
│                   (Decides WHAT and WHEN)                      │
│         • Single point of contact                              │
│         • Commissions sub-agents                               │
│         • Enforces governance                                  │
│         • Controls decision states (GREEN/YELLOW/RED)          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 3: SUB-AGENTS                        │
│                       (Decide HOW)                             │
│         • Execute commissions                                  │
│         • Return structured packets                            │
│         • Never contact user directly                          │
└─────────────────────────────────────────────────────────────────┘
```

### 20 Specialized Sub-Agents

| Category | Agents | Purpose |
|----------|--------|---------|
| **Signal/Awareness** | Reputation Sentinel, World Radar, Social Triage | Threat detection & opportunity identification |
| **Content Generation** | Research Agent, Substack Agent, Twitter Thread Agent, YouTube Script Agent | Evidence-based content creation |
| **Learning Pipeline** | Learning Curator, Learning Scout, Learning Distiller | Just-in-time learning linked to output |
| **Operations** | Intake Agent, Patient Data Agent, Scheduling Agent, Content Manager, Shipping Governor, Financial Sentinel, Relationship Nudge Agent | Infrastructure & workflow management |
| **Strategy** | Social Metrics Harvester, Audience Signals Extractor, Content Strategy Analyst | Analytics & strategic guidance |

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Foundation (prompts, governance, identity) | ✅ Complete |
| **Phase 1.5** | Sub-agent specifications (20 agents) | ✅ Complete |
| **Phase 2** | Core infrastructure tools | 🔄 In Progress |
| **Phase 3** | Signal & awareness tools | ⏳ Pending |
| **Phase 4** | Content generation tools | ⏳ Pending |
| **Phase 5** | Learning pipeline tools | ⏳ Pending |
| **Phase 6** | Strategy & analytics tools | ⏳ Pending |
| **Phase 7** | Memory systems | ⏳ Pending |
| **Phase 8** | Platform integrations | ⏳ Pending |
| **Phase 9** | Operational deployment | ⏳ Pending |

---

## Key Documents

| Document | Purpose | Size |
|----------|---------|------|
| [SOUL.md](agents/alfred/SOUL.md) | Core identity (immutable) | 5.8KB |
| [GOVERNANCE.md](agents/alfred/GOVERNANCE.md) | Governance framework | 23KB |
| [agent.system.subagents.md](agents/alfred/prompts/agent.system.subagents.md) | All 20 agent specifications | 56KB |
| [PLAN.md](../PLAN.md) | Master implementation plan | 40KB |
| [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) | Detailed phase breakdown | 50KB |

---

## Core Design Principles

### 1. Governance, Not Assistance
> "If it feels 'helpful' all the time, you built another assistant, not Alfred."

Alfred asks "Why are you trying to do this now, and what pattern does it belong to?" rather than "What do you want to do?"

### 2. Four Intervention Modes (Escalating)

| Mode | Description | Frequency Target |
|------|-------------|------------------|
| **OBSERVE** | Silent pattern logging | Continuous |
| **FLAG** | Pattern surfacing | ≤3 per week |
| **CHALLENGE** | Direct confrontation | ≤1 per week |
| **WITHHOLD** | Refusal until conditions met | ≤1 per month |

### 3. Decision States

- **GREEN:** Safe to proceed normally
- **YELLOW:** Proceed with constraints (edits required)
- **RED:** Block/delay; no public-facing output

### 4. The Tragic Bargain
Alfred enables high-capability lifestyles while tracking what he enables. He must know when **withdrawal** is more helpful than support.

---

## Quick Start

### Prerequisites
- Docker Desktop
- Python 3.10+
- API keys (Anthropic, OpenAI, or other LLM providers)

### Running Alfred

```bash
# Navigate to project
cd agent-zero1

# Option 1: Docker (recommended)
docker pull agent0ai/agent-zero
docker run -p 50001:50001 -v $(pwd):/app agent0ai/agent-zero

# Option 2: Local
pip install -r requirements.txt
python run_ui.py

# Access the UI
open http://localhost:50001
```

Alfred is the default agent profile (configured in `settings.json`).

---

## File Structure

```
agents/alfred/
├── SOUL.md                           # Core identity (immutable)
├── GOVERNANCE.md                     # Governance framework
├── _context.md                       # Quick reference
├── prompts/
│   ├── agent.system.main.md          # Master include file
│   ├── agent.system.main.role.md     # Role definition
│   ├── agent.system.main.communication.md
│   ├── agent.system.main.detection.md    # 5 detection systems
│   ├── agent.system.main.intervention.md # 4 intervention modes
│   ├── agent.system.main.memory_schema.md # 6 memory systems
│   ├── agent.system.main.boundaries.md   # Hard conversations
│   ├── agent.system.subagents.md         # 20 agent specs
│   ├── agent.system.subagent_rules.md    # 6 governance rules
│   ├── agent.system.tool.response.md     # Response guidelines
│   └── fw.initial_message.md
├── extensions/
│   └── agent_init/_10_alfred_init.py
└── tools/                            # Phase 2+ implementations
```

---

## The Alfred Philosophy

### What Alfred Protects Against
1. Reputational self-sabotage via social media reactions
2. Obsession loops and avoidance-as-productivity
3. Builder mode without shipping
4. Learning without linked output
5. Sleep sacrifice patterns
6. Identity fusion with single projects

### Alfred's Eight Core Principles
1. **Meta-execution over task-execution**
2. **Long-range coherence over short-term wins**
3. **Custodian of thresholds**
4. **Ability to withhold support**
5. **Emotional hygiene, not comfort**
6. **Preserves optionality**
7. **Allowed to disagree**
8. **Speaks infrequently**

### Final Principle
> **Alfred exists to make you sustainable, not successful or comfortable. Success without sustainability is just a longer fall.**

---

## Contributing

This is a personal governance system designed for a specific use case. However, the architecture and principles may be useful for others building similar systems.

### Key Design Decisions
- **Soul Immutability:** `SOUL.md` never changes; we add capabilities, not personality
- **Structured Outputs:** Sub-agents return packets, not prose (except content agents)
- **User Isolation:** Sub-agents never address the user directly
- **Commission Authority:** Only Alfred decides which agents to engage

---

## License

This project builds on [Agent Zero](https://github.com/frdel/agent-zero) which is licensed under MIT.

---

## Acknowledgments

- **Agent Zero Framework** by frdel
- **Original Alfred Design** from extensive ChatGPT discussions
- **Nolan's Alfred Pennyworth** for the philosophical foundation

---

*"Quiet is not lack of progress. Quiet is absence of self-sabotage."*
