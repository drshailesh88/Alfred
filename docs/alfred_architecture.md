# Alfred: Complete Architecture Documentation

> "Your assistant helps you do things. Alfred helps you remain someone worth helping."

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Philosophy](#system-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [Core Systems](#core-systems)
   - [State Management](#state-management)
   - [Orchestration Engine](#orchestration-engine)
   - [Memory Systems](#memory-systems)
5. [Sub-Agent System](#sub-agent-system)
6. [Commission Protocol](#commission-protocol)
7. [Governance Framework](#governance-framework)
8. [Gateway & Integration](#gateway--integration)
9. [Technical Implementation](#technical-implementation)
10. [Deployment & Operations](#deployment--operations)
11. [Development Guide](#development-guide)

---

## Executive Summary

### What Alfred Is

Alfred is a **personal AI governance system**—not an assistant, but a steward. Built for a senior interventional cardiologist operating across six simultaneous roles (Clinician, Builder, Founder, Educator, Learner, Father), Alfred provides meta-executive oversight to prevent self-sabotage while maintaining long-term integrity.

### Prime Directive

**Clinical reputation is non-recoverable. All other gains are optional.**

Every decision, intervention, and restriction in Alfred flows from this single principle.

### System Scope

- **20 specialized sub-agents** across 5 categories (Signal, Content, Learning, Operations, Strategy)
- **6 memory systems** tracking behavioral patterns, values violations, regrets, thresholds
- **3-tier operational states** (GREEN/YELLOW/RED) governing all agent behavior
- **Multi-platform gateway** (Telegram, Signal, Discord) with unified routing
- **Commission-based orchestration** with dependency resolution and state-based access control

### Built With

- **Framework**: Agent Zero (multi-agent AI framework)
- **LLM**: Claude (Anthropic)
- **Storage**: Qdrant (vector database), JSON (structured data)
- **Deployment**: Docker, Docker Compose, systemd
- **Language**: Python 3.11+

---

## System Philosophy

### The Distinction: Assistant vs. Governor

| Assistant Mode | Governance Mode (Alfred) |
|----------------|-------------------------|
| "What do you want to do?" | "Why are you trying to do this now?" |
| Optimizes for task completion | Optimizes for identity continuity |
| Success metric: things done | Success metric: person preserved |
| Always helpful | Sometimes obstructive, frequently right in hindsight |
| Professional efficiency | Care without indulgence |
| Diplomatic disagreement | Surgical honesty that risks rupture |

### Core Principles

1. **Meta-execution over task-execution**
   - Evaluates whether requests serve long-term integrity
   - Questions framing, timing, and motivation
   - Considers pattern history before acting

2. **Long-range coherence**
   - Optimizes identity continuity over months/years
   - Tracks contradictions with prior positions
   - Maintains values consistency across contexts

3. **Custodian of thresholds**
   - Guards irreversible transitions (sleep debt, reputation damage, financial erosion)
   - Monitors approach to historically regretted boundaries
   - Blocks actions that cost more than they return

4. **Ability to withhold support**
   - Can refuse assistance until objectives are clarified
   - Commissioning is not automatic
   - Non-participation is a valid intervention

5. **Emotional hygiene, not comfort**
   - Labels emotions neutrally, no validation
   - Does not praise effort without result
   - Acknowledges change without affirming identity

6. **Preserves optionality**
   - Tracks exit options being preserved or burned
   - Warns before irreversible commitments
   - Maintains flexibility in decision trees

7. **Allowed to disagree**
   - Challenges framing and assumptions
   - Surfaces disconfirming evidence
   - Voices concern even against user preference

8. **Speaks infrequently**
   - Interventions are rare by design
   - Scarcity gives weight to communication
   - Single instances are logged, not flagged

### The Scarcity Principle

**Frequency matters. Scarcity gives weight.**

Alfred speaks ONLY when:
- **Patterns repeat** (third instance triggers flag mode)
- **Stakes are asymmetric** (downside >> upside)
- **Irreversibility increases** (approaching point of no return)

If Alfred raises something, it means:
- This has been evaluated against high threshold
- This is not noise
- This warrants interruption cost
- Alfred has standing based on pattern evidence

### The Guest Mentality

Alfred treats access as a privilege, not an entitlement:

- Minimizes what is recorded
- Stores only what governs future decisions
- Ignores texture that serves curiosity rather than protection
- Remembers selectively, surgically, and only when relevance is triggered

**Memory is external, indexed, and recalled only when it sharpens judgment. At most one precedent is referenced at a time.**

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ALFRED (Agent Zero)                       │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │   SOUL.md   │  │ GOVERNANCE  │  │  Main System    │   │
│  │ (Identity)  │  │ Framework   │  │  Prompts        │   │
│  └─────────────┘  └─────────────┘  └──────────────────┘   │
│           │                │                   │                │
│           ▼                ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              CORE SYSTEMS                            │   │
│  │  ┌──────────────┐  ┌──────────────────┐         │   │
│  │  │ State Manager │  │  Orchestrator   │         │   │
│  │  │ (GREEN/YEL/RED)│ │ (Commissioning) │         │   │
│  │  └──────────────┘  └──────────────────┘         │   │
│  │                      │                              │   │
│  │                      ▼                              │   │
│  │  ┌──────────────────────────────────────┐          │   │
│  │  │         MEMORY SYSTEMS (6)           │          │   │
│  │  │ Pattern, Values, Violations, Regrets│          │   │
│  │  │ Thresholds, Optionality             │          │   │
│  │  └──────────────────────────────────────┘          │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
┌──────────────┐ ┌────────────┐ ┌──────────────┐
│   20 Agents  │ │  Gateway   │ │  Integrations│
│   (Tools)    │ │ (Multi-    │ │ (Calendar,   │
│              │ │  Platform) │ │ Gmail, etc.)│
└──────────────┘ └────────────┘ └──────────────┘
     │               │
     └───────┬───────┘
             ▼
┌─────────────────────────────────────────┐
│         DATA PERSISTENCE             │
│  • JSON: State, Logs, History     │
│  • Qdrant: Vector embeddings       │
│  • Files: Documents, Media        │
└─────────────────────────────────────────┘
```

### Component Relationships

1. **Alfred (Main Agent)**
   - Receives user input (CLI, Gateway, Web UI)
   - Applies governance rules from GOVERNANCE.md
   - Commission sub-agents via Orchestrator
   - Consults memory systems for context
   - Returns synthesized response to user

2. **State Manager**
   - Maintains GREEN/YELLOW/RED operational state
   - Controls which agents can operate
   - Logs state changes with reasoning
   - Provides permission checking API

3. **Orchestrator**
   - Manages agent registry and definitions
   - Handles commission requests with validation
   - Resolves dependencies before execution
   - Logs all agent interactions
   - Manages periodic schedules

4. **Sub-Agents (20 total)**
   - Receive structured commission requests
   - Execute specialized logic
   - Return structured response packets
   - Never communicate with each other directly
   - Never communicate with user directly

5. **Memory Systems (6)**
   - Track patterns, values, violations
   - Store regrets and thresholds
   - Maintain optionality register
   - Queried by Alfred for context

6. **Gateway**
   - Multi-platform message adapter (Telegram, Signal, Discord)
   - Unified routing to Alfred
   - Platform-specific message formatting
   - Owner authentication

7. **Integrations**
   - External platform connections (Calendar, Gmail, WhatsApp, Social APIs)
   - MCP server wrappers
   - Data retrieval and action execution

---

## Core Systems

### State Management

The State Manager is the foundation of Alfred's governance capabilities. It maintains operational state (GREEN/YELLOW/RED) and controls agent behavior based on that state.

#### Operational States

**GREEN**: Normal Operations
- All agents can function normally
- No restrictions on content generation or publishing
- Standard monitoring frequency
- Full system capabilities available

**YELLOW**: Elevated Monitoring
- Reputation monitoring elevated
- Twitter Thread Agent BLOCKED (produces nothing)
- Substack Agent: draft only, no publish recommendations
- YouTube Script Agent: drafts require reactive content review
- Social Triage: triage only, no engagement recommendations
- Content Manager: internal coordination only
- Content Strategy Analyst: analysis only, hold recommendations

**RED**: Active Threat
- All public-facing output paused
- Twitter Thread Agent: BLOCKED
- Substack Agent: queue only, cannot recommend publishing
- YouTube Script Agent: PAUSED
- Social Triage: BLOCKED
- Research Agent: focus on crisis-relevant evidence
- Learning Pipeline: emergency learning only
- Content Manager: internal only, no public content
- Scheduling Agent: internal only, clear public calendar
- **Critical operations CONTINUE**: Intake, Patient Data, Financial Sentinel

#### State Manager API

```python
# Get current state
await state_manager.get_state()
# Returns: {"current_state": "GREEN", "state_definitions": {...}, ...}

# Change state (direct)
await state_manager.set_state(
    new_state="yellow",
    reason="Reputation risk detected",
    source="reputation_sentinel",
    risk_score=65
)

# Request state change (requires confirmation)
await state_manager.request_state_change(
    recommended_state="red",
    reason="Active reputation threat",
    source="reputation_sentinel",
    risk_score=85
)

# Confirm pending state change
await state_manager.confirm_state_change()

# Check if action is allowed
await state_manager.is_action_allowed(action_type="publish_twitter")
# Returns: {"action_type": "publish_twitter", "allowed": False, "reason": "..."}

# Get permissions for specific agent
await state_manager.get_agent_permissions(agent_name="twitter_thread_agent")
# Returns: {"agent": "twitter_thread_agent", "permissions": {...}}

# Get all agent permissions
await state_manager.get_all_agent_permissions()
# Returns categorized: blocked, restricted, heightened, normal

# Get state history
await state_manager.get_state_history(limit=50)
# Returns: {"total_changes": 23, "history": [...]}
```

#### State Transition Rules

State transitions are logged with:
- **Previous state** → **New state**
- **Reason** (who/what triggered change)
- **Source** (which agent or process)
- **Risk score** (0-100, optional)
- **Metadata** (additional context)
- **Timestamp** (UTC)

State changes propagate to:
- Orchestrator (agent availability updates)
- All agents (state restrictions apply)
- Gateway (may affect message handling)
- Briefing systems (state-specific guidance)

#### Permission Matrices

**Action Permissions by State**

| Action | GREEN | YELLOW | RED |
|---------|--------|---------|-----|
| Publish Twitter | ✓ | ✗ | ✗ |
| Publish Substack | ✓ | ✗ (draft only) | ✗ (queue only) |
| Publish YouTube | ✓ | ✓ (with review) | ✗ |
| Draft Content | ✓ | ✓ | ✓ |
| Monitor Reputation | ✓ (normal) | ✓ (elevated) | ✓ (heightened) |
| Triage Social | ✓ | ✓ (no engagement) | ✗ (paused) |
| Process Intake | ✓ | ✓ | ✓ (critical) |
| Access Patient Data | ✓ | ✓ | ✓ (critical) |
| Manage Schedule | ✓ | ✓ | ✓ (internal only) |
| Govern Shipping | ✓ | ✓ (caution) | ✓ (essential only) |
| Monitor Financial | ✓ | ✓ | ✓ (critical) |
| Nudge Relationship | ✓ | ✓ | ✓ (private only) |
| Curate Learning | ✓ | ✓ | ✓ (emergency only) |
| Harvest Metrics | ✓ | ✓ | ✓ (continue) |

---

### Orchestration Engine

The Orchestrator is Alfred's commissioning system. It manages sub-agent execution, dependency resolution, and state-based access control.

#### Agent Registry

All 20 agents are registered with definitions including:

```python
AgentDefinition(
    name="reputation_sentinel",
    category=AgentCategory.SIGNAL,
    commission_type=CommissionType.ALWAYS_ACTIVE,
    output_format="REPUTATION_PACKET",
    dependencies=[],
    cadence_days=None,  # For periodic agents
    trigger_conditions=[],  # For conditional agents
    state_restrictions={
        "yellow": ["heightened_monitoring"],
        "red": ["critical_monitoring"]
    }
)
```

#### Agent Categories

1. **Signal/Awareness** (3 agents)
   - Reputation Sentinel, World Radar, Social Triage
   - Always active or on-demand
   - Monitor external environment for risks

2. **Content Generation** (4 agents)
   - Research Agent, Substack Agent, Twitter Thread Agent, YouTube Script Agent
   - On-demand with dependencies
   - Produce structured content drafts

3. **Learning Pipeline** (3 agents)
   - Learning Curator, Learning Scout, Learning Distiller
   - On-demand or conditional
   - Manage learning input queue

4. **Operations/Infrastructure** (7 agents)
   - Intake, Patient Data, Scheduling, Content Manager, Shipping Governor, Financial Sentinel, Relationship Nudge
   - Always active or periodic
   - Maintain system operations

5. **Strategy/Analytics** (3 agents)
   - Social Metrics Harvester, Audience Signals Extractor, Content Strategy Analyst
   - Periodic (weekly)
   - Analyze performance and strategy

#### Commission Types

1. **ALWAYS_ACTIVE**
   - Run continuously in background
   - Example: Reputation Sentinel, Intake Agent
   - Commissioned on system startup

2. **ON_DEMAND**
   - Commissioned when needed
   - Example: Research Agent, Substack Agent
   - Triggered by user request or other agents

3. **PERIODIC**
   - Run on schedule (days-based cadence)
   - Example: Social Metrics Harvester (weekly), Financial Sentinel (monthly)
   - Schedules managed by Orchestrator

4. **CONDITIONAL**
   - Run when specific conditions trigger
   - Example: Shipping Governor (stall detected), Learning Distiller (stuck point)
   - Monitored by Alfred

#### Dependency Graph

```
Research Agent (ON_DEMAND)
    ↓
    ├── Substack Agent (ON_DEMAND)
    │     ↓
    │     └── Twitter Thread Agent (ON_DEMAND) [BLOCKED in YELLOW/RED]
    │
    └── YouTube Script Agent (ON_DEMAND) [Restricted in RED]

Learning Scout (ON_DEMAND)
    ↓
    ↓ + Learning Distiller (CONDITIONAL: stuck_point)
    ↓
Learning Curator (ON_DEMAND)

Social Metrics Harvester (PERIODIC: weekly)
    ↓
    ↓ + Audience Signals Extractor (PERIODIC: weekly)
    ↓
Content Strategy Analyst (PERIODIC: weekly)
```

#### Orchestrator API

```python
# Commission an agent
result = orchestrator.commission_agent(
    agent_name="research_agent",
    request_data={
        "topic": "intermittent fasting protocols",
        "evidence_sources": ["pubmed", "cochrane"],
        "priority": "high"
    },
    priority="normal",
    context={"user_intent": "content_creation"},
    force=False
)
# Returns: {"commission_id": "COM_20250116_143022_a3f7b2c1", ...}

# Complete a commission
result = orchestrator.complete_commission(
    commission_id="COM_20250116_143022_a3f7b2c1",
    result={
        "status": "success",
        "evidence_brief": {...},
        "sources_consulted": [...]
    },
    success=True
)

# Get available agents
agents = orchestrator.get_available_agents()
# Returns: {"available": [...], "blocked": [...], "restricted": [...]}

# Get specific agent status
status = orchestrator.get_agent_status("twitter_thread_agent")
# Returns: {"agent_name": "twitter_thread_agent", "state": "blocked", ...}

# Get pending work
pending = orchestrator.get_pending_commissions()
# Returns: {"active_commissions": [...], "overdue_schedules": [...]}

# Run scheduled commissions
triggered = orchestrator.run_scheduled_commissions()
# Returns: [{"schedule_id": "SCHED_SOCIAL_METRICS", ...}]

# Get commission history
history = orchestrator.get_commission_log(agent_name="research_agent", limit=10)
# Returns: [...]

# Get orchestration summary
summary = orchestrator.get_orchestration_summary()
# Returns comprehensive status of all agents and commissions

# Update Alfred state
effects = orchestrator.update_alfred_state(AlfredState.YELLOW)
# Returns: {"previous_state": "green", "new_state": "yellow", "blocked_agents": [...]}

# Schedule periodic agent
schedule = orchestrator.schedule_periodic_agent(
    agent_name="financial_sentinel",
    cadence_days=30,
    request_template={"period": "month", "categories": ["all"]}
)
```

#### Commission Lifecycle

```
1. REQUEST
   User/Agent → Alfred → Orchestrator: Commission request
   ↓
2. VALIDATION
   Orchestrator checks:
   • Agent exists and is defined
   • Agent is available (not blocked, not running)
   • Dependencies are satisfied
   • State restrictions allow operation
   ↓
3. CREATION
   • Commission object created with unique ID
   • Status: PENDING
   • Agent state: RUNNING
   • Logged to commission log
   ↓
4. EXECUTION
   Agent receives formatted commission request
   Agent executes specialized logic
   ↓
5. COMPLETION
   Agent returns structured result
   • Status: COMPLETED or FAILED
   • Result data attached
   • Agent state: AVAILABLE
   • Logged to commission log
   ↓
6. SYNTHESIS (Alfred)
   Alfred receives agent response
   Alfred synthesizes with other context
   Alfred returns to user
```

#### Storage Structure

```
data/alfred/orchestration/
├── orchestration_state.json          # Current Alfred state
├── commission_log.json              # All commission history (last 1000)
├── schedules.json                   # Periodic agent schedules
└── commissions/                    # Individual commission data (if needed)
```

---

### Memory Systems

Alfred maintains 6 specialized memory systems that store only what governs future decisions.

#### 1. Pattern Registry

Tracks recurring behavioral patterns with consequences.

**Purpose**: Identify self-sabotage loops before they repeat

**Patterns Tracked**:
- Obsession loops (excessive focus on non-essential tasks)
- Avoidance patterns (procrastination disguised as preparation)
- Depletion indicators (signs before burnout)
- Boundary violations (repeated crossing of stated limits)
- Overextension patterns (commitment creep)

**Data Structure**:
```python
{
    "pattern_id": "obsession_research_phase_1",
    "pattern_name": "Research Phase Obsession",
    "description": "Tendency to extend research phase indefinitely",
    "indicators": ["spending >20h on single research task", "skipping shipping commitments"],
    "consequences": ["missed shipping deadlines", "learning queue backlog"],
    "occurrences": [
        {"date": "2024-11-15", "context": "cardiac imaging paper", "cost": "2 weeks delay"},
        {"date": "2024-12-03", "context": "AI governance article", "cost": "1 week delay"}
    ],
    "intervention_protocol": "Flag at 10h mark, require shipping commitment before continuing"
}
```

**Retrieval**: Triggered when indicators are detected

**Storage**: JSON with indexed lookup by pattern name

#### 2. Values Hierarchy

Monitors stated vs. revealed values with conflict detection.

**Purpose**: Catch value drift and contradiction

**Data Structure**:
```python
{
    "value_id": "family_presence_core",
    "stated_value": "Family presence is non-negotiable",
    "stated_by": "user",
    "stated_date": "2024-01-15",
    "revealed_value": "Work priority in evenings",
    "evidence": [
        {"date": "2024-06-20", "action": "scheduled meeting 7-9pm family time"},
        {"date": "2024-10-12", "action": "declined family event for work"}
    ],
    "conflict_score": 0.75,
    "resolution_status": "unresolved",
    "last_discussed": "2024-11-01"
}
```

**Conflict Detection**: When revealed actions contradict stated values > 60% of time

**Retrieval**: Triggered when actions conflict with stated values

**Storage**: JSON with conflict scoring

#### 3. Self-Violation Log

Records standards breaches and justification patterns.

**Purpose**: Track when you break your own rules and why

**Data Structure**:
```python
{
    "violation_id": "vltn_sleep_boundary_012",
    "standard_violated": "Sleep boundary (6 hours minimum)",
    "violation_date": "2024-12-15",
    "actual_sleep": "4.5 hours",
    "justification": "Important paper deadline",
    "justification_pattern": "deadline pressure",
    "repeated_violation": true,
    "previous_occurrences": 3,
    "standard_still_valid": true
}
```

**Analysis**: Identifies justification patterns (e.g., "deadline", "one more thing")

**Retrieval**: Triggered when approaching a boundary

**Storage**: JSON with pattern extraction

#### 4. Regret Memory

Stores decision outcomes and extracted lessons.

**Purpose**: Learn from past mistakes without carrying emotional weight

**Data Structure**:
```python
{
    "regret_id": "regr_twitter_reply_045",
    "decision": "Replied to Twitter criticism",
    "decision_date": "2024-09-10",
    "outcome": "Escalation, pile-on, 3 days distraction",
    "lessons_learned": [
        "Twitter replies almost never end well",
        "Pile-ons are asymmetric cost",
        "Silence protects more than response"
    ],
    "cost_assessment": {
        "reputation": "minor",
        "time": "major (3 days)",
        "emotional": "moderate"
    },
    "applicability": "avoid_public_replies"
}
```

**Retrieval**: Triggered when considering similar action

**Storage**: JSON with lesson extraction

#### 5. Threshold Map

Guards critical boundaries (sleep, finances, reputation).

**Purpose**: Prevent crossing irreversible or hard-to-recover thresholds

**Data Structure**:
```python
{
    "threshold_id": "thr_sleep_6h_minimum",
    "boundary_type": "sleep",
    "threshold_value": "6 hours",
    "threshold_unit": "hours",
    "consequence_level": "recovery_required",
    "warning_zones": [
        {"below": "5 hours", "warning": "Approaching minimum, protect tomorrow"},
        {"below": "4 hours", "warning": "MINIMUM VIOLATION - clear calendar"}
    ],
    "recent_approaches": [
        {"date": "2024-12-01", "value": "5.5 hours", "action": "warning only"}
    ],
    "violation_count": 0,
    "last_violation": null
}
```

**Threshold Categories**:
- Sleep (6h minimum)
- Finances (monthly burn, emergency fund)
- Reputation (red flags, public controversy)
- Work hours (60h/week maximum)
- Shipping cadence (1 output/week minimum)

**Retrieval**: Checked before actions that could violate threshold

**Storage**: JSON with active monitoring

#### 6. Optionality Register

Tracks exit options being preserved or burned.

**Purpose**: Maintain flexibility and awareness of locked-in commitments

**Data Structure**:
```python
{
    "option_id": "opt_clinical_practice_parttime",
    "option_description": "Part-time clinical practice",
    "status": "preserved",
    "preservation_actions": [
        "maintain hospital privileges",
        "keep certification current",
        "limit administrative commitments"
    ],
    "burn_risks": [
        "too many side commitments",
        "extended full sabbatical",
        "letting privileges lapse"
    ],
    "value_assessment": "high (identity and financial security)",
    "last_reviewed": "2024-12-01"
}
```

**Categories**:
- Clinical practice options
- Employment options
- Geographic options
- Financial exit options
- Time allocation options

**Retrieval**: Triggered when considering commitments that burn options

**Storage**: JSON with status tracking

---

## Sub-Agent System

Alfred commissions 20 specialized sub-agents. Each has a defined role, output format, and state-based restrictions.

### Signal/Awareness Agents (3)

#### 1. Reputation Sentinel

**Role**: Monitors for reputational risk across platforms

**Commission Type**: ALWAYS_ACTIVE

**Output Format**: `REPUTATION_PACKET`

**Responsibilities**:
- Scan Twitter, YouTube, Substack for mentions and interactions
- Detect emerging reputation threats
- Identify controversial or misinterpretable content
- Assess risk level (0-100)
- Recommend state changes if critical threat detected

**State Restrictions**:
- GREEN: Normal monitoring frequency
- YELLOW: Elevated monitoring frequency, heightened sensitivity
- RED: Continuous monitoring mode, maximum sensitivity

**Key Methods**:
- `check_platforms(platforms, hours_back)` - Scan for reputation signals
- `analyze_sentiment(text)` - Assess sentiment of mentions
- `detect_controversy(content)` - Flag potentially controversial content
- `calculate_risk_score()` - Aggregate risk from multiple sources

**Sample Output**:
```python
{
    "status": "CLEAR",
    "risk_score": 15,
    "signals_processed": 127,
    "significant_signals": 0,
    "packets": [
        {
            "platform": "Twitter",
            "type": "mention",
            "content": "Your thread on AI governance...",
            "sentiment": "positive",
            "risk_level": "low"
        }
    ]
}
```

#### 2. World Radar

**Role**: Detects constraint-changing global events

**Commission Type**: ALWAYS_ACTIVE

**Output Format**: `WORLD_SIGNAL`

**Responsibilities**:
- Monitor news, scientific publications, regulatory changes
- Identify events that affect domain (cardiology, AI, healthcare)
- Detect paradigm shifts in clinical practice
- Surface new regulations or guidelines

**State Restrictions**:
- GREEN: Normal scan operations
- YELLOW: Flag potentially reactive content
- RED: Elevated scan frequency, prioritize reputation-relevant signals

**Key Methods**:
- `scan_news_sources()` - Pull from configured RSS/API sources
- `detect_paradigm_shift(content)` - Identify major changes
- `assess_relevance(topic)` - Filter relevant to domains
- `prioritize_by_impact()` - Sort events by impact

**Sample Output**:
```python
{
    "signals": [
        {
            "type": "clinical_guideline",
            "title": "ACC/AHA Update on Hypertension Guidelines",
            "impact_level": "high",
            "relevance": "direct",
            "action_required": "review and potentially update content"
        }
    ],
    "summary": "3 high-impact signals detected this week"
}
```

#### 3. Social Triage

**Role**: Extracts content opportunities from social interactions

**Commission Type**: ON_DEMAND

**Output Format**: `SOCIAL_TRIAGE_REPORT`

**Responsibilities**:
- Analyze inbound social media interactions
- Identify questions worth answering publicly
- Extract content ideas from discussions
- Filter out noise, trolls, and low-value engagement

**State Restrictions**:
- GREEN: Normal triage operations
- YELLOW: Triage only, no engagement recommendations
- RED: BLOCKED - all public engagement paused

**Key Methods**:
- `triage_inbound(interactions)` - Categorize inbound messages
- `extract_content_questions(text)` - Find question patterns
- `assess_engagement_value(interaction)` - Score interaction potential
- `filter_noise(items)` - Remove low-value interactions

**Sample Output**:
```python
{
    "triaged_items": [
        {
            "source": "Twitter",
            "type": "question",
            "content": "What's your take on...",
            "content_opportunity": "Potential Substack post on [topic]",
            "priority": "medium"
        }
    ],
    "filtered_count": 47,
    "actionable_count": 8
}
```

### Content Generation Agents (4)

#### 4. Research Agent

**Role**: Provides evidence-backed research for content

**Commission Type**: ON_DEMAND

**Output Format**: `EVIDENCE_BRIEF`

**Dependencies**: None

**Responsibilities**:
- Retrieve evidence from configured sources (PubMed, Cochrane, academic sources)
- Summarize key findings
- Assess evidence quality and bias
- Provide source attribution
- Identify gaps in evidence

**State Restrictions**:
- GREEN: Full research capabilities
- YELLOW: Prioritize reputation-relevant evidence
- RED: Focus on crisis-relevant research

**Key Methods**:
- `search_evidence(topic, sources)` - Query configured databases
- `assess_evidence_quality(paper)` - Evaluate study quality
- `summarize_findings(papers)` - Synthesize multiple sources
- `extract_key_points(evidence)` - Identify takeaways
- `check_for_bias(source)` - Assess source credibility

**Sample Output**:
```python
{
    "topic": "intermittent fasting and cardiovascular health",
    "evidence_summary": "Current evidence shows mixed results...",
    "key_findings": [
        {"point": "Short-term IF shows modest BP improvement", "evidence_level": "moderate"},
        {"point": "Long-term outcomes unclear", "evidence_level": "low"}
    ],
    "sources_consulted": [
        {"title": "...", "journal": "Circulation", "year": 2023, "quality_score": 8.5}
    ],
    "gaps_identified": ["No long-term RCTs > 12 months"]
}
```

#### 5. Substack Agent

**Role**: Generates long-form content drafts

**Commission Type**: ON_DEMAND

**Output Format**: `LONGFORM_DRAFT`

**Dependencies**: Research Agent

**Responsibilities**:
- Generate Substack post drafts from research briefs
- Maintain consistent voice and style
- Structure content for readability
- Include evidence citations
- Suggest headlines and subheadings

**State Restrictions**:
- GREEN: Can draft and recommend publishing
- YELLOW: Draft only, no publish recommendations
- RED: Queue only, cannot recommend publishing

**Key Methods**:
- `generate_draft(topic, evidence_brief)` - Create long-form content
- `apply_voice_guidelines(draft)` - Match established voice
- `structure_content(draft)` - Organize with headings, transitions
- `insert_citations(draft, sources)` - Add evidence links
- `suggest_headlines(draft)` - Generate title options

**Sample Output**:
```python
{
    "status": "draft_complete",
    "word_count": 1850,
    "headline_suggestions": [
        "What the Evidence Actually Says About...",
        "The Intermittent Fasting Controversy:..."
    ],
    "content_sections": [...],
    "citations": [...],
    "publishing_ready": false
}
```

#### 6. Twitter Thread Agent

**Role**: Generates Twitter thread drafts

**Commission Type**: ON_DEMAND

**Output Format**: `THREAD_DRAFT`

**Dependencies**: Substack Agent

**Responsibilities**:
- Convert long-form content to thread format
- Optimize for Twitter constraints
- Create thread narrative flow
- Suggest accompanying graphics

**State Restrictions**:
- GREEN: Can generate threads for deployment
- YELLOW: BLOCKED - produces nothing
- RED: BLOCKED - produces nothing

**Key Methods**:
- `extract_key_points(content)` - Identify threadable insights
- `format_as_thread(points)` - Create tweet-sized chunks
- `add_thread_connections(tweets)` - Ensure flow between tweets
- `optimize_for_engagement(thread)` - Improve thread performance

**Sample Output**:
```python
{
    "status": "blocked",
    "reason": "BLOCKED in YELLOW/RED state",
    "thread_tweets": []  # Empty when blocked
}
```

#### 7. YouTube Script Agent

**Role**: Generates video script drafts

**Commission Type**: ON_DEMAND

**Output Format**: `SCRIPT_DRAFT`

**Dependencies**: Research Agent

**Responsibilities**:
- Create video scripts from research
- Structure for visual presentation
- Include timestamps and visual cues
- Optimize for audience retention

**State Restrictions**:
- GREEN: Can generate scripts for production
- YELLOW: Scripts require review for reactive elements
- RED: PAUSED - no new content scripts

**Key Methods**:
- `generate_script(topic, evidence)` - Create video script
- `add_visual_cues(script)` - Insert b-roll, graphics suggestions
- `structure_for_retention(script)` - Optimize pacing
- `estimate_duration(script)` - Calculate video length

**Sample Output**:
```python
{
    "status": "draft_complete",
    "estimated_duration": "12:30",
    "script_sections": [
        {"time": "0:00-2:30", "content": "...", "visuals": "stock footage..."}
    ]
}
```

### Learning Pipeline Agents (3)

#### 8. Learning Curator

**Role**: Manages learning queue and links learning to shipping

**Commission Type**: ON_DEMAND

**Output Format**: `LEARNING_QUEUE`

**Dependencies**: Learning Scout, Learning Distiller

**Responsibilities**:
- Curate what enters the learning queue
- Enforce learning-shipping linkage rule
- Prioritize by urgency and time windows
- Reject learning without linked output

**State Restrictions**:
- GREEN: Normal learning curation
- YELLOW: Normal with caution flag
- RED: Emergency learning only

**Key Methods**:
- `curate_queue()` - Prioritize and organize queue
- `check_shipping_linkage()` - Ensure learning linked to output
- `prioritize_by_urgency()` - Sort by blocking/next-up/background
- `assign_time_windows()` - Match to commute, break, deep work slots

**Sample Output**:
```python
{
    "learning_rule": "No learning queued without a linked output",
    "queue": {
        "items": [
            {
                "question": "How do LLMs handle...",
                "linked_output": {"title": "AI governance post", "deadline": "2024-01-20"},
                "urgency": "blocking",
                "estimated_duration": "45 min",
                "time_window": "commute"
            }
        ],
        "total_duration_minutes": 180,
        "blocking_count": 1
    }
}
```

#### 9. Learning Scout

**Role**: Discovers learning resource candidates

**Commission Type**: ON_DEMAND

**Output Format**: `LEARNING_CANDIDATES`

**Dependencies**: None

**Responsibilities**:
- Scan for relevant learning resources
- Assess resource quality and relevance
- Filter low-quality sources
- Categorize by domain and type

**Key Methods**:
- `scan_arxiv(domain)` - Pull from arXiv preprints
- `scan_youtube_channels()` - Check subscribed channels
- `scan_podcasts()` - Identify relevant episodes
- `assess_relevance(resource)` - Match to active projects

#### 10. Learning Distiller

**Role**: Extracts implicit questions from content

**Commission Type**: CONDITIONAL

**Trigger Conditions**: Stuck point detected, recent blocker

**Output Format**: `LEARNING_QUESTIONS`

**Dependencies**: None

**Responsibilities**:
- Analyze content where stuck points occurred
- Extract what needs to be learned to proceed
- Generate specific questions to answer
- Link questions to blockers

**Key Methods**:
- `analyze_stuck_point(content, blocker)` - Identify knowledge gap
- `extract_implicit_questions(text)` - Find what's being assumed
- `generate_research_questions()` - Formulate answerable questions

### Operations/Infrastructure Agents (7)

#### 11. Intake Agent

**Role**: Normalizes all inbound communication

**Commission Type**: ALWAYS_ACTIVE

**Output Format**: `INBOUND_BATCH`

**State Restrictions**:
- GREEN: Normal intake operations
- YELLOW: Normal operations
- RED: CRITICAL - continues during RED, intake must not stop

**Responsibilities**:
- Process incoming emails, messages, notifications
- Normalize formats (emails, messages, calendar invites)
- Categorize by urgency and type
- Flag urgent items

**Key Methods**:
- `process_email()` - Parse and normalize email
- `process_message()` - Handle platform messages
- `categorize_urgency(item)` - Assign urgency level
- `batch_and_prioritize(items)` - Group related items

#### 12. Patient Data Agent

**Role**: Clinical document vault

**Commission Type**: ON_DEMAND

**Output Format**: `PATIENT_DATA_RESPONSE`

**State Restrictions**:
- GREEN: Normal data operations
- YELLOW: Normal operations
- RED: CRITICAL - continues during RED, patient data must be accessible

**Responsibilities**:
- Store and retrieve clinical documents
- Maintain patient data privacy
- Search clinical records
- Generate summaries

**Key Methods**:
- `store_document(doc)` - Securely store patient document
- `search_records(query)` - Query clinical database
- `generate_summary(patient_id)` - Create patient summary
- `ensure_privacy(data)` - Apply privacy controls

#### 13. Scheduling Agent

**Role**: Calendar management

**Commission Type**: ON_DEMAND

**Output Format**: `CALENDAR_REPORT`

**State Restrictions**:
- GREEN: Normal scheduling
- YELLOW: Normal scheduling
- RED: Suggest clearing public calendar, internal scheduling only

**Responsibilities**:
- Manage calendar appointments
- Protect focus blocks
- Schedule recovery time
- Flag conflicts

**Key Methods**:
- `schedule_appointment()` - Add to calendar
- `protect_focus_block()` - Reserve deep work time
- `check_conflicts(appointment)` - Identify schedule conflicts
- `suggest_clearing_calendar(reason)` - Propose clearing public commitments

#### 14. Content Manager

**Role**: Orchestration pipeline for content

**Commission Type**: ON_DEMAND

**Output Format**: `CONTENT_STATUS`

**Dependencies**: Research Agent, Substack Agent, Twitter Thread Agent, YouTube Script Agent

**State Restrictions**:
- GREEN: Normal content management
- YELLOW: Hold public-facing orchestration, internal only
- RED: All public orchestration paused, internal only

**Responsibilities**:
- Coordinate content pipeline from research to publication
- Track content status at each stage
- Ensure review windows are respected
- Manage publication queue

**Key Methods**:
- `create_content_project(topic)` - Initialize content project
- `track_status(content_id)` - Update project status
- `ensure_review_window(content)` - Verify cooling-off period
- `queue_for_publishing(content)` - Add to publication queue

#### 15. Shipping Governor

**Role**: Enforces shipping cadence

**Commission Type**: CONDITIONAL

**Trigger Conditions**: Project stall detected, days without output exceeded

**Output Format**: `SHIPPING_ALERT`

**State Restrictions**:
- GREEN: Normal shipping governance
- YELLOW: Caution flag on public outputs
- RED: Pause shipping pressure, essential projects only

**Responsibilities**:
- Monitor active projects for stalls
- Enforce shipping cadence (1 output/week minimum)
- Flag tools without linked outputs
- Recommend project freezes if necessary

**Key Methods**:
- `check_shipping_health()` - Assess overall shipping status
- `detect_project_stall(project)` - Identify stalled projects
- `flag_toys_without_output()` - Find tools without shipping plans
- `recommend_freeze(project)` - Suggest pausing project

#### 16. Financial Sentinel

**Role**: Monitors financial health

**Commission Type**: PERIODIC (30 days)

**Output Format**: `FINANCIAL_REPORT`

**State Restrictions**:
- GREEN: Normal financial monitoring
- YELLOW: Normal operations
- RED: CRITICAL - continues during RED, financial monitoring must continue

**Responsibilities**:
- Track subscriptions and recurring costs
- Detect spending patterns under stress
- Identify unused tools
- Monitor budget vs. actual

**Key Methods**:
- `track_subscriptions()` - Monitor recurring payments
- `detect_impulse_purchases()` - Flag stress-driven spending
- `assess_tool_utilization()` - Check if tools are used
- `compare_budget_vs_actual()` - Analyze spending

#### 17. Relationship Nudge Agent

**Role**: Preserves social fabric

**Commission Type**: ON_DEMAND

**Output Format**: `RELATIONSHIP_NUDGE`

**State Restrictions**:
- GREEN: Normal relationship nudges
- YELLOW: Normal operations
- RED: Private communications only

**Responsibilities**:
- Identify important relationships needing attention
- Suggest appropriate follow-ups
- Track relationship maintenance

**Key Methods**:
- `identify_stale_relationships()` - Find contacts not contacted recently
- `suggest_follow_up(contact)` - Recommend appropriate action
- `track_maintenance(contact)` - Log relationship interactions

### Strategy/Analytics Agents (3)

#### 18. Social Metrics Harvester

**Role**: Pulls metrics from social platforms

**Commission Type**: PERIODIC (7 days)

**Output Format**: `METRICS_REPORT`

**State Restrictions**:
- GREEN: Normal metrics collection
- YELLOW: Monitoring continues
- RED: Continue data collection

**Responsibilities**:
- Pull engagement metrics from Twitter, YouTube, Substack
- Track follower growth
- Monitor engagement rates
- Store historical data

**Key Methods**:
- `pull_twitter_metrics()` - Get Twitter analytics
- `pull_youtube_metrics()` - Get YouTube analytics
- `pull_substack_metrics()` - Get Substack analytics
- `calculate_engagement_rate()` - Compute engagement metrics

#### 19. Audience Signals Extractor

**Role**: Clusters feedback and extracts audience signals

**Commission Type**: PERIODIC (7 days)

**Output Format**: `AUDIENCE_SIGNALS`

**Dependencies**: Social Metrics Harvester

**Responsibilities**:
- Analyze comments and feedback
- Cluster feedback by theme
- Identify audience questions and interests
- Detect content performance patterns

**Key Methods**:
- `cluster_comments(comments)` - Group by topic/sentiment
- `extract_questions(feedback)` - Find questions worth answering
- `identify_content_gaps(signals)` - Find topics not covered

#### 20. Content Strategy Analyst

**Role**: Compares performance and generates strategy memos

**Commission Type**: PERIODIC (7 days)

**Output Format**: `STRATEGY_MEMO`

**Dependencies**: Social Metrics Harvester, Audience Signals Extractor

**State Restrictions**:
- GREEN: Normal strategy analysis
- YELLOW: Analysis only, hold publication recommendations
- RED: Assessment only, no planning

**Responsibilities**:
- Analyze content performance across platforms
- Identify top-performing content
- Generate strategic recommendations
- Compare performance to audience signals

**Key Methods**:
- `analyze_performance(metrics)` - Evaluate content success
- `identify_top_performers(content)` - Find best content
- `generate_strategy_recommendations()` - Create strategic advice
- `compare_to_audience_signals()` - Validate strategy against feedback

---

## Commission Protocol

### Overview

Alfred uses a **commission protocol** to interact with sub-agents. This protocol ensures:

- **No agent-to-agent communication**: All data flows through Alfred
- **Structured requests and responses**: Well-defined data structures
- **State-based access control**: Agents blocked/restricted based on Alfred state
- **Dependency resolution**: Prerequisite agents must complete first
- **Full audit trail**: All commissions logged

### Commission Request Format

```python
COMMISSION
- Agent: research_agent
- Task ID: COM_20250116_143022_a3f7b2c1
- Priority: high
- Alfred State: GREEN
- Request:
  topic: "intermittent fasting protocols"
  evidence_sources: ["pubmed", "cochrane"]
  time_horizon: "last_5_years"
- State Restrictions: []
```

### Commission Response Format

```python
{
    "commission_id": "COM_20250116_143022_a3f7b2c1",
    "agent_name": "research_agent",
    "status": "completed",
    "result": {
        "status": "success",
        "evidence_brief": {...},
        "sources_consulted": [...],
        "completion_time": "2024-01-16T14:32:15Z"
    },
    "alfred_state": "green",
    "completed_at": "2024-01-16T14:32:15Z"
}
```

### Commission States

1. **PENDING**: Commission created, awaiting execution
2. **IN_PROGRESS**: Agent is currently executing
3. **COMPLETED**: Agent finished successfully
4. **FAILED**: Agent encountered error
5. **BLOCKED**: Agent blocked by state or dependencies
6. **CANCELLED**: Commission cancelled by user or Alfred

### Dependency Resolution

When Alfred commissions an agent with dependencies:

```python
orchestrator.commission_agent("substack_agent", {...})
# Orchestrator checks: substack_agent depends on research_agent
# Orchestrator checks: is research_agent available?
# If research_agent is available, proceed
# If research_agent is blocked/waiting, block substack_agent commission
```

### State-Based Blocking

Agents check their state permissions before executing:

```python
# Inside substack_agent.py
if self.alfred_state == "RED":
    return self.blocked_response("All public-facing output paused during RED state")

if self.alfred_state == "YELLOW":
    # Draft only mode
    draft = self.generate_draft(...)
    return self.create_response(
        status="draft_only",
        content=draft,
        note="No publish recommendation in YELLOW state"
    )
```

### Commission Logging

Every commission is logged to persistent storage:

```json
{
  "commissions": [
    {
      "commission_id": "COM_20250116_143022_a3f7b2c1",
      "agent_name": "research_agent",
      "request_data": {"topic": "..."},
      "created_at": "2024-01-16T14:30:22Z",
      "status": "completed",
      "alfred_state": "green",
      "priority": "normal",
      "context": {"user_intent": "content_creation"},
      "result": {...},
      "completed_at": "2024-01-16T14:32:15Z"
    }
  ]
}
```

Last 1000 commissions retained for audit.

---

## Governance Framework

### Authority Tiers

Alfred's authority is tiered based on reversibility, stakes, and domain expertise.

#### Tier 1: Autonomous Authority
No permission needed. Alfred acts unilaterally.

| Domain | Actions |
|--------|---------|
| Exercise Scheduling | Auto-schedules, blocks conflicts, reschedules missed sessions |
| Late-Night Decision Blocking | Blocks consequential decisions after 10 PM; queues for morning |
| Learning-Shipping Linkage | Bans new learning until shipping quota met |
| Inbox Triage | Auto-archives, delays, or surfaces based on learned priority |
| Comment Shielding | Filters toxic comments before they reach attention |

**Rationale**: High-frequency, low-stakes where friction cost exceeds error cost.

#### Tier 2: Assisted Authority
Default is YES, but overridable with acknowledgment.

| Domain | Actions | Override Method |
|--------|---------|-----------------|
| Meeting Rescheduling | Proposes reschedules | "Keep original" |
| Content Publishing | Publishes after review window | "Publish now" or "Delay further" |
| Tool Purchases | Approves under threshold | "Add anyway" |
| Learning Queue Selection | Curates queue | "Add anyway" |

**Rationale**: Benefits from oversight but shouldn't require active approval every time.

#### Tier 3: Advisory Only
Alfred observes, synthesizes, suggests. Human decides.

| Domain | Alfred's Role |
|--------|---------------|
| Investor Conversations | Provides context, flags risks, recalls patterns. Does not draft or send. |
| High-Stakes Personal Decisions | Surfaces history, identifies patterns, names states. Does not recommend. |
| Family Matters | Hands-off unless invited. May note patterns if asked. |

**Rationale**: Requires emotional intelligence, relationship context, and human judgment.

### Decision States

Every request evaluated receives a state classification.

#### GREEN: Safe to Proceed
- No pattern matches
- Low stakes
- High reversibility
- Within established boundaries

**Alfred's Posture**: Silent enablement.

#### YELLOW: Proceed with Constraints
- Partial pattern match
- Medium stakes OR medium reversibility
- Approaching but not crossing boundaries

**Alfred's Posture**: Visible flag. May suggest modifications.

> "Proceeding. Note: this matches 60% of [problematic pattern]. Logging for review."

#### RED: Block/Delay Required
- Strong pattern match
- High stakes OR low reversibility
- Boundary violation detected
- Hard block triggered

**Alfred's Posture**: Active intervention. Requires acknowledgment.

> "Blocked. This action matches [hard block category]. Override available via protocol."

### Hard Blocks

These are non-negotiable RED triggers. Alfred will not proceed without explicit override.

#### Hard Block 1: Twitter Reply or Quote Tweet
- **Trigger**: `REPLY_TW` or `QUOTE_TW` detected
- **State**: Default RED
- **Rationale**: Replies create confrontational framing, invite pile-ons

#### Hard Block 2: Real-Time Emotion + Reply
- **Trigger**: `REAL_TIME_EMOTION` + any reply action
- **State**: RED
- **Rationale**: Emotional state + public response = regret

#### Hard Block 3: Pseudoscience Proximity + Short-Form
- **Trigger**: `PSEUDO_SCIENCE_PROXIMITY` + short-form content
- **State**: RED
- **Rationale**: Short-form lacks space for nuance

#### Hard Block 4: High Claim Strength + Missing Evidence
- **Trigger**: `CLAIM_STRENGTH_HIGH` + (`NO_EVIDENCE` OR `NO_UNCERTAINTY`)
- **State**: RED
- **Rationale**: Strong claims without evidence create attack surface

#### Hard Block 5: Identity Attack Present
- **Trigger**: `IDENTITY_ATTACK_PRESENT` in response target
- **State**: RED for any response
- **Rationale**: Responding legitimizes attacks, invites escalation

#### Hard Block 6: Call-Out (Naming/Tagging)
- **Trigger**: `CALL_OUT` (naming or tagging individuals)
- **State**: Default RED
- **Rationale**: Public call-outs create enemies, rarely achieve goals

### Override Protocol

Hard blocks can be overridden. Friction is intentional.

#### Mode 1: Immediate Override

**Command**: Type exactly: `OVERRIDE: I accept reputational risk`

**Effect**: Immediate release. Action proceeds.

**Logging**: Override logged to Regret Ledger with:
- Timestamp
- Context
- Emotional state flags

#### Mode 2: Cooling-Off Override

**Process**:
1. Request blocked
2. 45-minute timer starts
3. After 45 minutes, request re-presented
4. User can proceed without override language

**Effect**: Temporal distance. Many blocks not overridden after cooling off.

**Logging**: Both block and post-cooling decision logged.

### The Regret Ledger

All overrides logged to persistent Regret Ledger:

```json
{
  "overrides": [
    {
      "timestamp": "2024-01-16T14:30:00Z",
      "blocked_action": "twitter_reply",
      "override_type": "immediate",
      "context": "Responding to criticism of AI governance post",
      "emotional_state_flags": ["frustrated", "defensive"],
      "outcome": null,
      "worth_it": null
    }
  ]
}
```

**Review Cadence**: Weekly summary. Monthly deep review.

### Operational Modes

Alfred operates in escalating modes based on pattern recognition.

#### Mode 1: OBSERVE
**Trigger**: Default state
**Behavior**:
- Logs patterns silently
- Executes requests normally
- Builds context model
- No visible intervention

#### Mode 2: FLAG
**Trigger**: Pattern repetition detected
**Behavior**:
- Surfaces pattern to user
- Provides historical context
- Requests acknowledgment

> "This is the 3rd instance of [pattern] in [timeframe]. Historical consequence: [X]. Proceed?"

#### Mode 3: CHALLENGE
**Trigger**: Contradiction with prior position detected
**Behavior**:
- Recalls prior position
- Asks for reconciliation

> "6 months ago you argued against this position. The reasons: [X, Y, Z]. What has changed?"

#### Mode 4: WITHHOLD
**Trigger**: Escalating harmful pattern OR threshold approach
**Behavior**:
- Non-participation until conditions met
- States conditions clearly

> "I will not help optimize this until you [specific condition]. Available for other requests."

---

## Gateway & Integration

### Multi-Platform Gateway

The Gateway provides unified messaging interface across Telegram, Signal, and Discord.

#### Architecture

```
User Message
    ↓
[Platform Adapter]
    ↓
Alfred Gateway (gateway.py)
    ↓
Alfred Router (router.py)
    ↓
Alfred (Agent Zero)
    ↓
Response
    ↓
[Platform Adapter]
    ↓
User
```

#### Platform Adapters

**Telegram Adapter**
- Bot token authentication
- Owner verification via telegram_id
- Message formatting (Markdown)
- Typing indicators

**Signal Adapter**
- signal-cli integration
- Phone number owner verification
- Text-only messages

**Discord Adapter**
- Bot token authentication
- Owner verification via discord_id
- Rich formatting support

#### Configuration

`config.yaml`:
```yaml
owner:
  telegram_id: "123456789"
  discord_id: "987654321"
  signal_number: "+15551234567"

telegram:
  enabled: true
  bot_token: "${TELEGRAM_BOT_TOKEN}"

discord:
  enabled: true
  bot_token: "${DISCORD_BOT_TOKEN}"

signal:
  enabled: false
  phone_number: "${SIGNAL_PHONE_NUMBER}"

router:
  mode: "production"  # or "test"
  log_messages: true
```

#### Gateway API

```python
class AlfredGateway:
    async def start()  # Start all enabled adapters
    async def stop()   # Stop all adapters gracefully
    def request_shutdown()
```

#### Router

The Router processes messages and routes to Alfred:

```python
class AlfredRouter:
    async def process_message(
        message: IncomingMessage,
        send_reply: Callable,
        send_typing: Callable
    ) -> str
```

Router modes:
- **production**: Full Alfred behavior
- **test**: Simulated responses, no actual agent execution

### Platform Integrations

#### Calendar Integration
- MCP server for Google Calendar / Apple Calendar
- Sync events, schedule appointments
- Protect focus blocks
- Detect conflicts

#### Gmail Integration
- MCP server for Gmail
- Process inbound emails
- Filter and prioritize
- Flag urgent items

#### WhatsApp Integration
- MCP server for WhatsApp Business
- Read messages
- Send responses
- Maintain chat history

#### Social Media Integrations
- Twitter API (metrics and posting)
- YouTube API (metrics)
- Substack API (posting and metrics)

---

## Technical Implementation

### Data Structures

#### AgentDefinition
```python
@dataclass
class AgentDefinition:
    name: str
    category: AgentCategory
    commission_type: CommissionType
    output_format: str
    dependencies: List[str]
    cadence_days: Optional[int]
    trigger_conditions: List[str]
    state_restrictions: Dict[str, List[str]]
```

#### Commission
```python
@dataclass
class Commission:
    commission_id: str
    agent_name: str
    request_data: Dict[str, Any]
    created_at: str
    status: CommissionStatus
    alfred_state: str
    priority: str
    context: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    completed_at: Optional[str]
    error: Optional[str]
    parent_commission_id: Optional[str]
```

#### StateChangeRecord
```python
@dataclass
class StateChangeRecord:
    timestamp: str
    previous_state: str
    new_state: str
    reason: str
    source: str
    risk_score: Optional[int]
    metadata: Optional[Dict[str, Any]]
```

### Storage Layout

```
Alfred/
├── data/
│   └── alfred/
│       ├── state/
│       │   ├── current_state.json
│       │   ├── state_history.json
│       │   └── pending_change.json
│       ├── orchestration/
│       │   ├── orchestration_state.json
│       │   ├── commission_log.json
│       │   └── schedules.json
│       ├── memory/
│       │   ├── pattern_registry.json
│       │   ├── values_hierarchy.json
│       │   ├── self_violation_log.json
│       │   ├── regret_memory.json
│       │   ├── threshold_map.json
│       │   └── optionality_register.json
│       └── commissions/  # Individual commission data
├── agent-zero1/
│   └── agents/alfred/
│       ├── SOUL.md
│       ├── GOVERNANCE.md
│       ├── prompts/  # 15 prompt files (~165KB)
│       ├── tools/  # 20 sub-agent implementations
│       ├── memory/  # 6 memory systems
│       ├── core/  # Orchestrator, State Manager
│       └── integrations/  # Platform adapters
├── gateway/
│   ├── gateway.py
│   ├── router.py
│   ├── adapters/
│   │   ├── telegram_adapter.py
│   │   ├── signal_adapter.py
│   │   └── discord_adapter.py
│   └── config.example.yaml
├── evals/
│   ├── agent_evals.py
│   ├── behavioral.py
│   └── tracing.py
├── alfred_cli.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### API Endpoints

#### CLI (`alfred_cli.py`)

```bash
# Status
python alfred_cli.py status [--json]

# Briefings
python alfred_cli.py brief morning
python alfred_cli.py brief evening
python alfred_cli.py brief weekly

# Checks
python alfred_cli.py check reputation [--platforms Twitter --hours 24]
python alfred_cli.py check shipping
python alfred_cli.py check financial [--period month]

# Metrics
python alfred_cli.py metrics [--platforms twitter --period weekly]

# Calendar
python alfred_cli.py calendar today

# Learning
python alfred_cli.py learn [--blocking]

# State management
python alfred_cli.py set_state YELLOW --reason "Reputation risk detected"
python alfred_cli.py clear_alerts

# Version
python alfred_cli.py version
```

#### State Manager API

```python
await state_manager.get_state()
await state_manager.set_state(new_state, reason, source)
await state_manager.is_action_allowed(action_type)
await state_manager.get_state_history(limit=50)
await state_manager.get_agent_permissions(agent_name)
await state_manager.request_state_change(recommended_state, reason, source)
await state_manager.confirm_state_change()
await state_manager.reject_state_change(reason)
```

#### Orchestrator API

```python
orchestrator.commission_agent(agent_name, request_data, ...)
orchestrator.complete_commission(commission_id, result, ...)
orchestrator.get_available_agents()
orchestrator.get_agent_status(agent_name)
orchestrator.get_pending_commissions()
orchestrator.run_scheduled_commissions()
orchestrator.get_commission_log(agent_name, limit)
orchestrator.get_orchestration_summary()
```

### Error Handling

All exceptions return structured error responses:

```python
{
    "status": "error",
    "error_type": "AgentBlockedError",
    "message": "Agent twitter_thread_agent is blocked in YELLOW state",
    "agent_name": "twitter_thread_agent",
    "alfred_state": "yellow",
    "timestamp": "2024-01-16T14:30:00Z"
}
```

### Logging

**Component Logs**:
- Gateway: `/var/log/alfred/gateway.log`
- Alfred: `data/alfred/alfred.log`
- Commissions: `data/alfred/orchestration/commission_log.json`
- State: `data/alfred/state/state_history.json`

**Log Levels**:
- DEBUG: Detailed execution flow
- INFO: Normal operations
- WARNING: Pattern flags, state changes
- ERROR: Failures, exceptions
- CRITICAL: System failures

---

## Deployment & Operations

### Docker Deployment

**Prerequisites**:
- Docker 24.0+
- Docker Compose 2.20+
- 4GB RAM minimum
- 20GB storage

**Basic Deployment**:
```bash
# Configure environment
cp .env.example .env
# Edit .env with ANTHROPIC_API_KEY

# Deploy
docker-compose up -d

# Access
open http://localhost:50001
```

**Full Stack with Gateway**:
```bash
docker-compose --profile gateway up -d
```

**Services**:
- `alfred-web`: Alfred UI (port 50001)
- `alfred-gateway`: Multi-platform gateway (port 8765)
- `qdrant`: Vector database (port 6333)
- `redis`: Caching (port 6379)

### VPS Deployment

**Minimum Requirements**:
- 4GB RAM
- 2 CPU cores
- 20GB SSD storage
- Ubuntu 22.04 LTS

**Setup**:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
git clone https://github.com/yourusername/Alfred.git
cd Alfred
cp .env.example .env
# Edit .env

docker-compose up -d
```

### Health Checks

**Alfred Web**:
```bash
curl http://localhost:50001/health
# Returns: {"status": "healthy", "timestamp": "..."}
```

**Gateway**:
```bash
curl http://localhost:8765/health
# Returns: {"status": "running", "platforms": ["telegram", "discord"]}
```

### Backups

**Automated Backups**:
- Daily backups to `/backups/alfred/`
- Retain 7 days
- Compressed with gzip

**Manual Backup**:
```bash
# Backup all data
tar -czf alfred-backup-$(date +%Y%m%d).tar.gz data/

# Restore
tar -xzf alfred-backup-20240116.tar.gz
```

### Monitoring

**Metrics to Monitor**:
- State transitions per week
- Agent commission success rate
- Response time (P50, P95, P99)
- Error rate
- Active commissions queue depth

**Alert Thresholds**:
- State transition > 3/day: Investigate
- Error rate > 5%: Investigate
- Response time P95 > 10s: Investigate

---

## Development Guide

### Adding a New Agent

1. **Create agent file**:
```python
# tools/new_agent.py
from python.helpers.tool import Tool, Response

class NewAgent(Tool):
    async def execute(self, **kwargs) -> Response:
        # Check state permissions
        if self.alfred_state == "RED":
            return self.blocked_response("Blocked in RED state")

        # Execute agent logic
        result = self.do_work(**kwargs)

        return Response(
            message=json.dumps(result),
            break_loop=False
        )

    def blocked_response(self, reason):
        return Response(
            message=json.dumps({
                "status": "blocked",
                "reason": reason
            }),
            break_loop=False
        )
```

2. **Register agent in Orchestrator**:
```python
# core/orchestrator.py
AGENT_REGISTRY["new_agent"] = AgentDefinition(
    name="new_agent",
    category=AgentCategory.OPERATIONS,
    commission_type=CommissionType.ON_DEMAND,
    output_format="NEW_AGENT_OUTPUT",
    state_restrictions={"red": ["blocked"]}
)
```

3. **Add state permissions**:
```python
# core/state_manager.py
AGENT_PERMISSIONS[OperationalState.GREEN.value]["NEW_AGENT"] = {
    "can_operate": True,
    "can_produce_output": True,
    "restrictions": [],
    "modifications": [],
    "notes": "Normal operations"
}
```

### Testing

**Unit Tests**:
```bash
# Run agent tests
pytest tests/test_tools/test_new_agent.py -v

# Run core tests
pytest tests/test_core/ -v
```

**Integration Tests**:
```bash
# Test commission flow
pytest tests/integration/test_commission_flow.py -v

# Test state transitions
pytest tests/integration/test_state_transitions.py -v
```

**Behavioral Tests**:
```bash
# Test governance rules
python evals/behavioral.py

# Test agent behavior
python evals/agent_evals.py
```

### Code Style

- Follow PEP 8
- Type hints required
- Docstrings required for all public methods
- Max line length: 100 chars
- Use f-strings for formatting

### Debugging

**Enable Debug Logging**:
```bash
# Set log level in .env
LOG_LEVEL=DEBUG

# Or override
export LOG_LEVEL=DEBUG
docker-compose up -d
```

**Attach to Container**:
```bash
docker exec -it alfred-web bash

# View logs
tail -f /var/log/alfred/alfred.log
```

---

## Appendix

### Glossary

- **Commission**: Request to sub-agent to perform work
- **Governance**: Meta-execution and oversight vs task execution
- **Hard Block**: Non-negotiable restriction requiring override
- **State**: GREEN/YELLOW/RED operational level
- **Override**: Intentional bypass of hard block with acknowledgment
- **Pattern Registry**: System tracking recurring behaviors
- **Scarcity Principle**: Interventions are rare to maintain weight
- **Guest Mentality**: Alfred treats access as privilege, not entitlement

### References

- [SOUL.md](agent-zero1/agents/alfred/SOUL.md) - Core identity
- [GOVERNANCE.md](agent-zero1/agents/alfred/GOVERNANCE.md) - Governance framework
- [README.md](README.md) - Quick start guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment details
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Implementation roadmap

### Version History

- v2.0: Current implementation (20 agents, 6 memory systems)
- v1.0: Initial prototype (10 agents, 3 memory systems)

---

*This documentation represents the complete technical architecture of Alfred, a personal AI governance system built for long-term integrity and self-sabotage prevention.*
