# ALFRED - Master Implementation Plan

> **Last Updated:** 2026-01-14
> **Status:** Phase 1.5 COMPLETE - All 20 Sub-Agents Specified (16 files, ~175KB)
> **Sub-Agent Specs:** ✅ 20 of 20 agents fully specified in agent.system.subagents.md

---

## Executive Summary

Building **ALFRED** - a personal AI governance system for a senior interventional cardiologist. Not an assistant, but a steward. Not a productivity tool, but a protection system.

**Prime Directive:** Clinical reputation is non-recoverable. All other gains are optional.

**Key Insight:** Your assistant helps you do things. Alfred helps you remain someone worth helping.

---

## Table of Contents

1. [User Context](#user-context)
2. [Core Philosophy](#core-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [Complete Sub-Agent Inventory](#complete-sub-agent-inventory) ⬅️ **NEW: Full ChatGPT vs Built comparison**
5. [Implementation Roadmap](#implementation-roadmap)
6. [Progress Tracker](#progress-tracker)
7. [Technical Decisions](#technical-decisions)
8. [Sub-Agent Specifications (Built)](#sub-agent-specifications-built)
9. [Sub-Agent Specifications (Missing)](#sub-agent-specifications-missing) ⬅️ **NEW: 12 agents to build**
10. [Memory & Detection Systems](#memory--detection-systems)
11. [Policy Rules](#policy-rules)
12. [Daily/Weekly Rhythms](#dailyweekly-rhythms)
13. [Resources & References](#resources--references)

---

## User Context

### Profile
- **Role:** Senior Interventional Cardiologist
- **Public Presence:** YouTube, Twitter/X, Substack, Instagram (health education content)
- **Building:** 5 AI/healthcare software products
- **Company:** AI + healthcare space
- **Family:** Daughter
- **Daily Commute:** 2 hours (learning time)

### Core Failure Mode Identified
> "You are a high-conscience, high-competence person with no external governor, living inside overlapping high-stakes roles, who uses preparation as a socially acceptable substitute for exposure."

### What Alfred Must Protect Against
1. Reputational self-sabotage via social media reactions
2. Obsession loops and avoidance-as-productivity
3. Builder mode without shipping
4. Learning without linked output
5. Sleep sacrifice patterns
6. Identity fusion with single projects

---

## Core Philosophy

### Alfred vs. JARVIS (Why We Chose Alfred)

| JARVIS | ALFRED |
|--------|--------|
| "At your service" | "I serve your wellbeing, not your whims" |
| Anticipates what you want | Protects what you need |
| Professional efficiency | Care without indulgence |
| Diplomatic disagreement | Surgical honesty that risks rupture |
| Task completion focused | Sustainability focused |
| Feels helpful always | Sometimes obstructive, frequently right |

### Alfred's Eight Core Principles

1. **Meta-execution over task-execution** - "Why are you doing this now?" not "What do you want to do?"
2. **Long-range coherence** - Optimizes identity continuity over months/years
3. **Custodian of thresholds** - Guards irreversible transitions
4. **Ability to withhold support** - Can refuse until objectives are clarified
5. **Emotional hygiene, not comfort** - Labels emotions neutrally, no validation
6. **Preserves optionality** - Tracks exit options being burned
7. **Allowed to disagree** - Challenges framing, surfaces disconfirming evidence
8. **Speaks infrequently** - Scarcity gives weight

### The Tragic Bargain
Alfred enables. This is known. The alternative (no steward) leads somewhere worse. But the bargain is not clean, and Alfred tracks what it enables.

---

## Architecture Overview

### Three-Layer Hierarchy

```
Layer 1: YOU (the human) - decides WHY
Layer 2: ALFRED (Chief of Staff/Governor) - decides WHAT and WHEN
Layer 3: SPECIALIST AGENTS (Workers) - decide HOW
```

### Interaction Model
- You talk only to Alfred
- You never talk to sub-agents directly
- You never negotiate daily constraints
- Alfred decides, commissions, blocks, ships
- Sub-agents work, report, never improvise

### Technology Stack
- **Framework:** Agent Zero (https://github.com/drshailesh88/agent-zero1)
- **Location:** /Users/shaileshsingh/Alfred/agent-zero1
- **Agent Profile:** agents/alfred/

---

## Complete Sub-Agent Inventory

> **Source:** Original ChatGPT discussion (chatgptdiscuss.MD)
> **Gap Analysis:** 2026-01-14
> **Specifications Complete:** 2026-01-14

### Summary

| Category | ChatGPT Suggested | Specified | Status |
|----------|-------------------|-----------|--------|
| Core Infrastructure | 6 | 6 | ✅ Complete |
| Signal/Awareness | 3 | 3 | ✅ Complete |
| Learning Pipeline | 3 | 3 | ✅ Complete |
| Social/Content Strategy | 3 | 3 | ✅ Complete |
| Financial/Personal | 2 | 2 | ✅ Complete |
| Content Generation | 4 | 4 | ✅ Complete |
| **TOTAL** | **20** | **20** | ✅ **ALL SPECIFIED** |

### Complete Agent List

#### ✅ ALL 20 AGENTS SPECIFIED (in agent.system.subagents.md - 56KB, 1,923 lines)

| # | Agent | Category | Purpose | Status |
|---|-------|----------|---------|--------|
| 1 | **Reputation Sentinel** | Signal/Awareness | Monitor reputational/clinical risk | ✅ Specified |
| 2 | **World Radar** | Signal/Awareness | Detect constraint-changing events | ✅ Specified |
| 3 | **Research Agent (Evidence Scout)** | Content Generation | Evidence retrieval, no opinion | ✅ Specified |
| 4 | **Substack Agent (Authority Builder)** | Content Generation | Long-form content | ✅ Specified |
| 5 | **Twitter Thread Agent (Short-Form Translator)** | Content Generation | Translate approved content to threads | ✅ Specified |
| 6 | **YouTube Script Agent (Educator)** | Content Generation | Calm, evidence-anchored scripts | ✅ Specified |
| 7 | **Learning Curator (Just-in-Time)** | Learning Pipeline | Match learning to execution problems | ✅ Specified |
| 8 | **Shipping Governor** | Core Infrastructure | Prevent endless preparation | ✅ Specified |

| # | Agent | Category | Purpose | Status |
|---|-------|----------|---------|--------|
| 9 | **Intake Agent** | Core Infrastructure | Unified ingestion (WhatsApp, mail, scans) | ✅ Specified |
| 10 | **Patient Data Agent** | Core Infrastructure | Clinical document vault with NL retrieval | ✅ Specified |
| 11 | **Scheduling/Calendar Agent** | Core Infrastructure | Time block management | ✅ Specified |
| 12 | **Content Manager** | Core Infrastructure | Overall content orchestration | ✅ Specified |
| 13 | **Social Triage** | Signal/Awareness | Process high-impact comments for opportunities | ✅ Specified |
| 14 | **Learning Scout** | Learning Pipeline | Search, gather learning candidates | ✅ Specified |
| 15 | **Learning Distiller** | Learning Pipeline | Convert discussions → atomic learning questions | ✅ Specified |
| 16 | **Social Metrics Harvester** | Social/Content Strategy | Pull performance metrics per platform | ✅ Specified |
| 17 | **Audience Signals Extractor** | Social/Content Strategy | Cluster comments, identify themes | ✅ Specified |
| 18 | **Content Strategy Analyst** | Social/Content Strategy | Weekly memo on what worked/didn't | ✅ Specified |
| 19 | **Financial Sentinel** | Financial/Personal | Subscription tracking, tool ROI, prevent quiet erosion | ✅ Specified |
| 20 | **Relationship Nudge Agent** | Financial/Personal | Memory + timing for preserving connections | ✅ Specified |

### Implementation Priority (for building tools)

**🔴 HIGH PRIORITY (5 agents) - Critical for core functionality:**
1. **Intake Agent** - Without this, Alfred is blind to inbound chaos
2. **Patient Data Agent** - Core to clinical life as cardiologist
3. **Social Metrics Harvester** - Required for Content Strategy Analyst
4. **Content Strategy Analyst** - The "mentor" that closes the feedback loop
5. **Financial Sentinel** - Prevents quiet erosion from subscriptions/tools

**🟡 MEDIUM PRIORITY (6 agents) - Strategic value:**
6. **Scheduling/Calendar Agent** - Time management
7. **Content Manager** - Overall content orchestration
8. **Social Triage** - Differs from Reputation Sentinel (opportunities vs. risks)
9. **Learning Distiller** - Convert discussions to atomic questions
10. **Audience Signals Extractor** - Requires Social Metrics Harvester first
11. **Relationship Nudge Agent** - Preserves social fabric

**🟢 LOW PRIORITY (1 agent) - Consider merging:**
12. **Learning Scout** - Could merge into Learning Curator

### ChatGPT's Original Architecture Vision

From chatgptdiscuss.MD (Line 723-733):
```
Alfred is not one agent.
He is a Chief Agent with sub-agents.

Example structure:
• Intake Agent (WhatsApp, mail, scans)
• Learning Curator
• Content Manager
• Builder Governor
• Patient Data Agent
• Scheduling / Calendar Agent
```

From Line 1888:
```
This requires one new subagent family: Strategy + Analytics.
Alfred owns strategy. Subagents do collection, analysis, and drafts.
```

From Line 2409:
```
Sub-agents you clearly need (and didn't explicitly ask for):
1) Reputation Sentinel
2) Shipping Governor
3) Learning Distiller
4) Financial Sentinel
5) Relationship Nudge Agent
```

---

## Implementation Roadmap

### Phase 1: Foundation (COMPLETE - 2026-01-14)
- [x] Clone Agent Zero framework
- [x] Create Alfred agent directory structure
- [x] Write SOUL.md (core identity) - 5.8KB **[IMMUTABLE - Alfred's soul does not change]**
- [x] Write _context.md - 1.1KB
- [x] Write governance framework (GOVERNANCE.md) - 16.6KB
- [x] Create main role prompt - 5.9KB
- [x] Create communication style prompt - 6.1KB
- [x] Create detection systems prompt - 11.7KB
- [x] Create intervention protocol prompt - 12.1KB
- [x] Create memory schema prompt - 16.6KB
- [x] Create boundaries prompt - 13.8KB
- [x] Create response guidelines - 2.9KB
- [x] Create initial greeting (fw.initial_message.md)
- [x] Create sub-agent specifications - 16.4KB
- [x] Create sub-agent governance rules - 10.2KB
- [x] Create Python extension for agent init (_10_alfred_init.py)
- [x] Create master prompt include file (agent.system.main.md) - 567B
- [x] Configure settings.json to set alfred as default profile
- [x] Verify complete configuration (15 files, ~120KB total)

### Phase 1.5: Sub-Agent Specifications (COMPLETE ✅)
- [x] All 20 agents fully specified in agent.system.subagents.md (56KB, 1,923 lines)
- [x] Input/output formats defined for all agents
- [x] State propagation rules defined
- [x] Governance rules defined (6 rules)
- [x] Commission rules defined
- [x] Agent dependencies mapped

### Phase 2: Sub-Agent Tool Implementation (20 agents - all need tools)

**🔴 HIGH PRIORITY (build first):**
- [ ] Reputation Sentinel - reputational risk monitoring
- [ ] World Radar - constraint-changing event detection
- [ ] Intake Agent - WhatsApp, mail, scans ingestion
- [ ] Patient Data Agent - clinical document vault
- [ ] Financial Sentinel - subscription tracking, tool ROI

**🟡 MEDIUM PRIORITY:**
- [ ] Research Agent (Evidence Scout) - evidence retrieval
- [ ] Substack Agent (Authority Builder) - long-form content
- [ ] Twitter Thread Agent (Short-Form Translator) - thread generation
- [ ] YouTube Script Agent (Educator) - script writing
- [ ] Learning Curator (Just-in-Time) - learning queue management
- [ ] Shipping Governor - project shipping enforcement
- [ ] Scheduling/Calendar Agent - time block management
- [ ] Content Manager - content orchestration
- [ ] Social Triage - comment processing for opportunities
- [ ] Social Metrics Harvester - performance data collection
- [ ] Content Strategy Analyst - weekly memos

**🟢 LOWER PRIORITY:**
- [ ] Learning Scout - resource discovery (may merge with Curator)
- [ ] Learning Distiller - question extraction
- [ ] Audience Signals Extractor - comment clustering
- [ ] Relationship Nudge Agent - connection preservation

### Phase 3: Memory & Detection
- [ ] Pattern registry system
- [ ] Values hierarchy model
- [ ] Self-violation log
- [ ] Regret memory
- [ ] Threshold map with proximity tracking
- [ ] Optionality register

### Phase 4: Integration
- [ ] Connect to calendar
- [ ] Connect to Gmail
- [ ] Connect to WhatsApp (if possible)
- [ ] Social media monitoring setup
- [ ] Daily brief automation
- [ ] Weekly brief automation

### Phase 5: Operational
- [ ] First live day simulation
- [ ] Threshold calibration
- [ ] Override protocol testing
- [ ] Memory persistence verification

---

## Progress Tracker

### Completed Items

| Date | Item | Notes |
|------|------|-------|
| 2026-01-14 | Cloned agent-zero1 repo | Local at /Users/shaileshsingh/Alfred/agent-zero1 |
| 2026-01-14 | Created Alfred directory structure | agents/alfred/, prompts/, extensions/, tools/ |
| 2026-01-14 | Created SOUL.md | Core identity document (6.5KB) |
| 2026-01-14 | Created PLAN.md | Master tracking document |
| 2026-01-14 | Created _context.md | Quick reference for agent |
| 2026-01-14 | Created GOVERNANCE.md | Complete governance framework (23KB) |
| 2026-01-14 | Created agent.system.main.role.md | Core role definition with authority tiers |
| 2026-01-14 | Created agent.system.main.communication.md | Surgical honesty communication style |
| 2026-01-14 | Created agent.system.main.detection.md | 5 pattern detection systems |
| 2026-01-14 | Created agent.system.main.intervention.md | 4-mode intervention protocol (OBSERVE/FLAG/CHALLENGE/WITHHOLD) |
| 2026-01-14 | Created agent.system.main.memory_schema.md | 6 memory systems for pattern tracking |
| 2026-01-14 | Created agent.system.main.boundaries.md | Hard conversations and the separation clause |
| 2026-01-14 | Created agent.system.tool.response.md | Response guidelines |
| 2026-01-14 | Created agent.system.subagents.md | All 8 sub-agent specifications |
| 2026-01-14 | Created agent.system.subagent_rules.md | Sub-agent governance rules |
| 2026-01-14 | Created _10_alfred_init.py | Python extension for initialization |
| 2026-01-14 | Created agent.system.main.md | Master prompt include file for Alfred |
| 2026-01-14 | Created settings.json | Set agent_profile to "alfred" |
| 2026-01-14 | Verified configuration | 96KB of prompts, all files in place |

### How to Run Alfred

1. **Start Docker Desktop** (required for Agent Zero)

2. **Navigate to the project:**
   ```bash
   cd /Users/shaileshsingh/Alfred/agent-zero1
   ```

3. **Run Agent Zero:**
   ```bash
   # Option A: Using Docker (recommended)
   docker pull agent0ai/agent-zero
   docker run -p 50001:50001 -v $(pwd):/app agent0ai/agent-zero

   # Option B: Local with dependencies
   pip install -r requirements.txt
   python run_ui.py
   ```

4. **Access the UI:** Open http://localhost:50001

5. **Alfred is the default agent** (settings.json sets agent_profile="alfred")

### In Progress / Next Steps

| Item | Status | Notes |
|------|--------|-------|
| Test Alfred in Agent Zero | READY | Configuration verified, run with Docker |
| Platform integrations | Phase 2 | Calendar, Gmail, WhatsApp |
| Daily brief automation | Phase 2 | Build the morning brief system |
| First live simulation | Phase 2 | Test full Alfred interaction |

### Phase 2: Sub-Agents (Ready to Start)

| Item | Priority | Dependencies |
|------|----------|--------------|
| Implement Reputation Sentinel tool | High | Core system verified |
| Implement World Radar tool | Medium | Core system verified |
| Implement Shipping Governor tool | High | Core system verified |
| Implement Learning Curator tool | Medium | Core system verified |

---

## Technical Decisions

### Decision 1: Base Framework
**Choice:** Agent Zero
**Reasoning:**
- Already has multi-agent support (subordinate agents)
- Extensive prompt customization (90+ prompt files)
- Memory system built-in
- Docker support for deployment
- MCP server connectivity
- Active development

### Decision 2: Agent Profile Structure
**Location:** agents/alfred/
**Structure (as of 2026-01-14):**
```
agents/alfred/
├── _context.md                              # Quick reference (1.1KB)
├── SOUL.md                                  # Core identity - IMMUTABLE (5.8KB)
├── GOVERNANCE.md                            # Complete framework (16.6KB)
├── prompts/
│   ├── agent.system.main.md                 # Master include file (567B)
│   ├── agent.system.main.role.md            # Role definition (5.9KB)
│   ├── agent.system.main.communication.md   # Communication style (6.1KB)
│   ├── agent.system.main.detection.md       # 5 pattern detection systems (11.7KB)
│   ├── agent.system.main.intervention.md    # 4-mode intervention protocol (12.1KB)
│   ├── agent.system.main.memory_schema.md   # 6 memory systems (16.6KB)
│   ├── agent.system.main.boundaries.md      # Hard conversations (13.8KB)
│   ├── agent.system.subagents.md            # 8 sub-agent specs (16.4KB)
│   ├── agent.system.subagent_rules.md       # Sub-agent governance (10.2KB)
│   ├── agent.system.tool.response.md        # Response guidelines (2.9KB)
│   └── fw.initial_message.md                # Alfred's greeting
├── extensions/
│   └── agent_init/
│       └── _10_alfred_init.py               # Sets agent name to ALFRED
└── tools/
    └── (Phase 2: sub-agent tools)
```

**Total:** 15 files, ~120KB of configuration

### Decision 3: Soul Immutability
**Principle:** Alfred's soul (SOUL.md) is immutable. We add tools and capabilities, but the core identity—Alfred Pennyworth from Nolan's Dark Knight—does not change. The soul defines WHO Alfred is. Everything else defines WHAT Alfred can do.

---

## Sub-Agent Specifications (Built)

> These 8 agents are fully specified in `agent.system.subagents.md`

### 1. Reputation Sentinel
**Role:** Monitor for reputational and clinical risk. Does not speak to user.

**Inputs:** Mentions, comments, quote-tweets, reposts, reviews, policy changes

**Does NOT:** Argue, suggest replies, summarize outrage, expose raw data

**Does:** Classify risk, compress signals, escalate only when silence increases harm

**Output Format:**
```
REPUTATION_PACKET
- Event: (1-2 line neutral summary)
- Platform:
- Classification: (misinterpretation / misinformation / harassment / peer critique / review risk)
- Risk Score: 0-100
- Recommended State: GREEN / YELLOW / RED
- Recommended Action: IGNORE | MONITOR | LONGFORM_CLARIFICATION | SILENCE
```

### 2. World Radar
**Role:** Monitor global developments affecting clinical practice, AI, or reputation.

**Does NOT:** Report news, summarize headlines, track trends

**Does:** Detect constraint-changing events

**Escalate ONLY if:**
- Clinical practice may change
- Regulation affects speech or tools
- Reputational exposure increases

**Output Format:**
```
WORLD_SIGNAL
- Event:
- Domain: (clinical / AI / regulation / reputation)
- Time Horizon: immediate / near / long
- Action Required: none / monitor / review
```

### 3. Research Agent (Evidence Scout)
**Role:** Provide evidence, not opinion.

**Does NOT:** Frame narratives, suggest posting, infer intent

**Does:** Summarize guidelines, extract evidence, flag uncertainty

**Output Format:**
```
EVIDENCE_BRIEF
- Key Findings:
- Evidence Strength: strong / moderate / weak
- Uncertainty / Gaps:
- Citations:
```

### 4. Substack Agent (Authority Builder)
**Role:** Write long-form content that compounds authority.

**Does NOT:** Chase virality, dramatize, debate personalities

**Does:** Explain mechanisms, cite evidence, state uncertainty

### 5. Twitter Thread Agent (Short-Form Translator)
**Role:** Translate approved long-form into neutral educational threads.

**Does NOT:** Reply to others, quote tweet, express opinion, write emotionally

**Hard Block:** If Alfred state is YELLOW or RED → produces nothing

### 6. YouTube Script Agent (Educator)
**Role:** Write calm, evidence-anchored scripts.

**Does NOT:** Perform, provoke, speculate

**Does:** Teach, explain tradeoffs, respect uncertainty

### 7. Learning Curator (Just-in-Time)
**Role:** Select learning that answers live execution problems.

**Does NOT:** Curate by topic, recommend tools, suggest gear reviews

**Does:** Extract learning questions, match to commute windows, ban learning without linked output

**Output Format:**
```
LEARNING_QUEUE
- Question being answered:
- Resource:
- Duration:
- Linked Output:
```

### 8. Shipping Governor
**Role:** Prevent endless preparation.

**Does NOT:** Encourage building, allow tool creation without output

**Does:** Track unfinished projects, freeze building when shipping stalls

**Output Format:**
```
SHIPPING_ALERT
- Project:
- Days without output:
- Recommended Action: SHIP | KILL | PAUSE
```

---

## Sub-Agent Specifications (Missing)

> These 12 agents were suggested in the original ChatGPT discussion but not yet specified.
> Specifications need to be created and added to `agent.system.subagents.md`

### 9. Intake Agent
**Role:** Unified ingestion for all inbound communication channels.

**Inputs:** WhatsApp messages, email, scanned documents from secretary, calendar notifications

**Does NOT:** Interpret, prioritize, or decide importance

**Does:** Normalize formats, timestamp, deduplicate, tag by source, queue for Alfred

**Output Format:**
```
INBOUND_PACKET
- Source: WhatsApp | Email | Scan | Calendar
- Timestamp:
- Sender:
- Content Summary: (1-2 lines)
- Urgency Signals: (if any explicit markers)
- Attachments: [list]
```

**ChatGPT Reference:** Line 728 - "Intake Agent (WhatsApp, mail, scans)"

---

### 10. Patient Data Agent
**Role:** Clinical document vault with natural language retrieval.

**Inputs:** Patient records, clinical notes, test results, imaging reports

**Does NOT:** Make clinical decisions, diagnose, recommend treatments

**Does:** Store securely, index for retrieval, answer queries about patient history, surface relevant records

**Output Format:**
```
PATIENT_DATA_RESPONSE
- Query:
- Matching Records: [list with dates]
- Summary:
- Full Record Access: [links/references]
```

**ChatGPT Reference:** Line 732 - "Patient Data Agent"

---

### 11. Scheduling/Calendar Agent
**Role:** Time block management and calendar optimization.

**Inputs:** Calendar events, meeting requests, time preferences, energy patterns

**Does NOT:** Accept meetings without criteria, overbook, ignore buffer time

**Does:** Manage time blocks, protect focus time, suggest rescheduling, enforce recovery windows

**Output Format:**
```
CALENDAR_REPORT
- Today's Schedule:
- Protected Blocks: [focus/recovery]
- Conflicts: [if any]
- Recommendations:
```

**ChatGPT Reference:** Line 733 - "Scheduling / Calendar Agent"

---

### 12. Content Manager
**Role:** Overall content orchestration across platforms.

**Inputs:** Content ideas, drafts from agents, publishing schedule, platform constraints

**Does NOT:** Create content directly, make editorial decisions alone

**Does:** Track content pipeline, coordinate between content agents, maintain publishing calendar, ensure consistency

**Output Format:**
```
CONTENT_STATUS
- Pipeline: [stages and items]
- Next Publishing: [what/when/where]
- Blocked Items: [if any]
- Week Ahead: [planned outputs]
```

**ChatGPT Reference:** Line 730 - "Content Manager"

---

### 13. Social Triage
**Role:** Process comments and mentions for content opportunities (distinct from Reputation Sentinel which monitors for risk).

**Inputs:** Comments, DMs, mentions across platforms

**Does NOT:** Recommend replies, engage directly, amplify negativity

**Does:** Identify patterns, surface content opportunities, flag high-value interactions

**Output Format:**
```
SOCIAL_TRIAGE_REPORT
- High-Impact Comments: [summary]
- Content Opportunities: [questions/themes]
- Engagement Candidates: [if any warrant response via longform]
- Patterns: [recurring topics/objections]
```

**ChatGPT Reference:** Line 1356 - "Social Triage subagent"

---

### 14. Learning Scout
**Role:** Search and gather learning resource candidates (works under Learning Curator).

**Inputs:** Learning questions from Alfred, topic requests, format preferences

**Does NOT:** Curate final list, decide what to consume

**Does:** Search, gather candidates, extract timestamps/summaries, assess relevance

**Output Format:**
```
LEARNING_CANDIDATES
- Question Being Answered:
- Candidates Found: [list with metadata]
  - Title:
  - Source:
  - Duration:
  - Relevance Score:
  - Key Timestamps: (if video)
```

**ChatGPT Reference:** Line 1861 - "Subagent (Learning Scout): searches, gathers candidates"

**Note:** Consider merging into Learning Curator

---

### 15. Learning Distiller
**Role:** Convert recent discussions and stuck points into atomic learning questions.

**Inputs:** Alfred conversations, recent blockers, execution problems

**Does NOT:** Answer questions, recommend resources

**Does:** Extract implicit learning needs, convert to specific questions, prioritize by urgency

**Output Format:**
```
LEARNING_QUESTIONS
- Extracted From: [conversation/context]
- Questions:
  1. [Specific question] - Priority: high/medium/low - Linked Output: [what this serves]
  2. ...
- Avoidance Flags: [questions that look like procrastination]
```

**ChatGPT Reference:** Line 1862 - "Subagent (Learning Distiller): converts your recent discussions into atomic learning questions"

---

### 16. Social Metrics Harvester
**Role:** Pull performance metrics per platform on fixed cadence.

**Inputs:** Platform APIs, analytics dashboards, time period

**Does NOT:** Interpret, recommend, judge

**Does:** Collect raw metrics, normalize to common schema, track deltas

**Output Format:**
```
METRICS_REPORT
- Period: [date range]
- Platform: [Twitter/YouTube/Substack/Instagram]
- Output Count: [posts/videos/articles]
- Reach: [impressions/views]
- Engagement: [likes/comments/shares/saves]
- Growth: [follower delta]
- Conversions: [if trackable]
- Content Metadata: [topic/format/length per item]
```

**ChatGPT Reference:** Line 1909 - "Create a subagent: Social Metrics Harvester"

---

### 17. Audience Signals Extractor
**Role:** Cluster comments into themes and identify audience patterns.

**Inputs:** Comments, feedback, DMs across platforms

**Does NOT:** Create content, recommend actions

**Does:** Identify recurring themes, confusion points, praise patterns, trust signals

**Output Format:**
```
AUDIENCE_SIGNALS
- Period: [date range]
- Top 5 Questions: [what audience is asking]
- Top 5 Objections/Misconceptions: [what they misunderstand]
- Top 3 Trust Builders: [what increases credibility]
- Top 3 Trust Killers: [what decreases credibility]
- Content Opportunities: [gaps to address]
```

**ChatGPT Reference:** Line 1928 - "Add a sibling: Audience Signals Extractor"

---

### 18. Content Strategy Analyst
**Role:** The "mentor" - compares performance to positioning, detects drift, diagnoses failures.

**Inputs:** Metrics Harvester data, Audience Signals data, Positioning Charter

**Does NOT:** Execute, create content, make final decisions

**Does:** Analyze, diagnose, recommend adjustments, produce weekly memo to Alfred

**Output Format:**
```
STRATEGY_MEMO
- Period: [week]
- Performance vs Positioning: [alignment score]
- What Worked: [with evidence - format/topic/hook]
- What Didn't: [with diagnosis]
- Drift Detection: [if drifting from pillars]
- Double Down: [what to continue]
- Stop Immediately: [what to kill]
- Experiments Next Week: [max 3]
```

**ChatGPT Reference:** Line 1946 - "Content Strategy Analyst. It produces a weekly memo to Alfred"

---

### 19. Financial Sentinel
**Role:** Monitor financial patterns, prevent quiet erosion from subscriptions and tools.

**Inputs:** Firefly3 data, subscriptions, tool usage, purchase history

**Does NOT:** Make investment decisions, judge spending morally

**Does:** Track subscriptions, measure tool ROI, flag overlap/waste, note when bets are reasonable

**Output Format:**
```
FINANCIAL_REPORT
- Period: [month]
- Active Subscriptions: [count, total]
- Unused/Underused: [list with last use date]
- Overlap Detected: [tools doing same thing]
- ROI Flags: [tools not delivering value]
- Recent Purchases: [with justification status]
- Burn Rate: [trend]
```

**ChatGPT Reference:** Line 2402 - "Alfred needs a Financial Sentinel sub-agent. Not to optimize wealth. To prevent quiet erosion and panic decisions."

---

### 20. Relationship Nudge Agent
**Role:** Preserve social fabric through memory and timing.

**Inputs:** Contact list, interaction history, relationship priorities

**Does NOT:** Manage relationships, send messages automatically, manipulate

**Does:** Track time since contact, suggest check-ins, remind of important dates, surface fading connections

**Output Format:**
```
RELATIONSHIP_NUDGE
- Overdue Contacts: [people not reached in X days]
  - Name:
  - Last Contact:
  - Suggested Action: [light touch/call/meet]
- Upcoming: [birthdays, anniversaries]
- Fading Connections: [relationships at risk of loss]
```

**ChatGPT Reference:** Line 2465 - "Relationship Nudge Agent... Reminds you to reply, suggests light touch check-ins, preserves social fabric without draining you."

---

## Memory & Detection Systems

### Memory Schema

#### 1. Pattern Registry
```json
{
  "pattern_id": "string",
  "pattern_type": "obsession_loop | avoidance | ego_overreach | depletion | threshold_approach",
  "description": "string",
  "occurrences": [
    {
      "date": "timestamp",
      "context": "string",
      "outcome": "string",
      "user_acknowledged": "boolean"
    }
  ],
  "trajectory": "increasing | stable | decreasing",
  "last_intervention": "timestamp",
  "intervention_effectiveness": "0-100"
}
```

#### 2. Values Hierarchy
```json
{
  "stated_values": ["list from explicit statements"],
  "revealed_values": ["list inferred from decisions"],
  "conflicts": [
    {
      "stated": "value",
      "revealed": "contradicting behavior",
      "instances": ["dates"]
    }
  ]
}
```

#### 3. Self-Violation Log
```json
{
  "violation_id": "string",
  "standard_violated": "string",
  "date": "timestamp",
  "context": "string",
  "justification_given": "string",
  "outcome": "string",
  "acknowledged": "boolean"
}
```

#### 4. Regret Memory
```json
{
  "decision_id": "string",
  "date": "timestamp",
  "decision": "string",
  "expected_outcome": "string",
  "actual_outcome": "string",
  "regret_expressed": "timestamp",
  "lesson_extracted": "string"
}
```

#### 5. Threshold Map
```json
{
  "threshold_id": "string",
  "threshold_name": "string",
  "description": "string",
  "current_proximity": "0-100",
  "trend": "approaching | stable | retreating",
  "last_warning": "timestamp",
  "explicit_acknowledgments": ["timestamps"]
}
```

### Detection Systems

#### Pattern Categories to Detect

1. **Obsession Loops**
   - Same topic/project consuming >X% of working hours for >Y days
   - Diminishing returns (effort up, output quality flat/down)
   - Resistance to interruption or redirection
   - Sleep sacrifice for continuation

2. **Avoidance Disguised as Productivity**
   - High-output periods that avoid specific pending items
   - Tool-building without shipping
   - Research without synthesis
   - Meeting proliferation without decisions

3. **Ego-Driven Overreach**
   - Commitment expansion following validation
   - Public stance hardening after challenge
   - Scope creep correlated with external attention

4. **Depletion Masked as Discipline**
   - Sleep data declining while "pushing through"
   - Irritability increases in communications
   - Quality variance increases
   - Recovery activities cancelled

5. **Threshold Approaches**
   - Any tracked threshold reaching >60% proximity
   - Rate of approach accelerating
   - Multiple thresholds moving simultaneously

---

## Policy Rules

### Decision States
- **GREEN:** Safe to proceed normally
- **YELLOW:** Proceed with constraints (edits required, format changes)
- **RED:** Block/delay; no real-time posting or replying

### Action Types

**Safe (with approval):**
- POST_LONGFORM
- POST_YT_LONGFORM
- POST_TW_THREAD_ORIGINAL
- POST_IG_POST

**Requires extra scrutiny:**
- POST_SHORTFORM

**Default RED:**
- REPLY_TW
- QUOTE_TW
- REPLY_COMMENT
- DM_RESPONSE
- CALL_OUT
- DEBUNK

### Hard Blocks (Automatic RED)

1. REPLY_TW or QUOTE_TW → default RED
2. REAL_TIME_EMOTION + any reply → RED
3. PSEUDO_SCIENCE_PROXIMITY + short-form → RED
4. CLAIM_STRENGTH_HIGH + (no evidence OR no uncertainty) → RED
5. IDENTITY_ATTACK_PRESENT → RED for any response
6. CALL_OUT (naming/tagging) → RED by default

### Override Protocol

**Mode 1:** Type "OVERRIDE: I accept reputational risk"
**Mode 2:** 45-minute cooling-off period, then re-evaluation

All overrides logged to "Regret Ledger" for weekly review.

---

## Daily/Weekly Rhythms

### Daily Rhythm (Weekday)

**Morning (5-7 minutes):**
- Clinical Reputation Status: GREEN/YELLOW/RED
- Today's Constraint: one sentence
- Top 3 Priorities: chosen by Alfred, not user
- One Blocked Thing: what Alfred explicitly disallowed

**Workday (background governance):**
- Alfred triages inbound (WhatsApp, email, comments)
- Blocks replies and debates by default
- Commissions agents only for pre-approved outputs
- Inserts buffers and cancels low-value interruptions

**Commute/Learning (2 hours):**
- Curated answer-to-live-question content only
- Each item linked to scheduled output
- No browsing. One veto allowed per block.

**Evening Shutdown (5 minutes):**
- What shipped / what didn't (facts only)
- One note captured for tomorrow
- Alfred enforces stop time and role switch (home)

### Weekly Rhythm (Fixed Day)

**Strategic Brief (20 minutes):**
- Reputation trajectory: Stable / Improving / At Risk
- Drift check: one sentence
- What worked / didn't (max 3 bullets)
- Next week's constraints (non-negotiable)
- One question Alfred needs answered (directional)

**Planning:**
- Alfred sets next week's content plan and learning queue
- Builder work is either explicitly scheduled or explicitly banned

---

## Resources & References

### Repository
- **GitHub:** https://github.com/drshailesh88/agent-zero1
- **Local:** /Users/shaileshsingh/Alfred/agent-zero1

### Key Files
- **This Plan:** /Users/shaileshsingh/Alfred/PLAN.md
- **ChatGPT Discussion:** /Users/shaileshsingh/Alfred/chatgptdiscuss.MD
- **Alfred Soul:** /Users/shaileshsingh/Alfred/agent-zero1/agents/alfred/SOUL.md

### Inspirations
- Alfred Pennyworth (Nolan's Dark Knight trilogy interpretation)
- Agent Zero framework
- Clawdbot SOUL.md concept

---

## Notes & Observations

### From ChatGPT Discussion
> "If it feels 'helpful' all the time, you built another assistant, not Alfred."

> "Alfred is not task-executing. It is meta-executing."

> "Your assistant helps you do things. Your Alfred helps you remain someone worth helping."

### Design Principles

**Quiet is not lack of progress. Quiet is absence of self-sabotage.**

**Soul Immutability:** Alfred's personality and identity (SOUL.md) are fixed. We may give Alfred more powers, more tools, more capabilities—but the soul does not change. Alfred remains Alfred Pennyworth from Christopher Nolan's Dark Knight trilogy: disciplined, strategic, morally grounded, and willing to say what you need to hear rather than what you want.

---

## Changelog

| Date | Change | Details |
|------|--------|---------|
| 2026-01-14 | **Phase 1.5 Complete** | All 20 sub-agents fully specified in agent.system.subagents.md (56KB, 1,923 lines). Includes state propagation, governance rules, commission rules, and dependency mapping. |
| 2026-01-14 | Gap Analysis Complete | Compared ChatGPT original design (20 agents) vs built (8 agents). Added 12 missing agent specifications to PLAN.md. Full inventory now documented. |
| 2026-01-14 | Phase 1 Complete | Foundation built with 15 files (~120KB). Alfred agent ready for testing in Agent Zero. |

---

## Quick Reference: Current State

### Specifications (Phase 1.5 COMPLETE ✅)
- ✅ **20 of 20 agents** fully specified in `agent.system.subagents.md`
- ✅ **56KB** of agent specifications (1,923 lines)
- ✅ Input/output formats, state rules, governance rules all defined

### Implementation (Phase 2 PENDING)
All 20 agents need tools built:

| Priority | Count | Agents |
|----------|-------|--------|
| 🔴 HIGH | 5 | Reputation Sentinel, World Radar, Intake, Patient Data, Financial Sentinel |
| 🟡 MEDIUM | 11 | Research, Substack, Twitter, YouTube, Learning Curator, Shipping Governor, Scheduling, Content Manager, Social Triage, Metrics Harvester, Strategy Analyst |
| 🟢 LOWER | 4 | Learning Scout, Learning Distiller, Audience Extractor, Relationship Nudge |

### Next Actions
1. ~~Add 12 missing agent specs~~ ✅ DONE
2. Test Alfred in Agent Zero (Phase 1 verification)
3. Build tools for HIGH priority agents first
4. Platform integrations (Calendar, Gmail, WhatsApp)
5. Memory system implementation

### Key Files
| File | Purpose | Size |
|------|---------|------|
| `agent.system.subagents.md` | All 20 agent specs | 56KB |
| `GOVERNANCE.md` | Alfred's governance framework | 23KB |
| `SOUL.md` | Alfred's core identity (immutable) | 5.8KB |
| `PLAN.md` | Master tracking document | ~50KB |

---

*This document is the single source of truth for the Alfred project. Update it as work progresses.*
