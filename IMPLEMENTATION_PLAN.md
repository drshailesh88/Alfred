# ALFRED - Comprehensive Implementation Plan

> **Created:** 2026-01-14
> **Purpose:** Master implementation guide for all phases of Alfred development
> **Status:** Planning Complete - Ready for Execution
> **Total Estimated Scope:** 20 sub-agents, 6 memory systems, 4 platform integrations

---

## Executive Summary

This document provides the complete implementation roadmap for ALFRED - a personal AI governance system built on the Agent Zero framework. It synthesizes:

1. **Original Design Vision** (from ChatGPT discussions - 140KB)
2. **Completed Specifications** (20 sub-agents fully specified - 56KB)
3. **GitHub Tools Research** (open-source tools to integrate)
4. **Technical Architecture** (Agent Zero framework capabilities)

**The Goal:** Build a governance system that protects a senior interventional cardiologist's clinical reputation, prevents self-sabotage patterns, and maintains long-range identity coherence.

**The Constraint:** Alfred must be a governor, not an assistant. If it feels "helpful" all the time, we've built another assistant, not Alfred.

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1: Foundation (COMPLETE)](#phase-1-foundation-complete)
4. [Phase 2: Core Infrastructure Agents](#phase-2-core-infrastructure-agents)
5. [Phase 3: Signal & Awareness Agents](#phase-3-signal--awareness-agents)
6. [Phase 4: Content Generation Agents](#phase-4-content-generation-agents)
7. [Phase 5: Learning Pipeline Agents](#phase-5-learning-pipeline-agents)
8. [Phase 6: Strategy & Analytics Agents](#phase-6-strategy--analytics-agents)
9. [Phase 7: Memory Systems](#phase-7-memory-systems)
10. [Phase 8: Platform Integrations](#phase-8-platform-integrations)
11. [Phase 9: Operational Deployment](#phase-9-operational-deployment)
12. [GitHub Tools Integration Map](#github-tools-integration-map)
13. [Dependency Graph](#dependency-graph)
14. [Risk Mitigation](#risk-mitigation)
15. [Success Criteria](#success-criteria)

---

## Current State Assessment

### What's Built (Phase 1 - COMPLETE)

| Component | File | Size | Status |
|-----------|------|------|--------|
| Core Identity | SOUL.md | 5.8KB | ✅ Immutable |
| Governance Framework | GOVERNANCE.md | 23KB | ✅ Complete |
| Role Definition | agent.system.main.role.md | 5.9KB | ✅ Complete |
| Communication Style | agent.system.main.communication.md | 6.1KB | ✅ Complete |
| Detection Systems | agent.system.main.detection.md | 11.7KB | ✅ Complete |
| Intervention Protocol | agent.system.main.intervention.md | 12.1KB | ✅ Complete |
| Memory Schema | agent.system.main.memory_schema.md | 16.6KB | ✅ Complete |
| Boundaries | agent.system.main.boundaries.md | 13.8KB | ✅ Complete |
| Sub-Agent Specs | agent.system.subagents.md | 56KB | ✅ 20/20 Complete |
| Sub-Agent Rules | agent.system.subagent_rules.md | 10.2KB | ✅ Complete |
| Response Guidelines | agent.system.tool.response.md | 2.9KB | ✅ Complete |
| Initial Message | fw.initial_message.md | 100B | ✅ Complete |
| Python Extension | _10_alfred_init.py | 500B | ✅ Complete |
| **TOTAL** | 15 files | ~165KB | ✅ Phase 1 Complete |

### What's Specified But Not Built (Phase 2+)

All 20 sub-agents have complete specifications including:
- Input formats
- Output formats
- Governance rules
- State propagation rules
- Dependencies

**No tools are implemented yet.** Phase 2 begins the actual Python tool development.

---

## Architecture Overview

### Three-Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         LAYER 1: USER                          │
│                     (Decides WHY)                              │
│         • Sets values and constraints                          │
│         • Final authority on all decisions                     │
│         • Only talks to Alfred                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER 2: ALFRED                         │
│                  (Decides WHAT and WHEN)                       │
│         • Single point of contact                              │
│         • Commissions sub-agents                               │
│         • Synthesizes outputs                                  │
│         • Enforces governance                                  │
│         • Controls decision states (GREEN/YELLOW/RED)          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 3: SUB-AGENTS                        │
│                      (Decide HOW)                              │
│         • Execute commissions                                  │
│         • Return structured packets                            │
│         • Never contact user directly                          │
│         • No memory across commissions                         │
│         • No scope expansion                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Sub-Agent Categories

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           20 SUB-AGENTS                                      │
├──────────────────┬───────────────────────────────────────────────────────────┤
│ SIGNAL/AWARENESS │ Reputation Sentinel, World Radar, Social Triage          │
│ (3 agents)       │                                                           │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ CONTENT GEN      │ Research Agent, Substack Agent, Twitter Thread Agent,    │
│ (4 agents)       │ YouTube Script Agent                                      │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ LEARNING         │ Learning Curator, Learning Scout, Learning Distiller     │
│ (3 agents)       │                                                           │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ OPERATIONS       │ Intake Agent, Patient Data Agent, Scheduling Agent,      │
│ (7 agents)       │ Content Manager, Shipping Governor, Financial Sentinel,  │
│                  │ Relationship Nudge Agent                                  │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ STRATEGY         │ Social Metrics Harvester, Audience Signals Extractor,    │
│ (3 agents)       │ Content Strategy Analyst                                  │
└──────────────────┴───────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation (COMPLETE)

**Status:** ✅ 100% Complete

All foundation work is done:
- Agent Zero framework cloned and configured
- Alfred profile created with 15 files
- All prompt systems defined
- All 20 sub-agent specifications written
- Governance rules established

**Next:** Test Alfred in Agent Zero to verify prompts load correctly.

---

## Phase 2: Core Infrastructure Agents

**Priority:** 🔴 HIGH (Build First)
**Estimated Scope:** 5 agents

These agents form the foundation that other agents depend on.

### 2.1 Intake Agent

**Purpose:** Unified ingestion for all inbound communication

**GitHub Tools to Integrate:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **whatsapp-mcp** | MCP | WhatsApp messages | MCP Server |
| **mcp-gsuite** | MCP | Gmail integration | MCP Server |
| **PaddleOCR** | 60k | Document OCR | Python library |
| **MinerU** | 51k | PDF extraction | Python library |
| **Paperless-ngx** | 35k | Document management | API integration |

**Implementation Tasks:**

```
[ ] 2.1.1 Create intake_agent.py in agents/alfred/tools/
[ ] 2.1.2 Implement WhatsApp channel adapter (whatsapp-mcp)
[ ] 2.1.3 Implement Email channel adapter (mcp-gsuite)
[ ] 2.1.4 Implement Scan/Document adapter (PaddleOCR + MinerU)
[ ] 2.1.5 Implement Calendar adapter (google-calendar-mcp)
[ ] 2.1.6 Create INBOUND_PACKET output format
[ ] 2.1.7 Add queue management for batch processing
[ ] 2.1.8 Test with sample inputs from each channel
[ ] 2.1.9 Document configuration requirements
```

**Output Format:**
```json
{
  "packet_type": "INBOUND_BATCH",
  "timestamp": "ISO8601",
  "items": [
    {
      "source": "WhatsApp|Email|Scan|Calendar",
      "timestamp": "ISO8601",
      "sender": "string",
      "subject": "string (if applicable)",
      "preview": "string (1-2 lines)",
      "attachments": ["list"],
      "urgency_markers": ["list"]
    }
  ]
}
```

---

### 2.2 Patient Data Agent

**Purpose:** Clinical document vault with natural language retrieval

**GitHub Tools to Integrate:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **MedCAT** | 500 | Medical NLP/NER | Python library |
| **Qdrant** | 22k | Vector database | Self-hosted |
| **LlamaIndex** | 40k | RAG framework | Python library |
| **Presidio** | 3.5k | PHI de-identification | Python library |
| **mcp-simple-pubmed** | MCP | Medical literature | MCP Server |

**Implementation Tasks:**

```
[ ] 2.2.1 Create patient_data_agent.py
[ ] 2.2.2 Set up Qdrant vector database (Docker)
[ ] 2.2.3 Implement document ingestion pipeline
[ ] 2.2.4 Integrate MedCAT for medical entity extraction
[ ] 2.2.5 Implement Presidio for PHI redaction
[ ] 2.2.6 Build natural language query interface
[ ] 2.2.7 Create patient record retrieval functions
[ ] 2.2.8 Implement timeline generation
[ ] 2.2.9 Add access logging for compliance
[ ] 2.2.10 Test with sample patient records
```

**Security Requirements:**
- All data encrypted at rest
- Access logging for every query
- PHI redaction before any output
- No data leaves local environment

---

### 2.3 Scheduling/Calendar Agent

**Purpose:** Time block management and calendar optimization

**GitHub Tools to Integrate:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **Cal.com** | 39k | Scheduling platform | API/Self-hosted |
| **google-calendar-mcp** | MCP | Calendar access | MCP Server |
| **mcp-gsuite** | MCP | Full Google suite | MCP Server |

**Implementation Tasks:**

```
[ ] 2.3.1 Create scheduling_agent.py
[ ] 2.3.2 Integrate Google Calendar MCP
[ ] 2.3.3 Define block types (CLINICAL, DEEP_WORK, BUFFER, etc.)
[ ] 2.3.4 Implement protection rules for each block type
[ ] 2.3.5 Create conflict detection system
[ ] 2.3.6 Build meeting load calculator
[ ] 2.3.7 Implement buffer enforcement
[ ] 2.3.8 Create CALENDAR_REPORT output format
[ ] 2.3.9 Add protection alerts (overload, buffer breach)
[ ] 2.3.10 Test with sample calendar data
```

---

### 2.4 Financial Sentinel

**Purpose:** Monitor financial patterns, prevent quiet erosion

**GitHub Tools to Integrate:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **Firefly III** | 22k | Finance tracking | API integration |
| **Wallos** | 7k | Subscription tracking | Self-hosted |
| **Actual Budget** | 24k | Budgeting | API integration |

**Implementation Tasks:**

```
[ ] 2.4.1 Create financial_sentinel.py
[ ] 2.4.2 Integrate with Firefly III API
[ ] 2.4.3 Set up Wallos for subscription tracking
[ ] 2.4.4 Implement subscription overlap detection
[ ] 2.4.5 Build tool ROI calculation
[ ] 2.4.6 Create unused/underused tool detection
[ ] 2.4.7 Implement impulse purchase pattern detection
[ ] 2.4.8 Create FINANCIAL_REPORT output format
[ ] 2.4.9 Add renewal alerts
[ ] 2.4.10 Test with sample financial data
```

---

### 2.5 Shipping Governor

**Purpose:** Prevent endless preparation, enforce shipping

**Implementation Tasks:**

```
[ ] 2.5.1 Create shipping_governor.py
[ ] 2.5.2 Define project tracking schema
[ ] 2.5.3 Implement output tracking per project
[ ] 2.5.4 Create "days without output" calculator
[ ] 2.5.5 Build building-vs-shipping ratio tracker
[ ] 2.5.6 Implement freeze recommendations
[ ] 2.5.7 Create SHIPPING_REPORT output format
[ ] 2.5.8 Add shipping alerts (WARNING, CRITICAL)
[ ] 2.5.9 Test with sample project data
```

---

## Phase 3: Signal & Awareness Agents

**Priority:** 🔴 HIGH (Critical for Protection)
**Estimated Scope:** 3 agents

These agents protect against reputational damage and detect threats.

### 3.1 Reputation Sentinel

**Purpose:** Monitor for reputational and clinical risk

**GitHub Tools to Integrate:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **twscrape** | 2k | Twitter scraping | Python library |
| **VADER** | - | Sentiment analysis | Python library |
| **Detoxify** | 1.1k | Toxicity detection | Python library |
| **BERTopic** | 6k | Topic clustering | Python library |
| **Botometer** | - | Bot detection | API |

**Implementation Tasks:**

```
[ ] 3.1.1 Create reputation_sentinel.py
[ ] 3.1.2 Set up Twitter monitoring (twscrape)
[ ] 3.1.3 Implement YouTube comment monitoring
[ ] 3.1.4 Add Instagram mention tracking
[ ] 3.1.5 Integrate sentiment analysis (VADER)
[ ] 3.1.6 Add toxicity detection (Detoxify)
[ ] 3.1.7 Implement risk classification (0-100)
[ ] 3.1.8 Create state recommendation logic (GREEN/YELLOW/RED)
[ ] 3.1.9 Build REPUTATION_PACKET output format
[ ] 3.1.10 Test with sample social data
```

**Risk Classifications:**
- MISINTERPRETATION
- MISINFORMATION
- HARASSMENT
- PEER_CRITIQUE
- REVIEW_RISK

**Hard Rule:** Never recommend comment-thread engagement.

---

### 3.2 World Radar

**Purpose:** Detect constraint-changing global events

**GitHub Tools to Integrate:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **mcp-simple-pubmed** | MCP | Medical literature | MCP Server |
| **WebSearch** | Built-in | News monitoring | Claude tool |

**Implementation Tasks:**

```
[ ] 3.2.1 Create world_radar.py
[ ] 3.2.2 Set up PubMed monitoring for clinical guidelines
[ ] 3.2.3 Implement regulatory news tracking
[ ] 3.2.4 Add AI governance monitoring
[ ] 3.2.5 Create constraint-change detection logic
[ ] 3.2.6 Build WORLD_SIGNAL output format
[ ] 3.2.7 Implement escalation criteria (5 conditions)
[ ] 3.2.8 Add time horizon classification
[ ] 3.2.9 Test with sample news events
```

**Escalation Only If:**
1. Clinical practice may change
2. Regulation affects speech or tools
3. Reputational exposure increases
4. Platform policy affects content distribution
5. AI governance impacts tool availability

---

### 3.3 Social Triage

**Purpose:** Process comments for content opportunities (not risks)

**GitHub Tools:**
- Same as Reputation Sentinel (twscrape, VADER, BERTopic)

**Implementation Tasks:**

```
[ ] 3.3.1 Create social_triage.py
[ ] 3.3.2 Implement comment classification system
[ ] 3.3.3 Build question extraction pipeline
[ ] 3.3.4 Create confusion pattern detection
[ ] 3.3.5 Implement praise pattern recognition
[ ] 3.3.6 Build content opportunity scoring
[ ] 3.3.7 Create SOCIAL_TRIAGE_REPORT output format
[ ] 3.3.8 Test with sample comments
```

**Key Distinction:**
- Reputation Sentinel → RISK detection
- Social Triage → OPPORTUNITY detection

---

## Phase 4: Content Generation Agents

**Priority:** 🟡 MEDIUM
**Estimated Scope:** 4 agents

### 4.1 Research Agent (Evidence Scout)

**Purpose:** Evidence retrieval and synthesis

**GitHub Tools:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **mcp-simple-pubmed** | MCP | PubMed search | MCP Server |
| **LlamaIndex** | 40k | Document processing | Python library |
| **Perplexity API** | - | Research queries | API |

**Implementation Tasks:**

```
[ ] 4.1.1 Create research_agent.py
[ ] 4.1.2 Integrate PubMed MCP server
[ ] 4.1.3 Implement evidence strength classification
[ ] 4.1.4 Build contradiction detection
[ ] 4.1.5 Create citation formatting
[ ] 4.1.6 Add uncertainty flagging
[ ] 4.1.7 Create EVIDENCE_BRIEF output format
[ ] 4.1.8 Test with sample research queries
```

**Integrity Rule:** If evidence is weak, MUST say so.

---

### 4.2 Substack Agent (Authority Builder)

**Purpose:** Long-form content for intellectual authority

**Implementation Tasks:**

```
[ ] 4.2.1 Create substack_agent.py
[ ] 4.2.2 Build outline generation from evidence briefs
[ ] 4.2.3 Implement uncertainty disclosure insertion
[ ] 4.2.4 Create internal linking suggestions
[ ] 4.2.5 Add quality gate checks
[ ] 4.2.6 Build LONGFORM_DRAFT output format
[ ] 4.2.7 Integrate state-based blocking (blocked in RED)
[ ] 4.2.8 Test with sample topics
```

**Quality Gates:**
- No shock hooks
- No outrage framing
- All claims supported
- Uncertainty stated

---

### 4.3 Twitter Thread Agent

**Purpose:** Translate approved long-form to threads

**Implementation Tasks:**

```
[ ] 4.3.1 Create twitter_thread_agent.py
[ ] 4.3.2 Build source material parser
[ ] 4.3.3 Implement character count optimizer
[ ] 4.3.4 Create thread flow generator
[ ] 4.3.5 Add tone verification
[ ] 4.3.6 Build THREAD_DRAFT output format
[ ] 4.3.7 Implement state blocking (YELLOW/RED = nothing)
[ ] 4.3.8 Test with sample long-form content
```

**Hard Block:** Produces NOTHING if Alfred state is YELLOW or RED.

---

### 4.4 YouTube Script Agent

**Purpose:** Calm, evidence-anchored video scripts

**Implementation Tasks:**

```
[ ] 4.4.1 Create youtube_script_agent.py
[ ] 4.4.2 Build section-based script structure
[ ] 4.4.3 Implement timing estimation
[ ] 4.4.4 Create visual suggestion system
[ ] 4.4.5 Add uncertainty moment markers
[ ] 4.4.6 Build SCRIPT_DRAFT output format
[ ] 4.4.7 Integrate state blocking
[ ] 4.4.8 Test with sample topics
```

**Shorts/Reels Rule:** Repurposed from long-form only, never original short-form.

---

## Phase 5: Learning Pipeline Agents

**Priority:** 🟡 MEDIUM
**Estimated Scope:** 3 agents

### 5.1 Learning Curator (Just-In-Time)

**Purpose:** Match learning to execution problems

**GitHub Tools:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **ts-fsrs** | 506 | Spaced repetition | Algorithm |
| **Logseq** | 40k | PKM integration | MCP potential |
| **Obsidian MCP** | MCP | Note integration | MCP Server |

**Implementation Tasks:**

```
[ ] 5.1.1 Create learning_curator.py
[ ] 5.1.2 Build question extraction from context
[ ] 5.1.3 Implement learning-output linking
[ ] 5.1.4 Create commute window matching
[ ] 5.1.5 Add learning debt tracking
[ ] 5.1.6 Build LEARNING_QUEUE output format
[ ] 5.1.7 Test with sample learning requests
```

**Core Rule:** No learning queued without linked output.

---

### 5.2 Learning Scout

**Purpose:** Search and discover learning resources

**Implementation Tasks:**

```
[ ] 5.2.1 Create learning_scout.py
[ ] 5.2.2 Build resource search across platforms
[ ] 5.2.3 Implement credibility assessment
[ ] 5.2.4 Create relevance scoring
[ ] 5.2.5 Add timestamp extraction for videos
[ ] 5.2.6 Build LEARNING_CANDIDATES output format
[ ] 5.2.7 Test with sample questions
```

---

### 5.3 Learning Distiller

**Purpose:** Extract learning questions from conversations

**GitHub Tools:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **WhisperX** | - | Transcription | Python library |
| **KeyBERT** | 3.5k | Keyword extraction | Python library |

**Implementation Tasks:**

```
[ ] 5.3.1 Create learning_distiller.py
[ ] 5.3.2 Build conversation analyzer
[ ] 5.3.3 Implement stuck-point extraction
[ ] 5.3.4 Create priority classification
[ ] 5.3.5 Add avoidance pattern detection
[ ] 5.3.6 Build LEARNING_QUESTIONS output format
[ ] 5.3.7 Test with sample conversations
```

---

## Phase 6: Strategy & Analytics Agents

**Priority:** 🟡 MEDIUM
**Estimated Scope:** 3 agents

### 6.1 Social Metrics Harvester

**Purpose:** Automated metrics collection across platforms

**GitHub Tools:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **Postiz** | 14k | Multi-platform analytics | Self-hosted |
| **Umami** | 33k | Web analytics | Self-hosted |
| **twscrape** | 2k | Twitter metrics | Python library |
| **youtube-analyzer** | - | YouTube metrics | Python library |

**Implementation Tasks:**

```
[ ] 6.1.1 Create social_metrics_harvester.py
[ ] 6.1.2 Implement Twitter metrics collection
[ ] 6.1.3 Add YouTube metrics collection
[ ] 6.1.4 Integrate Instagram metrics
[ ] 6.1.5 Add Substack metrics
[ ] 6.1.6 Create cross-platform normalization
[ ] 6.1.7 Build METRICS_REPORT output format
[ ] 6.1.8 Set up weekly cadence
[ ] 6.1.9 Test with live accounts
```

---

### 6.2 Audience Signals Extractor

**Purpose:** Cluster comments into themes

**GitHub Tools:**

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **BERTopic** | 6k | Topic modeling | Python library |
| **KeyBERT** | 3.5k | Keyword extraction | Python library |
| **PyABSA** | 2.5k | Aspect sentiment | Python library |
| **sentence-transformers** | 16k | Clustering | Python library |

**Implementation Tasks:**

```
[ ] 6.2.1 Create audience_signals_extractor.py
[ ] 6.2.2 Build comment aggregation pipeline
[ ] 6.2.3 Implement topic clustering (BERTopic)
[ ] 6.2.4 Add question detection
[ ] 6.2.5 Create confusion pattern recognition
[ ] 6.2.6 Build trust signal identification
[ ] 6.2.7 Create AUDIENCE_SIGNALS output format
[ ] 6.2.8 Test with sample comments
```

---

### 6.3 Content Strategy Analyst

**Purpose:** Weekly memo comparing performance to positioning

**Implementation Tasks:**

```
[ ] 6.3.1 Create content_strategy_analyst.py
[ ] 6.3.2 Build positioning charter integration
[ ] 6.3.3 Implement drift detection
[ ] 6.3.4 Create performance diagnosis
[ ] 6.3.5 Build experiment recommendation
[ ] 6.3.6 Create STRATEGY_MEMO output format
[ ] 6.3.7 Set up weekly cadence
[ ] 6.3.8 Test with sample data
```

**Dependencies:**
- Requires Social Metrics Harvester data
- Requires Audience Signals Extractor data
- Requires Positioning Charter (to be created)

---

## Phase 7: Memory Systems

**Priority:** 🟡 MEDIUM
**Estimated Scope:** 6 memory systems

### GitHub Tools for Memory:

| Tool | Stars | Purpose | Integration Method |
|------|-------|---------|-------------------|
| **mcp-memory-service** | MCP | Persistent memory | MCP Server |
| **Qdrant** | 22k | Vector storage | Self-hosted |
| **ChromaDB** | 15k | Alternative vector DB | Python library |

### 7.1 Pattern Registry

```
[ ] 7.1.1 Create pattern_registry.py
[ ] 7.1.2 Define pattern schema (JSON)
[ ] 7.1.3 Implement occurrence tracking
[ ] 7.1.4 Build trajectory analysis
[ ] 7.1.5 Add intervention effectiveness tracking
[ ] 7.1.6 Create pattern query interface
[ ] 7.1.7 Test with sample patterns
```

### 7.2 Values Hierarchy

```
[ ] 7.2.1 Create values_hierarchy.py
[ ] 7.2.2 Define stated vs. revealed values schema
[ ] 7.2.3 Implement conflict detection
[ ] 7.2.4 Build inference rules
[ ] 7.2.5 Add awareness tracking
[ ] 7.2.6 Test with sample value conflicts
```

### 7.3 Self-Violation Log

```
[ ] 7.3.1 Create self_violation_log.py
[ ] 7.3.2 Define violation schema
[ ] 7.3.3 Implement justification type classification
[ ] 7.3.4 Build outcome tracking
[ ] 7.3.5 Add regret correlation
[ ] 7.3.6 Test with sample violations
```

### 7.4 Regret Memory

```
[ ] 7.4.1 Create regret_memory.py
[ ] 7.4.2 Define regret schema
[ ] 7.4.3 Implement decision tracking
[ ] 7.4.4 Build lesson extraction
[ ] 7.4.5 Add lesson application tracking
[ ] 7.4.6 Test with sample regrets
```

### 7.5 Threshold Map

```
[ ] 7.5.1 Create threshold_map.py
[ ] 7.5.2 Define threshold categories
[ ] 7.5.3 Implement proximity calculation
[ ] 7.5.4 Build trend analysis
[ ] 7.5.5 Add warning escalation
[ ] 7.5.6 Test with sample thresholds
```

### 7.6 Optionality Register

```
[ ] 7.6.1 Create optionality_register.py
[ ] 7.6.2 Define option schema
[ ] 7.6.3 Implement status tracking (open/narrowing/closed)
[ ] 7.6.4 Build closure risk assessment
[ ] 7.6.5 Add decision correlation
[ ] 7.6.6 Test with sample options
```

---

## Phase 8: Platform Integrations

**Priority:** 🟢 MEDIUM-LOW
**Estimated Scope:** 4 integrations

### 8.1 Calendar Integration

**MCP Server:** google-calendar-mcp, mcp-gsuite

```
[ ] 8.1.1 Configure google-calendar-mcp
[ ] 8.1.2 Set up OAuth credentials
[ ] 8.1.3 Test calendar read access
[ ] 8.1.4 Test event creation
[ ] 8.1.5 Implement block type enforcement
```

### 8.2 Gmail Integration

**MCP Server:** mcp-gsuite

```
[ ] 8.2.1 Configure Gmail in mcp-gsuite
[ ] 8.2.2 Set up OAuth credentials
[ ] 8.2.3 Test email read access
[ ] 8.2.4 Implement triage rules
[ ] 8.2.5 Test with sample emails
```

### 8.3 WhatsApp Integration

**MCP Server:** whatsapp-mcp

```
[ ] 8.3.1 Configure whatsapp-mcp
[ ] 8.3.2 Set up authentication
[ ] 8.3.3 Test message reading
[ ] 8.3.4 Implement channel routing
[ ] 8.3.5 Test with sample messages
```

### 8.4 Social Media Integration

**Tools:** twscrape, YouTube API, Instagram API

```
[ ] 8.4.1 Set up Twitter API credentials
[ ] 8.4.2 Configure YouTube API
[ ] 8.4.3 Set up Instagram access
[ ] 8.4.4 Implement monitoring loops
[ ] 8.4.5 Test with live accounts
```

---

## Phase 9: Operational Deployment

**Priority:** 🟢 LOW (After all agents built)
**Estimated Scope:** 5 milestones

### 9.1 Docker Deployment

```
[ ] 9.1.1 Create Alfred-specific Dockerfile
[ ] 9.1.2 Configure all MCP servers
[ ] 9.1.3 Set up persistent volumes
[ ] 9.1.4 Create docker-compose.yml
[ ] 9.1.5 Test complete deployment
```

### 9.2 Daily Brief Automation

```
[ ] 9.2.1 Create daily_brief.py
[ ] 9.2.2 Implement morning brief generation
[ ] 9.2.3 Add evening shutdown report
[ ] 9.2.4 Configure notification delivery
[ ] 9.2.5 Test with live data
```

### 9.3 Weekly Brief Automation

```
[ ] 9.3.1 Create weekly_brief.py
[ ] 9.3.2 Implement strategic brief generation
[ ] 9.3.3 Add planning integration
[ ] 9.3.4 Configure delivery
[ ] 9.3.5 Test with live data
```

### 9.4 First Live Simulation

```
[ ] 9.4.1 Define simulation scenario
[ ] 9.4.2 Run 1-day simulation
[ ] 9.4.3 Review all agent outputs
[ ] 9.4.4 Calibrate thresholds
[ ] 9.4.5 Document issues
```

### 9.5 Threshold Calibration

```
[ ] 9.5.1 Review all default thresholds
[ ] 9.5.2 Adjust based on simulation
[ ] 9.5.3 Test override protocol
[ ] 9.5.4 Verify memory persistence
[ ] 9.5.5 Final documentation
```

---

## GitHub Tools Integration Map

### MCP Servers (Direct Integration)

| MCP Server | Purpose | Priority | Phase |
|------------|---------|----------|-------|
| mcp-gsuite | Gmail + Calendar | HIGH | 2, 8 |
| google-calendar-mcp | Multi-account calendar | HIGH | 2, 8 |
| whatsapp-mcp | WhatsApp messages | HIGH | 2, 8 |
| mcp-simple-pubmed | Medical literature | HIGH | 3, 4 |
| mcp-memory-service | Persistent memory | MEDIUM | 7 |
| obsidian-mcp-server | Note integration | LOW | 5 |
| notion-mcp-server | Database access | LOW | 5 |
| mcp-server-qdrant | Vector search | MEDIUM | 2, 7 |

### Python Libraries (Import)

| Library | Purpose | Priority | Phase |
|---------|---------|----------|-------|
| PaddleOCR | Document OCR | HIGH | 2 |
| MinerU | PDF extraction | HIGH | 2 |
| MedCAT | Medical NLP | HIGH | 2 |
| Presidio | PHI redaction | HIGH | 2 |
| LlamaIndex | RAG framework | HIGH | 2, 4 |
| Qdrant-client | Vector DB | HIGH | 2, 7 |
| twscrape | Twitter scraping | HIGH | 3, 6 |
| VADER | Sentiment | HIGH | 3, 6 |
| Detoxify | Toxicity | HIGH | 3 |
| BERTopic | Topic modeling | MEDIUM | 3, 6 |
| KeyBERT | Keywords | MEDIUM | 5, 6 |
| PyABSA | Aspect sentiment | LOW | 6 |
| sentence-transformers | Embeddings | MEDIUM | 6 |

### Self-Hosted Services

| Service | Purpose | Priority | Phase |
|---------|---------|----------|-------|
| Firefly III | Finance tracking | HIGH | 2 |
| Wallos | Subscriptions | HIGH | 2 |
| Qdrant | Vector database | HIGH | 2, 7 |
| Paperless-ngx | Documents | MEDIUM | 2 |
| Cal.com | Scheduling | LOW | 2 |
| Postiz | Social analytics | MEDIUM | 6 |
| Umami | Web analytics | LOW | 6 |

---

## Dependency Graph

```
                    ┌─────────────────────────────────────────┐
                    │              ALFRED CORE                │
                    │  (Governance, State, Commission)        │
                    └──────────────────┬──────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐            ┌─────────────────┐            ┌─────────────────┐
│ INTAKE AGENT  │            │ PATIENT DATA    │            │ FINANCIAL       │
│ (Foundation)  │            │ AGENT           │            │ SENTINEL        │
└───────────────┘            └─────────────────┘            └─────────────────┘
        │                              │
        │ feeds                        │ supports
        ▼                              ▼
┌───────────────┐            ┌─────────────────┐
│ REPUTATION    │◄───────────│ RESEARCH AGENT  │
│ SENTINEL      │            │ (Evidence)      │
└───────────────┘            └────────┬────────┘
        │                             │
        │ informs state               │ feeds
        ▼                             ▼
┌───────────────┐            ┌─────────────────────────────────────────┐
│ SOCIAL TRIAGE │            │ CONTENT AGENTS                          │
│               │            │ (Substack, Twitter, YouTube)            │
└───────────────┘            └────────────────────┬────────────────────┘
        │                                         │
        │ provides                                │ publishes
        ▼                                         ▼
┌───────────────┐            ┌─────────────────────────────────────────┐
│ SOCIAL METRICS│            │ CONTENT MANAGER                         │
│ HARVESTER     │            │ (Pipeline Orchestration)                │
└───────┬───────┘            └─────────────────────────────────────────┘
        │
        │ data for
        ▼
┌───────────────┐
│ AUDIENCE      │
│ SIGNALS       │
│ EXTRACTOR     │
└───────┬───────┘
        │
        │ feeds
        ▼
┌───────────────┐
│ CONTENT       │
│ STRATEGY      │
│ ANALYST       │
└───────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                        LEARNING PIPELINE                           │
│                                                                     │
│  LEARNING DISTILLER ──► LEARNING SCOUT ──► LEARNING CURATOR        │
│  (extracts questions)   (finds resources)  (curates queue)         │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                        SUPPORTING AGENTS                            │
│                                                                     │
│  SCHEDULING AGENT       SHIPPING GOVERNOR      RELATIONSHIP NUDGE  │
│  WORLD RADAR                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| API rate limits | HIGH | Implement caching, batch processing |
| MCP server failures | MEDIUM | Fallback to direct API calls |
| Vector DB performance | MEDIUM | Index optimization, sharding |
| PHI data security | CRITICAL | Local-only processing, encryption |

### Design Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Alfred becomes assistant | CRITICAL | Strict governance rules, intervention scarcity |
| Sub-agent scope creep | HIGH | Enforce structured outputs |
| Memory bloat | MEDIUM | Decay protocol, archiving |
| Over-intervention | HIGH | Weekly frequency targets |

### Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| False positive reputation alerts | MEDIUM | Calibration period, confidence thresholds |
| Missed genuine threats | HIGH | Conservative initial thresholds |
| User bypass attempts | MEDIUM | Log all overrides, surface patterns |

---

## Success Criteria

### Phase 2 Success (Core Infrastructure)

- [ ] All 5 infrastructure agents operational
- [ ] Intake Agent processing all channels
- [ ] Patient Data Agent serving queries
- [ ] Financial Sentinel tracking subscriptions
- [ ] Shipping Governor enforcing outputs

### Phase 3-4 Success (Signal + Content)

- [ ] Reputation Sentinel monitoring 4 platforms
- [ ] World Radar detecting constraint changes
- [ ] Research Agent producing evidence briefs
- [ ] Content agents producing drafts

### Phase 5-6 Success (Learning + Strategy)

- [ ] Learning pipeline operational
- [ ] Metrics harvesting weekly
- [ ] Strategy memos generating weekly

### Phase 7-8 Success (Memory + Integration)

- [ ] All 6 memory systems persistent
- [ ] Calendar integration working
- [ ] Gmail integration working
- [ ] WhatsApp integration working (if possible)

### Phase 9 Success (Operational)

- [ ] Daily briefs generating automatically
- [ ] Weekly briefs generating automatically
- [ ] First live simulation completed
- [ ] User operating with Alfred for 1 week

### Ultimate Success Criteria

From the original design:

> "Alfred succeeds not when output is maximized, but when the person remains coherent and can still choose to exit."

Metrics:
- Reputation incidents: 0
- Sleep sacrifice patterns: Detected and flagged
- Shipping stalls: Identified and resolved
- Override frequency: Low (indicates appropriate calibration)
- User satisfaction: Sustainable, not always comfortable

---

## Appendix A: File Structure After Implementation

```
/Users/shaileshsingh/Alfred/
├── agent-zero1/
│   ├── agents/alfred/
│   │   ├── SOUL.md                           # Immutable identity
│   │   ├── GOVERNANCE.md                     # Governance framework
│   │   ├── _context.md                       # Quick reference
│   │   ├── prompts/
│   │   │   ├── agent.system.main.md          # Master include
│   │   │   ├── agent.system.main.role.md
│   │   │   ├── agent.system.main.communication.md
│   │   │   ├── agent.system.main.detection.md
│   │   │   ├── agent.system.main.intervention.md
│   │   │   ├── agent.system.main.memory_schema.md
│   │   │   ├── agent.system.main.boundaries.md
│   │   │   ├── agent.system.subagents.md
│   │   │   ├── agent.system.subagent_rules.md
│   │   │   ├── agent.system.tool.response.md
│   │   │   └── fw.initial_message.md
│   │   ├── extensions/
│   │   │   └── agent_init/
│   │   │       └── _10_alfred_init.py
│   │   ├── tools/                            # NEW: Phase 2+
│   │   │   ├── __init__.py
│   │   │   ├── intake_agent.py
│   │   │   ├── patient_data_agent.py
│   │   │   ├── scheduling_agent.py
│   │   │   ├── financial_sentinel.py
│   │   │   ├── shipping_governor.py
│   │   │   ├── reputation_sentinel.py
│   │   │   ├── world_radar.py
│   │   │   ├── social_triage.py
│   │   │   ├── research_agent.py
│   │   │   ├── substack_agent.py
│   │   │   ├── twitter_thread_agent.py
│   │   │   ├── youtube_script_agent.py
│   │   │   ├── learning_curator.py
│   │   │   ├── learning_scout.py
│   │   │   ├── learning_distiller.py
│   │   │   ├── social_metrics_harvester.py
│   │   │   ├── audience_signals_extractor.py
│   │   │   ├── content_strategy_analyst.py
│   │   │   ├── content_manager.py
│   │   │   └── relationship_nudge_agent.py
│   │   └── memory/                           # NEW: Phase 7
│   │       ├── __init__.py
│   │       ├── pattern_registry.py
│   │       ├── values_hierarchy.py
│   │       ├── self_violation_log.py
│   │       ├── regret_memory.py
│   │       ├── threshold_map.py
│   │       └── optionality_register.py
│   └── ... (Agent Zero framework files)
├── gateway/                                   # Multi-platform messaging
├── PLAN.md                                    # Original master plan
├── IMPLEMENTATION_PLAN.md                     # THIS DOCUMENT
├── chatgptdiscuss.MD                         # Original design discussions
└── .claude/
    └── settings.local.json                   # MCP configuration
```

---

## Appendix B: MCP Configuration

Add to `.mcp.json` for full integration:

```json
{
  "mcpServers": {
    "task-master-ai": {
      "command": "npx",
      "args": ["-y", "--package=task-master-ai", "task-master-ai"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    },
    "pubmed-mcp-server": {
      "command": "npx",
      "args": ["-y", "pubmed-mcp-server"]
    },
    "google-calendar": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/google-calendar-mcp"]
    },
    "gsuite": {
      "command": "npx",
      "args": ["-y", "mcp-gsuite"]
    },
    "whatsapp": {
      "command": "npx",
      "args": ["-y", "whatsapp-mcp"]
    },
    "memory-service": {
      "command": "npx",
      "args": ["-y", "mcp-memory-service"]
    },
    "qdrant": {
      "command": "npx",
      "args": ["-y", "@qdrant/mcp-server-qdrant"]
    }
  }
}
```

---

## Appendix C: Quick Start Commands

### Testing Alfred

```bash
cd /Users/shaileshsingh/Alfred/agent-zero1

# Option 1: Docker (recommended)
docker pull agent0ai/agent-zero
docker run -p 50001:50001 -v $(pwd):/app agent0ai/agent-zero

# Option 2: Local
pip install -r requirements.txt
python run_ui.py

# Access: http://localhost:50001
```

### Starting a New Phase

```bash
# Create branch for phase
git checkout -b phase-2-infrastructure

# Work on agents
cd agent-zero1/agents/alfred/tools/

# After completion
git add .
git commit -m "Phase 2: Implement core infrastructure agents"
git push origin phase-2-infrastructure
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-14*
*Author: Claude Code + Shailesh Singh*
