# ALFRED Codebase Recovery Plan

## What Happened
You had built substantial work locally (Phase 8 complete with 41 files), committed to GitHub, then imported a branch that had different architecture. The `git reset --hard` replaced your main with the branch, losing some critical components.

## Good News
**Your original code is fully recoverable from commit `9eb9d8a`**

---

## Recovery Strategy: Best of Both Worlds

### KEEP from Branch Import (Current HEAD)
These are the features you wanted:

| Component | Location | Lines | Purpose |
|-----------|----------|-------|---------|
| Docker deployment | `Dockerfile`, `docker-compose.yml` | ~150 | Root-level containerization |
| Deployment guide | `DEPLOYMENT.md` | 626 | VPS deployment instructions |
| CLI tool | `alfred_cli.py` | 1,271 | Terminal interface |
| Eval framework | `evals/` | 3,382 | Testing suite |
| Test suite | `tests/` | ~6 modules | Unit tests |
| Gateway | `gateway/` | 1,234 | Telegram/Signal/Discord |
| Gmail adapter | `integrations/gmail_adapter.py` | 1,084 | Email integration |
| WhatsApp adapter | `integrations/whatsapp_adapter.py` | 1,068 | Messaging |
| Calendar adapter | `integrations/calendar_adapter.py` | 1,058 | Enhanced calendar |
| Social adapter | `integrations/social_adapter.py` | 1,706 | API-based social |
| Package config | `pyproject.toml`, `requirements.txt` | ~80 | Dependencies |
| Environment | `.env.example`, `.mcp.json` | ~200 | Configuration |
| SOUL v2 | `SOUL.md` (already present) | ~400 | Personality updates |

### RECOVER from Your Original (9eb9d8a)
These are the components you built that were lost:

| Component | Location | Lines | Purpose |
|-----------|----------|-------|---------|
| **Orchestrator** | `core/orchestrator.py` | ~1,184 | Sub-agent commissioning |
| **State Manager** | `core/state_manager.py` | ~1,077 | Operational state (GREEN/YELLOW/RED) |
| Core init | `core/__init__.py` | ~115 | Core module bootstrap |
| Tools exports | `tools/__init__.py` | ~68 | **Exports all 20 agents** |
| Google Calendar | `integrations/google_calendar.py` | ~1,203 | Original calendar impl |
| YouTube adapter | `integrations/youtube.py` | ~1,382 | Original YouTube impl |
| Brief data | `data/alfred/briefs/` | ~5 files | Historical brief data |
| Agent Dockerfile | `agents/alfred/Dockerfile` | ~71 | Agent-specific container |
| Agent compose | `docker-compose.alfred.yml` | ~79 | Agent-specific orchestration |

### ALREADY RECOVERED (from previous session)
These were restored in commit 86598fb:

| Component | Location | Lines | Status |
|-----------|----------|-------|--------|
| Twitter scraper | `integrations/twitter_scraper.py` | 2,087 | ✅ Present |
| Instagram scraper | `integrations/instagram_scraper.py` | 1,831 | ✅ Present |
| Substack scraper | `integrations/substack_scraper.py` | 1,756 | ✅ Present |
| Manual input | `integrations/manual_input.py` | 1,784 | ✅ Present |

---

## Execution Plan

### Step 1: Recover Core Module
```bash
# Extract from original commit
git show 9eb9d8a:agent-zero1/agents/alfred/core/orchestrator.py > core/orchestrator.py
git show 9eb9d8a:agent-zero1/agents/alfred/core/state_manager.py > core/state_manager.py
git show 9eb9d8a:agent-zero1/agents/alfred/core/__init__.py > core/__init__.py
```

### Step 2: Fix Tools Exports
Replace current `tools/__init__.py` with original that exports all 20 agents properly.

### Step 3: Recover Original Integrations
```bash
# Keep as alternates (don't overwrite new adapters)
git show 9eb9d8a:agent-zero1/agents/alfred/integrations/google_calendar.py > integrations/google_calendar_original.py
git show 9eb9d8a:agent-zero1/agents/alfred/integrations/youtube.py > integrations/youtube_original.py
```

### Step 4: Recover Agent-Level Deployment
```bash
git show 9eb9d8a:agent-zero1/agents/alfred/Dockerfile > agents/alfred/Dockerfile
git show 9eb9d8a:agent-zero1/docker-compose.alfred.yml > docker-compose.alfred.yml
git show 9eb9d8a:agent-zero1/agents/alfred/.env.template > agents/alfred/.env.template
```

### Step 5: Recover Brief Data
```bash
mkdir -p data/alfred/briefs/weekly
git show 9eb9d8a:agent-zero1/data/alfred/briefs/... > data/...
```

### Step 6: Update Integration Exports
Merge the `integrations/__init__.py` to export both:
- New adapters (gmail, whatsapp, calendar_adapter, social_adapter)
- Original adapters (google_calendar, youtube)
- Scraping adapters (twitter_scraper, instagram_scraper, substack_scraper)

### Step 7: Validate & Commit
```bash
python3 -m py_compile **/*.py  # Syntax check
git add -A
git commit -m "feat: Recover complete ALFRED architecture (orchestrator, state_manager, all exports)"
git push origin main
```

---

## Final Architecture After Recovery

```
Alfred/
├── alfred_cli.py              # CLI tool (from branch) ✅
├── DEPLOYMENT.md              # Deployment guide (from branch) ✅
├── Dockerfile                 # Root container (from branch) ✅
├── docker-compose.yml         # Root orchestration (from branch) ✅
├── docker-compose.alfred.yml  # Agent orchestration (RECOVERED)
├── pyproject.toml             # Package config (from branch) ✅
├── requirements.txt           # Dependencies (from branch) ✅
├── .env.example               # Root env (from branch) ✅
├── .mcp.json                  # MCP config (from branch) ✅
├── evals/                     # Eval framework (from branch) ✅
├── tests/                     # Test suite (from branch) ✅
├── gateway/                   # Messaging gateway (from branch) ✅
└── agent-zero1/agents/alfred/
    ├── SOUL.md                # Personality v2 (from branch) ✅
    ├── GOVERNANCE.md          # Governance rules ✅
    ├── Dockerfile             # Agent container (RECOVERED)
    ├── .env.template          # Agent env (RECOVERED)
    ├── core/                  # RECOVERED
    │   ├── __init__.py
    │   ├── orchestrator.py    # Sub-agent commissioning
    │   ├── state_manager.py   # Operational state
    │   ├── daily_brief.py     # (also in tools/)
    │   └── weekly_brief.py    # (also in tools/)
    ├── tools/                 # All 20+ agents
    │   ├── __init__.py        # FIXED - exports all agents
    │   └── *.py               # All agent implementations ✅
    ├── memory/                # 6 memory systems ✅
    │   └── *.py
    └── integrations/          # MERGED
        ├── __init__.py        # Updated exports
        ├── calendar_adapter.py    # New (from branch)
        ├── gmail_adapter.py       # New (from branch)
        ├── whatsapp_adapter.py    # New (from branch)
        ├── social_adapter.py      # New (from branch)
        ├── google_calendar.py     # RECOVERED (original)
        ├── youtube.py             # RECOVERED (original)
        ├── twitter_scraper.py     # Already present ✅
        ├── instagram_scraper.py   # Already present ✅
        ├── substack_scraper.py    # Already present ✅
        └── manual_input.py        # Already present ✅
```

---

## Summary

| Category | Before Recovery | After Recovery |
|----------|-----------------|----------------|
| Core systems | ❌ Missing source | ✅ orchestrator + state_manager |
| Tool exports | 8 of 22 | ✅ All 22 exported |
| Integrations | 9 adapters | ✅ 11 adapters (merged) |
| Deployment | Root only | ✅ Root + Agent level |
| Features you wanted | ✅ Present | ✅ Preserved |

**Total recovered: ~4,000+ lines of critical infrastructure**

---

## Ready to Execute?

Approve this plan and I will execute all recovery steps.
