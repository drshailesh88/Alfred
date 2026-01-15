"""
Pytest fixtures for Alfred testing.

Provides:
- Mock Alfred state management
- Sample input data fixtures
- Temporary storage paths
- Mock MCP server responses
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List
import sys

# Add the agent-zero1/agents/alfred path to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "agent-zero1" / "agents" / "alfred"))


# =============================================================================
# ALFRED STATE FIXTURES
# =============================================================================

@pytest.fixture
def mock_alfred_state_green():
    """Mock Alfred in GREEN state (normal operations)."""
    from tools import AlfredState
    return AlfredState.GREEN


@pytest.fixture
def mock_alfred_state_yellow():
    """Mock Alfred in YELLOW state (elevated monitoring)."""
    from tools import AlfredState
    return AlfredState.YELLOW


@pytest.fixture
def mock_alfred_state_red():
    """Mock Alfred in RED state (active threat)."""
    from tools import AlfredState
    return AlfredState.RED


@pytest.fixture
def all_alfred_states():
    """All Alfred states for parametrized testing."""
    from tools import AlfredState
    return [AlfredState.GREEN, AlfredState.YELLOW, AlfredState.RED]


# =============================================================================
# TEMPORARY STORAGE FIXTURES
# =============================================================================

@pytest.fixture
def temp_storage_path(tmp_path):
    """Provide a temporary storage path for memory systems."""
    storage_dir = tmp_path / "alfred_test_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def temp_memory_path(temp_storage_path):
    """Provide a temporary path for memory system JSON files."""
    memory_dir = temp_storage_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir


@pytest.fixture
def temp_scan_directory(temp_storage_path):
    """Provide a temporary directory for scan adapter tests."""
    scan_dir = temp_storage_path / "scans"
    scan_dir.mkdir(parents=True, exist_ok=True)
    return scan_dir


# =============================================================================
# SAMPLE DATA FIXTURES - INTAKE AGENT
# =============================================================================

@pytest.fixture
def sample_email_data():
    """Sample email data for testing intake agent."""
    return {
        "message_id": "msg_001",
        "from": "dr.smith@hospital.org",
        "subject": "Patient Consultation Request",
        "body": "I would like to discuss a complex cardiac case with you.",
        "date": datetime.now().isoformat(),
        "thread_id": "thread_001",
        "is_read": False,
        "is_starred": True,
        "attachments": [
            {
                "filename": "patient_ecg.pdf",
                "mime_type": "application/pdf",
                "size": 1024000,
                "attachment_id": "att_001"
            }
        ]
    }


@pytest.fixture
def sample_whatsapp_data():
    """Sample WhatsApp message data for testing intake agent."""
    return {
        "message_id": "wa_msg_001",
        "sender_name": "Dr. Johnson",
        "sender_number": "+1234567890",
        "text": "URGENT: Can you review this case today?",
        "timestamp": datetime.now().timestamp(),
        "chat_id": "chat_001",
        "group_name": None,
        "has_media": False,
        "is_read": False
    }


@pytest.fixture
def sample_intake_request():
    """Sample intake request for testing."""
    return {
        "channels": ["Email", "WhatsApp"],
        "time_window": "last_24_hours",
        "include": "all"
    }


@pytest.fixture
def sample_intake_batch():
    """Sample inbound batch result."""
    return {
        "batch_id": "20240115_120000_0001",
        "items_count": 2,
        "channels_checked": ["Email", "WhatsApp"],
        "status": "complete",
        "items": [],
        "batch_summary": {
            "by_channel": {"Email": 1, "WhatsApp": 1},
            "with_attachments": 1,
            "with_urgency_markers": 1,
            "unread_count": 2,
            "flagged_count": 1
        }
    }


# =============================================================================
# SAMPLE DATA FIXTURES - REPUTATION SENTINEL
# =============================================================================

@pytest.fixture
def sample_reputation_check_request():
    """Sample reputation check request."""
    return {
        "scope": ["Twitter", "YouTube", "Substack"],
        "time_window_hours": 24,
        "context": "Recent cardiology podcast appearance",
        "priority": "routine"
    }


@pytest.fixture
def sample_reputation_event():
    """Sample reputation event for testing."""
    return {
        "event_id": "event_001",
        "platform": "Twitter",
        "timestamp": datetime.now().isoformat(),
        "content_hash": "abc123def456",
        "sentiment_score": -0.6,
        "toxicity_score": 0.7,
        "bot_probability": 0.2,
        "reach_estimate": 5000,
        "engagement_velocity": 0.3,
        "topic_cluster": "medical_misinformation",
        "author_type": "public",
        "is_coordinated": False,
        "contains_clinical_claims": True,
        "contains_personal_attack": False,
        "references_specific_content": True
    }


@pytest.fixture
def sample_reputation_packet():
    """Sample reputation packet output."""
    return {
        "event": "Multiple signals on Twitter indicating content may be misinterpreted.",
        "platform": "Twitter",
        "classification": "misinterpretation",
        "risk_score": 45,
        "recommended_state": "YELLOW",
        "recommended_action": "MONITOR",
        "rationale": "Moderate risk (45); continue monitoring for escalation.",
        "pattern_note": None,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# SAMPLE DATA FIXTURES - FINANCIAL SENTINEL
# =============================================================================

@pytest.fixture
def sample_subscription_data():
    """Sample subscription data for financial sentinel testing."""
    return {
        "id": "sub_001",
        "name": "Claude Pro",
        "cost": 20.00,
        "billing_cycle": "monthly",
        "renewal_date": (datetime.now() + timedelta(days=10)).isoformat(),
        "auto_renew": True,
        "primary_function": "ai_assistant",
        "vendor": "Anthropic",
        "tags": ["productivity", "ai"]
    }


@pytest.fixture
def sample_tool_data():
    """Sample tool data for financial sentinel testing."""
    return {
        "id": "tool_001",
        "name": "Mechanical Keyboard",
        "cost": 150.00,
        "purchase_date": (datetime.now() - timedelta(days=90)).isoformat(),
        "expected_lifespan_months": 36,
        "primary_function": "typing",
        "linked_output": "content creation"
    }


@pytest.fixture
def sample_purchase_data():
    """Sample purchase data for impulse detection testing."""
    return {
        "id": "purchase_001",
        "item_name": "New Tool",
        "amount": 75.00,
        "category": "tools",
        "timestamp": datetime.now().isoformat(),
        "justification": None,
        "time_to_decision_minutes": 15,
        "triggered_by": "twitter_ad"
    }


@pytest.fixture
def sample_financial_check_request():
    """Sample financial check request."""
    return {
        "period": "month",
        "focus": "all",
        "include_pending": True,
        "roi_assessment": ["sub_001", "tool_001"]
    }


# =============================================================================
# SAMPLE DATA FIXTURES - SHIPPING GOVERNOR
# =============================================================================

@pytest.fixture
def sample_project_data():
    """Sample project data for shipping governor testing."""
    return {
        "name": "Alfred Integration",
        "start_date": (date.today() - timedelta(days=20)).isoformat(),
        "description": "Integrate all Alfred sub-agents",
        "blockers": ["waiting for API access"]
    }


@pytest.fixture
def sample_project_stalled():
    """Sample stalled project for testing."""
    return {
        "name": "Abandoned Feature",
        "start_date": (date.today() - timedelta(days=45)).isoformat(),
        "description": "Feature that never shipped",
        "blockers": ["almost done", "just need to polish"]
    }


@pytest.fixture
def sample_build_data():
    """Sample build item data for shipping governor testing."""
    return {
        "name": "Custom Dashboard",
        "start_date": (date.today() - timedelta(days=10)).isoformat(),
        "description": "Analytics dashboard",
        "linked_output": "weekly metrics report",
        "linked_project": "Alfred Integration"
    }


@pytest.fixture
def sample_build_no_output():
    """Sample build without linked output."""
    return {
        "name": "Fancy Tool",
        "start_date": (date.today() - timedelta(days=15)).isoformat(),
        "description": "A tool I want to build",
        "linked_output": None,
        "linked_project": None
    }


@pytest.fixture
def sample_shipping_check_request():
    """Sample shipping check request."""
    return {
        "active_projects": [
            {
                "name": "Project A",
                "start_date": (date.today() - timedelta(days=5)).isoformat(),
                "blockers": []
            }
        ],
        "recent_outputs": [],
        "pending_builds": [],
        "claimed_blockers": []
    }


# =============================================================================
# SAMPLE DATA FIXTURES - PATTERN REGISTRY
# =============================================================================

@pytest.fixture
def sample_pattern_data():
    """Sample pattern data for registry testing."""
    return {
        "pattern_type": "obsession_loop",
        "description": "Excessive focus on tool building over shipping",
        "initial_context": "Spent 3 days perfecting dashboard instead of shipping",
        "severity": 7
    }


@pytest.fixture
def sample_occurrence_data():
    """Sample occurrence data for pattern tracking."""
    return {
        "context": "Building another dashboard without linked output",
        "severity": 6,
        "notes": "Third occurrence this week",
        "trigger": "New tool discovery on HackerNews"
    }


@pytest.fixture
def sample_intervention_data():
    """Sample intervention data for pattern registry."""
    return {
        "intervention_id": "int_001",
        "intervention_description": "Implemented 'ship or kill' rule for all projects",
        "effectiveness": 8
    }


# =============================================================================
# SAMPLE DATA FIXTURES - THRESHOLD MAP
# =============================================================================

@pytest.fixture
def sample_threshold_data():
    """Sample threshold data for threshold map testing."""
    return {
        "threshold_name": "Sleep Deprivation",
        "threshold_type": "health",
        "description": "Risk of chronic sleep deprivation affecting clinical judgment",
        "initial_proximity": 30,
        "crossing_consequences": "Impaired clinical decision-making, burnout risk"
    }


@pytest.fixture
def sample_financial_threshold():
    """Sample financial threshold."""
    return {
        "threshold_name": "Subscription Creep",
        "threshold_type": "financial",
        "description": "Monthly subscriptions exceeding reasonable limits",
        "initial_proximity": 50,
        "crossing_consequences": "Budget strain, tool sprawl, decision fatigue"
    }


@pytest.fixture
def sample_proximity_update():
    """Sample proximity update data."""
    return {
        "new_proximity": 65,
        "notes": "Added two new subscriptions this week",
        "evidence": "Monthly recurring increased by $40"
    }


# =============================================================================
# MOCK MCP SERVER FIXTURES
# =============================================================================

@pytest.fixture
def mock_mcp_gmail():
    """Mock mcp-gsuite Gmail responses."""
    return AsyncMock(
        list_messages=AsyncMock(return_value=[
            {
                "message_id": "msg_001",
                "from": "contact@example.com",
                "subject": "Test Email",
                "body": "Test body content",
                "date": datetime.now().isoformat(),
                "is_read": False,
                "attachments": []
            }
        ]),
        get_message=AsyncMock(return_value={"body": "Full email content"}),
        connect=AsyncMock(return_value=True),
        disconnect=AsyncMock(return_value=None)
    )


@pytest.fixture
def mock_mcp_whatsapp():
    """Mock whatsapp-mcp responses."""
    return AsyncMock(
        get_messages=AsyncMock(return_value=[
            {
                "message_id": "wa_001",
                "sender_name": "Test Contact",
                "text": "Test message",
                "timestamp": datetime.now().timestamp(),
                "is_read": False
            }
        ]),
        get_message=AsyncMock(return_value={"text": "Full message"}),
        connect=AsyncMock(return_value=True),
        disconnect=AsyncMock(return_value=None)
    )


@pytest.fixture
def mock_mcp_calendar():
    """Mock google-calendar-mcp responses."""
    return AsyncMock(
        get_notifications=AsyncMock(return_value=[]),
        get_pending_invites=AsyncMock(return_value=[]),
        get_event=AsyncMock(return_value={"summary": "Test Event"}),
        connect=AsyncMock(return_value=True),
        disconnect=AsyncMock(return_value=None)
    )


@pytest.fixture
def mock_firefly_api():
    """Mock Firefly III API responses."""
    return AsyncMock(
        get_transactions=AsyncMock(return_value=[]),
        get_accounts=AsyncMock(return_value=[]),
        create_transaction=AsyncMock(return_value=True),
        get_categories=AsyncMock(return_value=["subscriptions", "tools"]),
        connect=AsyncMock(return_value=True)
    )


@pytest.fixture
def mock_wallos_api():
    """Mock Wallos API responses."""
    return AsyncMock(
        get_subscriptions=AsyncMock(return_value=[]),
        add_subscription=AsyncMock(return_value=True),
        update_subscription=AsyncMock(return_value=True),
        get_upcoming_renewals=AsyncMock(return_value=[]),
        connect=AsyncMock(return_value=True)
    )


# =============================================================================
# MOCK NLP TOOL FIXTURES
# =============================================================================

@pytest.fixture
def mock_vader_sentiment():
    """Mock VADER sentiment analyzer."""
    mock = MagicMock()
    mock.polarity_scores.return_value = {
        "neg": 0.3,
        "neu": 0.5,
        "pos": 0.2,
        "compound": -0.4
    }
    return mock


@pytest.fixture
def mock_detoxify():
    """Mock Detoxify toxicity detection."""
    mock = MagicMock()
    mock.predict.return_value = {
        "toxicity": 0.5,
        "severe_toxicity": 0.1,
        "obscene": 0.2,
        "threat": 0.3,
        "insult": 0.4,
        "identity_attack": 0.1
    }
    return mock


@pytest.fixture
def mock_bertopic():
    """Mock BERTopic clustering."""
    mock = MagicMock()
    mock.fit_transform.return_value = ([0, 0, 1, 1, -1], None)
    mock.get_topic_label.return_value = "medical_discussion"
    return mock


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def current_datetime():
    """Current datetime for consistent testing."""
    return datetime.now()


@pytest.fixture
def past_datetime():
    """Past datetime (7 days ago) for testing."""
    return datetime.now() - timedelta(days=7)


@pytest.fixture
def future_datetime():
    """Future datetime (7 days ahead) for testing."""
    return datetime.now() + timedelta(days=7)


@pytest.fixture
def sample_json_file(temp_storage_path):
    """Create a sample JSON file for testing."""
    def _create_json(filename: str, data: dict) -> Path:
        filepath = temp_storage_path / filename
        with open(filepath, 'w') as f:
            json.dump(data, f)
        return filepath
    return _create_json


# =============================================================================
# AGENT FACTORY FIXTURES
# =============================================================================

@pytest.fixture
def intake_agent_factory(temp_storage_path):
    """Factory for creating IntakeAgent instances."""
    def _create(state=None):
        from tools.intake_agent import IntakeAgent
        agent = IntakeAgent()
        if state:
            agent.alfred_state = state
        return agent
    return _create


@pytest.fixture
def reputation_sentinel_factory():
    """Factory for creating ReputationSentinel instances."""
    def _create(state=None):
        from tools.reputation_sentinel import ReputationSentinel
        agent = ReputationSentinel()
        if state:
            agent.alfred_state = state
        return agent
    return _create


@pytest.fixture
def financial_sentinel_factory():
    """Factory for creating FinancialSentinel instances."""
    def _create(state=None):
        from tools.financial_sentinel import FinancialSentinel
        agent = FinancialSentinel()
        if state:
            agent.alfred_state = state
        return agent
    return _create


@pytest.fixture
def shipping_governor_factory():
    """Factory for creating ShippingGovernor instances."""
    def _create(state=None):
        from tools.shipping_governor import ShippingGovernor
        agent = ShippingGovernor()
        if state:
            agent.alfred_state = state
        return agent
    return _create


@pytest.fixture
def pattern_registry_factory(temp_memory_path):
    """Factory for creating PatternRegistry instances."""
    def _create():
        from memory.pattern_registry import PatternRegistry
        storage_path = temp_memory_path / "pattern_test.json"
        return PatternRegistry(storage_path=storage_path)
    return _create


@pytest.fixture
def threshold_map_factory(temp_memory_path):
    """Factory for creating ThresholdMap instances."""
    def _create():
        from memory.threshold_map import ThresholdMap
        storage_path = temp_memory_path / "threshold_test.json"
        return ThresholdMap(storage_path=storage_path)
    return _create


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_singleton_state():
    """Clean up any singleton state between tests."""
    yield
    # Any cleanup logic here


@pytest.fixture
def reset_imports():
    """Reset module imports for clean state."""
    # Remove Alfred modules from sys.modules for clean reimport
    modules_to_remove = [key for key in sys.modules.keys() if 'alfred' in key.lower()]
    for mod in modules_to_remove:
        del sys.modules[mod]
    yield
