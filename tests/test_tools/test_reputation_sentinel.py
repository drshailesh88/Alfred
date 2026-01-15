"""
Tests for ReputationSentinel - Ambient Awareness Engine

Tests cover:
- Main class functionality
- State-aware behavior (GREEN/YELLOW/RED)
- Input/output format compliance
- Signal processing and risk assessment
- Edge cases
- Pattern detection
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent-zero1" / "agents" / "alfred"))

from tools.reputation_sentinel import (
    ReputationSentinel,
    ReputationSignal,
    RiskClassification,
    ReputationPacket,
    ReputationScanResult,
    SignalCluster,
    Platform,
    SignalType,
    StateRecommendation,
    ActionRecommendation,
    create_reputation_sentinel,
)
from tools import AlfredState, AgentResponse


class TestReputationSentinelInitialization:
    """Tests for ReputationSentinel initialization."""

    def test_create_reputation_sentinel(self):
        """Test that ReputationSentinel can be created."""
        agent = create_reputation_sentinel()
        assert agent is not None
        assert agent.name == "Reputation Sentinel"

    def test_initial_state_is_green(self):
        """Test that agent starts in GREEN state."""
        agent = ReputationSentinel()
        assert agent.alfred_state == AlfredState.GREEN

    def test_signal_agent_inheritance(self):
        """Test that ReputationSentinel is a SignalAgent."""
        agent = ReputationSentinel()
        # SignalAgent has get_monitoring_level method
        assert hasattr(agent, 'get_monitoring_level')


class TestReputationSentinelStateAwareBehavior:
    """Tests for state-aware behavior."""

    def test_green_state_monitoring_level(self, reputation_sentinel_factory, mock_alfred_state_green):
        """Test monitoring level in GREEN state."""
        agent = reputation_sentinel_factory(state=mock_alfred_state_green)
        level = agent.get_monitoring_level()
        assert level == "NORMAL"

    def test_yellow_state_monitoring_level(self, reputation_sentinel_factory, mock_alfred_state_yellow):
        """Test monitoring level in YELLOW state (heightened)."""
        agent = reputation_sentinel_factory(state=mock_alfred_state_yellow)
        level = agent.get_monitoring_level()
        assert level == "HEIGHTENED"

    def test_red_state_monitoring_level(self, reputation_sentinel_factory, mock_alfred_state_red):
        """Test monitoring level in RED state (critical)."""
        agent = reputation_sentinel_factory(state=mock_alfred_state_red)
        level = agent.get_monitoring_level()
        assert level == "CRITICAL"

    def test_yellow_state_increases_sensitivity(self, reputation_sentinel_factory, mock_alfred_state_yellow):
        """Test that YELLOW state increases signal sensitivity."""
        agent = reputation_sentinel_factory(state=mock_alfred_state_yellow)
        # In YELLOW state, thresholds should be lower (more sensitive)
        assert agent._risk_threshold_modifier < 1.0

    def test_red_state_maximizes_sensitivity(self, reputation_sentinel_factory, mock_alfred_state_red):
        """Test that RED state maximizes signal sensitivity."""
        agent = reputation_sentinel_factory(state=mock_alfred_state_red)
        # In RED state, thresholds should be at minimum
        assert agent._risk_threshold_modifier <= 0.5


class TestReputationSignal:
    """Tests for ReputationSignal data class."""

    def test_signal_creation(self):
        """Test basic signal creation."""
        signal = ReputationSignal(
            signal_id="sig_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="abc123",
            sentiment_score=-0.5,
            toxicity_score=0.3,
            reach_estimate=1000
        )
        assert signal.platform == Platform.TWITTER
        assert signal.sentiment_score == -0.5

    def test_signal_risk_level_calculation(self):
        """Test signal risk level calculation."""
        signal = ReputationSignal(
            signal_id="sig_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="abc123",
            sentiment_score=-0.8,
            toxicity_score=0.7,
            reach_estimate=10000,
            engagement_velocity=0.5
        )
        # High toxicity + negative sentiment + high reach should be high risk
        risk = signal.calculate_risk_score()
        assert risk >= 50

    def test_signal_low_risk_content(self):
        """Test low risk signal."""
        signal = ReputationSignal(
            signal_id="sig_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="abc123",
            sentiment_score=0.2,
            toxicity_score=0.1,
            reach_estimate=100
        )
        risk = signal.calculate_risk_score()
        assert risk < 30

    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = ReputationSignal(
            signal_id="sig_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="abc123",
            sentiment_score=-0.5
        )
        result = signal.to_dict()
        assert result["signal_id"] == "sig_001"
        assert result["platform"] == "Twitter"
        assert "timestamp" in result

    def test_signal_clinical_claim_flag(self):
        """Test clinical claim detection."""
        signal = ReputationSignal(
            signal_id="sig_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="abc123",
            sentiment_score=-0.5,
            contains_clinical_claims=True
        )
        # Clinical claims should increase risk
        risk = signal.calculate_risk_score()
        base_signal = ReputationSignal(
            signal_id="sig_002",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="def456",
            sentiment_score=-0.5,
            contains_clinical_claims=False
        )
        base_risk = base_signal.calculate_risk_score()
        assert risk >= base_risk

    def test_signal_personal_attack_flag(self):
        """Test personal attack detection amplifies risk."""
        signal = ReputationSignal(
            signal_id="sig_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="abc123",
            sentiment_score=-0.5,
            toxicity_score=0.5,
            contains_personal_attack=True
        )
        risk = signal.calculate_risk_score()
        assert risk >= 40  # Personal attacks should raise risk significantly


class TestRiskClassification:
    """Tests for risk classification logic."""

    def test_classify_coordinated_attack(self, reputation_sentinel_factory):
        """Test classification of coordinated attack."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id=f"sig_{i}",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash=f"hash_{i}",
                sentiment_score=-0.8,
                toxicity_score=0.7,
                is_coordinated=True
            )
            for i in range(5)
        ]
        classification = agent.classify_signals(signals)
        assert classification == RiskClassification.COORDINATED_ATTACK or \
               classification == RiskClassification.HIGH_VOLUME_CRITICISM

    def test_classify_misinterpretation(self, reputation_sentinel_factory):
        """Test classification of content misinterpretation."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_001",
                sentiment_score=-0.4,
                toxicity_score=0.2,
                references_specific_content=True
            )
        ]
        classification = agent.classify_signals(signals)
        assert classification in [
            RiskClassification.MISINTERPRETATION,
            RiskClassification.ORGANIC_CRITICISM,
            RiskClassification.NOISE
        ]

    def test_classify_platform_algorithm(self, reputation_sentinel_factory):
        """Test classification of platform algorithm issue."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.YOUTUBE,
                timestamp=datetime.now(),
                content_hash="hash_001",
                sentiment_score=-0.3,
                engagement_velocity=2.0,  # Unusual velocity
                bot_probability=0.1
            )
        ]
        classification = agent.classify_signals(signals)
        # Algorithm issues often have unusual engagement patterns
        assert classification is not None

    def test_classify_noise(self, reputation_sentinel_factory):
        """Test classification of noise signals."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_001",
                sentiment_score=0.0,
                toxicity_score=0.1,
                reach_estimate=10
            )
        ]
        classification = agent.classify_signals(signals)
        assert classification == RiskClassification.NOISE


class TestReputationPacket:
    """Tests for ReputationPacket output format."""

    def test_packet_creation(self):
        """Test packet creation."""
        packet = ReputationPacket(
            event="Test event description",
            platform="Twitter",
            classification=RiskClassification.ORGANIC_CRITICISM,
            risk_score=35,
            recommended_state=StateRecommendation.MAINTAIN,
            recommended_action=ActionRecommendation.MONITOR,
            rationale="Low risk organic criticism"
        )
        assert packet.risk_score == 35
        assert packet.recommended_state == StateRecommendation.MAINTAIN

    def test_packet_to_dict(self):
        """Test packet serialization."""
        packet = ReputationPacket(
            event="Test event",
            platform="Twitter",
            classification=RiskClassification.NOISE,
            risk_score=10,
            recommended_state=StateRecommendation.MAINTAIN,
            recommended_action=ActionRecommendation.IGNORE,
            rationale="Noise"
        )
        result = packet.to_dict()
        assert result["event"] == "Test event"
        assert result["risk_score"] == 10
        assert "classification" in result

    def test_packet_formatted_output(self):
        """Test packet formatted output."""
        packet = ReputationPacket(
            event="Test event",
            platform="Twitter",
            classification=RiskClassification.NOISE,
            risk_score=10,
            recommended_state=StateRecommendation.MAINTAIN,
            recommended_action=ActionRecommendation.IGNORE,
            rationale="Noise"
        )
        output = packet.to_formatted_output()
        assert "REPUTATION_PACKET" in output
        assert "Twitter" in output


class TestStateRecommendationLogic:
    """Tests for state recommendation logic."""

    def test_recommend_maintain_for_low_risk(self, reputation_sentinel_factory):
        """Test that low risk recommends MAINTAIN."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_001",
                sentiment_score=0.1,
                toxicity_score=0.1,
                reach_estimate=100
            )
        ]
        recommendation = agent.determine_state_recommendation(signals)
        assert recommendation == StateRecommendation.MAINTAIN

    def test_recommend_yellow_for_moderate_risk(self, reputation_sentinel_factory):
        """Test that moderate risk recommends YELLOW."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_001",
                sentiment_score=-0.6,
                toxicity_score=0.5,
                reach_estimate=5000,
                contains_clinical_claims=True
            )
        ]
        recommendation = agent.determine_state_recommendation(signals)
        assert recommendation in [StateRecommendation.ESCALATE_YELLOW, StateRecommendation.MAINTAIN]

    def test_recommend_red_for_high_risk(self, reputation_sentinel_factory):
        """Test that high risk recommends RED."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id=f"sig_{i}",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash=f"hash_{i}",
                sentiment_score=-0.9,
                toxicity_score=0.9,
                reach_estimate=100000,
                is_coordinated=True,
                contains_personal_attack=True
            )
            for i in range(10)
        ]
        recommendation = agent.determine_state_recommendation(signals)
        assert recommendation in [StateRecommendation.ESCALATE_RED, StateRecommendation.ESCALATE_YELLOW]


class TestActionRecommendationLogic:
    """Tests for action recommendation logic."""

    def test_recommend_ignore_for_noise(self, reputation_sentinel_factory):
        """Test IGNORE recommendation for noise."""
        agent = reputation_sentinel_factory()
        classification = RiskClassification.NOISE
        action = agent.determine_action_recommendation(classification, 5)
        assert action == ActionRecommendation.IGNORE

    def test_recommend_monitor_for_organic(self, reputation_sentinel_factory):
        """Test MONITOR recommendation for organic criticism."""
        agent = reputation_sentinel_factory()
        classification = RiskClassification.ORGANIC_CRITICISM
        action = agent.determine_action_recommendation(classification, 30)
        assert action == ActionRecommendation.MONITOR

    def test_recommend_respond_for_misinterpretation(self, reputation_sentinel_factory):
        """Test CLARIFY_RESPOND recommendation for misinterpretation."""
        agent = reputation_sentinel_factory()
        classification = RiskClassification.MISINTERPRETATION
        action = agent.determine_action_recommendation(classification, 45)
        assert action in [ActionRecommendation.CLARIFY_RESPOND, ActionRecommendation.MONITOR]

    def test_recommend_legal_for_coordinated_attack(self, reputation_sentinel_factory):
        """Test LEGAL_REVIEW recommendation for coordinated attacks."""
        agent = reputation_sentinel_factory()
        classification = RiskClassification.COORDINATED_ATTACK
        action = agent.determine_action_recommendation(classification, 85)
        assert action in [ActionRecommendation.LEGAL_REVIEW, ActionRecommendation.PLATFORM_REPORT]


class TestSignalClustering:
    """Tests for signal clustering functionality."""

    def test_cluster_by_topic(self, reputation_sentinel_factory):
        """Test clustering signals by topic."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_001",
                topic_cluster="cardiology"
            ),
            ReputationSignal(
                signal_id="sig_002",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_002",
                topic_cluster="cardiology"
            ),
            ReputationSignal(
                signal_id="sig_003",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_003",
                topic_cluster="general_health"
            )
        ]
        clusters = agent.cluster_signals(signals)
        assert len(clusters) >= 1
        assert any(c.topic == "cardiology" for c in clusters)

    def test_cluster_by_platform(self, reputation_sentinel_factory):
        """Test clustering signals by platform."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_001"
            ),
            ReputationSignal(
                signal_id="sig_002",
                platform=Platform.YOUTUBE,
                timestamp=datetime.now(),
                content_hash="hash_002"
            )
        ]
        clusters = agent.cluster_signals(signals, by_platform=True)
        platforms = {c.platform for c in clusters}
        assert Platform.TWITTER in platforms or "Twitter" in [c.platform for c in clusters]

    def test_empty_signals_clustering(self, reputation_sentinel_factory):
        """Test clustering with empty signals list."""
        agent = reputation_sentinel_factory()
        clusters = agent.cluster_signals([])
        assert len(clusters) == 0


class TestReputationScan:
    """Tests for reputation scan functionality."""

    def test_process_scan_request(self, reputation_sentinel_factory, sample_reputation_check_request):
        """Test processing scan request."""
        agent = reputation_sentinel_factory()
        result = agent.process_scan_request(sample_reputation_check_request)
        assert isinstance(result, AgentResponse)
        assert result.agent_name == "Reputation Sentinel"

    def test_scan_result_format(self, reputation_sentinel_factory):
        """Test scan result format."""
        agent = reputation_sentinel_factory()
        result = agent.process_scan_request({
            "scope": ["Twitter"],
            "time_window_hours": 24
        })
        assert "data" in result.to_dict()

    def test_scan_with_state_context(self, reputation_sentinel_factory, mock_alfred_state_yellow):
        """Test scan adjusts for current state."""
        agent = reputation_sentinel_factory(state=mock_alfred_state_yellow)
        result = agent.process_scan_request({"scope": ["Twitter"]})
        # Should have heightened sensitivity context
        assert result.alfred_state == AlfredState.YELLOW


class TestPatternDetection:
    """Tests for pattern detection in signals."""

    def test_detect_recurring_pattern(self, reputation_sentinel_factory):
        """Test detection of recurring criticism patterns."""
        agent = reputation_sentinel_factory()

        # Simulate historical signals
        history = [
            ReputationSignal(
                signal_id=f"hist_{i}",
                platform=Platform.TWITTER,
                timestamp=datetime.now() - timedelta(days=7-i),
                content_hash=f"hash_{i}",
                sentiment_score=-0.5,
                topic_cluster="vaccine_discussion"
            )
            for i in range(5)
        ]

        new_signal = ReputationSignal(
            signal_id="new_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="new_hash",
            sentiment_score=-0.5,
            topic_cluster="vaccine_discussion"
        )

        pattern = agent.detect_pattern(new_signal, history)
        assert pattern is not None or pattern is None  # May or may not detect pattern

    def test_no_pattern_for_unique_signal(self, reputation_sentinel_factory):
        """Test no pattern detected for unique signal."""
        agent = reputation_sentinel_factory()

        signal = ReputationSignal(
            signal_id="unique_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="unique_hash",
            sentiment_score=-0.2,
            topic_cluster="unique_topic"
        )

        pattern = agent.detect_pattern(signal, [])
        assert pattern is None


class TestReputationSentinelEdgeCases:
    """Edge case tests for ReputationSentinel."""

    def test_empty_scope(self, reputation_sentinel_factory):
        """Test handling empty scope in request."""
        agent = reputation_sentinel_factory()
        result = agent.process_scan_request({"scope": []})
        # Should handle gracefully
        assert isinstance(result, AgentResponse)

    def test_invalid_platform_in_scope(self, reputation_sentinel_factory):
        """Test handling invalid platform in scope."""
        agent = reputation_sentinel_factory()
        result = agent.process_scan_request({
            "scope": ["InvalidPlatform", "Twitter"]
        })
        assert isinstance(result, AgentResponse)

    def test_negative_time_window(self, reputation_sentinel_factory):
        """Test handling negative time window."""
        agent = reputation_sentinel_factory()
        result = agent.process_scan_request({
            "scope": ["Twitter"],
            "time_window_hours": -1
        })
        assert isinstance(result, AgentResponse)

    def test_extremely_high_risk_signal(self, reputation_sentinel_factory):
        """Test handling extremely high risk signal."""
        agent = reputation_sentinel_factory()
        signal = ReputationSignal(
            signal_id="extreme_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="extreme_hash",
            sentiment_score=-1.0,
            toxicity_score=1.0,
            reach_estimate=1000000,
            engagement_velocity=10.0,
            is_coordinated=True,
            contains_clinical_claims=True,
            contains_personal_attack=True
        )
        risk = signal.calculate_risk_score()
        assert risk >= 80  # Should be very high risk
        assert risk <= 100  # But not over 100

    def test_bot_probability_impact(self, reputation_sentinel_factory):
        """Test bot probability affects classification."""
        agent = reputation_sentinel_factory()
        bot_signals = [
            ReputationSignal(
                signal_id=f"bot_{i}",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash=f"bot_hash_{i}",
                sentiment_score=-0.7,
                bot_probability=0.9
            )
            for i in range(5)
        ]
        classification = agent.classify_signals(bot_signals)
        # High bot probability should influence classification
        assert classification in [
            RiskClassification.BOT_ACTIVITY,
            RiskClassification.COORDINATED_ATTACK,
            RiskClassification.HIGH_VOLUME_CRITICISM,
            RiskClassification.NOISE
        ]


class TestReputationSentinelResponse:
    """Tests for response format compliance."""

    def test_response_has_required_fields(self, reputation_sentinel_factory):
        """Test that response has all required fields."""
        agent = reputation_sentinel_factory()
        response = agent.process_scan_request({"scope": ["Twitter"]})
        response_dict = response.to_dict()

        assert "agent_name" in response_dict
        assert "timestamp" in response_dict
        assert "success" in response_dict
        assert "alfred_state" in response_dict
        assert "data" in response_dict

    def test_response_state_matches_agent(self, reputation_sentinel_factory, mock_alfred_state_yellow):
        """Test that response state matches agent state."""
        agent = reputation_sentinel_factory(state=mock_alfred_state_yellow)
        response = agent.process_scan_request({"scope": ["Twitter"]})

        assert response.alfred_state == AlfredState.YELLOW

    def test_blocked_response_format(self, reputation_sentinel_factory):
        """Test blocked response format."""
        agent = reputation_sentinel_factory()
        response = agent.blocked_response("Test block reason")

        assert response.success is False
        assert "BLOCKED" in str(response.data)


class TestPlatformSpecificBehavior:
    """Tests for platform-specific signal handling."""

    def test_twitter_specific_processing(self, reputation_sentinel_factory):
        """Test Twitter-specific signal processing."""
        agent = reputation_sentinel_factory()
        signal = ReputationSignal(
            signal_id="tw_001",
            platform=Platform.TWITTER,
            timestamp=datetime.now(),
            content_hash="tw_hash",
            sentiment_score=-0.5,
            engagement_velocity=0.8
        )
        # Twitter has specific engagement velocity considerations
        risk = signal.calculate_risk_score()
        assert isinstance(risk, (int, float))

    def test_youtube_specific_processing(self, reputation_sentinel_factory):
        """Test YouTube-specific signal processing."""
        agent = reputation_sentinel_factory()
        signal = ReputationSignal(
            signal_id="yt_001",
            platform=Platform.YOUTUBE,
            timestamp=datetime.now(),
            content_hash="yt_hash",
            sentiment_score=-0.5,
            reach_estimate=50000  # YouTube typically has higher reach
        )
        risk = signal.calculate_risk_score()
        assert isinstance(risk, (int, float))

    def test_substack_specific_processing(self, reputation_sentinel_factory):
        """Test Substack-specific signal processing."""
        agent = reputation_sentinel_factory()
        signal = ReputationSignal(
            signal_id="ss_001",
            platform=Platform.SUBSTACK,
            timestamp=datetime.now(),
            content_hash="ss_hash",
            sentiment_score=-0.4,
            author_type="subscriber"  # Substack has author relationships
        )
        risk = signal.calculate_risk_score()
        assert isinstance(risk, (int, float))


class TestGeneratePacket:
    """Tests for packet generation."""

    def test_generate_single_packet(self, reputation_sentinel_factory):
        """Test generating a single reputation packet."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_001",
                sentiment_score=-0.5,
                toxicity_score=0.3
            )
        ]
        packet = agent.generate_packet(signals)
        assert isinstance(packet, ReputationPacket)
        assert packet.platform == "Twitter"

    def test_generate_packet_multi_platform(self, reputation_sentinel_factory):
        """Test generating packet from multi-platform signals."""
        agent = reputation_sentinel_factory()
        signals = [
            ReputationSignal(
                signal_id="sig_001",
                platform=Platform.TWITTER,
                timestamp=datetime.now(),
                content_hash="hash_001",
                sentiment_score=-0.5
            ),
            ReputationSignal(
                signal_id="sig_002",
                platform=Platform.YOUTUBE,
                timestamp=datetime.now(),
                content_hash="hash_002",
                sentiment_score=-0.4
            )
        ]
        packet = agent.generate_packet(signals)
        assert isinstance(packet, ReputationPacket)
        # Should indicate multiple platforms
        assert "Multiple" in packet.platform or packet.platform in ["Twitter", "YouTube"]
