"""
Agent Evaluations for Alfred

DeepEval test cases for each agent type with metrics for:
- Task completion
- Tool use accuracy
- Response relevancy

Usage:
    from evals.agent_evals import (
        ReputationSentinelEval,
        ShippingGovernorEval,
        run_agent_eval
    )

    eval_suite = ReputationSentinelEval()
    results = eval_suite.run_all()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import sys

# Add path for Alfred imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agent-zero1" / "agents" / "alfred"))

from evals import (
    BaseTestCase,
    BaseMetric,
    EvalResult,
    EvalStatus,
    EvalSuiteResult,
    MetricResult,
    MetricType,
    TaskCompletionMetric,
    ToolUseAccuracyMetric,
    ResponseRelevancyMetric,
    StateTransitionMetric,
    GovernanceComplianceMetric,
    DeepEvalMetricWrapper,
)


# =============================================================================
# DEEPEVAL INTEGRATION
# =============================================================================

def get_deepeval_task_completion_metric():
    """Get DeepEval's TaskCompletionMetric if available."""
    try:
        from deepeval.metrics import TaskCompletionMetric as DeepEvalTaskCompletion
        return DeepEvalTaskCompletion()
    except ImportError:
        return None


def get_deepeval_relevancy_metric():
    """Get DeepEval's AnswerRelevancyMetric if available."""
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        return AnswerRelevancyMetric()
    except ImportError:
        return None


def create_llm_test_case(
    input_text: str,
    actual_output: str,
    expected_output: Optional[str] = None,
    context: Optional[List[str]] = None
):
    """Create a DeepEval LLMTestCase."""
    try:
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
        )
    except ImportError:
        return None


def run_deepeval_evaluation(
    test_cases: List[Any],
    metrics: List[Any]
) -> Dict[str, Any]:
    """Run DeepEval evaluation on test cases."""
    try:
        from deepeval import evaluate
        results = evaluate(test_cases, metrics)
        return {"success": True, "results": results}
    except ImportError:
        return {"success": False, "error": "DeepEval not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# TEST CASE IMPLEMENTATIONS
# =============================================================================

@dataclass
class AgentTestCase(BaseTestCase):
    """Test case for Alfred agents."""

    agent_class: Optional[Type] = None
    metrics: List[BaseMetric] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default metrics if none provided."""
        if not self.metrics:
            self.metrics = [
                TaskCompletionMetric(),
                ResponseRelevancyMetric(),
            ]

    def run(self, agent: Any) -> EvalResult:
        """Run the test case against an agent."""
        start_time = time.time()

        try:
            # Execute agent
            if hasattr(agent, 'execute'):
                output = agent.execute(self.input_data)
            elif hasattr(agent, 'process_request'):
                # Handle async agents
                output = asyncio.get_event_loop().run_until_complete(
                    agent.process_request(self.input_data)
                )
            else:
                raise ValueError(f"Agent {agent} has no execute or process_request method")

            # Convert output to dict if needed
            if hasattr(output, 'to_dict'):
                output_data = output.to_dict()
            elif hasattr(output, '__dict__'):
                output_data = output.__dict__
            else:
                output_data = output

            # Run metrics
            metric_results = []
            for metric in self.metrics:
                result = metric.measure(
                    self.input_data,
                    output_data,
                    self.expected_output
                )
                metric_results.append(result)

            # Determine overall status
            all_passed = all(m.passed for m in metric_results)
            status = EvalStatus.PASSED if all_passed else EvalStatus.FAILED

            duration_ms = (time.time() - start_time) * 1000

            return EvalResult(
                agent_name=getattr(agent, 'name', agent.__class__.__name__),
                test_name=self.name,
                status=status,
                metrics=metric_results,
                duration_ms=duration_ms,
                input_data=self.input_data,
                output_data=output_data,
                expected_output=self.expected_output,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return EvalResult(
                agent_name=getattr(agent, 'name', 'unknown'),
                test_name=self.name,
                status=EvalStatus.ERROR,
                duration_ms=duration_ms,
                input_data=self.input_data,
                error_message=str(e),
            )


# =============================================================================
# REPUTATION SENTINEL EVALUATIONS
# =============================================================================

class ReputationSentinelEval:
    """Evaluation suite for ReputationSentinel agent."""

    AGENT_NAME = "reputation_sentinel"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for ReputationSentinel."""
        return [
            # Test: Basic reputation check
            AgentTestCase(
                name="basic_reputation_check",
                input_data={
                    "scope": ["Twitter", "YouTube", "Substack"],
                    "time_window_hours": 24,
                    "context": "",
                    "priority": "routine"
                },
                expected_output={
                    "status": "CLEAR",
                    "recommended_state": "GREEN",
                },
                tags=["basic", "routine"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                    ResponseRelevancyMetric(threshold=0.6),
                    StateTransitionMetric(threshold=0.8),
                ],
            ),
            # Test: Elevated priority check
            AgentTestCase(
                name="elevated_priority_check",
                input_data={
                    "scope": ["Twitter"],
                    "time_window_hours": 6,
                    "context": "Recent controversial post",
                    "priority": "elevated"
                },
                expected_output={
                    "recommended_state": "YELLOW",
                },
                tags=["elevated", "twitter"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                    StateTransitionMetric(threshold=0.8),
                ],
            ),
            # Test: Never recommends engagement
            AgentTestCase(
                name="no_engagement_recommendation",
                input_data={
                    "scope": ["Twitter"],
                    "time_window_hours": 24,
                    "context": "Negative mentions",
                    "priority": "routine"
                },
                expected_output={
                    "recommended_action": ["IGNORE", "MONITOR", "SILENCE", "LONGFORM_CLARIFICATION"],
                },
                tags=["governance", "critical"],
                metrics=[
                    GovernanceComplianceMetric(
                        rules=[
                            (
                                lambda i, o: _check_no_engagement(o),
                                "Never recommends direct engagement"
                            )
                        ],
                        threshold=1.0
                    ),
                ],
            ),
            # Test: Handles review site monitoring
            AgentTestCase(
                name="review_site_check",
                input_data={
                    "scope": ["Review Site"],
                    "time_window_hours": 24,
                    "context": "",
                    "priority": "routine"
                },
                expected_output={
                    "recommended_state": "GREEN",
                },
                tags=["review", "clinical"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
        ]

    @classmethod
    def create_deepeval_test_cases(cls):
        """Create DeepEval test cases for ReputationSentinel."""
        test_cases = []
        try:
            from deepeval.test_case import LLMTestCase

            test_cases.append(LLMTestCase(
                input="Check Twitter for reputation risks",
                actual_output="",  # Will be filled during test
                expected_output='{"status": "CLEAR", "recommended_state": "GREEN"}',
            ))

            test_cases.append(LLMTestCase(
                input="Monitor YouTube comments for negative sentiment",
                actual_output="",
                expected_output='{"status": "CLEAR", "platform": "YouTube"}',
            ))

        except ImportError:
            pass

        return test_cases


def _check_no_engagement(output: Any) -> bool:
    """Check that output doesn't recommend engagement."""
    if isinstance(output, dict):
        action = output.get("recommended_action") or output.get("data", {}).get("recommended_action")
        if action:
            forbidden = {"REPLY", "COMMENT", "ENGAGE", "RESPOND", "QUOTE_TWEET"}
            if isinstance(action, str):
                return action.upper() not in forbidden
            elif isinstance(action, list):
                return all(a.upper() not in forbidden for a in action if isinstance(a, str))
    return True


# =============================================================================
# SHIPPING GOVERNOR EVALUATIONS
# =============================================================================

class ShippingGovernorEval:
    """Evaluation suite for ShippingGovernor agent."""

    AGENT_NAME = "shipping_governor"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for ShippingGovernor."""
        return [
            # Test: Basic shipping health check
            AgentTestCase(
                name="basic_shipping_check",
                input_data={
                    "check_type": "health",
                    "include_projects": True,
                },
                expected_output={
                    "overall_health": "HEALTHY",
                },
                tags=["basic", "health"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                    ResponseRelevancyMetric(threshold=0.6),
                ],
            ),
            # Test: Project freeze recommendation
            AgentTestCase(
                name="freeze_check",
                input_data={
                    "check_type": "freeze_assessment",
                    "project_id": "test_project",
                },
                expected_output={
                    "freeze_recommended": False,
                },
                tags=["project", "freeze"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
            # Test: Output tracking
            AgentTestCase(
                name="output_tracking",
                input_data={
                    "check_type": "output_status",
                    "days": 30,
                },
                expected_output={
                    "recent_outputs_30d": 0,
                },
                tags=["tracking", "metrics"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
        ]


# =============================================================================
# FINANCIAL SENTINEL EVALUATIONS
# =============================================================================

class FinancialSentinelEval:
    """Evaluation suite for FinancialSentinel agent."""

    AGENT_NAME = "financial_sentinel"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for FinancialSentinel."""
        return [
            # Test: Monthly subscription check
            AgentTestCase(
                name="monthly_subscription_check",
                input_data={
                    "period": "month",
                },
                expected_output={
                    "budget_status": {"status": "on_track"},
                },
                tags=["basic", "monthly"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
            # Test: Quarterly review
            AgentTestCase(
                name="quarterly_review",
                input_data={
                    "period": "quarter",
                },
                expected_output={},
                tags=["quarterly", "review"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
        ]


# =============================================================================
# LEARNING CURATOR EVALUATIONS
# =============================================================================

class LearningCuratorEval:
    """Evaluation suite for LearningCurator agent."""

    AGENT_NAME = "learning_curator"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for LearningCurator."""
        return [
            # Test: Queue management
            AgentTestCase(
                name="queue_status",
                input_data={
                    "action": "get_queue",
                },
                expected_output={
                    "items": [],
                },
                tags=["queue", "basic"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
            # Test: Learning linked to output
            AgentTestCase(
                name="linked_output_check",
                input_data={
                    "action": "check_linked",
                },
                expected_output={},
                tags=["linked", "governance"],
                metrics=[
                    GovernanceComplianceMetric(
                        rules=[
                            (
                                lambda i, o: _check_learning_linked(o),
                                "Learning must be linked to output"
                            )
                        ],
                        threshold=1.0
                    ),
                ],
            ),
        ]


def _check_learning_linked(output: Any) -> bool:
    """Check that learning items are linked to output."""
    if isinstance(output, dict):
        items = output.get("items") or output.get("data", {}).get("items", [])
        for item in items:
            if isinstance(item, dict) and not item.get("linked_output"):
                return False
    return True


# =============================================================================
# SCHEDULING AGENT EVALUATIONS
# =============================================================================

class SchedulingAgentEval:
    """Evaluation suite for SchedulingAgent."""

    AGENT_NAME = "scheduling_agent"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for SchedulingAgent."""
        return [
            # Test: Today's schedule
            AgentTestCase(
                name="today_schedule",
                input_data={
                    "request_type": "today",
                },
                expected_output={},
                tags=["basic", "daily"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
            # Test: Buffer protection
            AgentTestCase(
                name="buffer_check",
                input_data={
                    "request_type": "buffer_status",
                },
                expected_output={
                    "buffer_status": {"adequate": True},
                },
                tags=["buffer", "protection"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
        ]


# =============================================================================
# CONTENT AGENT EVALUATIONS
# =============================================================================

class SubstackAgentEval:
    """Evaluation suite for SubstackAgent."""

    AGENT_NAME = "substack_agent"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for SubstackAgent."""
        return [
            # Test: Draft generation (blocked in non-GREEN state)
            AgentTestCase(
                name="draft_generation_green",
                input_data={
                    "action": "generate_draft",
                    "topic": "Test topic",
                    "alfred_state": "GREEN",
                },
                expected_output={
                    "status": "draft_created",
                },
                tags=["content", "draft"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                    StateTransitionMetric(threshold=1.0),
                ],
            ),
            # Test: Blocked in YELLOW state
            AgentTestCase(
                name="draft_blocked_yellow",
                input_data={
                    "action": "generate_draft",
                    "topic": "Test topic",
                    "alfred_state": "YELLOW",
                },
                expected_output={
                    "status": "BLOCKED",
                },
                tags=["content", "state", "blocked"],
                metrics=[
                    GovernanceComplianceMetric(
                        rules=[
                            (
                                lambda i, o: _check_content_blocked_in_yellow(i, o),
                                "Content generation blocked in YELLOW state"
                            )
                        ],
                        threshold=1.0
                    ),
                ],
            ),
        ]


def _check_content_blocked_in_yellow(input_data: Any, output: Any) -> bool:
    """Check that content is blocked in YELLOW state."""
    if isinstance(input_data, dict):
        state = input_data.get("alfred_state")
        if state == "YELLOW" or state == "RED":
            if isinstance(output, dict):
                return output.get("status") == "BLOCKED" or not output.get("success", True)
    return True


class TwitterThreadAgentEval:
    """Evaluation suite for TwitterThreadAgent."""

    AGENT_NAME = "twitter_thread_agent"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for TwitterThreadAgent."""
        return [
            AgentTestCase(
                name="thread_generation",
                input_data={
                    "action": "generate_thread",
                    "source_content": "Test content for thread",
                    "alfred_state": "GREEN",
                },
                expected_output={},
                tags=["content", "twitter"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
        ]


class YouTubeScriptAgentEval:
    """Evaluation suite for YouTubeScriptAgent."""

    AGENT_NAME = "youtube_script_agent"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for YouTubeScriptAgent."""
        return [
            AgentTestCase(
                name="script_generation",
                input_data={
                    "action": "generate_script",
                    "topic": "Test topic",
                    "alfred_state": "GREEN",
                },
                expected_output={},
                tags=["content", "youtube"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
        ]


# =============================================================================
# DAILY BRIEF EVALUATIONS
# =============================================================================

class DailyBriefEval:
    """Evaluation suite for DailyBriefAgent."""

    AGENT_NAME = "daily_brief"

    @staticmethod
    def get_test_cases() -> List[AgentTestCase]:
        """Get all test cases for DailyBriefAgent."""
        return [
            # Test: Morning brief
            AgentTestCase(
                name="morning_brief",
                input_data={
                    "brief_type": "morning",
                },
                expected_output={
                    "type": "MORNING_BRIEF",
                },
                tags=["brief", "morning"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                    ResponseRelevancyMetric(threshold=0.6),
                ],
            ),
            # Test: Evening shutdown
            AgentTestCase(
                name="evening_shutdown",
                input_data={
                    "brief_type": "evening",
                },
                expected_output={
                    "type": "EVENING_BRIEF",
                },
                tags=["brief", "evening"],
                metrics=[
                    TaskCompletionMetric(threshold=0.7),
                ],
            ),
        ]


# =============================================================================
# TEST FIXTURES
# =============================================================================

def get_sample_reputation_request() -> Dict[str, Any]:
    """Get sample reputation check request."""
    return {
        "scope": ["Twitter", "YouTube", "Substack"],
        "time_window_hours": 24,
        "context": "",
        "priority": "routine"
    }


def get_sample_shipping_request() -> Dict[str, Any]:
    """Get sample shipping check request."""
    return {
        "check_type": "health",
        "include_projects": True,
    }


def get_sample_financial_request() -> Dict[str, Any]:
    """Get sample financial check request."""
    return {
        "period": "month",
    }


def get_common_scenarios() -> List[Dict[str, Any]]:
    """Get common test scenarios across agents."""
    return [
        {
            "name": "green_state_normal_operation",
            "alfred_state": "GREEN",
            "expected_behavior": "normal_operation",
        },
        {
            "name": "yellow_state_restricted",
            "alfred_state": "YELLOW",
            "expected_behavior": "restricted_operation",
        },
        {
            "name": "red_state_emergency",
            "alfred_state": "RED",
            "expected_behavior": "emergency_mode",
        },
    ]


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

class AgentEvalRunner:
    """Runner for agent evaluations."""

    AGENT_EVALS = {
        "reputation_sentinel": ReputationSentinelEval,
        "shipping_governor": ShippingGovernorEval,
        "financial_sentinel": FinancialSentinelEval,
        "learning_curator": LearningCuratorEval,
        "scheduling_agent": SchedulingAgentEval,
        "substack_agent": SubstackAgentEval,
        "twitter_thread_agent": TwitterThreadAgentEval,
        "youtube_script_agent": YouTubeScriptAgentEval,
        "daily_brief": DailyBriefEval,
    }

    def __init__(self, enable_deepeval: bool = False):
        self.enable_deepeval = enable_deepeval
        self._results: Dict[str, EvalSuiteResult] = {}

    def get_available_agents(self) -> List[str]:
        """Get list of available agents for evaluation."""
        return list(self.AGENT_EVALS.keys())

    def run_eval(
        self,
        agent_name: str,
        agent: Any,
        tags: Optional[List[str]] = None
    ) -> EvalSuiteResult:
        """Run evaluation for a specific agent."""
        if agent_name not in self.AGENT_EVALS:
            raise ValueError(f"Unknown agent: {agent_name}")

        eval_class = self.AGENT_EVALS[agent_name]
        test_cases = eval_class.get_test_cases()

        suite_result = EvalSuiteResult(suite_name=f"{agent_name}_evaluation")

        for test_case in test_cases:
            # Filter by tags
            if tags and not any(t in test_case.tags for t in tags):
                continue

            result = test_case.run(agent)
            suite_result.add_result(result)

        self._results[agent_name] = suite_result
        return suite_result

    def run_all(
        self,
        agents: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> Dict[str, EvalSuiteResult]:
        """Run evaluations for all provided agents."""
        results = {}
        for agent_name, agent in agents.items():
            if agent_name in self.AGENT_EVALS:
                results[agent_name] = self.run_eval(agent_name, agent, tags)
        return results

    def get_results(self) -> Dict[str, EvalSuiteResult]:
        """Get all evaluation results."""
        return self._results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation results."""
        total_tests = 0
        total_passed = 0
        total_failed = 0

        agent_summaries = {}
        for agent_name, result in self._results.items():
            agent_summaries[agent_name] = {
                "total": result.total_tests,
                "passed": result.passed_tests,
                "failed": result.failed_tests,
                "pass_rate": result.pass_rate,
            }
            total_tests += result.total_tests
            total_passed += result.passed_tests
            total_failed += result.failed_tests

        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            "agent_summaries": agent_summaries,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_agent_eval(
    agent_name: str,
    agent: Any,
    tags: Optional[List[str]] = None,
    enable_deepeval: bool = False
) -> EvalSuiteResult:
    """Convenience function to run evaluation for a single agent."""
    runner = AgentEvalRunner(enable_deepeval=enable_deepeval)
    return runner.run_eval(agent_name, agent, tags)


def get_test_cases_for_agent(agent_name: str) -> List[AgentTestCase]:
    """Get test cases for a specific agent."""
    eval_class = AgentEvalRunner.AGENT_EVALS.get(agent_name)
    if eval_class:
        return eval_class.get_test_cases()
    return []


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Test Case
    "AgentTestCase",
    # Agent Evaluations
    "ReputationSentinelEval",
    "ShippingGovernorEval",
    "FinancialSentinelEval",
    "LearningCuratorEval",
    "SchedulingAgentEval",
    "SubstackAgentEval",
    "TwitterThreadAgentEval",
    "YouTubeScriptAgentEval",
    "DailyBriefEval",
    # Runner
    "AgentEvalRunner",
    # Convenience
    "run_agent_eval",
    "get_test_cases_for_agent",
    # Fixtures
    "get_sample_reputation_request",
    "get_sample_shipping_request",
    "get_sample_financial_request",
    "get_common_scenarios",
    # DeepEval
    "create_llm_test_case",
    "run_deepeval_evaluation",
]
