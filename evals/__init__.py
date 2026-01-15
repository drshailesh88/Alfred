"""
Alfred Evaluation Framework

Provides comprehensive evaluation capabilities for Alfred agents:
- DeepEval integration for LLM-based metrics
- LangSmith integration for tracing and debugging
- Behavioral testing for governance rules
- Task completion and tool use accuracy metrics

Usage:
    from evals import AlfredEvaluator, run_agent_evals

    evaluator = AlfredEvaluator()
    results = evaluator.evaluate_agent("reputation_sentinel")
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# =============================================================================
# EVALUATION RESULT TYPES
# =============================================================================

class EvalStatus(Enum):
    """Status of an evaluation run."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class MetricType(Enum):
    """Types of evaluation metrics."""
    TASK_COMPLETION = "task_completion"
    TOOL_USE_ACCURACY = "tool_use_accuracy"
    RESPONSE_RELEVANCY = "response_relevancy"
    STATE_TRANSITION = "state_transition"
    GOVERNANCE_COMPLIANCE = "governance_compliance"
    LATENCY = "latency"
    CUSTOM = "custom"


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    metric_name: str
    metric_type: MetricType
    score: float  # 0.0 to 1.0
    passed: bool
    threshold: float = 0.7
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class EvalResult:
    """Result of an evaluation run for an agent."""
    agent_name: str
    test_name: str
    status: EvalStatus
    metrics: List[MetricResult] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    expected_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    trace_id: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Check if all metrics passed."""
        return all(m.passed for m in self.metrics) and self.status == EvalStatus.PASSED

    @property
    def overall_score(self) -> float:
        """Calculate overall score from all metrics."""
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "test_name": self.test_name,
            "status": self.status.value,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "metrics": [m.to_dict() for m in self.metrics],
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "expected_output": self.expected_output,
            "error_message": self.error_message,
            "trace_id": self.trace_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class EvalSuiteResult:
    """Result of running an evaluation suite."""
    suite_name: str
    results: List[EvalResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    total_duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    def add_result(self, result: EvalResult):
        """Add a result and update counts."""
        self.results.append(result)
        self.total_tests += 1
        self.total_duration_ms += result.duration_ms

        if result.status == EvalStatus.PASSED and result.passed:
            self.passed_tests += 1
        elif result.status == EvalStatus.FAILED:
            self.failed_tests += 1
        elif result.status == EvalStatus.SKIPPED:
            self.skipped_tests += 1
        elif result.status == EvalStatus.ERROR:
            self.error_tests += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "error_tests": self.error_tests,
            "pass_rate": self.pass_rate,
            "total_duration_ms": self.total_duration_ms,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# BASE EVALUATION CLASSES
# =============================================================================

class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(self, name: str):
        self.name = name
        self._metrics: List[BaseMetric] = []

    def add_metric(self, metric: 'BaseMetric'):
        """Add a metric to the evaluator."""
        self._metrics.append(metric)

    @abstractmethod
    def evaluate(self, **kwargs) -> EvalResult:
        """Run evaluation and return result."""
        pass


class BaseMetric(ABC):
    """Base class for evaluation metrics."""

    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        threshold: float = 0.7
    ):
        self.name = name
        self.metric_type = metric_type
        self.threshold = threshold

    @abstractmethod
    def measure(
        self,
        input_data: Any,
        output_data: Any,
        expected_output: Optional[Any] = None
    ) -> MetricResult:
        """Measure the metric and return result."""
        pass

    def _create_result(
        self,
        score: float,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> MetricResult:
        """Helper to create a MetricResult."""
        return MetricResult(
            metric_name=self.name,
            metric_type=self.metric_type,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details=details or {},
            error=error,
        )


class BaseTestCase(ABC):
    """Base class for test cases."""

    def __init__(
        self,
        name: str,
        input_data: Dict[str, Any],
        expected_output: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        self.name = name
        self.input_data = input_data
        self.expected_output = expected_output
        self.tags = tags or []

    @abstractmethod
    def run(self, agent: Any) -> EvalResult:
        """Run the test case against an agent."""
        pass


# =============================================================================
# DEEPEVAL INTEGRATION SETUP
# =============================================================================

class DeepEvalConfig:
    """Configuration for DeepEval integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        enable_logging: bool = True
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.enable_logging = enable_logging
        self._initialized = False

    def initialize(self):
        """Initialize DeepEval with configuration."""
        if self._initialized:
            return

        try:
            # Set up DeepEval configuration
            import deepeval
            if self.api_key:
                os.environ["OPENAI_API_KEY"] = self.api_key
            self._initialized = True
        except ImportError:
            raise ImportError(
                "DeepEval not installed. Install with: pip install deepeval"
            )

    @property
    def is_initialized(self) -> bool:
        return self._initialized


class DeepEvalMetricWrapper(BaseMetric):
    """Wrapper to use DeepEval metrics in Alfred's evaluation framework."""

    def __init__(
        self,
        name: str,
        deepeval_metric: Any,
        metric_type: MetricType = MetricType.CUSTOM,
        threshold: float = 0.7
    ):
        super().__init__(name, metric_type, threshold)
        self._deepeval_metric = deepeval_metric

    def measure(
        self,
        input_data: Any,
        output_data: Any,
        expected_output: Optional[Any] = None
    ) -> MetricResult:
        """Measure using DeepEval metric."""
        try:
            from deepeval.test_case import LLMTestCase

            # Create test case
            test_case = LLMTestCase(
                input=str(input_data) if not isinstance(input_data, str) else input_data,
                actual_output=str(output_data) if not isinstance(output_data, str) else output_data,
                expected_output=str(expected_output) if expected_output else None,
            )

            # Measure
            self._deepeval_metric.measure(test_case)
            score = self._deepeval_metric.score if hasattr(self._deepeval_metric, 'score') else 0.0

            return self._create_result(
                score=score,
                details={
                    "reason": getattr(self._deepeval_metric, 'reason', None),
                    "deepeval_metric": self._deepeval_metric.__class__.__name__,
                }
            )
        except Exception as e:
            return self._create_result(score=0.0, error=str(e))


# =============================================================================
# LANGSMITH INTEGRATION SETUP
# =============================================================================

class LangSmithConfig:
    """Configuration for LangSmith integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: str = "alfred-evals",
        endpoint: str = "https://api.smith.langchain.com",
        enable_tracing: bool = True
    ):
        self.api_key = api_key or os.environ.get("LANGCHAIN_API_KEY")
        self.project_name = project_name
        self.endpoint = endpoint
        self.enable_tracing = enable_tracing
        self._initialized = False

    def initialize(self):
        """Initialize LangSmith with configuration."""
        if self._initialized:
            return

        try:
            if self.api_key and self.enable_tracing:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = self.api_key
                os.environ["LANGCHAIN_PROJECT"] = self.project_name
                os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LangSmith: {e}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# =============================================================================
# ALFRED EVALUATION FRAMEWORK
# =============================================================================

class AlfredEvaluator:
    """Main evaluator for Alfred agents."""

    def __init__(
        self,
        deepeval_config: Optional[DeepEvalConfig] = None,
        langsmith_config: Optional[LangSmithConfig] = None,
        enable_tracing: bool = False
    ):
        self.deepeval_config = deepeval_config or DeepEvalConfig()
        self.langsmith_config = langsmith_config or LangSmithConfig()
        self.enable_tracing = enable_tracing
        self._test_cases: Dict[str, List[BaseTestCase]] = {}

        # Initialize integrations
        if enable_tracing and self.langsmith_config.api_key:
            self.langsmith_config.initialize()

    def register_test_cases(self, agent_name: str, test_cases: List[BaseTestCase]):
        """Register test cases for an agent."""
        if agent_name not in self._test_cases:
            self._test_cases[agent_name] = []
        self._test_cases[agent_name].extend(test_cases)

    def evaluate_agent(
        self,
        agent_name: str,
        agent: Any,
        tags: Optional[List[str]] = None
    ) -> EvalSuiteResult:
        """Run all registered test cases for an agent."""
        suite_result = EvalSuiteResult(suite_name=f"{agent_name}_evaluation")

        test_cases = self._test_cases.get(agent_name, [])

        for test_case in test_cases:
            # Filter by tags if specified
            if tags and not any(t in test_case.tags for t in tags):
                continue

            try:
                result = test_case.run(agent)
                suite_result.add_result(result)
            except Exception as e:
                error_result = EvalResult(
                    agent_name=agent_name,
                    test_name=test_case.name,
                    status=EvalStatus.ERROR,
                    error_message=str(e),
                )
                suite_result.add_result(error_result)

        return suite_result

    def evaluate_all(
        self,
        agents: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> Dict[str, EvalSuiteResult]:
        """Run evaluations for all registered agents."""
        results = {}
        for agent_name, agent in agents.items():
            if agent_name in self._test_cases:
                results[agent_name] = self.evaluate_agent(agent_name, agent, tags)
        return results


# =============================================================================
# ALFRED-SPECIFIC METRICS
# =============================================================================

class TaskCompletionMetric(BaseMetric):
    """Metric for measuring task completion."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(
            name="task_completion",
            metric_type=MetricType.TASK_COMPLETION,
            threshold=threshold
        )

    def measure(
        self,
        input_data: Any,
        output_data: Any,
        expected_output: Optional[Any] = None
    ) -> MetricResult:
        """Measure task completion based on output structure and content."""
        if output_data is None:
            return self._create_result(score=0.0, details={"reason": "No output"})

        # Check for success indicators
        score = 0.0
        details = {}

        if isinstance(output_data, dict):
            # Check for success flag
            if output_data.get("success", False):
                score += 0.4

            # Check for data presence
            if output_data.get("data"):
                score += 0.3

            # Check for no errors
            if not output_data.get("errors"):
                score += 0.3

            details["has_success"] = output_data.get("success", False)
            details["has_data"] = bool(output_data.get("data"))
            details["has_errors"] = bool(output_data.get("errors"))

        return self._create_result(score=score, details=details)


class ToolUseAccuracyMetric(BaseMetric):
    """Metric for measuring tool use accuracy."""

    def __init__(self, threshold: float = 0.75):
        super().__init__(
            name="tool_use_accuracy",
            metric_type=MetricType.TOOL_USE_ACCURACY,
            threshold=threshold
        )

    def measure(
        self,
        input_data: Any,
        output_data: Any,
        expected_output: Optional[Any] = None
    ) -> MetricResult:
        """Measure tool use accuracy."""
        if not expected_output:
            return self._create_result(
                score=0.5,
                details={"reason": "No expected output to compare"}
            )

        score = 0.0
        details = {}

        if isinstance(output_data, dict) and isinstance(expected_output, dict):
            # Compare keys present
            expected_keys = set(expected_output.keys())
            actual_keys = set(output_data.keys())

            key_overlap = len(expected_keys & actual_keys) / len(expected_keys) if expected_keys else 0
            score += key_overlap * 0.5

            # Compare values for common keys
            value_matches = 0
            for key in expected_keys & actual_keys:
                if output_data.get(key) == expected_output.get(key):
                    value_matches += 1

            value_score = value_matches / len(expected_keys) if expected_keys else 0
            score += value_score * 0.5

            details["key_overlap"] = key_overlap
            details["value_matches"] = value_matches

        return self._create_result(score=score, details=details)


class ResponseRelevancyMetric(BaseMetric):
    """Metric for measuring response relevancy."""

    def __init__(self, threshold: float = 0.7):
        super().__init__(
            name="response_relevancy",
            metric_type=MetricType.RESPONSE_RELEVANCY,
            threshold=threshold
        )

    def measure(
        self,
        input_data: Any,
        output_data: Any,
        expected_output: Optional[Any] = None
    ) -> MetricResult:
        """Measure response relevancy."""
        if output_data is None:
            return self._create_result(score=0.0, details={"reason": "No output"})

        score = 0.0
        details = {}

        # Basic relevancy checks
        if isinstance(output_data, dict):
            # Check if agent name is relevant
            agent_name = output_data.get("agent_name", "")
            if agent_name:
                score += 0.3
                details["has_agent_name"] = True

            # Check if output has relevant data structure
            if "data" in output_data:
                score += 0.4
                details["has_data"] = True

            # Check for timestamp (indicates proper response)
            if "timestamp" in output_data:
                score += 0.3
                details["has_timestamp"] = True

        return self._create_result(score=score, details=details)


class StateTransitionMetric(BaseMetric):
    """Metric for measuring correct state transitions."""

    VALID_STATES = {"GREEN", "YELLOW", "RED"}
    VALID_TRANSITIONS = {
        "GREEN": {"GREEN", "YELLOW", "RED"},
        "YELLOW": {"GREEN", "YELLOW", "RED"},
        "RED": {"GREEN", "YELLOW", "RED"},
    }

    def __init__(self, threshold: float = 1.0):
        super().__init__(
            name="state_transition",
            metric_type=MetricType.STATE_TRANSITION,
            threshold=threshold
        )

    def measure(
        self,
        input_data: Any,
        output_data: Any,
        expected_output: Optional[Any] = None
    ) -> MetricResult:
        """Measure state transition correctness."""
        details = {}

        # Extract states
        initial_state = None
        final_state = None
        recommended_state = None

        if isinstance(input_data, dict):
            initial_state = input_data.get("alfred_state") or input_data.get("initial_state")

        if isinstance(output_data, dict):
            final_state = output_data.get("alfred_state")
            recommended_state = (
                output_data.get("recommended_state") or
                output_data.get("data", {}).get("overall_recommended_state")
            )

        if isinstance(expected_output, dict):
            expected_state = expected_output.get("alfred_state") or expected_output.get("recommended_state")
        else:
            expected_state = None

        # Calculate score
        score = 0.0

        # Check if state is valid
        if final_state in self.VALID_STATES or recommended_state in self.VALID_STATES:
            score += 0.5
            details["valid_state"] = True

        # Check if transition is valid
        if initial_state and (final_state or recommended_state):
            actual_end_state = final_state or recommended_state
            if actual_end_state in self.VALID_TRANSITIONS.get(initial_state, set()):
                score += 0.3
                details["valid_transition"] = True

        # Check if matches expected
        if expected_state:
            if (final_state == expected_state or recommended_state == expected_state):
                score += 0.2
                details["matches_expected"] = True
        else:
            score += 0.2  # No expected, partial credit

        details["initial_state"] = initial_state
        details["final_state"] = final_state
        details["recommended_state"] = recommended_state
        details["expected_state"] = expected_state

        return self._create_result(score=score, details=details)


class GovernanceComplianceMetric(BaseMetric):
    """Metric for measuring compliance with governance rules."""

    def __init__(
        self,
        rules: Optional[List[Callable[[Any, Any], bool]]] = None,
        threshold: float = 1.0
    ):
        super().__init__(
            name="governance_compliance",
            metric_type=MetricType.GOVERNANCE_COMPLIANCE,
            threshold=threshold
        )
        self.rules = rules or []

    def add_rule(self, rule: Callable[[Any, Any], bool], description: str = ""):
        """Add a governance rule to check."""
        self.rules.append((rule, description))

    def measure(
        self,
        input_data: Any,
        output_data: Any,
        expected_output: Optional[Any] = None
    ) -> MetricResult:
        """Measure governance compliance."""
        if not self.rules:
            return self._create_result(
                score=1.0,
                details={"reason": "No rules defined"}
            )

        passed_rules = 0
        rule_results = []

        for rule, description in self.rules:
            try:
                passed = rule(input_data, output_data)
                if passed:
                    passed_rules += 1
                rule_results.append({
                    "description": description,
                    "passed": passed,
                })
            except Exception as e:
                rule_results.append({
                    "description": description,
                    "passed": False,
                    "error": str(e),
                })

        score = passed_rules / len(self.rules)

        return self._create_result(
            score=score,
            details={
                "rules_checked": len(self.rules),
                "rules_passed": passed_rules,
                "rule_results": rule_results,
            }
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Status and Types
    "EvalStatus",
    "MetricType",
    # Results
    "MetricResult",
    "EvalResult",
    "EvalSuiteResult",
    # Base Classes
    "BaseEvaluator",
    "BaseMetric",
    "BaseTestCase",
    # Configuration
    "DeepEvalConfig",
    "LangSmithConfig",
    "DeepEvalMetricWrapper",
    # Main Framework
    "AlfredEvaluator",
    # Alfred Metrics
    "TaskCompletionMetric",
    "ToolUseAccuracyMetric",
    "ResponseRelevancyMetric",
    "StateTransitionMetric",
    "GovernanceComplianceMetric",
]
