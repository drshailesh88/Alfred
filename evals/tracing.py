"""
Tracing Module for Alfred

LangSmith integration for tracing and debugging Alfred agents.
Provides @observe decorator wrapper and trace logging.

Usage:
    from evals.tracing import observe, TracingContext, setup_tracing

    # Enable tracing
    setup_tracing(api_key="your-api-key", project="alfred-evals")

    # Decorate agent methods
    @observe(name="reputation_check")
    async def check_reputation(self, request):
        ...

    # Use tracing context
    with TracingContext(run_name="morning_brief") as ctx:
        result = agent.generate_brief()
        ctx.log_output(result)
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import asyncio

# Configure logging
logger = logging.getLogger("alfred.tracing")

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# CONFIGURATION
# =============================================================================

class TracingBackend(Enum):
    """Available tracing backends."""
    LANGSMITH = "langsmith"
    OPENTELEMETRY = "opentelemetry"
    LOCAL = "local"
    NONE = "none"


@dataclass
class TracingConfig:
    """Configuration for tracing."""
    backend: TracingBackend = TracingBackend.LOCAL
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "alfred-evals"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    otel_endpoint: Optional[str] = None
    otel_service_name: str = "alfred"
    local_log_path: Optional[Path] = None
    enabled: bool = True
    log_inputs: bool = True
    log_outputs: bool = True
    log_metadata: bool = True
    sample_rate: float = 1.0  # 0.0 to 1.0


# Global configuration
_config: Optional[TracingConfig] = None
_langsmith_client = None
_otel_tracer = None


def setup_tracing(
    backend: str = "local",
    api_key: Optional[str] = None,
    project: str = "alfred-evals",
    endpoint: Optional[str] = None,
    enabled: bool = True,
    log_path: Optional[str] = None,
    **kwargs
) -> TracingConfig:
    """
    Set up tracing for Alfred.

    Args:
        backend: Tracing backend ("langsmith", "opentelemetry", "local", "none")
        api_key: API key for LangSmith
        project: Project name for LangSmith
        endpoint: Endpoint URL (for LangSmith or OpenTelemetry)
        enabled: Whether tracing is enabled
        log_path: Path for local log files
        **kwargs: Additional configuration options

    Returns:
        TracingConfig instance
    """
    global _config, _langsmith_client, _otel_tracer

    _config = TracingConfig(
        backend=TracingBackend(backend),
        langsmith_api_key=api_key or os.environ.get("LANGCHAIN_API_KEY"),
        langsmith_project=project,
        langsmith_endpoint=endpoint or "https://api.smith.langchain.com",
        enabled=enabled,
        local_log_path=Path(log_path) if log_path else None,
        **{k: v for k, v in kwargs.items() if hasattr(TracingConfig, k)}
    )

    # Initialize backend
    if _config.enabled:
        if _config.backend == TracingBackend.LANGSMITH:
            _init_langsmith()
        elif _config.backend == TracingBackend.OPENTELEMETRY:
            _init_opentelemetry()
        elif _config.backend == TracingBackend.LOCAL:
            _init_local_logging()

    return _config


def _init_langsmith():
    """Initialize LangSmith client."""
    global _langsmith_client

    if not _config or not _config.langsmith_api_key:
        logger.warning("LangSmith API key not provided. Tracing disabled.")
        return

    try:
        from langsmith import Client

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = _config.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = _config.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = _config.langsmith_endpoint

        _langsmith_client = Client()
        logger.info(f"LangSmith tracing enabled for project: {_config.langsmith_project}")

    except ImportError:
        logger.warning("LangSmith not installed. Install with: pip install langsmith")
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}")


def _init_opentelemetry():
    """Initialize OpenTelemetry tracer."""
    global _otel_tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": _config.otel_service_name})
        provider = TracerProvider(resource=resource)

        # Add console exporter for debugging
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer("alfred.tracing")

        logger.info("OpenTelemetry tracing enabled")

    except ImportError:
        logger.warning(
            "OpenTelemetry not installed. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk"
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")


def _init_local_logging():
    """Initialize local logging."""
    if _config and _config.local_log_path:
        _config.local_log_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local tracing enabled at: {_config.local_log_path}")
    else:
        logger.info("Local tracing enabled (in-memory)")


def get_config() -> Optional[TracingConfig]:
    """Get current tracing configuration."""
    return _config


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return _config is not None and _config.enabled


# =============================================================================
# TRACE DATA STRUCTURES
# =============================================================================

@dataclass
class TraceSpan:
    """A single trace span."""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    parent_span_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, success, error
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Calculate duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {},
        })

    def set_output(self, key: str, value: Any):
        """Set an output value."""
        self.outputs[key] = value

    def set_metadata(self, key: str, value: Any):
        """Set a metadata value."""
        self.metadata[key] = value

    def finish(self, status: str = "success", error: Optional[str] = None):
        """Finish the span."""
        self.end_time = datetime.now()
        self.status = status
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metadata": self.metadata,
            "events": self.events,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Convert span to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class TraceRun:
    """A complete trace run (collection of spans)."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    spans: List[TraceSpan] = field(default_factory=list)
    root_span: Optional[TraceSpan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"

    def add_span(self, span: TraceSpan):
        """Add a span to the run."""
        if not self.root_span:
            self.root_span = span
        self.spans.append(span)

    def finish(self, status: str = "success"):
        """Finish the run."""
        self.end_time = datetime.now()
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Convert run to dictionary."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }


# =============================================================================
# TRACING CONTEXT
# =============================================================================

class TracingContext:
    """Context manager for tracing a block of code."""

    _current_context: Optional['TracingContext'] = None

    def __init__(
        self,
        run_name: str = "",
        parent_context: Optional['TracingContext'] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.run_name = run_name
        self.parent_context = parent_context or TracingContext._current_context
        self.metadata = metadata or {}
        self.run: Optional[TraceRun] = None
        self.current_span: Optional[TraceSpan] = None
        self._previous_context: Optional['TracingContext'] = None

    def __enter__(self) -> 'TracingContext':
        """Enter the context."""
        self._previous_context = TracingContext._current_context
        TracingContext._current_context = self

        if not is_tracing_enabled():
            return self

        # Create run
        self.run = TraceRun(name=self.run_name, metadata=self.metadata)

        # Create root span
        self.current_span = TraceSpan(
            name=self.run_name,
            trace_id=self.run.run_id,
        )
        self.run.add_span(self.current_span)

        # Log start with LangSmith if available
        self._langsmith_start_run()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        TracingContext._current_context = self._previous_context

        if not is_tracing_enabled() or not self.run:
            return False

        # Finish span and run
        if exc_type:
            self.current_span.finish(status="error", error=str(exc_val))
            self.run.finish(status="error")
        else:
            self.current_span.finish(status="success")
            self.run.finish(status="success")

        # Log end with appropriate backend
        self._langsmith_end_run()
        self._log_to_local()

        return False  # Don't suppress exceptions

    def log_input(self, key: str, value: Any):
        """Log an input value."""
        if self.current_span and _config and _config.log_inputs:
            self.current_span.inputs[key] = _sanitize_value(value)

    def log_output(self, value: Any, key: str = "output"):
        """Log an output value."""
        if self.current_span and _config and _config.log_outputs:
            self.current_span.set_output(key, _sanitize_value(value))

    def log_metadata(self, key: str, value: Any):
        """Log metadata."""
        if self.current_span and _config and _config.log_metadata:
            self.current_span.set_metadata(key, _sanitize_value(value))

    def log_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Log an event."""
        if self.current_span:
            self.current_span.add_event(name, attributes)

    def create_child_span(self, name: str) -> TraceSpan:
        """Create a child span."""
        if not self.run:
            return TraceSpan(name=name)

        span = TraceSpan(
            name=name,
            trace_id=self.run.run_id,
            parent_span_id=self.current_span.span_id if self.current_span else None,
        )
        self.run.add_span(span)
        return span

    def _langsmith_start_run(self):
        """Start run in LangSmith."""
        global _langsmith_client

        if not _langsmith_client or _config.backend != TracingBackend.LANGSMITH:
            return

        try:
            # LangSmith automatically traces via environment variables
            pass
        except Exception as e:
            logger.warning(f"Failed to start LangSmith run: {e}")

    def _langsmith_end_run(self):
        """End run in LangSmith."""
        global _langsmith_client

        if not _langsmith_client or _config.backend != TracingBackend.LANGSMITH:
            return

        try:
            # LangSmith automatically traces via environment variables
            pass
        except Exception as e:
            logger.warning(f"Failed to end LangSmith run: {e}")

    def _log_to_local(self):
        """Log to local file."""
        if not _config or _config.backend != TracingBackend.LOCAL:
            return

        if not self.run:
            return

        try:
            if _config.local_log_path:
                log_file = _config.local_log_path / f"trace_{self.run.run_id}.json"
                with open(log_file, "w") as f:
                    json.dump(self.run.to_dict(), f, indent=2, default=str)
            else:
                logger.debug(f"Trace: {self.run.to_json()}")
        except Exception as e:
            logger.warning(f"Failed to write local trace: {e}")

    @classmethod
    def get_current(cls) -> Optional['TracingContext']:
        """Get the current tracing context."""
        return cls._current_context


# =============================================================================
# OBSERVE DECORATOR
# =============================================================================

def observe(
    name: Optional[str] = None,
    run_type: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
    capture_input: bool = True,
    capture_output: bool = True
) -> Callable[[F], F]:
    """
    Decorator to observe (trace) a function.

    Args:
        name: Name for the trace (defaults to function name)
        run_type: Type of run ("chain", "llm", "tool", etc.)
        metadata: Additional metadata to attach
        capture_input: Whether to capture function inputs
        capture_output: Whether to capture function outputs

    Returns:
        Decorated function

    Usage:
        @observe(name="reputation_check")
        async def check_reputation(self, request):
            ...
    """
    def decorator(func: F) -> F:
        trace_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return await func(*args, **kwargs)

            with TracingContext(run_name=trace_name, metadata=metadata or {}) as ctx:
                # Capture inputs
                if capture_input:
                    ctx.log_input("args", args[1:] if args else [])  # Skip self
                    ctx.log_input("kwargs", kwargs)

                ctx.log_metadata("run_type", run_type)

                try:
                    result = await func(*args, **kwargs)

                    # Capture output
                    if capture_output:
                        ctx.log_output(result)

                    return result
                except Exception as e:
                    ctx.log_event("error", {"error": str(e)})
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)

            with TracingContext(run_name=trace_name, metadata=metadata or {}) as ctx:
                # Capture inputs
                if capture_input:
                    ctx.log_input("args", args[1:] if args else [])
                    ctx.log_input("kwargs", kwargs)

                ctx.log_metadata("run_type", run_type)

                try:
                    result = func(*args, **kwargs)

                    # Capture output
                    if capture_output:
                        ctx.log_output(result)

                    return result
                except Exception as e:
                    ctx.log_event("error", {"error": str(e)})
                    raise

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def trace_agent_call(
    agent_name: str,
    method_name: str,
    inputs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracing an agent call.

    Usage:
        with trace_agent_call("reputation_sentinel", "check", {"scope": ["Twitter"]}):
            result = agent.check(request)
    """
    run_name = f"{agent_name}.{method_name}"
    ctx = TracingContext(run_name=run_name, metadata=metadata or {})
    ctx.log_input("inputs", inputs)
    return ctx


# =============================================================================
# LANGSMITH-SPECIFIC UTILITIES
# =============================================================================

def get_langsmith_client():
    """Get the LangSmith client if available."""
    global _langsmith_client
    return _langsmith_client


def langsmith_log_feedback(
    run_id: str,
    key: str,
    score: float,
    comment: Optional[str] = None
):
    """Log feedback to LangSmith."""
    global _langsmith_client

    if not _langsmith_client:
        logger.warning("LangSmith client not initialized")
        return

    try:
        _langsmith_client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
        )
    except Exception as e:
        logger.warning(f"Failed to log feedback to LangSmith: {e}")


def langsmith_create_dataset(
    name: str,
    description: str = "",
    examples: Optional[List[Dict[str, Any]]] = None
):
    """Create a dataset in LangSmith."""
    global _langsmith_client

    if not _langsmith_client:
        logger.warning("LangSmith client not initialized")
        return None

    try:
        dataset = _langsmith_client.create_dataset(
            dataset_name=name,
            description=description,
        )

        if examples:
            for example in examples:
                _langsmith_client.create_example(
                    dataset_id=dataset.id,
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                )

        return dataset
    except Exception as e:
        logger.warning(f"Failed to create LangSmith dataset: {e}")
        return None


# =============================================================================
# OPENTELEMETRY UTILITIES
# =============================================================================

def get_otel_tracer():
    """Get the OpenTelemetry tracer if available."""
    global _otel_tracer
    return _otel_tracer


@contextmanager
def otel_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Create an OpenTelemetry span."""
    global _otel_tracer

    if not _otel_tracer:
        yield None
        return

    with _otel_tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        yield span


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _sanitize_value(value: Any, max_length: int = 10000) -> Any:
    """Sanitize a value for logging."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > max_length:
            return value[:max_length] + "...[truncated]"
        return value

    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value[:100]]  # Limit list size

    # For other objects, try to get dict representation
    if hasattr(value, "to_dict"):
        return _sanitize_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return _sanitize_value(value.__dict__)

    return str(value)[:max_length]


def get_trace_summary() -> Dict[str, Any]:
    """Get a summary of recent traces."""
    if not _config or not _config.local_log_path:
        return {"error": "Local logging not configured"}

    try:
        trace_files = list(_config.local_log_path.glob("trace_*.json"))
        traces = []

        for f in sorted(trace_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            with open(f) as file:
                data = json.load(file)
                traces.append({
                    "run_id": data.get("run_id"),
                    "name": data.get("name"),
                    "status": data.get("status"),
                    "start_time": data.get("start_time"),
                })

        return {"traces": traces, "total_files": len(trace_files)}

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "TracingBackend",
    "TracingConfig",
    "setup_tracing",
    "get_config",
    "is_tracing_enabled",
    # Data structures
    "TraceSpan",
    "TraceRun",
    # Context
    "TracingContext",
    # Decorators
    "observe",
    "trace_agent_call",
    # LangSmith
    "get_langsmith_client",
    "langsmith_log_feedback",
    "langsmith_create_dataset",
    # OpenTelemetry
    "get_otel_tracer",
    "otel_span",
    # Utilities
    "get_trace_summary",
]
