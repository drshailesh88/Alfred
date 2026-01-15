# Alfred Memory Systems
# Persistent memory systems for pattern tracking, values, thresholds, and more

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class MemoryType(Enum):
    """Types of memory systems."""
    PATTERN = "pattern"           # Pattern Registry
    VALUES = "values"             # Values Hierarchy
    VIOLATION = "violation"       # Self-Violation Log
    REGRET = "regret"            # Regret Memory
    THRESHOLD = "threshold"       # Threshold Map
    OPTIONALITY = "optionality"   # Optionality Register


@dataclass
class MemoryEntry:
    """Base class for all memory entries."""
    id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "data": self.data
        }


class BaseMemorySystem:
    """Base class for all Alfred memory systems."""

    def __init__(self, memory_type: MemoryType, storage_path: Optional[Path] = None):
        self.memory_type = memory_type
        self.storage_path = storage_path or Path(f"~/.alfred/memory/{memory_type.value}.json").expanduser()
        self._entries: Dict[str, MemoryEntry] = {}
        self._load()

    def _load(self):
        """Load memory from persistent storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = MemoryEntry(**entry_data)
                        self._entries[entry.id] = entry
            except Exception as e:
                print(f"Warning: Could not load memory from {self.storage_path}: {e}")

    def _save(self):
        """Save memory to persistent storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            data = {
                "memory_type": self.memory_type.value,
                "last_updated": datetime.now().isoformat(),
                "entries": [e.to_dict() for e in self._entries.values()]
            }
            json.dump(data, f, indent=2, default=str)

    def add(self, entry_id: str, data: Dict[str, Any]) -> MemoryEntry:
        """Add a new entry to memory."""
        entry = MemoryEntry(id=entry_id, data=data)
        self._entries[entry_id] = entry
        self._save()
        return entry

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve an entry by ID."""
        return self._entries.get(entry_id)

    def update(self, entry_id: str, data: Dict[str, Any]) -> Optional[MemoryEntry]:
        """Update an existing entry."""
        if entry_id in self._entries:
            self._entries[entry_id].data.update(data)
            self._entries[entry_id].updated_at = datetime.now().isoformat()
            self._save()
            return self._entries[entry_id]
        return None

    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()
            return True
        return False

    def list_all(self) -> List[MemoryEntry]:
        """List all entries."""
        return list(self._entries.values())

    def query(self, filter_fn) -> List[MemoryEntry]:
        """Query entries with a filter function."""
        return [e for e in self._entries.values() if filter_fn(e)]

    def clear(self):
        """Clear all entries."""
        self._entries.clear()
        self._save()


# Import all memory system implementations
from .pattern_registry import PatternRegistry, PatternType, Trajectory
from .values_hierarchy import ValuesHierarchy, ValueSource, ValueStrength
from .self_violation_log import SelfViolationLog, ViolationSeverity, ViolationCategory
from .regret_memory import RegretMemory, DecisionDomain, OutcomeType, RegretIntensity
from .threshold_map import ThresholdMap, ThresholdType, TrendDirection, AlertLevel
from .optionality_register import OptionalityRegister, OptionStatus, OptionCategory, ClosureReason

__all__ = [
    # Base classes
    "MemoryType",
    "MemoryEntry",
    "BaseMemorySystem",
    # Pattern Registry
    "PatternRegistry",
    "PatternType",
    "Trajectory",
    # Values Hierarchy
    "ValuesHierarchy",
    "ValueSource",
    "ValueStrength",
    # Self-Violation Log
    "SelfViolationLog",
    "ViolationSeverity",
    "ViolationCategory",
    # Regret Memory
    "RegretMemory",
    "DecisionDomain",
    "OutcomeType",
    "RegretIntensity",
    # Threshold Map
    "ThresholdMap",
    "ThresholdType",
    "TrendDirection",
    "AlertLevel",
    # Optionality Register
    "OptionalityRegister",
    "OptionStatus",
    "OptionCategory",
    "ClosureReason",
]
