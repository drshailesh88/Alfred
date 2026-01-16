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
class RecallCard:
    """
    Compressed memory recall format - max 600 characters.

    Memory must never pollute reasoning. Recall cards enforce
    surgical, minimal memory access. One card per response maximum.
    """
    type: str        # PATTERN | VIOLATION | REGRET | THRESHOLD | VALUE | OPTION
    id: str          # PAT-001, VIO-023, etc.
    name: str        # Short descriptive name
    last: str        # Last occurrence date (YYYY-MM-DD)
    cost: str        # What this cost (brief)
    move: str        # Recommended Alfred action

    MAX_LENGTH = 600

    def __str__(self) -> str:
        """Format as recall card string."""
        card = f"TYPE: {self.type}\nID: {self.id}\nNAME: \"{self.name}\"\nLAST: {self.last}\nCOST: \"{self.cost}\"\nMOVE: \"{self.move}\""
        if len(card) > self.MAX_LENGTH:
            # Truncate cost and move if too long
            available = self.MAX_LENGTH - len(f"TYPE: {self.type}\nID: {self.id}\nNAME: \"{self.name}\"\nLAST: {self.last}\nCOST: \"\"\nMOVE: \"\"")
            half = available // 2
            card = f"TYPE: {self.type}\nID: {self.id}\nNAME: \"{self.name}\"\nLAST: {self.last}\nCOST: \"{self.cost[:half]}...\"\nMOVE: \"{self.move[:half]}...\""
        return card

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "last": self.last,
            "cost": self.cost,
            "move": self.move
        }


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

    def to_recall_card(self, memory_type: str) -> RecallCard:
        """
        Convert to compressed recall card format.

        Subclasses should override this for type-specific formatting.
        """
        return RecallCard(
            type=memory_type.upper(),
            id=self.id,
            name=self.data.get("name", self.data.get("description", "Unknown"))[:50],
            last=self.updated_at[:10],
            cost=self.data.get("cost", self.data.get("outcome", "Unknown"))[:100],
            move=self.data.get("move", self.data.get("intervention", "Assess"))[:100]
        )


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

    def get_recall_card(self, entry_id: str) -> Optional[RecallCard]:
        """
        Get a single memory as a compressed recall card.

        Returns None if entry doesn't exist. This is the ONLY way
        memory should be accessed during conversation - never raw dumps.
        """
        entry = self.get(entry_id)
        if entry:
            return entry.to_recall_card(self.memory_type.value)
        return None

    def get_most_relevant(self, filter_fn=None) -> Optional[RecallCard]:
        """
        Get the single most relevant memory as a recall card.

        Enforces "one memory per response maximum" rule.
        Returns the most recently updated entry that matches the filter.
        """
        entries = self.query(filter_fn) if filter_fn else self.list_all()
        if not entries:
            return None
        # Sort by updated_at descending, return most recent
        most_recent = max(entries, key=lambda e: e.updated_at)
        return most_recent.to_recall_card(self.memory_type.value)

    def is_in_active_set(self, entry: MemoryEntry) -> bool:
        """
        Check if entry is in the "active set" eligible for recall.

        Active set criteria:
        - Occurred in last 60 days, OR
        - Recurrence count >= 3, OR
        - Marked "critical domain"
        """
        from datetime import datetime, timedelta
        sixty_days_ago = (datetime.now() - timedelta(days=60)).isoformat()

        # Check if updated in last 60 days
        if entry.updated_at >= sixty_days_ago:
            return True

        # Check recurrence count
        if entry.data.get("occurrences", 0) >= 3:
            return True

        # Check if marked critical
        if entry.data.get("critical", False) or entry.data.get("domain") == "critical":
            return True

        return False

    def get_active_set(self) -> List[MemoryEntry]:
        """Get all entries in the active set (eligible for recall)."""
        return [e for e in self._entries.values() if self.is_in_active_set(e)]


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
    "RecallCard",
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
