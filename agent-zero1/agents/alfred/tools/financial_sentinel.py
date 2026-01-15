"""
Financial Sentinel - Alfred Sub-Agent #19

Monitors financial patterns to prevent quiet erosion from subscriptions,
tools, and impulse purchases. Tracks ROI on tools, flags waste, and ensures
spending aligns with actual usage and goals.

Does NOT:
- Make investment decisions
- Judge spending morally
- Recommend penny-pinching
- Ignore legitimate tool needs
- Block purchases without reason
- Optimize for minimal spending
- Provide tax or legal advice

Does:
- Track all subscriptions and recurring costs
- Measure tool usage vs. cost
- Flag overlapping tools (same function)
- Identify unused subscriptions
- Note tools approaching renewal
- Assess purchase patterns
- Flag impulse purchase signals
- Calculate effective ROI where possible
"""

from . import OperationsAgent, AgentResponse, AlfredState
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
from abc import ABC, abstractmethod


# =============================================================================
# ENUMS
# =============================================================================

class FinancialCategory(Enum):
    """Categories of financial items."""
    SUBSCRIPTIONS = "subscriptions"  # Recurring software/services
    TOOLS = "tools"                  # One-time purchases (hardware/software)
    SERVICES = "services"            # Freelancers, contractors, support
    LEARNING = "learning"            # Courses, books, coaching
    INFRASTRUCTURE = "infrastructure"  # Hosting, domains, APIs


class UsageLevel(Enum):
    """Usage levels for subscriptions and tools."""
    HIGH = "high"        # Regular, frequent use
    MEDIUM = "medium"    # Occasional use
    LOW = "low"          # Rare use
    NONE = "none"        # No recent use


class ROIAssessment(Enum):
    """ROI assessment categories."""
    JUSTIFIED = "justified"          # Worth the cost
    QUESTIONABLE = "questionable"    # Unclear value
    UNJUSTIFIED = "unjustified"      # Not worth the cost
    PENDING = "pending"              # Too early to assess


class AlertType(Enum):
    """Types of financial alerts."""
    RENEWAL_APPROACHING = "renewal_approaching"
    OVERLAP = "overlap"
    UNUSED = "unused"
    IMPULSE = "impulse"
    BUDGET_BREACH = "budget_breach"


class BudgetStatus(Enum):
    """Budget status categories."""
    ON_TRACK = "on_track"
    WARNING = "warning"
    OVER = "over"


class RenewalRecommendation(Enum):
    """Recommendations for renewals."""
    RENEW = "renew"
    CANCEL = "cancel"
    EVALUATE = "evaluate"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UsageRecord:
    """Records a single usage event."""
    timestamp: datetime
    duration_minutes: Optional[int] = None
    action: Optional[str] = None
    context: Optional[str] = None


@dataclass
class FinancialItem:
    """Base class for all financial items."""
    id: str
    name: str
    cost: float
    category: FinancialCategory
    purchase_date: datetime
    description: Optional[str] = None
    vendor: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "cost": self.cost,
            "category": self.category.value,
            "purchase_date": self.purchase_date.isoformat(),
            "description": self.description,
            "vendor": self.vendor,
            "tags": self.tags,
            "notes": self.notes
        }


@dataclass
class Subscription(FinancialItem):
    """Recurring subscription or service."""
    billing_cycle: str = "monthly"  # monthly, quarterly, annually
    renewal_date: Optional[datetime] = None
    auto_renew: bool = True
    usage_records: List[UsageRecord] = field(default_factory=list)
    last_used: Optional[datetime] = None
    usage_level: UsageLevel = UsageLevel.MEDIUM
    roi_assessment: ROIAssessment = ROIAssessment.PENDING
    primary_function: Optional[str] = None  # For overlap detection
    alternative_names: List[str] = field(default_factory=list)  # Similar tools

    def __post_init__(self):
        # Ensure category is SUBSCRIPTIONS
        self.category = FinancialCategory.SUBSCRIPTIONS

    @property
    def monthly_cost(self) -> float:
        """Calculate normalized monthly cost."""
        if self.billing_cycle == "monthly":
            return self.cost
        elif self.billing_cycle == "quarterly":
            return self.cost / 3
        elif self.billing_cycle == "annually":
            return self.cost / 12
        return self.cost

    @property
    def days_until_renewal(self) -> Optional[int]:
        """Calculate days until renewal."""
        if not self.renewal_date:
            return None
        delta = self.renewal_date - datetime.now()
        return max(0, delta.days)

    def add_usage(self, duration_minutes: Optional[int] = None,
                  action: Optional[str] = None,
                  context: Optional[str] = None) -> None:
        """Record a usage event."""
        record = UsageRecord(
            timestamp=datetime.now(),
            duration_minutes=duration_minutes,
            action=action,
            context=context
        )
        self.usage_records.append(record)
        self.last_used = record.timestamp

    def calculate_usage_level(self, period_days: int = 30) -> UsageLevel:
        """Calculate usage level based on recent activity."""
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_uses = [u for u in self.usage_records if u.timestamp >= cutoff]

        if len(recent_uses) == 0:
            self.usage_level = UsageLevel.NONE
        elif len(recent_uses) >= 10:
            self.usage_level = UsageLevel.HIGH
        elif len(recent_uses) >= 3:
            self.usage_level = UsageLevel.MEDIUM
        else:
            self.usage_level = UsageLevel.LOW

        return self.usage_level

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "billing_cycle": self.billing_cycle,
            "renewal_date": self.renewal_date.isoformat() if self.renewal_date else None,
            "auto_renew": self.auto_renew,
            "monthly_cost": self.monthly_cost,
            "days_until_renewal": self.days_until_renewal,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_level": self.usage_level.value,
            "roi_assessment": self.roi_assessment.value,
            "primary_function": self.primary_function,
            "usage_count_30d": len([u for u in self.usage_records
                                   if u.timestamp >= datetime.now() - timedelta(days=30)])
        })
        return base


@dataclass
class Tool(FinancialItem):
    """One-time purchase tool (hardware or software)."""
    expected_lifespan_months: Optional[int] = None
    usage_records: List[UsageRecord] = field(default_factory=list)
    last_used: Optional[datetime] = None
    usage_level: UsageLevel = UsageLevel.MEDIUM
    roi_assessment: ROIAssessment = ROIAssessment.PENDING
    primary_function: Optional[str] = None
    linked_output: Optional[str] = None  # What this tool produces

    def __post_init__(self):
        # Ensure category is TOOLS
        self.category = FinancialCategory.TOOLS

    @property
    def monthly_amortized_cost(self) -> Optional[float]:
        """Calculate monthly amortized cost if lifespan is known."""
        if self.expected_lifespan_months and self.expected_lifespan_months > 0:
            return self.cost / self.expected_lifespan_months
        return None

    @property
    def months_owned(self) -> int:
        """Calculate how many months since purchase."""
        delta = datetime.now() - self.purchase_date
        return max(1, delta.days // 30)

    def add_usage(self, duration_minutes: Optional[int] = None,
                  action: Optional[str] = None,
                  context: Optional[str] = None) -> None:
        """Record a usage event."""
        record = UsageRecord(
            timestamp=datetime.now(),
            duration_minutes=duration_minutes,
            action=action,
            context=context
        )
        self.usage_records.append(record)
        self.last_used = record.timestamp

    def calculate_usage_level(self, period_days: int = 30) -> UsageLevel:
        """Calculate usage level based on recent activity."""
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_uses = [u for u in self.usage_records if u.timestamp >= cutoff]

        if len(recent_uses) == 0:
            self.usage_level = UsageLevel.NONE
        elif len(recent_uses) >= 10:
            self.usage_level = UsageLevel.HIGH
        elif len(recent_uses) >= 3:
            self.usage_level = UsageLevel.MEDIUM
        else:
            self.usage_level = UsageLevel.LOW

        return self.usage_level

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "expected_lifespan_months": self.expected_lifespan_months,
            "monthly_amortized_cost": self.monthly_amortized_cost,
            "months_owned": self.months_owned,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_level": self.usage_level.value,
            "roi_assessment": self.roi_assessment.value,
            "primary_function": self.primary_function,
            "linked_output": self.linked_output,
            "usage_count_30d": len([u for u in self.usage_records
                                   if u.timestamp >= datetime.now() - timedelta(days=30)])
        })
        return base


@dataclass
class Purchase:
    """Records a purchase event for pattern analysis."""
    id: str
    item_name: str
    amount: float
    category: FinancialCategory
    timestamp: datetime
    justification: Optional[str] = None
    justified: Optional[bool] = None  # None = pending review
    time_to_decision_minutes: Optional[int] = None  # Time from discovery to purchase
    triggered_by: Optional[str] = None  # What prompted the purchase

    @property
    def is_impulse(self) -> bool:
        """Check if purchase shows impulse characteristics."""
        # Impulse indicators:
        # - Very short decision time (< 30 minutes)
        # - No justification provided
        # - High cost relative to typical purchases
        if self.time_to_decision_minutes is not None and self.time_to_decision_minutes < 30:
            return True
        if not self.justification and self.amount > 50:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "item_name": self.item_name,
            "amount": self.amount,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "justification": self.justification,
            "justified": self.justified,
            "time_to_decision_minutes": self.time_to_decision_minutes,
            "triggered_by": self.triggered_by,
            "is_impulse": self.is_impulse
        }


@dataclass
class FinancialAlert:
    """Financial alert structure."""
    alert_type: AlertType
    item: str
    details: str
    amount_at_risk: float
    recommended_action: str
    decision_deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "item": self.item,
            "details": self.details,
            "amount_at_risk": self.amount_at_risk,
            "recommended_action": self.recommended_action,
            "decision_deadline": self.decision_deadline.isoformat() if self.decision_deadline else None,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged
        }

    def to_formatted_string(self) -> str:
        """Format alert for output."""
        lines = [
            "FINANCIAL_ALERT",
            f"- Alert Type: {self.alert_type.value}",
            f"- Item: {self.item}",
            f"- Details: {self.details}",
            f"- Amount at Risk: ${self.amount_at_risk:.2f}",
            f"- Recommended Action: {self.recommended_action}"
        ]
        if self.decision_deadline:
            lines.append(f"- Decision Deadline: {self.decision_deadline.strftime('%Y-%m-%d')}")
        return "\n".join(lines)


@dataclass
class Budget:
    """Budget tracking structure."""
    category: Optional[FinancialCategory] = None  # None = overall budget
    monthly_target: float = 0.0
    period_start: datetime = field(default_factory=lambda: datetime.now().replace(day=1))
    period_end: Optional[datetime] = None
    actual_spend: float = 0.0

    @property
    def remaining(self) -> float:
        return self.monthly_target - self.actual_spend

    @property
    def status(self) -> BudgetStatus:
        if self.actual_spend > self.monthly_target:
            return BudgetStatus.OVER
        elif self.actual_spend > self.monthly_target * 0.9:
            return BudgetStatus.WARNING
        return BudgetStatus.ON_TRACK

    @property
    def percentage_used(self) -> float:
        if self.monthly_target == 0:
            return 0.0
        return (self.actual_spend / self.monthly_target) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value if self.category else "overall",
            "monthly_target": self.monthly_target,
            "actual_spend": self.actual_spend,
            "remaining": self.remaining,
            "status": self.status.value,
            "percentage_used": self.percentage_used
        }


@dataclass
class OverlapGroup:
    """Group of tools serving the same function."""
    function: str
    items: List[str]  # Item names
    total_monthly_cost: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": self.function,
            "items": self.items,
            "total_monthly_cost": self.total_monthly_cost,
            "recommendation": self.recommendation
        }


# =============================================================================
# EXTERNAL TOOL INTERFACES (Prepared for Integration)
# =============================================================================

class FinanceTrackerInterface(ABC):
    """Abstract interface for finance tracking tools like Firefly III."""

    @abstractmethod
    async def get_transactions(self, start_date: datetime,
                               end_date: datetime) -> List[Dict[str, Any]]:
        """Retrieve transactions for a date range."""
        pass

    @abstractmethod
    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts."""
        pass

    @abstractmethod
    async def create_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Create a new transaction."""
        pass

    @abstractmethod
    async def get_categories(self) -> List[str]:
        """Get spending categories."""
        pass


class FireflyIIIInterface(FinanceTrackerInterface):
    """Interface for Firefly III finance tracking.

    Firefly III is a self-hosted personal finance manager.
    https://www.firefly-iii.org/

    Integration requires:
    - API endpoint URL
    - Personal Access Token
    """

    def __init__(self, api_url: Optional[str] = None,
                 api_token: Optional[str] = None):
        self.api_url = api_url
        self.api_token = api_token
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to Firefly III."""
        if not self.api_url or not self.api_token:
            return False
        # Implementation would make actual API call
        # Placeholder for integration
        self._connected = True
        return True

    async def get_transactions(self, start_date: datetime,
                               end_date: datetime) -> List[Dict[str, Any]]:
        """Retrieve transactions from Firefly III."""
        if not self._connected:
            return []
        # Implementation would make actual API call
        # GET /api/v1/transactions?start={start}&end={end}
        return []

    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts from Firefly III."""
        if not self._connected:
            return []
        # Implementation would make actual API call
        # GET /api/v1/accounts
        return []

    async def create_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Create transaction in Firefly III."""
        if not self._connected:
            return False
        # Implementation would make actual API call
        # POST /api/v1/transactions
        return True

    async def get_categories(self) -> List[str]:
        """Get categories from Firefly III."""
        if not self._connected:
            return []
        # Implementation would make actual API call
        # GET /api/v1/categories
        return []


class SubscriptionTrackerInterface(ABC):
    """Abstract interface for subscription tracking tools like Wallos."""

    @abstractmethod
    async def get_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all tracked subscriptions."""
        pass

    @abstractmethod
    async def add_subscription(self, subscription: Dict[str, Any]) -> bool:
        """Add a new subscription."""
        pass

    @abstractmethod
    async def update_subscription(self, id: str,
                                   updates: Dict[str, Any]) -> bool:
        """Update an existing subscription."""
        pass

    @abstractmethod
    async def get_upcoming_renewals(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get subscriptions renewing within specified days."""
        pass


class WallosInterface(SubscriptionTrackerInterface):
    """Interface for Wallos subscription tracking.

    Wallos is an open-source subscription tracker.
    https://github.com/ellite/Wallos

    Integration requires:
    - API endpoint URL
    - API key or session token
    """

    def __init__(self, api_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to Wallos."""
        if not self.api_url or not self.api_key:
            return False
        # Implementation would verify connection
        self._connected = True
        return True

    async def get_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all subscriptions from Wallos."""
        if not self._connected:
            return []
        # Implementation would make actual API call
        return []

    async def add_subscription(self, subscription: Dict[str, Any]) -> bool:
        """Add subscription to Wallos."""
        if not self._connected:
            return False
        # Implementation would make actual API call
        return True

    async def update_subscription(self, id: str,
                                   updates: Dict[str, Any]) -> bool:
        """Update subscription in Wallos."""
        if not self._connected:
            return False
        # Implementation would make actual API call
        return True

    async def get_upcoming_renewals(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get upcoming renewals from Wallos."""
        if not self._connected:
            return []
        # Implementation would make actual API call
        return []


class BudgetingInterface(ABC):
    """Abstract interface for budgeting tools like Actual Budget."""

    @abstractmethod
    async def get_budgets(self) -> List[Dict[str, Any]]:
        """Get all budget categories and their status."""
        pass

    @abstractmethod
    async def get_budget_status(self, month: Optional[datetime] = None) -> Dict[str, Any]:
        """Get overall budget status for a month."""
        pass

    @abstractmethod
    async def update_budget(self, category: str, amount: float) -> bool:
        """Update budget for a category."""
        pass


class ActualBudgetInterface(BudgetingInterface):
    """Interface for Actual Budget.

    Actual Budget is a privacy-focused budgeting app.
    https://actualbudget.com/

    Integration requires:
    - Server URL (if self-hosted)
    - Sync ID and encryption password
    """

    def __init__(self, server_url: Optional[str] = None,
                 sync_id: Optional[str] = None,
                 password: Optional[str] = None):
        self.server_url = server_url
        self.sync_id = sync_id
        self.password = password
        self._connected = False

    async def connect(self) -> bool:
        """Establish connection to Actual Budget."""
        if not self.server_url or not self.sync_id:
            return False
        # Implementation would verify connection
        self._connected = True
        return True

    async def get_budgets(self) -> List[Dict[str, Any]]:
        """Get budgets from Actual Budget."""
        if not self._connected:
            return []
        # Implementation would make actual API call
        return []

    async def get_budget_status(self, month: Optional[datetime] = None) -> Dict[str, Any]:
        """Get budget status from Actual Budget."""
        if not self._connected:
            return {}
        # Implementation would make actual API call
        return {}

    async def update_budget(self, category: str, amount: float) -> bool:
        """Update budget in Actual Budget."""
        if not self._connected:
            return False
        # Implementation would make actual API call
        return True


# =============================================================================
# FINANCIAL SENTINEL AGENT
# =============================================================================

class FinancialSentinel(OperationsAgent):
    """
    Financial Sentinel - Alfred Sub-Agent #19

    Monitors financial patterns to prevent quiet erosion from subscriptions,
    tools, and impulse purchases. Tracks ROI on tools, flags waste, and
    ensures spending aligns with actual usage and goals.
    """

    def __init__(self):
        super().__init__(name="Financial Sentinel")

        # Core data stores
        self._subscriptions: Dict[str, Subscription] = {}
        self._tools: Dict[str, Tool] = {}
        self._purchases: List[Purchase] = []
        self._alerts: List[FinancialAlert] = []
        self._budgets: Dict[str, Budget] = {}  # Key: category or "overall"

        # Configuration
        self._renewal_warning_days: int = 14  # Days before renewal to alert
        self._unused_threshold_days: int = 30  # Days without use = unused
        self._impulse_threshold_minutes: int = 30  # Quick decision = impulse
        self._impulse_pattern_count: int = 3  # Number of impulse purchases to trigger pattern alert

        # External tool interfaces (prepared for integration)
        self._firefly: Optional[FireflyIIIInterface] = None
        self._wallos: Optional[WallosInterface] = None
        self._actual_budget: Optional[ActualBudgetInterface] = None

        # Previous period data for delta calculation
        self._previous_monthly_total: Optional[float] = None

    # =========================================================================
    # EXTERNAL TOOL INTEGRATION
    # =========================================================================

    def configure_firefly(self, api_url: str, api_token: str) -> None:
        """Configure Firefly III integration."""
        self._firefly = FireflyIIIInterface(api_url, api_token)

    def configure_wallos(self, api_url: str, api_key: str) -> None:
        """Configure Wallos integration."""
        self._wallos = WallosInterface(api_url, api_key)

    def configure_actual_budget(self, server_url: str,
                                 sync_id: str,
                                 password: str) -> None:
        """Configure Actual Budget integration."""
        self._actual_budget = ActualBudgetInterface(server_url, sync_id, password)

    async def sync_external_tools(self) -> Dict[str, bool]:
        """Sync data from all configured external tools."""
        results = {}

        if self._firefly:
            results["firefly"] = await self._firefly.connect()

        if self._wallos:
            results["wallos"] = await self._wallos.connect()
            if results["wallos"]:
                # Import subscriptions from Wallos
                wallos_subs = await self._wallos.get_subscriptions()
                for sub_data in wallos_subs:
                    # Convert Wallos format to internal Subscription
                    pass  # Implementation depends on Wallos API response format

        if self._actual_budget:
            results["actual_budget"] = await self._actual_budget.connect()
            if results["actual_budget"]:
                # Import budget data from Actual Budget
                budget_data = await self._actual_budget.get_budgets()
                for budget in budget_data:
                    # Convert to internal Budget format
                    pass  # Implementation depends on Actual Budget API response

        return results

    # =========================================================================
    # DATA MANAGEMENT
    # =========================================================================

    def add_subscription(self, subscription: Subscription) -> None:
        """Add a subscription to tracking."""
        self._subscriptions[subscription.id] = subscription

    def remove_subscription(self, subscription_id: str) -> bool:
        """Remove a subscription from tracking."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to tracking."""
        self._tools[tool.id] = tool

    def remove_tool(self, tool_id: str) -> bool:
        """Remove a tool from tracking."""
        if tool_id in self._tools:
            del self._tools[tool_id]
            return True
        return False

    def record_purchase(self, purchase: Purchase) -> None:
        """Record a purchase for pattern analysis."""
        self._purchases.append(purchase)

    def set_budget(self, monthly_target: float,
                   category: Optional[FinancialCategory] = None) -> None:
        """Set budget target for overall or specific category."""
        key = category.value if category else "overall"
        self._budgets[key] = Budget(
            category=category,
            monthly_target=monthly_target
        )

    def record_usage(self, item_id: str,
                     duration_minutes: Optional[int] = None,
                     action: Optional[str] = None,
                     context: Optional[str] = None) -> bool:
        """Record usage of a subscription or tool."""
        if item_id in self._subscriptions:
            self._subscriptions[item_id].add_usage(duration_minutes, action, context)
            return True
        elif item_id in self._tools:
            self._tools[item_id].add_usage(duration_minutes, action, context)
            return True
        return False

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def analyze_subscriptions(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Analyze all subscriptions for usage, value, and issues.

        Returns analysis including:
        - Active subscriptions with usage levels
        - Unused/underused subscriptions
        - Monthly recurring total
        - Waste calculation
        """
        results = {
            "active": [],
            "unused": [],
            "underused": [],
            "monthly_total": 0.0,
            "monthly_waste": 0.0,
            "subscription_count": len(self._subscriptions)
        }

        for sub_id, sub in self._subscriptions.items():
            # Update usage level
            usage = sub.calculate_usage_level(period_days)

            sub_info = {
                "id": sub_id,
                "name": sub.name,
                "cost": sub.monthly_cost,
                "billing_cycle": sub.billing_cycle,
                "category": sub.category.value,
                "usage_level": usage.value,
                "roi_assessment": sub.roi_assessment.value,
                "last_used": sub.last_used.isoformat() if sub.last_used else None,
                "renewal_date": sub.renewal_date.isoformat() if sub.renewal_date else None,
                "primary_function": sub.primary_function
            }

            results["monthly_total"] += sub.monthly_cost
            results["active"].append(sub_info)

            if usage == UsageLevel.NONE:
                results["unused"].append(sub_info)
                results["monthly_waste"] += sub.monthly_cost
            elif usage == UsageLevel.LOW:
                results["underused"].append(sub_info)
                results["monthly_waste"] += sub.monthly_cost * 0.5  # Partial waste

        return results

    def check_renewals(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """
        Check for upcoming subscription renewals.

        Returns list of subscriptions renewing within specified days,
        with recommendations.
        """
        upcoming = []
        now = datetime.now()

        for sub_id, sub in self._subscriptions.items():
            if not sub.renewal_date:
                continue

            days_until = sub.days_until_renewal
            if days_until is not None and days_until <= days_ahead:
                # Determine recommendation based on usage
                if sub.usage_level == UsageLevel.NONE:
                    recommendation = RenewalRecommendation.CANCEL
                elif sub.usage_level == UsageLevel.LOW:
                    recommendation = RenewalRecommendation.EVALUATE
                elif sub.roi_assessment == ROIAssessment.UNJUSTIFIED:
                    recommendation = RenewalRecommendation.CANCEL
                elif sub.roi_assessment == ROIAssessment.QUESTIONABLE:
                    recommendation = RenewalRecommendation.EVALUATE
                else:
                    recommendation = RenewalRecommendation.RENEW

                renewal_info = {
                    "id": sub_id,
                    "name": sub.name,
                    "cost": sub.cost,
                    "billing_cycle": sub.billing_cycle,
                    "renewal_date": sub.renewal_date.isoformat(),
                    "days_until_renewal": days_until,
                    "usage_level": sub.usage_level.value,
                    "recommendation": recommendation.value,
                    "auto_renew": sub.auto_renew
                }
                upcoming.append(renewal_info)

                # Create alert if within warning threshold
                if days_until <= self._renewal_warning_days:
                    self._create_renewal_alert(sub, days_until, recommendation)

        # Sort by days until renewal
        upcoming.sort(key=lambda x: x["days_until_renewal"])
        return upcoming

    def detect_overlap(self) -> List[OverlapGroup]:
        """
        Detect overlapping tools/subscriptions serving the same function.

        Returns groups of items with the same primary function.
        """
        # Group by primary function
        function_groups: Dict[str, List[Tuple[str, float, str]]] = {}

        for sub_id, sub in self._subscriptions.items():
            if sub.primary_function:
                func = sub.primary_function.lower()
                if func not in function_groups:
                    function_groups[func] = []
                function_groups[func].append((sub.name, sub.monthly_cost, "subscription"))

        for tool_id, tool in self._tools.items():
            if tool.primary_function:
                func = tool.primary_function.lower()
                if func not in function_groups:
                    function_groups[func] = []
                monthly = tool.monthly_amortized_cost or 0
                function_groups[func].append((tool.name, monthly, "tool"))

        # Find overlaps (more than one item per function)
        overlaps = []
        for func, items in function_groups.items():
            if len(items) > 1:
                total_cost = sum(cost for _, cost, _ in items)
                item_names = [name for name, _, _ in items]

                # Generate recommendation
                recommendation = self._generate_overlap_recommendation(items)

                overlap_group = OverlapGroup(
                    function=func,
                    items=item_names,
                    total_monthly_cost=total_cost,
                    recommendation=recommendation
                )
                overlaps.append(overlap_group)

                # Create alert
                self._create_overlap_alert(overlap_group)

        return overlaps

    def _generate_overlap_recommendation(self,
                                          items: List[Tuple[str, float, str]]) -> str:
        """Generate recommendation for overlapping items."""
        # Simple heuristic: recommend keeping the cheapest one
        # More sophisticated logic could consider usage levels
        sorted_items = sorted(items, key=lambda x: x[1])
        cheapest = sorted_items[0][0]
        others = [item[0] for item in sorted_items[1:]]
        return f"Consider keeping {cheapest} and evaluating {', '.join(others)}"

    def calculate_roi(self, item_id: str) -> Dict[str, Any]:
        """
        Calculate ROI for a specific subscription or tool.

        ROI is assessed based on:
        - Cost vs. usage frequency
        - Linked outputs (for tools)
        - Time saved or value created
        """
        result = {
            "item_id": item_id,
            "item_name": None,
            "item_type": None,
            "cost": 0.0,
            "usage_metrics": {},
            "roi_assessment": ROIAssessment.PENDING.value,
            "rationale": ""
        }

        if item_id in self._subscriptions:
            sub = self._subscriptions[item_id]
            result["item_name"] = sub.name
            result["item_type"] = "subscription"
            result["cost"] = sub.monthly_cost

            # Calculate usage metrics
            period_days = 30
            cutoff = datetime.now() - timedelta(days=period_days)
            recent_uses = [u for u in sub.usage_records if u.timestamp >= cutoff]

            result["usage_metrics"] = {
                "uses_last_30_days": len(recent_uses),
                "total_minutes": sum(u.duration_minutes or 0 for u in recent_uses),
                "cost_per_use": sub.monthly_cost / len(recent_uses) if recent_uses else None,
                "usage_level": sub.usage_level.value
            }

            # Determine ROI assessment
            assessment, rationale = self._assess_subscription_roi(sub, result["usage_metrics"])
            result["roi_assessment"] = assessment.value
            result["rationale"] = rationale

            # Update the subscription's ROI assessment
            sub.roi_assessment = assessment

        elif item_id in self._tools:
            tool = self._tools[item_id]
            result["item_name"] = tool.name
            result["item_type"] = "tool"
            result["cost"] = tool.cost

            # Calculate usage metrics
            period_days = 30
            cutoff = datetime.now() - timedelta(days=period_days)
            recent_uses = [u for u in tool.usage_records if u.timestamp >= cutoff]

            result["usage_metrics"] = {
                "uses_last_30_days": len(recent_uses),
                "total_minutes": sum(u.duration_minutes or 0 for u in recent_uses),
                "months_owned": tool.months_owned,
                "monthly_amortized_cost": tool.monthly_amortized_cost,
                "linked_output": tool.linked_output,
                "usage_level": tool.usage_level.value
            }

            # Determine ROI assessment
            assessment, rationale = self._assess_tool_roi(tool, result["usage_metrics"])
            result["roi_assessment"] = assessment.value
            result["rationale"] = rationale

            # Update the tool's ROI assessment
            tool.roi_assessment = assessment

        return result

    def _assess_subscription_roi(self, sub: Subscription,
                                  metrics: Dict[str, Any]) -> Tuple[ROIAssessment, str]:
        """Assess ROI for a subscription based on usage metrics."""
        uses = metrics["uses_last_30_days"]
        cost_per_use = metrics["cost_per_use"]

        if uses == 0:
            return ROIAssessment.UNJUSTIFIED, "No usage in the last 30 days"

        if uses >= 10:
            return ROIAssessment.JUSTIFIED, f"High usage ({uses} uses) justifies ${sub.monthly_cost:.2f}/month"

        if cost_per_use and cost_per_use > 20:
            return ROIAssessment.QUESTIONABLE, f"High cost per use (${cost_per_use:.2f})"

        if uses >= 3:
            return ROIAssessment.JUSTIFIED, f"Moderate usage ({uses} uses) at acceptable cost"

        return ROIAssessment.QUESTIONABLE, f"Low usage ({uses} uses) - evaluate necessity"

    def _assess_tool_roi(self, tool: Tool,
                          metrics: Dict[str, Any]) -> Tuple[ROIAssessment, str]:
        """Assess ROI for a tool based on usage metrics."""
        uses = metrics["uses_last_30_days"]
        has_output = tool.linked_output is not None
        months_owned = metrics["months_owned"]

        if months_owned < 2:
            return ROIAssessment.PENDING, "Too early to assess (owned less than 2 months)"

        if uses == 0 and months_owned >= 3:
            return ROIAssessment.UNJUSTIFIED, f"No usage in 30 days despite owning for {months_owned} months"

        if has_output and uses >= 5:
            return ROIAssessment.JUSTIFIED, f"Regular use with linked output: {tool.linked_output}"

        if uses >= 10:
            return ROIAssessment.JUSTIFIED, f"High usage ({uses} uses/month)"

        if not has_output and uses < 3:
            return ROIAssessment.QUESTIONABLE, "Low usage without linked output"

        return ROIAssessment.QUESTIONABLE, f"Moderate usage ({uses} uses) - monitor"

    def detect_impulse_patterns(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Detect impulse purchase patterns.

        Returns analysis of recent purchases looking for:
        - Quick decision times
        - Missing justifications
        - Purchase clustering
        """
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_purchases = [p for p in self._purchases if p.timestamp >= cutoff]

        results = {
            "period_days": period_days,
            "total_purchases": len(recent_purchases),
            "total_spend": sum(p.amount for p in recent_purchases),
            "impulse_purchases": [],
            "pattern_detected": False,
            "pattern_details": None
        }

        impulse_count = 0
        impulse_total = 0.0

        for purchase in recent_purchases:
            if purchase.is_impulse:
                impulse_count += 1
                impulse_total += purchase.amount
                results["impulse_purchases"].append(purchase.to_dict())

        # Check for pattern
        if impulse_count >= self._impulse_pattern_count:
            results["pattern_detected"] = True
            results["pattern_details"] = {
                "impulse_count": impulse_count,
                "impulse_total": impulse_total,
                "percentage_of_purchases": (impulse_count / len(recent_purchases) * 100)
                    if recent_purchases else 0,
                "average_impulse_amount": impulse_total / impulse_count if impulse_count else 0
            }

            # Create alert
            self._create_impulse_alert(results["pattern_details"])

        return results

    def calculate_budget_status(self) -> Dict[str, Any]:
        """
        Calculate current budget status across all categories.

        Returns budget tracking with actual vs. target.
        """
        # Calculate actual spend for current month
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Sum subscription costs for current month
        monthly_subscriptions = sum(sub.monthly_cost for sub in self._subscriptions.values())

        # Sum purchases this month
        monthly_purchases = sum(
            p.amount for p in self._purchases
            if p.timestamp >= month_start
        )

        total_actual = monthly_subscriptions + monthly_purchases

        # Update overall budget
        if "overall" in self._budgets:
            self._budgets["overall"].actual_spend = total_actual

        # Build status report
        status = {
            "month": month_start.strftime("%Y-%m"),
            "subscriptions_total": monthly_subscriptions,
            "purchases_total": monthly_purchases,
            "total_actual": total_actual,
            "budgets": {}
        }

        for key, budget in self._budgets.items():
            budget_status = budget.to_dict()
            status["budgets"][key] = budget_status

            # Check for budget breach
            if budget.status == BudgetStatus.OVER:
                self._create_budget_breach_alert(budget)

        return status

    # =========================================================================
    # ALERT MANAGEMENT
    # =========================================================================

    def _create_renewal_alert(self, sub: Subscription,
                               days_until: int,
                               recommendation: RenewalRecommendation) -> None:
        """Create a renewal approaching alert."""
        alert = FinancialAlert(
            alert_type=AlertType.RENEWAL_APPROACHING,
            item=sub.name,
            details=f"Renews in {days_until} days. Usage: {sub.usage_level.value}",
            amount_at_risk=sub.cost,
            recommended_action=f"{recommendation.value.upper()} - {sub.billing_cycle} cost: ${sub.cost:.2f}",
            decision_deadline=sub.renewal_date
        )
        self._alerts.append(alert)

    def _create_overlap_alert(self, overlap: OverlapGroup) -> None:
        """Create an overlap detected alert."""
        alert = FinancialAlert(
            alert_type=AlertType.OVERLAP,
            item=f"Function: {overlap.function}",
            details=f"Multiple tools for same purpose: {', '.join(overlap.items)}",
            amount_at_risk=overlap.total_monthly_cost,
            recommended_action=overlap.recommendation
        )
        self._alerts.append(alert)

    def _create_impulse_alert(self, pattern_details: Dict[str, Any]) -> None:
        """Create an impulse pattern alert."""
        alert = FinancialAlert(
            alert_type=AlertType.IMPULSE,
            item="Purchase Pattern",
            details=f"Detected {pattern_details['impulse_count']} impulse purchases",
            amount_at_risk=pattern_details['impulse_total'],
            recommended_action="Review recent purchases and implement cooling-off period"
        )
        self._alerts.append(alert)

    def _create_budget_breach_alert(self, budget: Budget) -> None:
        """Create a budget breach alert."""
        category = budget.category.value if budget.category else "overall"
        alert = FinancialAlert(
            alert_type=AlertType.BUDGET_BREACH,
            item=f"{category.title()} Budget",
            details=f"Spent ${budget.actual_spend:.2f} of ${budget.monthly_target:.2f} budget",
            amount_at_risk=budget.actual_spend - budget.monthly_target,
            recommended_action="Review spending and defer non-essential purchases"
        )
        self._alerts.append(alert)

    def _create_unused_alert(self, sub: Subscription) -> None:
        """Create an unused subscription alert."""
        alert = FinancialAlert(
            alert_type=AlertType.UNUSED,
            item=sub.name,
            details=f"No usage detected in {self._unused_threshold_days} days",
            amount_at_risk=sub.monthly_cost,
            recommended_action="Cancel subscription or document intended use"
        )
        self._alerts.append(alert)

    def get_alerts(self, unacknowledged_only: bool = True) -> List[FinancialAlert]:
        """Get all alerts, optionally filtering to unacknowledged only."""
        if unacknowledged_only:
            return [a for a in self._alerts if not a.acknowledged]
        return self._alerts

    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert by index."""
        if 0 <= alert_index < len(self._alerts):
            self._alerts[alert_index].acknowledged = True
            return True
        return False

    def clear_acknowledged_alerts(self) -> int:
        """Clear all acknowledged alerts. Returns count of cleared alerts."""
        original_count = len(self._alerts)
        self._alerts = [a for a in self._alerts if not a.acknowledged]
        return original_count - len(self._alerts)

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_report(self,
                        period: str = "month",
                        focus: str = "all",
                        include_pending: bool = True,
                        roi_items: Optional[List[str]] = None) -> AgentResponse:
        """
        Generate comprehensive financial report.

        Args:
            period: "month" or "quarter" to analyze
            focus: "subscriptions", "tools", or "all"
            include_pending: Include upcoming renewals
            roi_items: Specific item IDs to evaluate ROI for

        Returns:
            AgentResponse with FINANCIAL_REPORT data
        """
        # Check state permission
        permitted, reason = self.check_state_permission()
        if not permitted:
            return self.blocked_response(reason)

        report_data = {
            "report_type": "FINANCIAL_REPORT",
            "report_date": datetime.now().isoformat(),
            "period": period,
            "focus": focus
        }

        # Analyze subscriptions
        subscription_analysis = self.analyze_subscriptions(
            period_days=30 if period == "month" else 90
        )

        # Calculate monthly recurring
        monthly_recurring = {
            "total": subscription_analysis["monthly_total"],
            "delta_from_last": None,
            "count": subscription_analysis["subscription_count"]
        }

        if self._previous_monthly_total is not None:
            monthly_recurring["delta_from_last"] = (
                subscription_analysis["monthly_total"] - self._previous_monthly_total
            )

        # Store for next comparison
        self._previous_monthly_total = subscription_analysis["monthly_total"]

        report_data["monthly_recurring"] = monthly_recurring

        # Active subscriptions
        if focus in ["subscriptions", "all"]:
            report_data["active_subscriptions"] = subscription_analysis["active"]

        # Detect overlaps
        overlaps = self.detect_overlap()
        if overlaps:
            report_data["overlap_detected"] = [o.to_dict() for o in overlaps]

        # Unused/underused
        report_data["unused_underused"] = {
            "unused": subscription_analysis["unused"],
            "underused": subscription_analysis["underused"],
            "monthly_waste": subscription_analysis["monthly_waste"]
        }

        # Upcoming renewals
        if include_pending:
            renewals_days = 30 if period == "month" else 90
            report_data["upcoming_renewals"] = self.check_renewals(renewals_days)

        # Recent purchases
        cutoff_days = 30 if period == "month" else 90
        cutoff = datetime.now() - timedelta(days=cutoff_days)
        recent_purchases = [
            p.to_dict() for p in self._purchases
            if p.timestamp >= cutoff
        ]
        report_data["recent_purchases"] = recent_purchases

        # Impulse patterns
        impulse_analysis = self.detect_impulse_patterns(cutoff_days)
        if impulse_analysis["pattern_detected"]:
            report_data["impulse_patterns"] = impulse_analysis["pattern_details"]

        # Budget status
        budget_status = self.calculate_budget_status()
        report_data["budget_status"] = budget_status

        # ROI assessments
        if roi_items:
            roi_assessments = []
            for item_id in roi_items:
                roi = self.calculate_roi(item_id)
                if roi["item_name"]:
                    roi_assessments.append(roi)
            report_data["roi_assessments"] = roi_assessments

        # Generate recommendations
        recommendations = self._generate_recommendations(
            subscription_analysis,
            overlaps,
            budget_status
        )
        report_data["recommendations"] = recommendations

        # Collect any new alerts
        report_data["new_alerts"] = [a.to_dict() for a in self.get_alerts()]

        return self.create_response(
            data=report_data,
            success=True,
            warnings=self._collect_warnings(subscription_analysis, budget_status)
        )

    def _generate_recommendations(self,
                                   subscription_analysis: Dict[str, Any],
                                   overlaps: List[OverlapGroup],
                                   budget_status: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Unused subscriptions
        unused = subscription_analysis["unused"]
        if unused:
            names = [u["name"] for u in unused[:3]]  # Top 3
            recommendations.append(
                f"Cancel unused subscriptions: {', '.join(names)} "
                f"(saving ${subscription_analysis['monthly_waste']:.2f}/month)"
            )

        # Overlaps
        for overlap in overlaps:
            recommendations.append(
                f"Consolidate {overlap.function} tools: {overlap.recommendation}"
            )

        # Budget issues
        if "overall" in budget_status.get("budgets", {}):
            overall = budget_status["budgets"]["overall"]
            if overall["status"] == "over":
                recommendations.append(
                    f"Reduce spending by ${-overall['remaining']:.2f} to meet budget target"
                )
            elif overall["status"] == "warning":
                recommendations.append(
                    f"Budget nearly exhausted - ${overall['remaining']:.2f} remaining"
                )

        # High-cost low-usage subscriptions
        underused = subscription_analysis["underused"]
        high_cost_underused = [u for u in underused if u["cost"] > 20]
        if high_cost_underused:
            names = [u["name"] for u in high_cost_underused[:2]]
            recommendations.append(
                f"Evaluate underused high-cost subscriptions: {', '.join(names)}"
            )

        return recommendations

    def _collect_warnings(self,
                          subscription_analysis: Dict[str, Any],
                          budget_status: Dict[str, Any]) -> List[str]:
        """Collect warning messages for the response."""
        warnings = []

        if subscription_analysis["monthly_waste"] > 50:
            warnings.append(
                f"High monthly waste detected: ${subscription_analysis['monthly_waste']:.2f}"
            )

        if "overall" in budget_status.get("budgets", {}):
            if budget_status["budgets"]["overall"]["status"] == "over":
                warnings.append("Budget exceeded for current month")

        return warnings

    # =========================================================================
    # OUTPUT FORMATTING
    # =========================================================================

    def format_report(self, response: AgentResponse) -> str:
        """Format the financial report as a readable string."""
        if not response.success:
            return f"FINANCIAL_REPORT\n- Status: BLOCKED\n- Reason: {response.errors[0]}"

        data = response.data
        lines = [
            "FINANCIAL_REPORT",
            f"- Report Date: {data['report_date']}",
            f"- Period: {data['period']}",
            ""
        ]

        # Monthly recurring
        mr = data["monthly_recurring"]
        delta_str = ""
        if mr["delta_from_last"] is not None:
            sign = "+" if mr["delta_from_last"] >= 0 else ""
            delta_str = f" ({sign}${mr['delta_from_last']:.2f})"

        lines.extend([
            "- Monthly Recurring:",
            f"  - Total: ${mr['total']:.2f}{delta_str}",
            f"  - Count: {mr['count']} subscriptions",
            ""
        ])

        # Active subscriptions
        if "active_subscriptions" in data:
            lines.append("- Active Subscriptions:")
            for sub in data["active_subscriptions"]:
                renewal = f", Renewal: {sub['renewal_date'][:10]}" if sub['renewal_date'] else ""
                lines.append(
                    f"  - {sub['name']}: ${sub['cost']:.2f}/mo | "
                    f"Usage: {sub['usage_level']} | ROI: {sub['roi_assessment']}{renewal}"
                )
            lines.append("")

        # Overlaps
        if "overlap_detected" in data:
            lines.append("- Overlap Detected:")
            for overlap in data["overlap_detected"]:
                lines.append(
                    f"  - {overlap['function']}: {', '.join(overlap['items'])} "
                    f"(${overlap['total_monthly_cost']:.2f}/mo)"
                )
                lines.append(f"    Recommendation: {overlap['recommendation']}")
            lines.append("")

        # Unused/Underused
        unused_data = data.get("unused_underused", {})
        if unused_data.get("unused") or unused_data.get("underused"):
            lines.append("- Unused/Underused:")
            for item in unused_data.get("unused", []):
                lines.append(f"  - {item['name']}: ${item['cost']:.2f}/mo (UNUSED)")
            for item in unused_data.get("underused", []):
                lines.append(f"  - {item['name']}: ${item['cost']:.2f}/mo (underused)")
            lines.append(f"  - Total Monthly Waste: ${unused_data.get('monthly_waste', 0):.2f}")
            lines.append("")

        # Upcoming renewals
        if data.get("upcoming_renewals"):
            lines.append("- Upcoming Renewals:")
            for renewal in data["upcoming_renewals"]:
                lines.append(
                    f"  - {renewal['renewal_date'][:10]}: {renewal['name']} - "
                    f"${renewal['cost']:.2f} - Recommend: {renewal['recommendation']}"
                )
            lines.append("")

        # Recent purchases
        if data.get("recent_purchases"):
            lines.append("- Recent Purchases:")
            for purchase in data["recent_purchases"][:5]:  # Show top 5
                justified = "Justified" if purchase["justified"] else (
                    "Pending review" if purchase["justified"] is None else "Unjustified"
                )
                lines.append(
                    f"  - {purchase['item_name']}: ${purchase['amount']:.2f} - {justified}"
                )
            lines.append("")

        # Impulse patterns
        if data.get("impulse_patterns"):
            patterns = data["impulse_patterns"]
            lines.extend([
                "- Impulse Patterns Detected:",
                f"  - Count: {patterns['impulse_count']} impulse purchases",
                f"  - Total: ${patterns['impulse_total']:.2f}",
                f"  - Average: ${patterns['average_impulse_amount']:.2f}",
                ""
            ])

        # Budget status
        if data.get("budget_status", {}).get("budgets"):
            lines.append("- Budget Status:")
            for category, budget in data["budget_status"]["budgets"].items():
                lines.append(
                    f"  - {category.title()}: ${budget['actual_spend']:.2f} / "
                    f"${budget['monthly_target']:.2f} ({budget['status'].upper()})"
                )
            lines.append("")

        # Recommendations
        if data.get("recommendations"):
            lines.append("- Recommendations:")
            for i, rec in enumerate(data["recommendations"], 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def format_alert(self, alert: FinancialAlert) -> str:
        """Format a single alert as a readable string."""
        return alert.to_formatted_string()

    # =========================================================================
    # REQUEST PROCESSING
    # =========================================================================

    def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process a FINANCIAL_CHECK_REQUEST.

        Expected request format:
        {
            "period": "month" | "quarter",
            "focus": "subscriptions" | "tools" | "all",
            "include_pending": true | false,
            "roi_assessment": ["item_id_1", "item_id_2", ...]
        }
        """
        period = request.get("period", "month")
        focus = request.get("focus", "all")
        include_pending = request.get("include_pending", True)
        roi_items = request.get("roi_assessment", None)

        return self.generate_report(
            period=period,
            focus=focus,
            include_pending=include_pending,
            roi_items=roi_items
        )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def export_data(self) -> Dict[str, Any]:
        """Export all financial data for persistence."""
        return {
            "subscriptions": {k: v.to_dict() for k, v in self._subscriptions.items()},
            "tools": {k: v.to_dict() for k, v in self._tools.items()},
            "purchases": [p.to_dict() for p in self._purchases],
            "alerts": [a.to_dict() for a in self._alerts],
            "budgets": {k: v.to_dict() for k, v in self._budgets.items()},
            "config": {
                "renewal_warning_days": self._renewal_warning_days,
                "unused_threshold_days": self._unused_threshold_days,
                "impulse_threshold_minutes": self._impulse_threshold_minutes,
                "impulse_pattern_count": self._impulse_pattern_count
            },
            "previous_monthly_total": self._previous_monthly_total,
            "exported_at": datetime.now().isoformat()
        }

    def to_json(self) -> str:
        """Export data as JSON string."""
        return json.dumps(self.export_data(), indent=2, default=str)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_financial_sentinel() -> FinancialSentinel:
    """Factory function to create a configured Financial Sentinel instance."""
    return FinancialSentinel()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create sentinel
    sentinel = FinancialSentinel()

    # Add example subscription
    netflix = Subscription(
        id="sub_001",
        name="Netflix",
        cost=15.99,
        category=FinancialCategory.SUBSCRIPTIONS,
        purchase_date=datetime(2024, 1, 1),
        billing_cycle="monthly",
        renewal_date=datetime.now() + timedelta(days=7),
        primary_function="streaming_entertainment"
    )
    sentinel.add_subscription(netflix)

    # Add example tool
    keyboard = Tool(
        id="tool_001",
        name="Mechanical Keyboard",
        cost=150.00,
        category=FinancialCategory.TOOLS,
        purchase_date=datetime(2024, 6, 1),
        expected_lifespan_months=36,
        primary_function="typing",
        linked_output="content creation"
    )
    sentinel.add_tool(keyboard)

    # Record some usage
    sentinel.record_usage("sub_001", duration_minutes=120, action="watched", context="movie")
    sentinel.record_usage("tool_001", duration_minutes=480, action="typing", context="writing article")

    # Set budget
    sentinel.set_budget(monthly_target=500.00)

    # Generate report
    response = sentinel.generate_report(
        period="month",
        focus="all",
        include_pending=True,
        roi_items=["sub_001", "tool_001"]
    )

    # Print formatted report
    print(sentinel.format_report(response))
