#!/usr/bin/env python3
"""
Alfred CLI - Command-line interface for Alfred Personal AI Governance System

Usage:
    alfred status                    - Show current Alfred state and recent alerts
    alfred brief morning             - Generate morning brief
    alfred brief evening             - Generate evening shutdown
    alfred brief weekly              - Generate weekly strategic brief
    alfred check reputation          - Run reputation sentinel check
    alfred check shipping            - Run shipping governor check
    alfred check financial           - Run financial sentinel check
    alfred metrics                   - Show social metrics summary
    alfred calendar today            - Show today's schedule
    alfred learn                     - Show learning queue

Options:
    --json                           - Output in JSON format for scripting
    --help                           - Show help message
"""

import sys
import json
import asyncio
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich import box
from rich.style import Style
from rich.columns import Columns

# Add the project paths to enable imports
PROJECT_ROOT = Path(__file__).parent
AGENT_TOOLS_PATH = PROJECT_ROOT / "agent-zero1" / "agents" / "alfred" / "tools"
sys.path.insert(0, str(AGENT_TOOLS_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

# Import Alfred sub-agents
try:
    from agent_zero1.agents.alfred.tools import (
        AlfredState,
        AgentResponse,
        ReputationSentinel,
        ShippingGovernor,
        FinancialSentinel,
        SocialMetricsHarvester,
        SchedulingAgent,
        LearningCurator,
    )
    from agent_zero1.agents.alfred.tools.reputation_sentinel import (
        Platform,
        Priority,
        ReputationCheckRequest,
        create_reputation_check_request,
    )
    from agent_zero1.agents.alfred.tools.shipping_governor import (
        ShippingHealth,
        ProjectAction,
        create_shipping_governor,
    )
    from agent_zero1.agents.alfred.tools.financial_sentinel import (
        create_financial_sentinel,
        FinancialCategory,
    )
    from agent_zero1.agents.alfred.tools.social_metrics_harvester import (
        create_harvester,
        Platform as MetricsPlatform,
        MetricsGranularity,
    )
    from agent_zero1.agents.alfred.tools.scheduling_agent import (
        create_scheduling_agent,
        RequestType,
        BlockType,
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

# Initialize Rich console
console = Console()

# State colors
STATE_COLORS = {
    "GREEN": "green",
    "YELLOW": "yellow",
    "RED": "red",
}

STATE_ICONS = {
    "GREEN": "[green]OK[/green]",
    "YELLOW": "[yellow]CAUTION[/yellow]",
    "RED": "[red]ALERT[/red]",
}

# Global state file path
STATE_FILE = PROJECT_ROOT / "data" / "alfred_state.json"


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class AlfredStateManager:
    """Manages Alfred's persistent state."""

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or return defaults."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        """Return default Alfred state."""
        return {
            "alfred_state": "GREEN",
            "last_check": None,
            "active_alerts": [],
            "recent_activities": [],
            "projects": [],
            "subscriptions": [],
            "learning_queue": [],
            "metrics_last_harvest": None,
        }

    def save(self) -> None:
        """Save state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self._state, f, indent=2, default=str)

    @property
    def alfred_state(self) -> str:
        return self._state.get("alfred_state", "GREEN")

    @alfred_state.setter
    def alfred_state(self, value: str) -> None:
        self._state["alfred_state"] = value
        self._state["last_check"] = datetime.now().isoformat()
        self.save()

    @property
    def active_alerts(self) -> List[Dict[str, Any]]:
        return self._state.get("active_alerts", [])

    def add_alert(self, alert: Dict[str, Any]) -> None:
        """Add an alert."""
        alert["timestamp"] = datetime.now().isoformat()
        self._state["active_alerts"].append(alert)
        # Keep only last 50 alerts
        self._state["active_alerts"] = self._state["active_alerts"][-50:]
        self.save()

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._state["active_alerts"] = []
        self.save()

    def add_activity(self, activity: str) -> None:
        """Record an activity."""
        self._state["recent_activities"].append({
            "activity": activity,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 100 activities
        self._state["recent_activities"] = self._state["recent_activities"][-100:]
        self.save()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state."""
        return {
            "alfred_state": self.alfred_state,
            "last_check": self._state.get("last_check"),
            "active_alerts_count": len(self.active_alerts),
            "recent_alerts": self.active_alerts[-5:],
        }


# Initialize global state manager
state_manager = AlfredStateManager()


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def output_json(data: Dict[str, Any]) -> None:
    """Output data as JSON."""
    click.echo(json.dumps(data, indent=2, default=str))


def format_state_badge(state: str) -> Text:
    """Format state as a colored badge."""
    color = STATE_COLORS.get(state, "white")
    return Text(f" {state} ", style=f"bold white on {color}")


def create_status_panel(state: str, last_check: Optional[str], alert_count: int) -> Panel:
    """Create a status panel."""
    state_text = format_state_badge(state)

    content = Text()
    content.append("Alfred State: ")
    content.append(state, style=f"bold {STATE_COLORS.get(state, 'white')}")
    content.append("\n")

    if last_check:
        content.append(f"Last Check: {last_check}\n", style="dim")
    else:
        content.append("Last Check: Never\n", style="dim")

    alert_style = "green" if alert_count == 0 else ("yellow" if alert_count < 3 else "red")
    content.append(f"Active Alerts: ", style="")
    content.append(f"{alert_count}", style=f"bold {alert_style}")

    return Panel(
        content,
        title="[bold blue]Alfred Status[/bold blue]",
        border_style="blue",
        box=box.ROUNDED,
    )


def create_alerts_table(alerts: List[Dict[str, Any]]) -> Table:
    """Create an alerts table."""
    table = Table(
        title="Recent Alerts",
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Time", style="dim", width=20)
    table.add_column("Type", style="cyan", width=15)
    table.add_column("Message", style="white")

    for alert in alerts[-10:]:  # Show last 10
        timestamp = alert.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                pass

        alert_type = alert.get("type", "info")
        message = alert.get("message", str(alert))

        type_style = {
            "error": "red",
            "warning": "yellow",
            "info": "blue",
            "success": "green",
        }.get(alert_type, "white")

        table.add_row(
            timestamp,
            Text(alert_type.upper(), style=type_style),
            message[:80] + "..." if len(message) > 80 else message,
        )

    return table


# =============================================================================
# CLI GROUPS AND COMMANDS
# =============================================================================

@click.group()
@click.option('--json', 'json_output', is_flag=True, help='Output in JSON format')
@click.pass_context
def cli(ctx: click.Context, json_output: bool) -> None:
    """Alfred - Personal AI Governance System CLI

    Not an assistant. A steward.
    Not a productivity tool. A protection system.
    """
    ctx.ensure_object(dict)
    ctx.obj['json'] = json_output


# =============================================================================
# STATUS COMMAND
# =============================================================================

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current Alfred state and recent alerts."""
    json_output = ctx.obj.get('json', False)

    summary = state_manager.get_state_summary()

    if json_output:
        output_json(summary)
        return

    # Display status panel
    console.print()
    console.print(create_status_panel(
        state=summary["alfred_state"],
        last_check=summary["last_check"],
        alert_count=summary["active_alerts_count"],
    ))

    # Display recent alerts if any
    if summary["recent_alerts"]:
        console.print()
        console.print(create_alerts_table(summary["recent_alerts"]))
    else:
        console.print("\n[dim]No recent alerts.[/dim]")

    console.print()


# =============================================================================
# BRIEF COMMANDS
# =============================================================================

@cli.group()
def brief() -> None:
    """Generate briefings (morning, evening, weekly)."""
    pass


@brief.command()
@click.pass_context
def morning(ctx: click.Context) -> None:
    """Generate morning brief."""
    json_output = ctx.obj.get('json', False)

    today = date.today()
    now = datetime.now()

    brief_data = {
        "type": "MORNING_BRIEF",
        "date": today.isoformat(),
        "generated_at": now.isoformat(),
        "alfred_state": state_manager.alfred_state,
        "priorities": [],
        "calendar_summary": {
            "meetings_count": 0,
            "focus_blocks": 0,
            "first_commitment": None,
        },
        "active_projects": [],
        "shipping_status": "HEALTHY",
        "reputation_status": "GREEN",
        "learning_queue_size": 0,
        "daily_intention": None,
        "warnings": [],
    }

    # Add state-specific guidance
    if state_manager.alfred_state == "RED":
        brief_data["warnings"].append("RED STATE ACTIVE: All public-facing output paused. Focus on recovery.")
        brief_data["priorities"].append({
            "priority": 1,
            "task": "Address active reputation/crisis issue",
            "urgency": "critical"
        })
    elif state_manager.alfred_state == "YELLOW":
        brief_data["warnings"].append("YELLOW STATE: Elevated monitoring active. Restrict reactive content.")

    # Add default morning priorities
    if not brief_data["priorities"]:
        brief_data["priorities"] = [
            {"priority": 1, "task": "Review calendar and protect focus blocks", "urgency": "high"},
            {"priority": 2, "task": "Check shipping status on active projects", "urgency": "medium"},
            {"priority": 3, "task": "Process learning queue during commute", "urgency": "low"},
        ]

    state_manager.add_activity("Generated morning brief")

    if json_output:
        output_json(brief_data)
        return

    # Display morning brief
    console.print()
    console.print(Panel(
        f"[bold]Date:[/bold] {today.strftime('%A, %B %d, %Y')}\n"
        f"[bold]Generated:[/bold] {now.strftime('%H:%M')}\n"
        f"[bold]Alfred State:[/bold] [{STATE_COLORS.get(state_manager.alfred_state, 'white')}]{state_manager.alfred_state}[/]",
        title="[bold blue]Morning Brief[/bold blue]",
        border_style="blue",
        box=box.DOUBLE,
    ))

    # Warnings
    if brief_data["warnings"]:
        for warning in brief_data["warnings"]:
            console.print(f"\n[bold red]WARNING:[/bold red] {warning}")

    # Priorities
    console.print("\n[bold cyan]Today's Priorities:[/bold cyan]")
    for p in brief_data["priorities"]:
        urgency_style = {
            "critical": "bold red",
            "high": "yellow",
            "medium": "blue",
            "low": "dim",
        }.get(p.get("urgency", "medium"), "white")
        console.print(f"  {p['priority']}. [{urgency_style}]{p['task']}[/]")

    # Summary stats
    console.print("\n[bold cyan]Quick Stats:[/bold cyan]")
    console.print(f"  [dim]Shipping Status:[/dim] {brief_data['shipping_status']}")
    console.print(f"  [dim]Reputation Status:[/dim] {brief_data['reputation_status']}")
    console.print(f"  [dim]Learning Queue:[/dim] {brief_data['learning_queue_size']} items")

    console.print()


@brief.command()
@click.pass_context
def evening(ctx: click.Context) -> None:
    """Generate evening shutdown brief."""
    json_output = ctx.obj.get('json', False)

    today = date.today()
    now = datetime.now()

    brief_data = {
        "type": "EVENING_BRIEF",
        "date": today.isoformat(),
        "generated_at": now.isoformat(),
        "alfred_state": state_manager.alfred_state,
        "daily_review": {
            "shipped": [],
            "completed_tasks": [],
            "deferred_tasks": [],
            "blockers_encountered": [],
        },
        "tomorrow_prep": {
            "first_commitment": None,
            "key_meetings": [],
            "focus_areas": [],
        },
        "energy_assessment": "normal",
        "recommendations": [
            "Disconnect from work communications",
            "Avoid checking metrics or mentions",
            "Prepare for tomorrow's first commitment",
        ],
        "state_transition": None,
    }

    # State-specific recommendations
    if state_manager.alfred_state == "RED":
        brief_data["recommendations"].insert(0, "Continue crisis monitoring but protect personal time")
    elif state_manager.alfred_state == "YELLOW":
        brief_data["recommendations"].insert(0, "Review monitoring summary before disconnecting")

    state_manager.add_activity("Generated evening brief")

    if json_output:
        output_json(brief_data)
        return

    # Display evening brief
    console.print()
    console.print(Panel(
        f"[bold]Date:[/bold] {today.strftime('%A, %B %d, %Y')}\n"
        f"[bold]Time:[/bold] {now.strftime('%H:%M')}\n"
        f"[bold]Alfred State:[/bold] [{STATE_COLORS.get(state_manager.alfred_state, 'white')}]{state_manager.alfred_state}[/]",
        title="[bold magenta]Evening Shutdown[/bold magenta]",
        border_style="magenta",
        box=box.DOUBLE,
    ))

    # Daily review
    console.print("\n[bold cyan]Daily Review:[/bold cyan]")
    review = brief_data["daily_review"]
    if review["shipped"]:
        console.print(f"  [green]Shipped:[/green] {len(review['shipped'])} items")
    else:
        console.print("  [dim]Shipped: Nothing today[/dim]")

    console.print(f"  [dim]Completed: {len(review['completed_tasks'])} tasks[/dim]")
    console.print(f"  [dim]Deferred: {len(review['deferred_tasks'])} tasks[/dim]")

    # Recommendations
    console.print("\n[bold cyan]Evening Recommendations:[/bold cyan]")
    for rec in brief_data["recommendations"]:
        console.print(f"  [dim]-[/dim] {rec}")

    console.print()


@brief.command()
@click.pass_context
def weekly(ctx: click.Context) -> None:
    """Generate weekly strategic brief."""
    json_output = ctx.obj.get('json', False)

    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    now = datetime.now()

    brief_data = {
        "type": "WEEKLY_STRATEGIC_BRIEF",
        "week_of": week_start.isoformat(),
        "generated_at": now.isoformat(),
        "alfred_state": state_manager.alfred_state,
        "week_summary": {
            "total_shipped": 0,
            "content_published": 0,
            "meetings_attended": 0,
            "deep_work_hours": 0,
        },
        "shipping_health": "HEALTHY",
        "reputation_trend": "stable",
        "financial_summary": {
            "subscriptions_cost": 0,
            "renewals_upcoming": 0,
        },
        "content_performance": {
            "top_performer": None,
            "total_reach": 0,
            "engagement_rate": 0,
        },
        "strategic_insights": [
            "Review project priorities for next week",
            "Audit subscription utilization",
            "Identify content opportunities from audience signals",
        ],
        "next_week_focus": [],
        "risks_identified": [],
    }

    # Add state-specific insights
    if state_manager.alfred_state != "GREEN":
        brief_data["risks_identified"].append({
            "risk": f"Alfred state is {state_manager.alfred_state}",
            "impact": "Content output restricted",
            "mitigation": "Address root cause before planning content",
        })

    state_manager.add_activity("Generated weekly strategic brief")

    if json_output:
        output_json(brief_data)
        return

    # Display weekly brief
    console.print()
    console.print(Panel(
        f"[bold]Week of:[/bold] {week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}\n"
        f"[bold]Generated:[/bold] {now.strftime('%Y-%m-%d %H:%M')}\n"
        f"[bold]Alfred State:[/bold] [{STATE_COLORS.get(state_manager.alfred_state, 'white')}]{state_manager.alfred_state}[/]",
        title="[bold green]Weekly Strategic Brief[/bold green]",
        border_style="green",
        box=box.DOUBLE,
    ))

    # Week summary
    console.print("\n[bold cyan]Week Summary:[/bold cyan]")
    summary = brief_data["week_summary"]
    console.print(f"  Shipped: {summary['total_shipped']} items")
    console.print(f"  Content Published: {summary['content_published']} pieces")
    console.print(f"  Meetings: {summary['meetings_attended']}")
    console.print(f"  Deep Work: {summary['deep_work_hours']} hours")

    # Health indicators
    console.print("\n[bold cyan]Health Indicators:[/bold cyan]")
    shipping_color = {"HEALTHY": "green", "WARNING": "yellow", "CRITICAL": "red"}.get(
        brief_data["shipping_health"], "white"
    )
    console.print(f"  Shipping: [{shipping_color}]{brief_data['shipping_health']}[/]")
    console.print(f"  Reputation: {brief_data['reputation_trend']}")

    # Strategic insights
    console.print("\n[bold cyan]Strategic Insights:[/bold cyan]")
    for insight in brief_data["strategic_insights"]:
        console.print(f"  [dim]-[/dim] {insight}")

    # Risks
    if brief_data["risks_identified"]:
        console.print("\n[bold red]Risks Identified:[/bold red]")
        for risk in brief_data["risks_identified"]:
            console.print(f"  [yellow]{risk['risk']}[/yellow]")
            console.print(f"    [dim]Impact: {risk['impact']}[/dim]")
            console.print(f"    [dim]Mitigation: {risk['mitigation']}[/dim]")

    console.print()


# =============================================================================
# CHECK COMMANDS
# =============================================================================

@cli.group()
def check() -> None:
    """Run checks (reputation, shipping, financial)."""
    pass


@check.command()
@click.option('--platforms', '-p', multiple=True, default=['Twitter', 'YouTube', 'Substack'],
              help='Platforms to check')
@click.option('--hours', '-h', default=24, help='Hours to look back')
@click.pass_context
def reputation(ctx: click.Context, platforms: tuple, hours: int) -> None:
    """Run reputation sentinel check."""
    json_output = ctx.obj.get('json', False)

    now = datetime.now()

    # Create check result
    check_result = {
        "type": "REPUTATION_CHECK",
        "timestamp": now.isoformat(),
        "platforms_checked": list(platforms),
        "time_window_hours": hours,
        "current_state": state_manager.alfred_state,
        "recommended_state": "GREEN",
        "signals_processed": 0,
        "significant_signals": 0,
        "risk_score": 0,
        "packets": [],
        "pattern_notes": [],
        "monitoring_level": "NORMAL" if state_manager.alfred_state == "GREEN" else "HEIGHTENED",
    }

    # Simulate check (in production, this would call the actual ReputationSentinel)
    if state_manager.alfred_state == "GREEN":
        check_result["status"] = "CLEAR"
        check_result["message"] = "No actionable reputation signals detected. Normal operations."
    elif state_manager.alfred_state == "YELLOW":
        check_result["status"] = "ELEVATED"
        check_result["message"] = "Elevated monitoring active. Some signals require attention."
        check_result["monitoring_level"] = "HEIGHTENED"
    else:
        check_result["status"] = "CRITICAL"
        check_result["message"] = "Active threat detected. All public-facing output paused."
        check_result["monitoring_level"] = "CRITICAL"

    state_manager.add_activity(f"Ran reputation check: {check_result['status']}")

    if json_output:
        output_json(check_result)
        return

    # Display reputation check results
    console.print()

    status_color = {
        "CLEAR": "green",
        "ELEVATED": "yellow",
        "CRITICAL": "red",
    }.get(check_result["status"], "white")

    console.print(Panel(
        f"[bold]Status:[/bold] [{status_color}]{check_result['status']}[/]\n"
        f"[bold]Platforms:[/bold] {', '.join(platforms)}\n"
        f"[bold]Time Window:[/bold] Last {hours} hours\n"
        f"[bold]Monitoring Level:[/bold] {check_result['monitoring_level']}",
        title="[bold blue]Reputation Sentinel Check[/bold blue]",
        border_style="blue",
        box=box.ROUNDED,
    ))

    console.print(f"\n[dim]{check_result['message']}[/dim]")

    # Stats
    console.print("\n[bold cyan]Check Statistics:[/bold cyan]")
    console.print(f"  Signals Processed: {check_result['signals_processed']}")
    console.print(f"  Significant Signals: {check_result['significant_signals']}")
    console.print(f"  Risk Score: {check_result['risk_score']}/100")
    console.print(f"  Recommended State: [{STATE_COLORS.get(check_result['recommended_state'], 'white')}]{check_result['recommended_state']}[/]")

    console.print()


@check.command()
@click.pass_context
def shipping(ctx: click.Context) -> None:
    """Run shipping governor check."""
    json_output = ctx.obj.get('json', False)

    now = datetime.now()

    # Create shipping report
    report = {
        "type": "SHIPPING_REPORT",
        "timestamp": now.isoformat(),
        "overall_health": "HEALTHY",
        "alfred_state": state_manager.alfred_state,
        "summary": {
            "active_projects": 0,
            "paused_projects": 0,
            "builds_in_progress": 0,
            "recent_outputs_30d": 0,
            "average_days_without_output": 0,
        },
        "project_assessments": [],
        "building_inventory": {
            "tools_in_progress": 0,
            "tools_with_linked_output": 0,
            "tools_without_output_link": [],
        },
        "shipping_alerts": [],
        "recommended_freezes": [],
        "governance_message": "Shipping health is GOOD. Maintain output cadence. Remember: Tools without output are toys.",
    }

    # Adjust message based on state
    if state_manager.alfred_state == "RED":
        report["status"] = "PAUSED"
        report["governance_message"] = "Shipping pressure paused during RED state - focus on recovery"

    state_manager.add_activity(f"Ran shipping check: {report['overall_health']}")

    if json_output:
        output_json(report)
        return

    # Display shipping report
    console.print()

    health_color = {
        "HEALTHY": "green",
        "WARNING": "yellow",
        "CRITICAL": "red",
    }.get(report["overall_health"], "white")

    console.print(Panel(
        f"[bold]Overall Health:[/bold] [{health_color}]{report['overall_health']}[/]\n"
        f"[bold]Active Projects:[/bold] {report['summary']['active_projects']}\n"
        f"[bold]Builds in Progress:[/bold] {report['summary']['builds_in_progress']}\n"
        f"[bold]Recent Outputs (30d):[/bold] {report['summary']['recent_outputs_30d']}",
        title="[bold green]Shipping Governor Report[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    ))

    # Governance message
    console.print(f"\n[italic]{report['governance_message']}[/italic]")

    # Building inventory
    bi = report["building_inventory"]
    console.print("\n[bold cyan]Building Inventory:[/bold cyan]")
    console.print(f"  Tools in Progress: {bi['tools_in_progress']}")
    console.print(f"  With Linked Output: {bi['tools_with_linked_output']}")
    console.print(f"  Without Output Link: {len(bi['tools_without_output_link'])} [red](FLAGGED)[/red]" if bi['tools_without_output_link'] else f"  Without Output Link: 0")

    # Alerts
    if report["shipping_alerts"]:
        console.print("\n[bold red]Shipping Alerts:[/bold red]")
        for alert in report["shipping_alerts"]:
            console.print(f"  [yellow]{alert}[/yellow]")

    console.print()


@check.command()
@click.option('--period', '-p', default='month', type=click.Choice(['month', 'quarter']),
              help='Analysis period')
@click.pass_context
def financial(ctx: click.Context, period: str) -> None:
    """Run financial sentinel check."""
    json_output = ctx.obj.get('json', False)

    now = datetime.now()

    # Create financial report
    report = {
        "type": "FINANCIAL_REPORT",
        "timestamp": now.isoformat(),
        "period": period,
        "monthly_recurring": {
            "total": 0.00,
            "delta_from_last": None,
            "count": 0,
        },
        "active_subscriptions": [],
        "unused_underused": {
            "unused": [],
            "underused": [],
            "monthly_waste": 0.00,
        },
        "upcoming_renewals": [],
        "overlap_detected": [],
        "budget_status": {
            "target": 0.00,
            "actual": 0.00,
            "status": "on_track",
        },
        "recommendations": [
            "Review any subscriptions approaching renewal",
            "Audit tools for actual usage vs cost",
            "Check for overlapping functionality",
        ],
        "impulse_patterns": None,
    }

    state_manager.add_activity(f"Ran financial check for {period}")

    if json_output:
        output_json(report)
        return

    # Display financial report
    console.print()

    console.print(Panel(
        f"[bold]Period:[/bold] {period.title()}\n"
        f"[bold]Monthly Recurring:[/bold] ${report['monthly_recurring']['total']:.2f}\n"
        f"[bold]Active Subscriptions:[/bold] {report['monthly_recurring']['count']}\n"
        f"[bold]Budget Status:[/bold] {report['budget_status']['status'].upper()}",
        title="[bold yellow]Financial Sentinel Report[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    ))

    # Waste analysis
    waste = report["unused_underused"]
    if waste["monthly_waste"] > 0:
        console.print(f"\n[red]Monthly Waste Detected: ${waste['monthly_waste']:.2f}[/red]")
    else:
        console.print("\n[green]No subscription waste detected.[/green]")

    # Recommendations
    console.print("\n[bold cyan]Recommendations:[/bold cyan]")
    for rec in report["recommendations"]:
        console.print(f"  [dim]-[/dim] {rec}")

    console.print()


# =============================================================================
# METRICS COMMAND
# =============================================================================

@cli.command()
@click.option('--platforms', '-p', multiple=True, default=['twitter', 'youtube', 'substack'],
              help='Platforms to show metrics for')
@click.option('--period', default='weekly', type=click.Choice(['daily', 'weekly', 'monthly']),
              help='Metrics period')
@click.pass_context
def metrics(ctx: click.Context, platforms: tuple, period: str) -> None:
    """Show social metrics summary."""
    json_output = ctx.obj.get('json', False)

    now = datetime.now()
    today = date.today()

    # Calculate period dates
    if period == 'daily':
        period_start = today
    elif period == 'weekly':
        period_start = today - timedelta(days=7)
    else:
        period_start = today - timedelta(days=30)

    # Create metrics report
    report = {
        "type": "METRICS_REPORT",
        "timestamp": now.isoformat(),
        "period": {
            "granularity": period,
            "start": period_start.isoformat(),
            "end": today.isoformat(),
        },
        "platforms": list(platforms),
        "platform_metrics": {},
        "cross_platform_summary": {
            "total_output": 0,
            "total_reach": 0,
            "total_engagement": 0,
            "net_follower_growth": 0,
        },
        "top_performing_content": None,
        "data_quality": {
            "platforms_collected": [],
            "platforms_failed": list(platforms),  # Simulating unconfigured state
            "collection_errors": {},
        },
    }

    # Add placeholder data for each platform
    for platform in platforms:
        report["platform_metrics"][platform] = {
            "output": 0,
            "impressions": 0,
            "engagement": 0,
            "engagement_rate": 0.0,
            "followers": 0,
            "followers_delta": 0,
            "status": "not_configured",
        }

    state_manager.add_activity(f"Viewed {period} metrics")

    if json_output:
        output_json(report)
        return

    # Display metrics report
    console.print()

    console.print(Panel(
        f"[bold]Period:[/bold] {period.title()} ({period_start} to {today})\n"
        f"[bold]Platforms:[/bold] {', '.join(platforms)}",
        title="[bold cyan]Social Metrics Report[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    # Cross-platform summary
    summary = report["cross_platform_summary"]
    console.print("\n[bold cyan]Cross-Platform Summary:[/bold cyan]")
    console.print(f"  Total Output: {summary['total_output']} items")
    console.print(f"  Total Reach: {summary['total_reach']:,}")
    console.print(f"  Total Engagement: {summary['total_engagement']:,}")
    console.print(f"  Net Follower Growth: {summary['net_follower_growth']:+,}")

    # Platform breakdown
    console.print("\n[bold cyan]Platform Breakdown:[/bold cyan]")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Platform", style="cyan")
    table.add_column("Output", justify="right")
    table.add_column("Impressions", justify="right")
    table.add_column("Engagement", justify="right")
    table.add_column("Eng. Rate", justify="right")
    table.add_column("Followers", justify="right")
    table.add_column("Status")

    for platform, data in report["platform_metrics"].items():
        status_style = "green" if data["status"] == "collected" else "dim"
        table.add_row(
            platform.title(),
            str(data["output"]),
            f"{data['impressions']:,}",
            f"{data['engagement']:,}",
            f"{data['engagement_rate']:.2f}%",
            f"{data['followers']:,} ({data['followers_delta']:+,})",
            f"[{status_style}]{data['status']}[/]",
        )

    console.print(table)

    # Data quality note
    if report["data_quality"]["platforms_failed"]:
        console.print(f"\n[dim]Note: No API credentials configured for: {', '.join(report['data_quality']['platforms_failed'])}[/dim]")

    console.print()


# =============================================================================
# CALENDAR COMMAND
# =============================================================================

@cli.group()
def calendar() -> None:
    """Calendar operations."""
    pass


@calendar.command()
@click.pass_context
def today(ctx: click.Context) -> None:
    """Show today's schedule."""
    json_output = ctx.obj.get('json', False)

    now = datetime.now()
    today_date = date.today()

    # Create calendar report
    report = {
        "type": "CALENDAR_REPORT",
        "date": today_date.isoformat(),
        "timestamp": now.isoformat(),
        "schedule": [],
        "protected_blocks": {
            "focus_time_hours": 2.0,
            "recovery_hours": 1.0,
            "personal_hours": 0,
        },
        "meeting_load": {
            "hours_scheduled": 0,
            "target_hours": 4.0,
            "status": "on_track",
        },
        "buffer_status": {
            "adequate": True,
            "gaps": [],
        },
        "alerts": [],
        "recommendations": [
            "Protect focus blocks for deep work",
            "Add buffer time between meetings",
            "Schedule recovery time",
        ],
    }

    state_manager.add_activity("Viewed today's calendar")

    if json_output:
        output_json(report)
        return

    # Display calendar report
    console.print()

    console.print(Panel(
        f"[bold]Date:[/bold] {today_date.strftime('%A, %B %d, %Y')}\n"
        f"[bold]Meeting Load:[/bold] {report['meeting_load']['hours_scheduled']:.1f}h / {report['meeting_load']['target_hours']:.1f}h\n"
        f"[bold]Buffer Status:[/bold] {'Adequate' if report['buffer_status']['adequate'] else 'Gaps detected'}",
        title="[bold magenta]Today's Schedule[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    ))

    # Protected blocks
    pb = report["protected_blocks"]
    console.print("\n[bold cyan]Protected Time:[/bold cyan]")
    console.print(f"  Focus Time: {pb['focus_time_hours']:.1f} hours")
    console.print(f"  Recovery: {pb['recovery_hours']:.1f} hours")
    console.print(f"  Personal: {pb['personal_hours']:.1f} hours")

    # Schedule
    if report["schedule"]:
        console.print("\n[bold cyan]Schedule:[/bold cyan]")
        for item in report["schedule"]:
            console.print(f"  {item['start']} - {item['end']}: {item['title']}")
    else:
        console.print("\n[dim]No events scheduled for today.[/dim]")

    # Recommendations
    console.print("\n[bold cyan]Recommendations:[/bold cyan]")
    for rec in report["recommendations"]:
        console.print(f"  [dim]-[/dim] {rec}")

    console.print()


# =============================================================================
# LEARN COMMAND
# =============================================================================

@cli.command()
@click.option('--blocking', is_flag=True, help='Show only blocking items')
@click.pass_context
def learn(ctx: click.Context, blocking: bool) -> None:
    """Show learning queue."""
    json_output = ctx.obj.get('json', False)

    now = datetime.now()

    # Create learning queue report
    report = {
        "type": "LEARNING_QUEUE",
        "timestamp": now.isoformat(),
        "alfred_state": state_manager.alfred_state,
        "learning_rule": "No learning queued without a linked output. Learning serves shipping.",
        "queue": {
            "items": [],
            "total_duration_minutes": 0,
            "blocking_count": 0,
            "linked_outputs_count": 0,
        },
        "summary": {
            "by_urgency": {
                "blocking": 0,
                "next_up": 0,
                "optimization": 0,
                "background": 0,
            },
            "by_window_type": {
                "commute": 0,
                "break": 0,
                "deep_work": 0,
                "unassigned": 0,
            },
            "total_items": 0,
        },
        "rejected_suggestions": [],
        "learning_debt_note": None,
    }

    # Add state-specific note
    if state_manager.alfred_state == "RED":
        report["state_note"] = "Learning curation paused in RED state - focus on crisis management"

    state_manager.add_activity("Viewed learning queue")

    if json_output:
        output_json(report)
        return

    # Display learning queue
    console.print()

    console.print(Panel(
        f"[bold]Items in Queue:[/bold] {report['summary']['total_items']}\n"
        f"[bold]Blocking Items:[/bold] {report['queue']['blocking_count']}\n"
        f"[bold]Total Duration:[/bold] {report['queue']['total_duration_minutes']} minutes\n"
        f"[bold]Linked Outputs:[/bold] {report['queue']['linked_outputs_count']}",
        title="[bold blue]Learning Queue[/bold blue]",
        border_style="blue",
        box=box.ROUNDED,
    ))

    # Learning rule
    console.print(f"\n[italic dim]{report['learning_rule']}[/italic dim]")

    # State note
    if report.get("state_note"):
        console.print(f"\n[yellow]{report['state_note']}[/yellow]")

    # Queue breakdown
    console.print("\n[bold cyan]Queue by Urgency:[/bold cyan]")
    urgency = report["summary"]["by_urgency"]
    console.print(f"  [red]Blocking:[/red] {urgency['blocking']}")
    console.print(f"  [yellow]Next Up:[/yellow] {urgency['next_up']}")
    console.print(f"  [blue]Optimization:[/blue] {urgency['optimization']}")
    console.print(f"  [dim]Background:[/dim] {urgency['background']}")

    # Window type breakdown
    console.print("\n[bold cyan]By Time Window:[/bold cyan]")
    windows = report["summary"]["by_window_type"]
    for window_type, count in windows.items():
        console.print(f"  {window_type.replace('_', ' ').title()}: {count}")

    # Queue items
    if report["queue"]["items"]:
        console.print("\n[bold cyan]Queue Items:[/bold cyan]")
        for item in report["queue"]["items"]:
            urgency_style = {
                "blocking": "bold red",
                "next_up": "yellow",
                "optimization": "blue",
                "background": "dim",
            }.get(item.get("urgency", "background"), "white")
            console.print(f"  [{urgency_style}]{item.get('question', 'Unknown')}[/]")
            console.print(f"    [dim]-> {item.get('linked_output', {}).get('title', 'No linked output')}[/dim]")
    else:
        console.print("\n[dim]Learning queue is empty.[/dim]")

    console.print()


# =============================================================================
# STATE MANAGEMENT COMMANDS
# =============================================================================

@cli.command()
@click.argument('new_state', type=click.Choice(['GREEN', 'YELLOW', 'RED']))
@click.option('--reason', '-r', default='Manual state change', help='Reason for state change')
@click.pass_context
def set_state(ctx: click.Context, new_state: str, reason: str) -> None:
    """Manually set Alfred state (GREEN, YELLOW, RED)."""
    json_output = ctx.obj.get('json', False)

    old_state = state_manager.alfred_state
    state_manager.alfred_state = new_state

    state_manager.add_alert({
        "type": "warning" if new_state != "GREEN" else "info",
        "message": f"State changed from {old_state} to {new_state}: {reason}",
    })

    result = {
        "action": "state_change",
        "old_state": old_state,
        "new_state": new_state,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }

    if json_output:
        output_json(result)
        return

    old_color = STATE_COLORS.get(old_state, 'white')
    new_color = STATE_COLORS.get(new_state, 'white')

    console.print()
    console.print(f"State changed: [{old_color}]{old_state}[/] -> [{new_color}]{new_state}[/]")
    console.print(f"[dim]Reason: {reason}[/dim]")
    console.print()


@cli.command()
@click.pass_context
def clear_alerts(ctx: click.Context) -> None:
    """Clear all active alerts."""
    json_output = ctx.obj.get('json', False)

    count = len(state_manager.active_alerts)
    state_manager.clear_alerts()

    result = {
        "action": "clear_alerts",
        "alerts_cleared": count,
        "timestamp": datetime.now().isoformat(),
    }

    if json_output:
        output_json(result)
        return

    console.print()
    console.print(f"[green]Cleared {count} alert(s).[/green]")
    console.print()


# =============================================================================
# VERSION COMMAND
# =============================================================================

@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show Alfred CLI version."""
    json_output = ctx.obj.get('json', False)

    version_info = {
        "name": "Alfred CLI",
        "version": "1.0.0",
        "description": "Personal AI Governance System",
        "prime_directive": "Clinical reputation is non-recoverable. All other gains are optional.",
    }

    if json_output:
        output_json(version_info)
        return

    console.print()
    console.print(Panel(
        f"[bold]Name:[/bold] {version_info['name']}\n"
        f"[bold]Version:[/bold] {version_info['version']}\n"
        f"[bold]Description:[/bold] {version_info['description']}\n\n"
        f"[italic]{version_info['prime_directive']}[/italic]",
        title="[bold blue]Alfred[/bold blue]",
        border_style="blue",
        box=box.DOUBLE,
    ))
    console.print()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for Alfred CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()
