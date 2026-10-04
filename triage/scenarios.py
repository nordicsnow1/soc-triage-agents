"""Labeled triage scenarios: an alert, the context the tools return, and the correct answer."""
import json
from pathlib import Path

from pydantic import BaseModel, Field

from triage.schema import Verdict

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"

# Tools the L1 agent will have. Scenarios may only require these.
KNOWN_TOOLS = {
    "get_rule_playbook",
    "get_actor_history",
    "get_related_events",
    "check_ip_reputation",
    "is_aws_ip",
}


class ActorHistory(BaseModel):
    actor: str
    is_automation: bool
    usual_source_ips: list[str]
    usual_hours_utc: str = Field(description="e.g. '08-18'")
    typical_actions: list[str]
    notes: str = ""


class RelatedEvent(BaseModel):
    eventID: str
    eventTime: str
    eventSource: str
    eventName: str
    sourceIPAddress: str
    errorCode: str | None = None
    summary: str = ""


class IpReputation(BaseModel):
    abuse_score: int = Field(ge=0, le=100)
    country: str
    isp: str
    is_tor_exit: bool = False


class Context(BaseModel):
    actor_history: ActorHistory
    related_events: list[RelatedEvent]
    ip_reputation: dict[str, IpReputation] = Field(default_factory=dict)


class Expected(BaseModel):
    verdict: Verdict
    must_cite: list[str] = Field(min_length=1, description="eventIDs the verdict must cite")
    must_call: list[str] = Field(min_length=1, description="tools the agent must call")
    rationale: str


class Scenario(BaseModel):
    id: str
    rule_id: str = Field(pattern=r"^[A-Z]+-\d{3}$")
    description: str
    alert: dict
    context: Context
    expected: Expected


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[Scenario]:
    return [Scenario(**json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(directory.glob("*.json"))]
