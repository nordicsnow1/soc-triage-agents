"""Fixed output format for the L1 triage agent.

Every verdict must parse into TriageResult. The evaluation compares these
fields with the labeled scenarios, so the format must not drift.
"""
from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    TRUE_POSITIVE = "true_positive"    # malicious or unauthorized activity
    BENIGN = "benign"                  # rule fired correctly, activity was authorized
    FALSE_POSITIVE = "false_positive"  # rule fired on activity outside its intent
    ESCALATE = "escalate"              # not enough evidence; send to L2


class TriageResult(BaseModel):
    alert_id: str = Field(description="ID of the alert that was triaged")
    rule_id: str = Field(pattern=r"^[A-Z]+-\d{3}$", description="Detection rule ID, e.g. LOG-001")
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 = guess, 1.0 = certain")
    evidence_event_ids: list[str] = Field(
        min_length=1, description="CloudTrail eventIDs the verdict relies on")
    reasoning: str = Field(min_length=1, max_length=1500, description="Short explanation")
    next_steps: list[str] = Field(default_factory=list, description="Recommended actions")
