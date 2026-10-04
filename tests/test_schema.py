import pytest
from pydantic import ValidationError

from triage.schema import TriageResult, Verdict

VALID = {
    "alert_id": "alert-001",
    "rule_id": "LOG-001",
    "verdict": "true_positive",
    "confidence": 0.9,
    "evidence_event_ids": ["11111111-2222-3333-4444-000000000001"],
    "reasoning": "StopLogging by an unknown user from a new IP address.",
    "next_steps": ["Restart logging", "Disable the user's access keys"],
}


def test_valid_result_parses():
    result = TriageResult(**VALID)
    assert result.verdict is Verdict.TRUE_POSITIVE


@pytest.mark.parametrize("field, bad_value", [
    ("verdict", "maybe"),
    ("confidence", 1.5),
    ("confidence", -0.1),
    ("evidence_event_ids", []),
    ("rule_id", "log-1"),
    ("reasoning", ""),
])
def test_invalid_values_are_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        TriageResult(**{**VALID, field: bad_value})


def test_missing_evidence_is_rejected():
    data = dict(VALID)
    del data["evidence_event_ids"]
    with pytest.raises(ValidationError):
        TriageResult(**data)
