"""Check that every labeled scenario is complete and consistent."""
import ipaddress

import pytest

from triage.scenarios import KNOWN_TOOLS, load_scenarios

SCENARIOS = load_scenarios()
IDS = [s.id for s in SCENARIOS]


def is_public_ip(value):
    try:
        return ipaddress.ip_address(value).is_global or value.startswith(("192.0.2.", "198.51.100.", "203.0.113."))
    except ValueError:
        return False  # e.g. an AWS service name such as "cloudtrail.amazonaws.com"


def test_scenarios_exist_and_ids_are_unique():
    assert len(SCENARIOS) >= 10
    assert len(IDS) == len(set(IDS))


def test_every_verdict_is_covered():
    verdicts = {s.expected.verdict.value for s in SCENARIOS}
    assert verdicts == {"true_positive", "benign", "false_positive", "escalate"}


@pytest.mark.parametrize("s", SCENARIOS, ids=IDS)
def test_alert_has_cloudtrail_detail(s):
    detail = s.alert["detail"]
    for field in ("eventID", "eventName", "eventSource", "sourceIPAddress", "userIdentity"):
        assert field in detail, field


@pytest.mark.parametrize("s", SCENARIOS, ids=IDS)
def test_cited_events_exist(s):
    known = {s.alert["detail"]["eventID"]} | {e.eventID for e in s.context.related_events}
    missing = set(s.expected.must_cite) - known
    assert not missing, "must_cite lists unknown eventIDs: " + str(missing)


@pytest.mark.parametrize("s", SCENARIOS, ids=IDS)
def test_required_tools_exist(s):
    unknown = set(s.expected.must_call) - KNOWN_TOOLS
    assert not unknown, "unknown tools: " + str(unknown)


@pytest.mark.parametrize("s", SCENARIOS, ids=IDS)
def test_reputation_data_covers_public_ips(s):
    """If the agent must check IP reputation, the mock must have data for every public IP."""
    if "check_ip_reputation" not in s.expected.must_call:
        pytest.skip("reputation not required")
    ips = {s.alert["detail"]["sourceIPAddress"]} | {e.sourceIPAddress for e in s.context.related_events}
    missing = {ip for ip in ips if is_public_ip(ip)} - set(s.context.ip_reputation)
    assert not missing, "no reputation data for: " + str(missing)
