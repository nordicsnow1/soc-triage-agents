"""Read-only triage tools.

This version answers from a scenario's mock context, so tests need no AWS and
cost nothing. A later version will read real CloudTrail and reputation APIs
behind the same tool names and docstrings.

Every call is recorded, so the evaluation can check which tools the agent used.
"""
import ipaddress
import json
from pathlib import Path

from strands import tool

from triage.scenarios import Context

PLAYBOOK_FILE = Path(__file__).resolve().parent.parent / "data" / "playbooks.json"
DOC_RANGES = ("192.0.2.", "198.51.100.", "203.0.113.")  # RFC 5737 example ranges, treated as public


def load_playbooks() -> dict:
    return json.loads(PLAYBOOK_FILE.read_text(encoding="utf-8"))["playbooks"]


def _is_public_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global or value.startswith(DOC_RANGES)
    except ValueError:
        return False


def build_tools(context: Context, playbooks: dict, calls: list, aws_ips: frozenset = frozenset()):
    """Return the tool functions for one triage run; each call is appended to `calls`."""

    def record(name: str, **args):
        calls.append({"tool": name, "args": args})

    @tool
    def get_rule_playbook(rule_id: str) -> dict:
        """Get the detection rule's description, known false positives, tuning guidance and response steps.

        Use it first, to learn what the rule is meant to catch and what usually causes benign alerts.

        Args:
            rule_id: Detection rule ID, for example "LOG-001".
        """
        record("get_rule_playbook", rule_id=rule_id)
        return playbooks.get(rule_id, {"error": f"no playbook for {rule_id}"})

    @tool
    def get_actor_history(actor: str) -> dict:
        """Get the normal behaviour of an actor: usual source IPs, usual hours (UTC), typical actions,
        whether it is automation, and notes.

        Use it to decide if the alert's activity is normal for this actor.

        Args:
            actor: User name, role name, or "root" (from detail.userIdentity).
        """
        record("get_actor_history", actor=actor)
        history = context.actor_history
        if actor.lower() in history.actor.lower() or history.actor.lower() in actor.lower():
            return history.model_dump()
        return {"error": f"no history for actor {actor}"}

    @tool
    def get_related_events(actor: str) -> dict:
        """Get other CloudTrail events by the same actor shortly before and after the alert,
        including their eventIDs, times, source IPs and error codes.

        Use it to see what happened around the alert (preparation, follow-up actions, restarts).

        Args:
            actor: User name, role name, or "root" (from detail.userIdentity).
        """
        record("get_related_events", actor=actor)
        return {"events": [e.model_dump(exclude_none=True) for e in context.related_events]}

    @tool
    def check_ip_reputation(ip: str) -> dict:
        """Look up the reputation of a public IP address: abuse score 0-100, country, ISP, Tor exit.

        Use it for external sourceIPAddress values. Do NOT use it for AWS service names
        (for example "cloudtrail.amazonaws.com") or private IPs.

        Args:
            ip: A public IPv4 or IPv6 address.
        """
        record("check_ip_reputation", ip=ip)
        if not _is_public_ip(ip):
            return {"error": f"{ip} is not a public IP address; reputation does not apply"}
        rep = context.ip_reputation.get(ip)
        return rep.model_dump() if rep else {"error": f"no reputation data for {ip}"}

    @tool
    def is_aws_ip(value: str) -> dict:
        """Check whether a sourceIPAddress belongs to AWS itself (an AWS service name or an AWS IP range).

        Use it when the source might be an AWS service acting on your behalf.

        Args:
            value: The sourceIPAddress value from the event.
        """
        record("is_aws_ip", value=value)
        return {"value": value,
                "is_aws_service_name": value.endswith(".amazonaws.com"),
                "is_aws_ip_range": value in aws_ips}

    return [get_rule_playbook, get_actor_history, get_related_events, check_ip_reputation, is_aws_ip]
