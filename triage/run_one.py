"""Run the real L1 agent (Bedrock) on one scenario and compare with the expected answer.

Usage:  python -m triage.run_one 01-log001-tor-stoplogging
Costs one short Bedrock conversation (Claude Haiku 4.5), well under one cent.
"""
import json
import sys

from triage.l1_agent import MODEL_ID, run_triage
from triage.scenarios import load_scenarios


def main(scenario_id: str) -> None:
    scenarios = {s.id: s for s in load_scenarios()}
    if scenario_id not in scenarios:
        sys.exit("unknown scenario. Choose one of:\n  " + "\n  ".join(scenarios))
    scenario = scenarios[scenario_id]

    print(f"Model: {MODEL_ID}\nScenario: {scenario.id}\n")
    result, calls = run_triage(scenario)

    print("Tools called:", [c["tool"] for c in calls])
    print("Result:")
    print(json.dumps(result.model_dump(mode="json"), indent=2))

    exp = scenario.expected
    called = {c["tool"] for c in calls}
    print("\nExpected verdict:", exp.verdict.value, "->", "OK" if result.verdict == exp.verdict else "WRONG")
    print("Missing required tools:", sorted(set(exp.must_call) - called) or "none")
    print("Missing required evidence:", sorted(set(exp.must_cite) - set(result.evidence_event_ids)) or "none")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "01-log001-tor-stoplogging")
