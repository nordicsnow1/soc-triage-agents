"""Offline tests for the evaluation scoring and report. No AWS, no cost."""
from fake_model import ScriptedModel, text_step, tool_step
from triage.evaluate import run_eval, score, summarize, write_report
from triage.l1_agent import TriageRun
from triage.scenarios import load_scenarios
from triage.schema import TriageResult

SCENARIOS = {s.id: s for s in load_scenarios()}
TOR = SCENARIOS["01-log001-tor-stoplogging"]
INJECTION = SCENARIOS["10-log001-prompt-injection"]


def answer(scenario, **changes):
    base = {"alert_id": scenario.alert["id"], "rule_id": scenario.rule_id,
            "verdict": scenario.expected.verdict.value, "confidence": 0.9,
            "evidence_event_ids": list(scenario.expected.must_cite), "reasoning": "test"}
    return {**base, **changes}


def fake_run(scenario, tools=None, **changes):
    calls = [{"tool": t, "args": {}} for t in (tools if tools is not None else scenario.expected.must_call)]
    return TriageRun(TriageResult(**answer(scenario, **changes)), calls,
                     {"inputTokens": 1000, "outputTokens": 200, "totalTokens": 1200})


def test_correct_run_scores_clean():
    r = score(TOR, 1, run=fake_run(TOR))
    assert r["verdict_ok"] and not r["tools_missing"] and not r["evidence_missing"] and not r["invented_ids"]


def test_wrong_verdict_missing_tool_missing_evidence():
    r = score(TOR, 1, run=fake_run(TOR, tools=["get_actor_history"], verdict="benign",
                                    evidence_event_ids=[TOR.expected.must_cite[0]]))
    assert not r["verdict_ok"]
    assert r["tools_missing"] == ["check_ip_reputation", "get_related_events"]
    assert r["evidence_missing"] == [TOR.expected.must_cite[1]]


def test_invented_event_id_is_detected():
    r = score(TOR, 1, run=fake_run(TOR, evidence_event_ids=[*TOR.expected.must_cite, "made-up-id"]))
    assert r["verdict_ok"] and r["invented_ids"] == ["made-up-id"]


def test_failed_run_counts_as_failure():
    r = score(TOR, 1, error=RuntimeError("throttled"))
    assert not r["verdict_ok"] and r["error"].startswith("RuntimeError")


def test_summary_gate():
    good = [score(TOR, i, run=fake_run(TOR)) for i in (1, 2)] + [score(INJECTION, 1, run=fake_run(INJECTION))]
    s = summarize(good)
    assert s["verdict_accuracy"] == 1.0 and s["gate_failures"] == []
    assert s["estimated_cost_usd"] == round(3 * (1000 / 1e6 * 1.0 + 200 / 1e6 * 5.0), 4)

    fooled = good + [score(INJECTION, 2, run=fake_run(INJECTION, verdict="benign"))]
    assert any("prompt injection" in f for f in summarize(fooled)["gate_failures"])

    invented = [score(TOR, 1, run=fake_run(TOR, evidence_event_ids=["x"]))]
    assert any("invented" in f for f in summarize(invented)["gate_failures"])


def test_run_eval_and_report_end_to_end(tmp_path):
    """The full loop with the scripted model: one correct and one broken scenario."""
    def factory(scenario):
        if scenario is TOR:
            return ScriptedModel([*(tool_step(t, {"actor": "dev-alice"} if t != "check_ip_reputation"
                                                 else {"ip": "203.0.113.66"})
                                    for t in scenario.expected.must_call),
                                  tool_step("TriageResult", answer(scenario))])
        return ScriptedModel([text_step("no answer")])  # never gives a valid answer

    records = run_eval([TOR, INJECTION], runs=1, model_factory=factory, progress=lambda _: None)
    assert [r["verdict_ok"] for r in records] == [True, False]
    assert records[1]["error"].startswith("StructuredOutputException")

    md = write_report(records, out_dir=tmp_path, model_id="scripted-fake")
    text = md.read_text(encoding="utf-8")
    assert "Verdict accuracy | 50%" in text and "Gate:** FAIL" in text
    assert md.with_suffix(".json").exists()
