# SOC triage agents

L1 / L2 / lead agents that triage alerts from
[aws-cloudtrail-detection-rules](https://github.com/nordicsnow1/aws-cloudtrail-detection-rules).

- `triage/schema.py` — the fixed verdict format every agent answer must follow.
- `tests/` — run with `python -m pytest`.

Status: step 1 (verdict format). The agent itself comes next.
