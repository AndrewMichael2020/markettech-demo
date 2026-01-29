from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from openai import OpenAI


# -----------------------------
# Contract: allowed operations
# -----------------------------

@dataclass(frozen=True)
class CleaningStep:
    op: str
    args: Dict[str, Any]


@dataclass(frozen=True)
class CleaningPlan:
    version: str
    steps: List[CleaningStep]
    notes: str = ""


def _now_iso() -> str:
    # DuckDB NOW() uses server time. For the workshop, we keep the judge anchored to runtime "now".
    import datetime as dt
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


ALLOWED_OPS = {
    "drop_conversions_where",  # args: {"where_sql": "revenue < 0"}
    "drop_orphan_conversions", # args: {"sessions_table": "raw_sessions", "conversions_table": "raw_conversions"}
    "fix_negative_revenue_abs",# args: {"revenue_col": "revenue"}
    "drop_future_sessions",    # args: {"ts_col": "ts"}
}


def summarize_quality(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """
    Returns counts for the same three checks used in the notebook.
    """
    checks = {
        "negative_revenue": int(con.execute("SELECT COUNT(*) FROM raw_conversions WHERE revenue < 0").fetchone()[0]),
        "orphaned_conversions": int(con.execute("""
            SELECT COUNT(*)
            FROM raw_conversions c
            LEFT JOIN raw_sessions s ON c.session_id = s.session_id
            WHERE s.session_id IS NULL
        """).fetchone()[0]),
        "future_session_timestamps": int(con.execute("SELECT COUNT(*) FROM raw_sessions WHERE CAST(ts AS TIMESTAMP) > NOW()").fetchone()[0]),
    }
    return checks


def apply_plan(
    sessions: pd.DataFrame,
    conversions: pd.DataFrame,
    channels: pd.DataFrame,
    plan: CleaningPlan,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Apply a cleaning plan deterministically using DuckDB as the execution engine.
    Returns cleaned tables + final quality summary.
    """
    con = duckdb.connect(database=":memory:")
    con.register("raw_sessions", sessions)
    con.register("raw_conversions", conversions)
    con.register("dim_channels", channels)

    for step in plan.steps:
        if step.op not in ALLOWED_OPS:
            raise ValueError(f"Plan contains disallowed op: {step.op}")

        if step.op == "drop_conversions_where":
            where_sql = str(step.args.get("where_sql", "")).strip()
            if not where_sql:
                raise ValueError("drop_conversions_where requires where_sql")
            con.execute(f"CREATE OR REPLACE TABLE raw_conversions AS SELECT * FROM raw_conversions WHERE NOT ({where_sql})")

        elif step.op == "drop_orphan_conversions":
            con.execute("""
                CREATE OR REPLACE TABLE raw_conversions AS
                SELECT c.*
                FROM raw_conversions c
                JOIN raw_sessions s ON c.session_id = s.session_id
            """)

        elif step.op == "fix_negative_revenue_abs":
            revenue_col = str(step.args.get("revenue_col", "revenue"))
            con.execute(f"""
                CREATE OR REPLACE TABLE raw_conversions AS
                SELECT
                    event_type,
                    session_id,
                    ts,
                    CASE WHEN {revenue_col} < 0 THEN ABS({revenue_col}) ELSE {revenue_col} END AS {revenue_col}
                FROM raw_conversions
            """)

        elif step.op == "drop_future_sessions":
            ts_col = str(step.args.get("ts_col", "ts"))
            con.execute(f"""
                CREATE OR REPLACE TABLE raw_sessions AS
                SELECT *
                FROM raw_sessions
                WHERE CAST({ts_col} AS TIMESTAMP) <= NOW()
            """)

    sess_clean = con.execute("SELECT * FROM raw_sessions").df()
    conv_clean = con.execute("SELECT * FROM raw_conversions").df()
    chan_clean = con.execute("SELECT * FROM dim_channels").df()
    final_checks = summarize_quality(con)

    return sess_clean, conv_clean, chan_clean, final_checks


def _parse_plan_json(obj: Dict[str, Any]) -> CleaningPlan:
    version = str(obj.get("version", "1.0"))
    notes = str(obj.get("notes", ""))
    steps_raw = obj.get("steps", [])
    if not isinstance(steps_raw, list):
        raise ValueError("Plan steps must be a list")

    steps: List[CleaningStep] = []
    for s in steps_raw:
        if not isinstance(s, dict):
            raise ValueError("Each step must be an object")
        op = str(s.get("op", "")).strip()
        args = s.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Step args must be an object")
        steps.append(CleaningStep(op=op, args=args))

    return CleaningPlan(version=version, steps=steps, notes=notes)


def propose_cleaning_plan(
    model: str,
    checks_before: Dict[str, Any],
    sample_sessions: pd.DataFrame,
    sample_conversions: pd.DataFrame,
    max_steps: int = 4,
) -> CleaningPlan:
    """
    Ask an OpenAI model for a cleaning plan, constrained to ALLOWED_OPS.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()

    system = (
        "You are a data cleaning agent for a workshop. "
        "You must output ONLY valid JSON, no markdown. "
        "You must use only the allowed ops. "
        "Goal: reduce all quality check counts to zero while minimizing data loss."
    )

    allowed_ops = sorted(list(ALLOWED_OPS))
    user = {
        "now": _now_iso(),
        "quality_checks_before": checks_before,
        "allowed_ops": allowed_ops,
        "max_steps": max_steps,
        "samples": {
            "raw_sessions_head": sample_sessions.head(10).to_dict(orient="records"),
            "raw_conversions_head": sample_conversions.head(10).to_dict(orient="records"),
        },
        "instruction": (
            "Return a JSON object: "
            "{version: string, notes: string, steps: [{op: string, args: object}, ...]}. "
            "Do not include any keys other than version, notes, steps. "
            "Each step.op must be one of allowed_ops."
        ),
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        # Ask for strict JSON output
        text={"format": {"type": "json_object"}},
    )

    raw = resp.output_text
    obj = json.loads(raw)
    return _parse_plan_json(obj)


def judge_cleaning_result(
    model: str,
    checks_before: Dict[str, Any],
    checks_after: Dict[str, Any],
    rows_before: Dict[str, int],
    rows_after: Dict[str, int],
    plan: CleaningPlan,
) -> Dict[str, Any]:
    """
    Ask an OpenAI model to act as an AI judge.
    Returns JSON verdict with: {verdict: "pass"|"fail", reasons: [...], next_action: "..."}.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()

    system = (
        "You are a strict data quality judge. "
        "Decide PASS only if all check counts are zero and the plan is reasonable "
        "(no unnecessary data loss). "
        "Output ONLY valid JSON (no markdown)."
    )

    payload = {
        "now": _now_iso(),
        "checks_before": checks_before,
        "checks_after": checks_after,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "plan": {
            "version": plan.version,
            "notes": plan.notes,
            "steps": [{"op": s.op, "args": s.args} for s in plan.steps],
        },
        "rules": {
            "must_zero_all_checks": True,
            "max_drop_fraction_soft_limit": 0.05,  # soft guideline for the judge
        },
        "instruction": 'Return JSON: {"verdict":"pass"|"fail","reasons":[...],"next_action":"..."}',
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload)},
        ],
        text={"format": {"type": "json_object"}},
    )
    return json.loads(resp.output_text)


def run_agentic_cleaning_loop(
    sessions: pd.DataFrame,
    conversions: pd.DataFrame,
    channels: pd.DataFrame,
    planner_model: str = "gpt-5.2-instant",
    judge_model: str = "gpt-5.2-thinking",
    max_iters: int = 2,
) -> Dict[str, Any]:
    """
    End-to-end "agentic" loop:
      1) measure quality
      2) model proposes plan
      3) we apply plan deterministically
      4) judge evaluates pass/fail
      5) optionally iterate once

    Returns dict with plan, judge verdict, cleaned tables, and checks.
    """
    # Snapshot before
    con0 = duckdb.connect(database=":memory:")
    con0.register("raw_sessions", sessions)
    con0.register("raw_conversions", conversions)
    con0.register("dim_channels", channels)

    checks_before = summarize_quality(con0)
    rows_before = {
        "raw_sessions": len(sessions),
        "raw_conversions": len(conversions),
    }

    plan: Optional[CleaningPlan] = None
    last_verdict: Optional[Dict[str, Any]] = None
    sess, conv, chan = sessions, conversions, channels
    checks_after: Dict[str, Any] = checks_before

    for i in range(max_iters):
        plan = propose_cleaning_plan(
            model=planner_model,
            checks_before=checks_after,
            sample_sessions=sess,
            sample_conversions=conv,
            max_steps=4,
        )
        sess2, conv2, chan2, checks2 = apply_plan(sess, conv, chan, plan)

        rows_after = {"raw_sessions": len(sess2), "raw_conversions": len(conv2)}
        verdict = judge_cleaning_result(
            model=judge_model,
            checks_before=checks_before,
            checks_after=checks2,
            rows_before=rows_before,
            rows_after=rows_after,
            plan=plan,
        )

        sess, conv, chan = sess2, conv2, chan2
        checks_after = checks2
        last_verdict = verdict

        if verdict.get("verdict") == "pass":
            break

    return {
        "checks_before": checks_before,
        "checks_after": checks_after,
        "rows_before": rows_before,
        "rows_after": {"raw_sessions": len(sess), "raw_conversions": len(conv)},
        "plan": {
            "version": plan.version if plan else None,
            "notes": plan.notes if plan else None,
            "steps": [{"op": s.op, "args": s.args} for s in (plan.steps if plan else [])],
        },
        "judge": last_verdict,
        "sessions_clean": sess,
        "conversions_clean": conv,
        "channels": chan,
    }
