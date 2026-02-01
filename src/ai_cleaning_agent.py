from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
    "drop_negative_revenue",   # drop rows where revenue < 0
    "drop_orphan_conversions", # drop conversions whose session_id is missing in raw_sessions
    "drop_future_sessions",    # drop sessions with future timestamps
}


def _df_sample_records_json_safe(df: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    """Return up to n rows as JSON-serializable records.

    Pandas Timestamps and other non-JSON-native types are converted to strings so
    they can be embedded safely in the LLM prompt payload.
    """
    sample = df.head(n).copy()

    # Convert any datetime-like columns (including tz-aware) to ISO-ish strings.
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]) or pd.api.types.is_datetime64tz_dtype(sample[col]):
            sample[col] = sample[col].astype(str)

    return sample.to_dict(orient="records")


def summarize_quality(con: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """Return quality check counts for data validation.

    Checks for:
      - Negative revenue in conversions
      - Orphaned conversions (no matching session)
      - Invalid session timestamps
      - Future session timestamps

    Args:
        con: DuckDB connection with registered tables.

    Returns:
        Dict[str, Any]: Dictionary of check names to count values.
    """
    checks = {
        "negative_revenue": int(
            con.execute("SELECT COUNT(*) FROM raw_conversions WHERE revenue < 0").fetchone()[0]
        ),
        "orphaned_conversions": int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM raw_conversions c
                LEFT JOIN raw_sessions s ON c.session_id = s.session_id
                WHERE s.session_id IS NULL
                """
            ).fetchone()[0]
        ),
        # Any ts values that cannot be parsed into a TIMESTAMP are counted as invalid
        "invalid_session_timestamps": int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM raw_sessions
                WHERE ts IS NOT NULL
                  AND TRY_CAST(ts AS TIMESTAMP) IS NULL
                """
            ).fetchone()[0]
        ),
        # Future timestamps are computed using TRY_CAST and comparing against NOW() cast to TIMESTAMP
        "future_session_timestamps": int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM raw_sessions
                WHERE TRY_CAST(ts AS TIMESTAMP) > CAST(NOW() AS TIMESTAMP)
                """
            ).fetchone()[0]
        ),
    }
    return checks


def apply_plan(
    sessions: pd.DataFrame,
    conversions: pd.DataFrame,
    channels: pd.DataFrame,
    plan: CleaningPlan,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Apply a cleaning plan deterministically using DuckDB, stage-by-stage.

    Instead of mutating registered views in-place (which can be surprising in DuckDB
    when views come from pandas DataFrames), we treat each step as:

      pandas DataFrame -> DuckDB SELECT -> new pandas DataFrame

    This keeps behavior explicit and makes it easy to reason about row deltas.

    Args:
        sessions: Session data DataFrame.
        conversions: Conversion data DataFrame.
        channels: Channel dimension DataFrame.
        plan: Cleaning plan to execute.

    Returns:
        Tuple containing:
          - Cleaned sessions DataFrame
          - Cleaned conversions DataFrame
          - Channels DataFrame (unchanged)
          - Final quality check summary

    Raises:
        ValueError: If plan contains disallowed operations.
        RuntimeError: If cleaning steps increase row counts.
    """

    sess_cur = sessions.copy()
    conv_cur = conversions.copy()
    chan_cur = channels.copy()

    for step in plan.steps:
        if step.op not in ALLOWED_OPS:
            raise ValueError(f"Plan contains disallowed op: {step.op}")

        prev_sess_rows = len(sess_cur)
        prev_conv_rows = len(conv_cur)

        con = duckdb.connect(database=":memory:")
        con.register("raw_sessions", sess_cur)
        con.register("raw_conversions", conv_cur)
        con.register("dim_channels", chan_cur)

        if step.op == "drop_negative_revenue":
            # Drop only rows that have negative revenue; original rows with
            # non-negative revenue remain, so duplicates introduced by corruption
            # can be fully removed.
            conv_cur = con.execute(
                """
                SELECT *
                FROM raw_conversions
                WHERE revenue >= 0 OR revenue IS NULL
                """
            ).df()

        elif step.op == "drop_orphan_conversions":
            # Keep only conversions whose session_id exists in raw_sessions, but
            # do so without duplicating rows even if session_ids collide.
            conv_cur = con.execute(
                """
                SELECT c.*
                FROM raw_conversions c
                WHERE EXISTS (
                    SELECT 1 FROM raw_sessions s WHERE s.session_id = c.session_id
                )
                """
            ).df()

        elif step.op == "drop_future_sessions":
            ts_col = str(step.args.get("ts_col", "ts"))
            sess_cur = con.execute(
                f"""
                SELECT *
                FROM raw_sessions
                WHERE TRY_CAST({ts_col} AS TIMESTAMP) <= CAST(NOW() AS TIMESTAMP)
                """
            ).df()

        # Close per-step connection
        con.close()

        # Hard guard: cleaning steps must not increase row counts.
        new_sess_rows = len(sess_cur)
        new_conv_rows = len(conv_cur)
        if new_sess_rows > prev_sess_rows or new_conv_rows > prev_conv_rows:
            raise RuntimeError(
                "Cleaning step expanded row counts, which is not allowed. "
                f"op={step.op}, "
                f"sessions_before={prev_sess_rows}, sessions_after={new_sess_rows}, "
                f"conversions_before={prev_conv_rows}, conversions_after={new_conv_rows}"
            )

    # Final quality summary on the cleaned tables
    con_final = duckdb.connect(database=":memory:")
    con_final.register("raw_sessions", sess_cur)
    con_final.register("raw_conversions", conv_cur)
    con_final.register("dim_channels", chan_cur)
    final_checks = summarize_quality(con_final)
    con_final.close()

    return sess_cur, conv_cur, chan_cur, final_checks


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
    """Ask an OpenAI model for a cleaning plan, constrained to ALLOWED_OPS.

    Args:
        model: OpenAI model name to use for planning.
        checks_before: Quality check results before cleaning.
        sample_sessions: Sample of session data for context.
        sample_conversions: Sample of conversion data for context.
        max_steps: Maximum number of cleaning steps to propose.

    Returns:
        CleaningPlan: Proposed cleaning plan with steps.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
        Exception: If API call fails or response is invalid.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        logger.info(f"Requesting cleaning plan from OpenAI model: {model}")
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
                "raw_sessions_head": _df_sample_records_json_safe(sample_sessions, n=10),
                "raw_conversions_head": _df_sample_records_json_safe(sample_conversions, n=10),
            },
            "instruction": (
                "Return a JSON object: "
                "{version: string, notes: string, steps: [{op: string, args: object}, ...]}. "
                "Do not include any keys other than version, notes, steps. "
                "Each step.op must be one of allowed_ops."
            ),
        }

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                # Use default=str so any remaining non-JSON-native values (e.g. Timestamps)
                # are safely converted to strings.
                {"role": "user", "content": json.dumps(user, default=str)},
            ],
            # Ask for strict JSON output
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content
        obj = json.loads(raw)
        plan = _parse_plan_json(obj)
        logger.info(f"Successfully generated cleaning plan with {len(plan.steps)} steps")
        return plan

    except Exception as e:
        logger.error(f"Failed to propose cleaning plan: {type(e).__name__}: {e}")
        raise


def judge_cleaning_result(
    model: str,
    checks_before: Dict[str, Any],
    checks_after: Dict[str, Any],
    rows_before: Dict[str, int],
    rows_after: Dict[str, int],
    plan: CleaningPlan,
) -> Dict[str, Any]:
    """Ask an OpenAI model to act as an AI judge for cleaning results.

    Args:
        model: OpenAI model name to use for judging.
        checks_before: Quality check results before cleaning.
        checks_after: Quality check results after cleaning.
        rows_before: Row counts before cleaning.
        rows_after: Row counts after cleaning.
        plan: The cleaning plan that was executed.

    Returns:
        Dict[str, Any]: Verdict with keys: verdict, reasons, next_action.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
        Exception: If API call fails or response is invalid.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        logger.info(f"Requesting judge verdict from OpenAI model: {model}")
        client = OpenAI()

        system = (
            "You are a strict data quality judge. "
            "Decide PASS if and only if: (a) all quality check counts after cleaning are zero, "
            "(b) no table's row count increased compared to before cleaning, and (c) the "
            "fraction of rows dropped is not obviously excessive relative to the starting data. "
            "If these conditions hold you MUST return verdict=\"pass\". Otherwise return "
            "verdict=\"fail\" with clear reasons. Output ONLY valid JSON (no markdown)."
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
                "max_drop_fraction_soft_limit": 0.05,
                "non_expansion_enforced_in_code": True,
                "instructional_note": (
                    "If all checks_after are zero and no rows increased, and the fraction "
                    "of dropped rows in any table is <= max_drop_fraction_soft_limit, you "
                    "should treat the cleaning as successful and set verdict to 'pass'."
                ),
            },
            "instruction": 'Return JSON: {"verdict":"pass"|"fail","reasons":[...],"next_action":"..."}',
        }

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        logger.info(f"Received judge verdict: {result.get('verdict', 'unknown')}")
        return result

    except Exception as e:
        logger.error(f"Failed to get judge verdict: {type(e).__name__}: {e}")
        raise


def run_agentic_cleaning_loop(
    sessions: pd.DataFrame,
    conversions: pd.DataFrame,
    channels: pd.DataFrame,
    planner_model: str = "gpt-5.2-chat-latest",
    judge_model: str = "gpt-5.2",
    max_iters: int = 2,
) -> Dict[str, Any]:
    """Execute end-to-end agentic cleaning loop with AI planning and judging.

    This function:
      1) Measures initial quality
      2) Uses AI model to propose cleaning plan
      3) Applies plan deterministically
      4) Uses AI judge to evaluate pass/fail
      5) Optionally iterates once

    Args:
        sessions: Session data DataFrame.
        conversions: Conversion data DataFrame.
        channels: Channel dimension DataFrame.
        planner_model: OpenAI model for generating cleaning plans.
        judge_model: OpenAI model for judging results.
        max_iters: Maximum number of cleaning iterations.

    Returns:
        Dict[str, Any]: Results including plan, judge verdict, cleaned tables, and checks.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
        Exception: If cleaning process fails.
    """
    logger.info("Starting agentic cleaning loop")
    logger.info(f"Initial data: {len(sessions)} sessions, {len(conversions)} conversions")

    try:
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
        logger.info(f"Initial quality checks: {checks_before}")

        plan: Optional[CleaningPlan] = None
        last_verdict: Optional[Dict[str, Any]] = None
        sess, conv, chan = sessions, conversions, channels
        checks_after: Dict[str, Any] = checks_before

        for i in range(max_iters):
            logger.info(f"Iteration {i+1}/{max_iters} - current checks: {checks_after}")

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

            logger.info(f"Iteration {i+1} complete - checks: {checks2}, verdict: {verdict.get('verdict')}")

            sess, conv, chan = sess2, conv2, chan2
            checks_after = checks2
            last_verdict = verdict

            if verdict.get("verdict") == "pass":
                logger.info("Cleaning passed, exiting loop")
                break

        result = {
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

        logger.info("Agentic cleaning loop completed successfully")
        return result

    except Exception as e:
        logger.error(f"Agentic cleaning loop failed: {type(e).__name__}: {e}")
        raise
