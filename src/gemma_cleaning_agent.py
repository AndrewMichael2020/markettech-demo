from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import requests

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
    import datetime as dt

    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


ALLOWED_OPS = {
    "drop_negative_revenue",
    "drop_orphan_conversions",
    "drop_future_sessions",
}


def _df_sample_records_json_safe(df: pd.DataFrame, n: int = 10) -> List[Dict[str, Any]]:
    """Convert DataFrame sample to JSON-serializable records.

    Args:
        df: Source DataFrame.
        n: Number of records to sample.

    Returns:
        List of dictionaries representing DataFrame rows.
    """
    sample = df.head(n).copy()

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
    """Apply a cleaning plan deterministically using DuckDB.

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
            conv_cur = con.execute(
                """
                SELECT *
                FROM raw_conversions
                WHERE revenue >= 0 OR revenue IS NULL
                """
            ).df()

        elif step.op == "drop_orphan_conversions":
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

        con.close()

        new_sess_rows = len(sess_cur)
        new_conv_rows = len(conv_cur)
        if new_sess_rows > prev_sess_rows or new_conv_rows > prev_conv_rows:
            raise RuntimeError(
                "Cleaning step expanded row counts, which is not allowed. "
                f"op={step.op}, "
                f"sessions_before={prev_sess_rows}, sessions_after={new_sess_rows}, "
                f"conversions_before={prev_conv_rows}, conversions_after={new_conv_rows}"
            )

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


# -----------------------------
# Local Gemma llamafile client
# -----------------------------


def _gemma_api_base() -> str:
    base = os.getenv("GEMMA_API_BASE", "http://127.0.0.1:8080/v1")
    return base.rstrip("/")


def _gemma_model_name() -> str:
    return os.getenv("GEMMA_MODEL_NAME", "google_gemma-3-4b-it-Q6_K")


def _extract_first_json_object(text: str) -> str:
    """Best-effort extraction of the first JSON object from an LLM reply.

    Gemma may wrap JSON in prose or multiple code fences. This helper finds
    the first balanced {...} block and returns it as a string.
    """

    candidate = text.strip()

    # Fast path: already valid JSON
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        pass

    # Strip simple Markdown fences if present
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if "\n" in candidate:
            candidate = candidate.split("\n", 1)[1]

    # Scan for first balanced {...}
    s = candidate
    start = s.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text!r}")

    depth = 0
    end = None
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError(f"Unbalanced JSON braces in model output: {text!r}")

    obj_str = s[start:end].strip()
    # Validate
    json.loads(obj_str)
    return obj_str


def _post_chat(messages: List[Dict[str, Any]], max_tokens: int = 1024, temperature: float = 0.1) -> str:
    """Send chat request to local Gemma llamafile.

    Args:
        messages: List of message dictionaries with role and content.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        str: Model response content.

    Raises:
        requests.HTTPError: If API request fails.
        RuntimeError: If response format is unexpected.
    """
    try:
        url = f"{_gemma_api_base()}/chat/completions"
        payload = {
            "model": _gemma_model_name(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        logger.debug(f"Sending request to Gemma API at {url}")
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"]
    except KeyError as exc:
        logger.error(f"Unexpected Gemma response format: {data}")
        raise RuntimeError(f"Unexpected Gemma response format: {data}") from exc
    except requests.RequestException as exc:
        logger.error(f"Gemma API request failed: {exc}")
        raise


def propose_cleaning_plan_gemma(
    checks_before: Dict[str, Any],
    sample_sessions: pd.DataFrame,
    sample_conversions: pd.DataFrame,
    max_steps: int = 4,
) -> CleaningPlan:
    """Ask local Gemma model for a cleaning plan.

    Args:
        checks_before: Quality check results before cleaning.
        sample_sessions: Sample of session data for context.
        sample_conversions: Sample of conversion data for context.
        max_steps: Maximum number of cleaning steps to propose.

    Returns:
        CleaningPlan: Proposed cleaning plan with steps.

    Raises:
        RuntimeError: If Gemma API is not available or returns invalid response.
        ValueError: If response cannot be parsed as valid plan.
    """
    logger.info("Requesting cleaning plan from local Gemma model")
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, default=str)},
    ]

    raw = _post_chat(messages, max_tokens=1024, temperature=0.0)

    obj_str = _extract_first_json_object(raw)
    obj = json.loads(obj_str)
    return _parse_plan_json(obj)


def judge_cleaning_result_gemma(
    checks_before: Dict[str, Any],
    checks_after: Dict[str, Any],
    rows_before: Dict[str, int],
    rows_after: Dict[str, int],
    plan: CleaningPlan,
) -> Dict[str, Any]:
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]

    raw = _post_chat(messages, max_tokens=512, temperature=0.0)
    obj_str = _extract_first_json_object(raw)
    return json.loads(obj_str)


def run_agentic_cleaning_loop_gemma(
    sessions: pd.DataFrame,
    conversions: pd.DataFrame,
    channels: pd.DataFrame,
    max_iters: int = 2,
) -> Dict[str, Any]:
    """End-to-end agentic cleaning loop using local Gemma llamafile.

    Assumes a llamafile HTTP server is running and reachable at GEMMA_API_BASE.
    """

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
        print(f"[GEMMA CLEANING] Iteration {i+1} - checks_before: {checks_after}")

        plan = propose_cleaning_plan_gemma(
            checks_before=checks_after,
            sample_sessions=sess,
            sample_conversions=conv,
            max_steps=4,
        )
        sess2, conv2, chan2, checks2 = apply_plan(sess, conv, chan, plan)

        rows_after = {"raw_sessions": len(sess2), "raw_conversions": len(conv2)}
        verdict = judge_cleaning_result_gemma(
            checks_before=checks_before,
            checks_after=checks2,
            rows_before=rows_before,
            rows_after=rows_after,
            plan=plan,
        )

        print(f"[GEMMA CLEANING] Iteration {i+1} - checks_after: {checks2}, verdict: {verdict.get('verdict')}")

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
