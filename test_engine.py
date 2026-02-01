from __future__ import annotations

import datetime as dt
import os
from unittest.mock import Mock, patch

import duckdb
import pandas as pd
import pytest

from markettech_workshop import generate_stream, build_semantic_view
from ai_cleaning_agent import (
    run_agentic_cleaning_loop,
    summarize_quality,
    apply_plan,
    CleaningPlan,
    CleaningStep,
)


def test_generate_stream_is_deterministic() -> None:
    """Test that generate_stream produces deterministic results."""
    s1, c1, ch1 = generate_stream(days=14, start_date=dt.date(2025, 9, 1))
    s2, c2, ch2 = generate_stream(days=14, start_date=dt.date(2025, 9, 1))

    # Same shapes
    assert len(s1) == len(s2)
    assert len(c1) == len(c2)
    assert len(ch1) == len(ch2)

    # Same heads for a quick deterministic check
    assert s1.head(5).to_dict() == s2.head(5).to_dict()
    assert c1.head(5).to_dict() == c2.head(5).to_dict()
    assert ch1.to_dict() == ch2.to_dict()


def test_metric_contract_changes_counts() -> None:
    """Test that changing the attribution window affects conversion counts."""
    df_sess, df_conv, df_chan = generate_stream(days=30, start_date=dt.date(2025, 9, 1))

    con = duckdb.connect(database=":memory:")
    con.register("raw_sessions", df_sess)
    con.register("raw_conversions", df_conv)
    con.register("dim_channels", df_chan)

    build_semantic_view(con, window_days=7)
    trusted_7 = int(con.execute("SELECT SUM(is_attributed) FROM f_attribution").fetchone()[0])

    build_semantic_view(con, window_days=30)
    trusted_30 = int(con.execute("SELECT SUM(is_attributed) FROM f_attribution").fetchone()[0])

    # A wider attribution window should never decrease attributed conversions
    assert trusted_30 >= trusted_7


def test_quality_checks_default_data_passes() -> None:
    """Test that default generated data passes all quality checks."""
    df_sess, df_conv, df_chan = generate_stream(days=14, start_date=dt.date(2025, 9, 1))

    con = duckdb.connect(database=":memory:")
    con.register("raw_sessions", df_sess)
    con.register("raw_conversions", df_conv)
    con.register("dim_channels", df_chan)

    neg_rev = int(con.execute("SELECT COUNT(*) FROM raw_conversions WHERE revenue < 0").fetchone()[0])
    orphan = int(con.execute("""
        SELECT COUNT(*)
        FROM raw_conversions c
        LEFT JOIN raw_sessions s ON c.session_id = s.session_id
        WHERE s.session_id IS NULL
    """).fetchone()[0])
    future = int(con.execute("SELECT COUNT(*) FROM raw_sessions WHERE CAST(ts AS TIMESTAMP) > NOW()").fetchone()[0])

    assert neg_rev == 0
    assert orphan == 0
    assert future == 0


def test_agentic_cleaning_requires_openai_key() -> None:
    """Document that the AI cleaning loop depends on OPENAI_API_KEY.

    The workshop's agentic cleaning uses OpenAI. In automated tests we avoid
    hitting the external API and instead assert that, when the key is missing,
    a clear RuntimeError is raised.
    """
    # Prepare a small deterministic dataset
    df_sess, df_conv, df_chan = generate_stream(days=7, start_date=dt.date(2025, 9, 1))

    # Temporarily clear any OPENAI_API_KEY so the behavior is consistent in CI
    prev_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(RuntimeError):
            run_agentic_cleaning_loop(
                sessions=df_sess,
                conversions=df_conv,
                channels=df_chan,
                planner_model="gpt-5.2-instant",
                judge_model="gpt-5.2-thinking",
                max_iters=1,
            )
    finally:
        if prev_key is not None:
            os.environ["OPENAI_API_KEY"] = prev_key


def test_summarize_quality() -> None:
    """Test the summarize_quality function with known data."""
    df_sess, df_conv, df_chan = generate_stream(days=7, start_date=dt.date(2025, 9, 1))

    con = duckdb.connect(database=":memory:")
    con.register("raw_sessions", df_sess)
    con.register("raw_conversions", df_conv)
    con.register("dim_channels", df_chan)

    checks = summarize_quality(con)

    assert isinstance(checks, dict)
    assert "negative_revenue" in checks
    assert "orphaned_conversions" in checks
    assert "invalid_session_timestamps" in checks
    assert "future_session_timestamps" in checks

    # Clean data should have zero issues
    assert checks["negative_revenue"] == 0
    assert checks["orphaned_conversions"] == 0
    assert checks["invalid_session_timestamps"] == 0
    assert checks["future_session_timestamps"] == 0


def test_apply_plan_with_empty_plan() -> None:
    """Test that applying an empty plan returns unchanged data."""
    df_sess, df_conv, df_chan = generate_stream(days=7, start_date=dt.date(2025, 9, 1))

    empty_plan = CleaningPlan(version="1.0", steps=[], notes="Empty plan")

    sess_clean, conv_clean, chan_clean, checks = apply_plan(
        df_sess, df_conv, df_chan, empty_plan
    )

    # Data should be unchanged
    assert len(sess_clean) == len(df_sess)
    assert len(conv_clean) == len(df_conv)
    assert len(chan_clean) == len(df_chan)


def test_apply_plan_with_invalid_op() -> None:
    """Test that applying a plan with invalid operation raises ValueError."""
    df_sess, df_conv, df_chan = generate_stream(days=7, start_date=dt.date(2025, 9, 1))

    invalid_plan = CleaningPlan(
        version="1.0",
        steps=[CleaningStep(op="invalid_operation", args={})],
        notes="Invalid plan"
    )

    with pytest.raises(ValueError, match="disallowed op"):
        apply_plan(df_sess, df_conv, df_chan, invalid_plan)


def test_build_semantic_view() -> None:
    """Test that build_semantic_view creates the expected view."""
    df_sess, df_conv, df_chan = generate_stream(days=7, start_date=dt.date(2025, 9, 1))

    con = duckdb.connect(database=":memory:")
    con.register("raw_sessions", df_sess)
    con.register("raw_conversions", df_conv)
    con.register("dim_channels", df_chan)

    build_semantic_view(con, window_days=7)

    # Check that the view exists and has expected columns
    result = con.execute("SELECT * FROM f_attribution LIMIT 1").df()
    expected_cols = [
        "session_id", "channel_name", "cost_per_acquisition",
        "session_ts", "revenue", "conversion_ts", "days_to_convert", "is_attributed"
    ]
    for col in expected_cols:
        assert col in result.columns
