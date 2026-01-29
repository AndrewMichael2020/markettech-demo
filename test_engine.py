from __future__ import annotations

import datetime as dt
import os

import duckdb
import pandas as pd
import pytest

from markettech_workshop import generate_stream, build_semantic_view
from ai_cleaning_agent import run_agentic_cleaning_loop


def test_generate_stream_is_deterministic():
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


def test_metric_contract_changes_counts():
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


def test_quality_checks_default_data_passes():
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


def test_agentic_cleaning_requires_openai_key():
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
