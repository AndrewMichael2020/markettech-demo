from __future__ import annotations

import os

import datetime as dt
from dataclasses import dataclass
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from ai_cleaning_agent import run_agentic_cleaning_loop

# Deterministic demo
np.random.seed(2026)


@dataclass(frozen=True)
class Channel:
    id: str
    name: str
    cac: float


def generate_stream(days: int = 60, start_date: dt.date = dt.date(2025, 9, 1)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    channels = [
        Channel("CH_ORG", "Organic", 0.00),
        Channel("CH_SOC", "Paid Social", 12.50),
        Channel("CH_EML", "Email", 0.50),
    ]
    df_chan = pd.DataFrame([c.__dict__ for c in channels])

    day_idx = np.arange(days)
    dates = np.array([start_date + dt.timedelta(days=int(i)) for i in day_idx])
    is_weekend = np.array([d.weekday() >= 5 for d in dates], dtype=bool)
    base = np.where(is_weekend, 1500, 800)
    vol = np.clip(np.random.normal(loc=base, scale=100).astype(int), 300, None)
    total_sessions = int(vol.sum())

    session_id = np.random.randint(100000, 999999, size=total_sessions).astype(str)
    day_for_session = np.repeat(np.arange(days), vol)
    session_date = dates[day_for_session]

    chan_ids = np.random.choice(["CH_ORG", "CH_SOC", "CH_EML"], p=[0.40, 0.40, 0.20], size=total_sessions)

    hour = np.random.randint(0, 24, size=total_sessions)
    minute = np.random.randint(0, 60, size=total_sessions)
    second = np.random.randint(0, 60, size=total_sessions)

    session_ts = np.array([dt.datetime.combine(d, dt.time(int(h), int(m), int(s)))
                           for d, h, m, s in zip(session_date, hour, minute, second)])

    engagement = np.where(
        chan_ids == "CH_SOC",
        np.clip(np.random.normal(loc=30, scale=20, size=total_sessions).astype(int), 1, None),
        np.clip(np.random.normal(loc=120, scale=60, size=total_sessions).astype(int), 5, None),
    )

    sessions = pd.DataFrame({
        "event_type": "session_start",
        "session_id": session_id,
        "channel_id": chan_ids,
        "ts": pd.to_datetime(session_ts),
        "engagement_sec": engagement,
    })

    base_prob = np.where(engagement > 60, 0.05, 0.005)
    is_convert = np.random.random(size=total_sessions) < base_prob

    conv_sessions = sessions.loc[is_convert, ["session_id", "ts"]].copy()
    lag_days = np.random.randint(0, 11, size=len(conv_sessions))
    lag_minutes = np.random.randint(5, 121, size=len(conv_sessions))
    conv_ts = conv_sessions["ts"].to_numpy() + pd.to_timedelta(lag_days, unit="D") + pd.to_timedelta(lag_minutes, unit="m")
    revenue = np.round(np.random.uniform(50, 200, size=len(conv_sessions)), 2)

    conversions = pd.DataFrame({
        "event_type": "purchase",
        "session_id": conv_sessions["session_id"].to_numpy(),
        "ts": pd.to_datetime(conv_ts),
        "revenue": revenue,
    })

    return sessions, conversions, df_chan


def inject_corruption(df_sess: pd.DataFrame, df_conv: pd.DataFrame, frac: float = 0.003) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sess2 = df_sess.copy()
    df_conv2 = df_conv.copy()

    n_sess = len(df_sess2)
    n_conv = len(df_conv2)

    k = max(1, int(n_conv * frac))
    idx = df_conv2.sample(k, random_state=2026).index
    df_conv2.loc[idx, "revenue"] = -df_conv2.loc[idx, "revenue"].abs()

    # Future timestamps (sessions) - keep timezone-naive to avoid mixed dtypes
    k2 = max(1, int(n_sess * frac))
    idx2 = df_sess2.sample(k2, random_state=2027).index
    df_sess2.loc[idx2, "ts"] = dt.datetime.utcnow() + dt.timedelta(days=30)

    k3 = max(1, int(n_conv * frac))
    idx3 = df_conv2.sample(k3, random_state=2028).index
    df_conv2.loc[idx3, "session_id"] = "999999999999"

    return df_sess2, df_conv2


def build_semantic_view(con: duckdb.DuckDBPyConnection, window_days: int) -> None:
    window_days = int(window_days)
    con.execute(f"""
    CREATE OR REPLACE VIEW f_attribution AS
    SELECT
        s.session_id,
        c.name AS channel_name,
        c.cac AS cost_per_acquisition,
        CAST(s.ts AS TIMESTAMP) AS session_ts,
        conv.revenue,
        CAST(conv.ts AS TIMESTAMP) AS conversion_ts,
        DATE_DIFF('day', CAST(s.ts AS TIMESTAMP), CAST(conv.ts AS TIMESTAMP)) AS days_to_convert,
        CASE
            WHEN conv.session_id IS NOT NULL
             AND DATE_DIFF('day', CAST(s.ts AS TIMESTAMP), CAST(conv.ts AS TIMESTAMP)) <= {window_days}
            THEN 1 ELSE 0
        END AS is_attributed
    FROM raw_sessions s
    LEFT JOIN raw_conversions conv ON s.session_id = conv.session_id
    JOIN dim_channels c ON s.channel_id = c.id
    """)


def channel_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    SELECT
        channel_name,
        COUNT(*) AS total_traffic,
        COUNT(revenue) AS naive_conversions,
        SUM(is_attributed) AS trusted_conversions,
        COUNT(revenue) - SUM(is_attributed) AS out_of_window,
        SUM(is_attributed) * AVG(cost_per_acquisition) AS est_spend_attributed
    FROM f_attribution
    GROUP BY 1
    ORDER BY trusted_conversions DESC
    """
    return con.execute(sql).df()


@st.cache_data(show_spinner=False)
def load_engine(days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return generate_stream(days=days)


def main() -> None:
    st.set_page_config(page_title="MarketTech Truth Engine", layout="wide")
    st.title("MarketTech Truth Engine")
    st.caption("Prototype: metric contracts + reproducible event replay")

    with st.sidebar:
        st.header("Controls")
        days = st.slider("Days of simulated history", min_value=14, max_value=120, value=60, step=7)
        window = st.slider("Attribution window (days)", min_value=1, max_value=30, value=7, step=1)
        inject_clicked = st.button("Inject bad data (for demo)")
        ai_clicked = st.button("Use AI cleaning agent")
        st.caption("AI requires OPENAI_API_KEY on the server.")

    # Session state for current engine data and AI results
    if "engine_days" not in st.session_state:
        st.session_state["engine_days"] = None
        st.session_state["base_sess"] = None
        st.session_state["base_conv"] = None
        st.session_state["base_chan"] = None
        st.session_state["df_sess"] = None
        st.session_state["df_conv"] = None
        st.session_state["df_chan"] = None
        st.session_state["corrupt_applied"] = False
        st.session_state["ai_result"] = None

    # If the days slider changes, reset the engine state
    if st.session_state["engine_days"] != days:
        base_sess, base_conv, base_chan = load_engine(days=days)
        st.session_state["engine_days"] = days
        st.session_state["base_sess"] = base_sess
        st.session_state["base_conv"] = base_conv
        st.session_state["base_chan"] = base_chan
        st.session_state["df_sess"] = base_sess
        st.session_state["df_conv"] = base_conv
        st.session_state["df_chan"] = base_chan
        st.session_state["corrupt_applied"] = False
        st.session_state["ai_result"] = None

    # One-shot: inject bad data into the current base dataset
    if inject_clicked:
        base_sess = st.session_state["base_sess"]
        base_conv = st.session_state["base_conv"]
        if base_sess is not None and base_conv is not None:
            sess_bad, conv_bad = inject_corruption(base_sess, base_conv)
            st.session_state["df_sess"] = sess_bad
            st.session_state["df_conv"] = conv_bad
            st.session_state["corrupt_applied"] = True
            st.session_state["ai_result"] = None

    # Current working tables for this run
    df_sess = st.session_state["df_sess"]
    df_conv = st.session_state["df_conv"]
    df_chan = st.session_state["df_chan"]

    # Optional AI cleaning loop (one-shot button)
    ai_result = st.session_state.get("ai_result")
    if ai_clicked:
        api_key_present = os.getenv("OPENAI_API_KEY", "").strip() != ""
        if not api_key_present:
            st.warning("OPENAI_API_KEY is not set on this server. AI cleaning is disabled.")
        else:
            try:
                ai_result = run_agentic_cleaning_loop(
                    sessions=df_sess,
                    conversions=df_conv,
                    channels=df_chan,
                    planner_model="gpt-5.2-chat-latest",
                    judge_model="gpt-5.2",
                    max_iters=2,
                )
            except Exception as e:
                # Surface a friendly error in the UI without breaking the whole app.
                st.error(f"AI cleaning agent failed: {e}")
                ai_result = None
            else:
                df_sess = ai_result["sessions_clean"]
                df_conv = ai_result["conversions_clean"]
                st.session_state["df_sess"] = df_sess
                st.session_state["df_conv"] = df_conv
                st.session_state["ai_result"] = ai_result

    # Build a fresh DuckDB connection from the current working tables
    con = duckdb.connect(database=":memory:")
    con.register("raw_sessions", df_sess)
    con.register("raw_conversions", df_conv)
    con.register("dim_channels", df_chan)

    build_semantic_view(con, window_days=window)
    df = channel_summary(con)

    # Executive metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", f"{len(df_sess):,}")
    c2.metric("Purchases (raw)", f"{len(df_conv):,}")
    c3.metric("Contract window", f"{window} days")
    c4.metric("Trusted conversions", f"{int(df['trusted_conversions'].sum()):,}")

    if ai_result is not None:
        st.subheader("AI cleaning verdict")

        checks_before = ai_result.get("checks_before", {}) or {}
        checks_after = ai_result.get("checks_after", {}) or {}
        plan = ai_result.get("plan", {}) or {}
        judge = ai_result.get("judge", {}) or {}

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Quality checks (before → after)**")
            rows = []
            for key, before_val in checks_before.items():
                after_val = checks_after.get(key, None)
                rows.append({
                    "check": key,
                    "before": before_val,
                    "after": after_val,
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("**Cleaning steps proposed by AI**")
            steps = plan.get("steps", []) or []
            if steps:
                step_rows = [
                    {"step": i + 1, "op": s.get("op"), "args": str(s.get("args"))}
                    for i, s in enumerate(steps)
                ]
                st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No cleaning steps were proposed.")

        verdict = judge.get("verdict", "unknown")
        st.markdown(f"**Judge verdict:** `{verdict}`")
        if judge.get("reasons"):
            with st.expander("Show judge reasoning"):
                st.json(judge)


    st.subheader("Channel performance")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Trusted conversions by channel")
    chart_df = df.set_index("channel_name")[["trusted_conversions", "naive_conversions", "out_of_window"]]
    st.bar_chart(chart_df)

    st.subheader("Weekly trusted conversions trend")
    trend_sql = """
    SELECT
        DATE_TRUNC('week', session_ts) AS week,
        channel_name,
        SUM(is_attributed) AS sales
    FROM f_attribution
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    df_trend = con.execute(trend_sql).df()
    pivot = df_trend.pivot(index="week", columns="channel_name", values="sales").fillna(0)
    st.line_chart(pivot)

    with st.expander("Show the semantic SQL (metric contract)"):
        st.code(f"Attribution window: {window} days\n\nSee build_semantic_view() in app.py", language="text")


if __name__ == "__main__":
    main()
