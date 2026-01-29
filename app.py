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

    # Mirror the workshop corruption strategy: add bad duplicates so that
    # dropping them can fully restore the original dataset.

    # 1) Negative revenue duplicates
    k = max(1, int(n_conv * frac))
    idx = df_conv2.sample(k, random_state=2026).index
    neg_dupes = df_conv2.loc[idx].copy()
    neg_dupes["revenue"] = -neg_dupes["revenue"].abs()
    df_conv2 = pd.concat([df_conv2, neg_dupes], ignore_index=True)

    # 2) Future session duplicates (timezone-naive to avoid mixed dtypes)
    k2 = max(1, int(n_sess * frac))
    idx2 = df_sess2.sample(k2, random_state=2027).index
    future_dupes = df_sess2.loc[idx2].copy()
    future_dupes["ts"] = dt.datetime.utcnow() + dt.timedelta(days=30)
    df_sess2 = pd.concat([df_sess2, future_dupes], ignore_index=True)

    # 3) Orphan conversion duplicates with guaranteed-missing session_id
    k3 = max(1, int(n_conv * frac))
    idx3 = df_conv2.sample(k3, random_state=2028).index
    orphan_dupes = df_conv2.loc[idx3].copy()
    orphan_dupes["session_id"] = "999999999999"
    df_conv2 = pd.concat([df_conv2, orphan_dupes], ignore_index=True)

    return df_sess2, df_conv2


def build_semantic_view(con: duckdb.DuckDBPyConnection, window_days: int) -> None:
    """(Re)build the semantic view used by the app.

    Uses TRY_CAST for timestamps so that any corrupted values become NULL
    instead of crashing the query engine.
    """
    window_days = int(window_days)
    con.execute(f"""
    CREATE OR REPLACE VIEW f_attribution AS
    SELECT
        s.session_id,
        c.name AS channel_name,
        c.cac AS cost_per_acquisition,
        TRY_CAST(s.ts AS TIMESTAMP) AS session_ts,
        conv.revenue,
        TRY_CAST(conv.ts AS TIMESTAMP) AS conversion_ts,
        DATE_DIFF('day', TRY_CAST(s.ts AS TIMESTAMP), TRY_CAST(conv.ts AS TIMESTAMP)) AS days_to_convert,
        CASE
            WHEN conv.session_id IS NOT NULL
             AND DATE_DIFF('day', TRY_CAST(s.ts AS TIMESTAMP), TRY_CAST(conv.ts AS TIMESTAMP)) <= {window_days}
            THEN 1 ELSE 0
        END AS is_attributed
    FROM raw_sessions s
    LEFT JOIN raw_conversions conv ON s.session_id = conv.session_id
    JOIN dim_channels c ON s.channel_id = c.id
    """)


def semantic_contract_sql(window_days: int) -> str:
    """Return the semantic SQL for the metric contract shown in the UI.

    This mirrors build_semantic_view so the contract is visible to non-coders.
    """
    window_days = int(window_days)
    return f"""
-- ============================================================
-- METRIC CONTRACT (DATA CONTRACT): f_attribution
-- ============================================================
-- Purpose:
-- This view is the “single source of truth” for one metric:
-- whether a purchase is counted as a conversion for a session.
--
-- Contract idea (plain language):
-- We only credit a purchase to a session if it happens within
-- the attribution window after that session.
--
-- Why it is a contract:
-- Everyone (Marketing, Finance, Analytics) agrees that this SQL
-- definition is the rule. If the rule changes, the metric changes.
-- ============================================================

-- Metric contract: attribution window = {window_days} day(s)
CREATE OR REPLACE VIEW f_attribution AS
SELECT
    -- Identifiers we need to join and group results reliably
    s.session_id,

    -- Dimension fields (what we will group by in reports)
    c.name AS channel_name,
    c.cac AS cost_per_acquisition,

    -- Timestamps (standardized to TIMESTAMP so time rules are consistent)
    TRY_CAST(s.ts AS TIMESTAMP) AS session_ts,
    TRY_CAST(conv.ts AS TIMESTAMP) AS conversion_ts,

    -- Business value (revenue of the purchase event, if it exists)
    conv.revenue,

    -- Contract field: time between session and purchase
    -- This is the key input to the attribution rule.
    DATE_DIFF(
        'day',
        TRY_CAST(s.ts AS TIMESTAMP),
        TRY_CAST(conv.ts AS TIMESTAMP)
    ) AS days_to_convert,

    -- Contract output: is_attributed (0/1)
    -- Rule:
    -- 1) there must be a purchase event linked to this session
    -- 2) it must occur within the attribution window (<= 7 days)
    CASE
        WHEN conv.session_id IS NOT NULL
         AND DATE_DIFF(
                'day',
                TRY_CAST(s.ts AS TIMESTAMP),
                TRY_CAST(conv.ts AS TIMESTAMP)
             ) <= 7
        THEN 1 ELSE 0
    END AS is_attributed

FROM raw_sessions s

-- Keep all sessions, even those with no purchase, so conversion rate is accurate
LEFT JOIN raw_conversions conv
    ON s.session_id = conv.session_id

-- Attach channel metadata (names and cost assumptions)
JOIN dim_channels c
    ON s.channel_id = c.id;
"""


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
    st.set_page_config(page_title="NorthPeak Retail", layout="wide")

    # Global style tweaks for workshop presentation (typography, cards, spacing).
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }
        .hero-block {
            padding: 1.25rem 1.5rem 1.5rem 1.5rem;
            border-radius: 0.85rem;
            background: linear-gradient(135deg, #e0f2fe 0%, #ecfeff 40%, #ffffff 100%);
            border: 1px solid rgba(15, 118, 110, 0.12);
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.5rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: #0f172a;
        }
        .hero-subtitle {
            font-size: 1.0rem;
            color: #0f766e;
            margin-bottom: 0.75rem;
        }
        .hero-body {
            font-size: 0.95rem;
            color: #334155;
            margin-bottom: 0.5rem;
        }
        .hero-try-list {
            margin-top: 0.4rem;
            margin-bottom: 0;
            padding-left: 1.3rem;
            color: #1e293b;
            font-size: 0.9rem;
        }
        .hero-try-list li {
            margin-bottom: 0.15rem;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 0.9rem;
            padding: 0.85rem 1.1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 600;
            color: #0f766e;
        }
        .metric-footnote {
            font-size: 0.85rem;
            color: #4b5563;
            margin-top: 0.75rem;
        }
        .section-spacer {
            margin-top: 1.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero section
    st.markdown(
        """
        <div class="hero-block">
          <h1 class="hero-title">NorthPeak Retail</h1>
          <p class="hero-subtitle">
            Learn about the workflow: event data → Database tables → SQL metric rules → quality checks → an enterprise dashboard.
          </p>
          <p class="hero-body">
            In this workshop you will act not as the data analyst only but as the analytics engineer. <br>
            You will replay events, define a conversion rule in SQL, and see how metrics change when the definition changes.
          </p>
          <ul class="hero-try-list">
            <li>Move the attribution contract window from 7 → 30 days.</li>
            <li>Turn on demo anomalies and watch the quality checks respond.</li>
            <li>Optionally run the AI cleaner and review the plan + judge verdict.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Filters")
        days = st.slider("History window (days)", min_value=14, max_value=120, value=120, step=7)
        st.caption("How many days of events we load into the database for analysis.")
        window = st.slider("Attribution contract window (days)", min_value=1, max_value=30, value=7, step=1)
        st.caption("How many days after a visit we still credit a purchase to that channel")
        inject_clicked = st.button("Inject demo anomalies")
        st.caption("Adds a small number of intentional issues so the quality checks have something to catch.")
        ai_clicked = st.button("Run AI cleaner + judge")
        st.caption("AI proposes a fix plan. A judge verifies the result. The pipeline stays deterministic.")
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

    # Executive summary cards
    st.subheader("Executive summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Sessions</div>
              <div class="metric-value">{len(df_sess):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Purchases (raw)</div>
              <div class="metric-value">{len(df_conv):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Attribution contract window</div>
              <div class="metric-value">{window} days</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Trusted conversions</div>
              <div class="metric-value">{int(df['trusted_conversions'].sum()):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p class=\"metric-footnote\">These KPIs are produced by the contract below. Change the contract window to change the results.</p>",
        unsafe_allow_html=True,
    )

    # Channel performance and conversion views (move higher in the layout)
    st.markdown("<div class=\"section-spacer\"></div>", unsafe_allow_html=True)
    st.subheader("Channel performance")
    st.markdown("What to look for: compare naive vs contract-based conversions to see the impact of the definition.")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Conversions by channel")
    chart_df = df.set_index("channel_name")[["trusted_conversions", "naive_conversions", "out_of_window"]]
    st.bar_chart(chart_df)

    st.subheader("Weekly conversions trend")
    st.markdown("What to look for: watch how the contract window changes the trendlines.")
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

    # Metric contract + quality section
    st.markdown("<div class=\"section-spacer\"></div>", unsafe_allow_html=True)
    st.subheader("Metric contract + quality checks")
    st.markdown(
        "This SQL is a rule ('data contract'). It defines which purchases we attribute as conversions from sessions. Clear rules make metrics consistent.")

    with st.expander("View the contract SQL"):
        st.code(semantic_contract_sql(window), language="sql")

    # AI cleaning section
    st.markdown("<div class=\"section-spacer\"></div>", unsafe_allow_html=True)
    st.subheader("Data cleaning with an GenAI agent")
    if ai_result is None:
        st.info("Quality checks run deterministically. GenAI agentic workflows are becoming crucial for data quality analysis and improvement.")
    else:
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

        verdict_raw = (judge or {}).get("verdict", "")
        verdict_label = "PASS" if str(verdict_raw).lower() == "pass" else "REVIEW"
        st.markdown(f"**Judge verdict:** `{verdict_label}`")
        if judge.get("reasons"):
            with st.expander("Show judge reasoning"):
                st.json(judge)


if __name__ == "__main__":
    main()
