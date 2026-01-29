# %% [markdown]
# # MarketTech: The Truth Engine
# **Engineering Data Products for Growth Strategy**
#
# **Objective:** Build a small, production-shaped analytics pipeline from scratch.
# **Stack:** Python (logic), DuckDB (warehouse), Matplotlib (viz).
#
# Scenario:
# - Marketing says **Paid Social** is the best channel.
# - Finance says we are losing money on it.
# - We will decide using a **metric contract** expressed as SQL.

# %%
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2026)

# %% [markdown]
# ## Phase 1: Ingestion (Event stream simulation)
# We simulate two event types:
# - `session_start`
# - `purchase`
#
# Key concept: the raw event stream is not “report-ready.”

# %%
@dataclass(frozen=True)
class Channel:
    id: str
    name: str
    cac: float  # cost per acquisition, dollars


def generate_stream(days: int = 60, start_date: dt.date = dt.date(2025, 9, 1), seed: int = 2026) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate a deterministic, workshop-friendly event stream.
    Returns:
      - sessions: session_start events
      - conversions: purchase events
      - channels: channel dimension
    """
    rng = np.random.default_rng(seed)

    channels = [
        Channel("CH_ORG", "Organic", 0.00),
        Channel("CH_SOC", "Paid Social", 12.50),
        Channel("CH_EML", "Email", 0.50),
    ]
    df_chan = pd.DataFrame([c.__dict__ for c in channels])

    # Traffic volume per day: weekends higher
    day_idx = np.arange(days)
    dates = np.array([start_date + dt.timedelta(days=int(i)) for i in day_idx])
    is_weekend = np.array([d.weekday() >= 5 for d in dates], dtype=bool)
    base = np.where(is_weekend, 1500, 800)
    vol = np.clip(rng.normal(loc=base, scale=100).astype(int), 300, None)
    total_sessions = int(vol.sum())

    # Session IDs
    session_id = rng.integers(100000, 999999, size=total_sessions).astype(str)

    # Map each session to a day
    day_for_session = np.repeat(np.arange(days), vol)
    session_date = dates[day_for_session]

    # Channel assignment (weights chosen to make arguments plausible)
    chan_ids = rng.choice(["CH_ORG", "CH_SOC", "CH_EML"], p=[0.40, 0.40, 0.20], size=total_sessions)

    # Time within day
    hour = rng.integers(0, 24, size=total_sessions)
    minute = rng.integers(0, 60, size=total_sessions)
    second = rng.integers(0, 60, size=total_sessions)

    session_ts = np.array([dt.datetime.combine(d, dt.time(int(h), int(m), int(s)))
                           for d, h, m, s in zip(session_date, hour, minute, second)])

    # Engagement: Paid Social has shorter engagement
    engagement = np.where(
        chan_ids == "CH_SOC",
        np.clip(rng.normal(loc=30, scale=20, size=total_sessions).astype(int), 1, None),
        np.clip(rng.normal(loc=120, scale=60, size=total_sessions).astype(int), 5, None),
    )

    sessions = pd.DataFrame({
        "event_type": "session_start",
        "session_id": session_id,
        "channel_id": chan_ids,
        "ts": pd.to_datetime(session_ts),
        "engagement_sec": engagement,
    })

    # Conversion probability depends on engagement
    # - If engagement > 60 seconds: higher conversion chance
    base_prob = np.where(engagement > 60, 0.05, 0.005)
    is_convert = rng.random(size=total_sessions) < base_prob

    # For converted sessions, add lag (0-10 days) and revenue (50-200)
    conv_sessions = sessions.loc[is_convert, ["session_id", "ts"]].copy()
    lag_days = rng.integers(0, 11, size=len(conv_sessions))
    lag_minutes = rng.integers(5, 121, size=len(conv_sessions))
    conv_ts = conv_sessions["ts"].to_numpy() + pd.to_timedelta(lag_days, unit="D") + pd.to_timedelta(lag_minutes, unit="m")
    revenue = np.round(rng.uniform(50, 200, size=len(conv_sessions)), 2)

    conversions = pd.DataFrame({
        "event_type": "purchase",
        "session_id": conv_sessions["session_id"].to_numpy(),
        "ts": pd.to_datetime(conv_ts),
        "revenue": revenue,
    })

    return sessions, conversions, df_chan


df_sess, df_conv, df_chan = generate_stream(days=60)
print(f"Stream Online: {len(df_sess):,} sessions | {len(df_conv):,} purchases")
df_sess.head(3)

# %% [markdown]
# ## Phase 2: Storage (DuckDB)
# We load raw tables into an in-process OLAP engine, then speak SQL.

# %%
con = duckdb.connect(database=":memory:")
con.register("raw_sessions", df_sess)
con.register("raw_conversions", df_conv)
con.register("dim_channels", df_chan)

con.execute("SELECT COUNT(*) AS sessions FROM raw_sessions").df()

# %% [markdown]
# ## Phase 3: Metric contract (Semantic Layer)
# Contract:
# - A purchase is attributed to a session only if it occurs within **N days** of session_start.
#
# This is where dashboard disagreements are born.

# %%
def build_semantic_view(con: duckdb.DuckDBPyConnection, window_days: int = 7) -> None:
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
    LEFT JOIN raw_conversions conv
        ON s.session_id = conv.session_id
    JOIN dim_channels c
        ON s.channel_id = c.id
    """)


build_semantic_view(con, window_days=7)
con.execute("SELECT * FROM f_attribution WHERE revenue IS NOT NULL LIMIT 5").df()

# %% [markdown]
# ## Phase 4: Truth vs Fiction
# Compare:
# - Naive conversions: every purchase counts
# - Trusted conversions: purchases within the attribution window count

# %%
def channel_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    sql = """
    SELECT
        channel_name,
        COUNT(*) AS total_traffic,
        COUNT(revenue) AS naive_conversions,
        SUM(is_attributed) AS trusted_conversions,
        COUNT(revenue) - SUM(is_attributed) AS out_of_window
    FROM f_attribution
    GROUP BY 1
    ORDER BY trusted_conversions DESC
    """
    return con.execute(sql).df()


df_truth = channel_summary(con)
df_truth

# %% [markdown]
# ### Exercise: change the contract
# Change the attribution window to 30 days and re-run the summary.
# What shifts? Who “wins” now?

# %%
build_semantic_view(con, window_days=30)
channel_summary(con)

# %% [markdown]
# ## Phase 5: Data Quality Gates
# We add assertions that stop the pipeline if basic rules fail.

# %%
def run_quality_checks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    checks = [
        ("Negative revenue", "SELECT COUNT(*) AS n FROM raw_conversions WHERE revenue < 0"),
        ("Orphaned conversions", """
            SELECT COUNT(*) AS n
            FROM raw_conversions c
            LEFT JOIN raw_sessions s ON c.session_id = s.session_id
            WHERE s.session_id IS NULL
        """),
        ("Future session timestamps", "SELECT COUNT(*) AS n FROM raw_sessions WHERE CAST(ts AS TIMESTAMP) > NOW()"),
    ]

    rows = []
    for name, sql in checks:
        n = int(con.execute(sql).fetchone()[0])
        rows.append({"check": name, "errors": n, "status": "PASS" if n == 0 else "FAIL"})
    return pd.DataFrame(rows)


dq = run_quality_checks(con)
dq


# %% [markdown]
# ## Phase 5A: Optional AI cleaning agent + AI judge
# This step demonstrates a minimal agentic loop:
# - Planner model proposes a cleaning plan (constrained operations)
# - We apply the plan deterministically (DuckDB)
# - Judge model verifies the outcome
#
# Requirements:
# - Set environment variable `OPENAI_API_KEY`
# - Internet access for API calls
#
# If `OPENAI_API_KEY` is not set, this section will skip.

# %%
import os
from ai_cleaning_agent import run_agentic_cleaning_loop

def inject_corruption(df_sess: pd.DataFrame, df_conv: pd.DataFrame, frac: float = 0.003) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject a small amount of bad data to make the cleaning demo real.
    - Negative revenue in conversions
    - Future session timestamps
    - Orphan conversions
    """
    df_sess2 = df_sess.copy()
    df_conv2 = df_conv.copy()

    n_sess = len(df_sess2)
    n_conv = len(df_conv2)

    # Negative revenue
    k = max(1, int(n_conv * frac))
    idx = df_conv2.sample(k, random_state=2026).index
    df_conv2.loc[idx, "revenue"] = -df_conv2.loc[idx, "revenue"].abs()

    # Future timestamps (sessions) - keep timezone-naive to avoid mixed dtypes
    k2 = max(1, int(n_sess * frac))
    idx2 = df_sess2.sample(k2, random_state=2027).index
    df_sess2.loc[idx2, "ts"] = dt.datetime.utcnow() + dt.timedelta(days=30)

    # Orphan conversions: create fake session ids
    k3 = max(1, int(n_conv * frac))
    idx3 = df_conv2.sample(k3, random_state=2028).index
    df_conv2.loc[idx3, "session_id"] = "999999999999"  # guaranteed orphan

    return df_sess2, df_conv2


USE_CORRUPT_DATA = True  # flip to False if you want the checks to already pass

df_sess_demo, df_conv_demo = (inject_corruption(df_sess, df_conv) if USE_CORRUPT_DATA else (df_sess, df_conv))

# Re-register demo data for checks
con_demo = duckdb.connect(database=":memory:")
con_demo.register("raw_sessions", df_sess_demo)
con_demo.register("raw_conversions", df_conv_demo)
con_demo.register("dim_channels", df_chan)

print("Quality checks BEFORE AI cleaning (demo dataset):")
dq_before = run_quality_checks(con_demo)
dq_before

# %%
if __name__ == "__main__":
    if os.getenv("OPENAI_API_KEY", "").strip():
        result = run_agentic_cleaning_loop(
            sessions=df_sess_demo,
            conversions=df_conv_demo,
            channels=df_chan,
            planner_model="gpt-5.2-chat-latest",
            judge_model="gpt-5.2",
            max_iters=2,
        )

        print("AI plan:")
        print(result["plan"])
        print("AI judge verdict:")
        print(result["judge"])

        # Use cleaned data going forward (optional)
        df_sess_clean = result["sessions_clean"]
        df_conv_clean = result["conversions_clean"]

        con_clean = duckdb.connect(database=":memory:")
        con_clean.register("raw_sessions", df_sess_clean)
        con_clean.register("raw_conversions", df_conv_clean)
        con_clean.register("dim_channels", df_chan)

        print("Quality checks AFTER AI cleaning:")
        run_quality_checks(con_clean)
    else:
        print("OPENAI_API_KEY is not set. Skipping AI cleaning demo.")


# %% [markdown]
# ## Phase 6: Prototype Visualization
# We plot weekly trusted conversions by channel.

# %%
# Rebuild with the default 7-day contract
build_semantic_view(con, window_days=7)

trend_sql = """
SELECT
    DATE_TRUNC('week', session_ts) AS week,
    channel_name,
    SUM(is_attributed) AS sales
FROM f_attribution
GROUP BY 1, 2
ORDER BY 1, 2
"""
df_viz = con.execute(trend_sql).df()

pivot = df_viz.pivot(index="week", columns="channel_name", values="sales").fillna(0)

pivot.plot(kind="line", figsize=(10, 5), title="Weekly Attributed Sales by Channel")
plt.ylabel("Trusted conversions (count)")
plt.xlabel("Week")
plt.show()
