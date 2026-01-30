## NorthPeak Retail / MarketTech demo

---

This repo contains a **teaching demo** for data analytics students. It was developed with assistance from ChatGPT, GitHub Copilot, GitHub Workspaces, GitHub Actions, Google Cloud Run, and Google Cloud Build. 

You start with raw events (website sessions and purchases) and gradually turn
them into a **trustworthy metric** behind a Streamlit app and a Cloud Run URL.

You don’t need to *start* as a software engineer to work through this, but the goal
is to gently pull you in that direction. Most of the work is changing parameters,
running notebooks, and reading charts – plus a bit of real coding that shows you the
"back of the dashboard" and builds the habits of a full‑stack analytics engineer.

The bigger theme: modern analytics is **not only Power BI / dashboards**.
With a bit of Python and SQL, you can build your own small but robust
analytics products: local apps, hosted apps, and even CI/CD pipelines with
role-based access and AI assistance – all using the same core ideas you
already know from analytics work.

---

![CI](https://github.com/AndrewMichael2020/markettech-demo/actions/workflows/ci.yml/badge.svg)
![Docker build](https://github.com/AndrewMichael2020/markettech-demo/actions/workflows/docker.yml/badge.svg)
![Tests](https://img.shields.io/github/actions/workflow/status/AndrewMichael2020/markettech-demo/ci.yml?label=tests&logo=github)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![DuckDB](https://img.shields.io/badge/Warehouse-DuckDB-yellow?logo=duckdb&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![OpenAI](https://img.shields.io/badge/AI-OpenAI-412991?logo=openai&logoColor=white)
![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

---

## What you will learn

By working through this repo, students see the full story:

1. **Event data → tables**  
   Simulated sessions and purchases land in DuckDB as `raw_sessions` and
   `raw_conversions`.
2. **Metric contract**  
   A SQL view (`f_attribution`) defines when a purchase “counts” as
   attributed to a session (within N days).
3. **Quality checks**  
   Simple rules catch bad data (negative revenue, orphan conversions,
   broken timestamps) before it ships.
4. **Product surface**  
   A Streamlit app (“NorthPeak Retail”) shows the same engine behind a URL
   with sliders and charts.
5. **Optional AI cleaner**  
   An AI "cleaning agent" (OpenAI or a local Gemma model) proposes how to
   fix bad rows, and an AI judge checks the result.

The goal is to move from an **Excel mindset** (“numbers just appear”) to a
**systems mindset** (“numbers come from code, contracts, and checks”).

---

## 60‑minute teaching flow (for instructors)

This is a suggested 1‑hour lesson plan if you are teaching this workshop.

### 0–10: Setup and premise

- Show two conflicting numbers for “conversions” from different dashboards.
- Explain that both are “correct” for their own definition, but confusing
  together.
- Open the Streamlit app URL (local or Cloud Run). This is the **product**
  students are trying to understand.

**Q&A (2–3 minutes):**  
Ask: “Where have you seen ‘dueling dashboards’ in real life?”

### 10–25: Notebook – the raw reality

- Generate deterministic `session_start` and `purchase` events.
- Register them as `raw_sessions` and `raw_conversions` in DuckDB.
- Discuss why event data is messy for business users (multiple timestamps,
  missing links, strange edge cases).

**Q&A (3–5 minutes):**  
Ask: “Which of these raw columns would you *not* show directly to a VP, and why?”

### 25–40: Notebook – the metric contract

- Build `f_attribution` as a semantic view on top of the raw tables.
- Define the contract: a purchase only counts if it happens within **N days**
  of a session.
- Compare:
  - naive conversions (every purchase)
  - trusted conversions (within the contract window)
  - out-of-window conversions.
- Flip the window from 7 → 30 days and see how “truth” changes.

**Q&A (3–5 minutes):**  
Ask: “If Finance and Marketing disagree, which ‘truth’ should win—and who decides?”

### 40–50: Quality gates and product surface

- Add and run simple checks:
  - negative revenue
  - orphan conversions (no matching session)
  - invalid or future timestamps.
- Message: if checks fail, we don’t ship the metric.
- Open `app.py` and point out it uses the same generator and SQL view.
- In the Streamlit app, change:
  - **History window (days)**
  - **Attribution window (days)**
  - Toggle “Inject demo anomalies” and watch quality metrics react.

**Q&A (3–5 minutes):**  
Ask: “Which check would you add next for your own company’s data?”

### 50–60: Optional AI + wrap‑up

- Briefly show that the AI cleaner (OpenAI or local Gemma) proposes a plan
  and that a judge reviews it.
- Emphasize that the AI is constrained by **your** contract and quality rules.
- Connect to real teams: how this pattern maps to BI tools, data platforms,
  and ML systems students might see on co‑op.

**Final close (2–3 minutes):**

- One-line summary: **Metrics are code, contracts are governance, and
  products are URLs.**
- Invite questions about how they might adapt this pattern to their own
  internships or projects.

---

## Repo overview (files you will touch)

- `markettech_workshop.py` / `markettech_workshop.ipynb`  
  The main workshop notebook (Python script + Jupyter notebook). You can run
  either version.
- `app.py`  
  The Streamlit app (NorthPeak Retail). Uses the same data generator and SQL
  contract as the notebook.
- `ai_cleaning_agent.py`  
  Optional agentic loop: planner model, deterministic cleaning code,
  and judge model.
- `test_engine.py`  
  A few small tests that prove:
  - data generation is deterministic
  - the metric contract behaves as described
  - the AI cleaner only runs when a key is present.
- `.github/workflows/ci.yml`  
  CI pipeline: runs tests and, on `main`, builds and deploys to Cloud Run.
- `docker.yml`  
  GitHub Actions workflow that builds the Docker image (no push) for quick
  feedback.
- `Dockerfile`  
  How the app is containerized for Cloud Run.
- `main.tf`, `variables.tf`, `versions.tf`  
  Terraform files that describe the Cloud Run service and Artifact Registry.
- `deploy_cloud_run.sh`, `cleanup_cloud_run.sh`, `set_openai_key_cloud_run.sh`  
  Helper scripts for instructors to deploy / tear down resources and configure
  the OpenAI key in Secret Manager.

You can safely ignore the Terraform and shell scripts if you are just a
student following the workshop. An instructor or DevOps engineer will usually
prepare the Cloud Run URL for you.

---

## Quickstart for students (local run)

### 1. Create and activate a virtual environment

This keeps Python packages for the workshop separate from your system.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Run the tests (optional but recommended)

```bash
pytest -q
```

If this passes, your environment matches the expected behavior.

### 4. Start the Streamlit app

```bash
streamlit run app.py
```

Your browser should open at `http://localhost:8501` (or Streamlit will show
you the exact URL). This is the **NorthPeak Retail** app.

What to try in the app:

- Change the **History window (days)** to see how much raw data you replay.
- Change the **Attribution contract window (days)** to see how the metric
  definition changes conversions.
- Turn on **Inject demo anomalies** to introduce a small amount of bad data.
- (If enabled) click **Run AI cleaner + judge** to see an AI plan and verdict.

---

## Optional: AI cleaning agent and AI judge

There are two ways to run the AI cleaning loop:

1. **OpenAI-hosted models** (default, requires an OpenAI API key)
2. **Local Gemma llamafile** from Mozilla AI (runs fully in this Codespace)

### Option A: OpenAI

If you want to explore the OpenAI-based AI cleaner, you need an OpenAI API
key. **Never commit this key to Git or share it in screenshots.**

1. Set your key in the terminal:

   ```bash
   export OPENAI_API_KEY="YOUR_REAL_KEY_HERE"
   ```

   On Windows PowerShell:

   ```powershell
   $env:OPENAI_API_KEY = "YOUR_REAL_KEY_HERE"
   ```

2. Use the AI cleaner:

   - In the notebook (`markettech_workshop.py` / `.ipynb`), find the AI cleaning
     phase and run those cells.
   - In the Streamlit app, turn on **Run AI cleaner + judge** in the sidebar.

Behind the scenes:

- The **planner model** suggests a small set of allowed operations
  (e.g. “drop rows with negative revenue”).
- Your Python code applies those steps deterministically using DuckDB.
- The **judge model** checks before/after quality checks and decides
  whether the cleaning is acceptable.

If `OPENAI_API_KEY` is not set (and no local Gemma server is configured),
the app will simply skip the AI part and explain that AI is optional.

### Option B: Local Gemma llamafile (no external API calls)

You can also run the AI cleaning loop entirely locally using the
`google_gemma-3-4b-it-Q6_K.llamafile` from Mozilla AI.

1. **Download the Gemma llamafile** (once):

   ```bash
   chmod +x download_gemma_llamafile.sh
   ./download_gemma_llamafile.sh
   ```

   This fetches the llamafile into the repo and marks it executable.

2. **Start the local Gemma HTTP server** in a terminal:

   ```bash
   chmod +x run_gemma_llamafile.sh
   ./run_gemma_llamafile.sh
   ```

   By default this starts an OpenAI-compatible server at
   `http://127.0.0.1:8080/v1`. You can make that explicit for the Python
   code via:

   ```bash
   export GEMMA_API_BASE=http://127.0.0.1:8080/v1
   ```

3. **Use the Gemma-based cleaning agent**:

   - In the notebook / script, the helper `_run_ai_demo` in
     `markettech_workshop.py` will automatically prefer the Gemma-based
     agent when `GEMMA_API_BASE` is set.
   - Programmatically, you can call the local agent directly:

     ```python
     import datetime as dt
     from markettech_workshop import generate_stream, inject_corruption
     from gemma_cleaning_agent import run_agentic_cleaning_loop_gemma

     df_sess, df_conv, df_chan = generate_stream(days=14, start_date=dt.date(2025, 9, 1))
     df_sess_bad, df_conv_bad = inject_corruption(df_sess, df_conv)

     result = run_agentic_cleaning_loop_gemma(
         sessions=df_sess_bad,
         conversions=df_conv_bad,
         channels=df_chan,
         max_iters=2,
     )

     print(result["plan"])
     print(result["judge"])
     ```

The `gemma_cleaning_agent.py` module mirrors the behavior of
`ai_cleaning_agent.py` but talks to the local llamafile via an
OpenAI-style `/v1/chat/completions` HTTP endpoint instead of the
hosted OpenAI API.

### Why local models matter (privacy & sovereignty)

Running the Gemma 4B llamafile entirely inside a GitHub Codespace (or on
your own machine) means:

- **No event data leaves your environment for AI planning/judging.** You can
  experiment with agentic data cleaning without sending payloads to a third
  party API.
- You practice thinking about **data residency and governance**: which
  workloads are safe to push to external services, and which should stay
  close to the data.
- In classroom or corporate settings where external AI services are
  restricted, you can still use this workshop with a fully local model.

For teaching purposes, you can frame the choice of backend (OpenAI vs local
Gemma) as part of the architectural trade‑offs students should learn to
reason about: latency and model quality vs. control and privacy.

---

## Appendix A: Cloud Run and CI/CD (for instructors / DevOps)

You do **not** need this section to learn analytics concepts. This is for
people setting up the hosted version of the app.

### Overview

- Terraform files (`main.tf`, `variables.tf`, `versions.tf`) describe:
  - enabling required Google Cloud APIs
  - an Artifact Registry repository
  - a Cloud Run service.
- GitHub Actions workflow `.github/workflows/ci.yml`:
  - runs tests on every push / PR
  - on `main`, builds the Docker image with Cloud Build
  - deploys to Cloud Run using Workload Identity Federation.
- The OpenAI key is stored **only** in Google Secret Manager as
  `openai-api-key`. Cloud Run reads it via `OPENAI_API_KEY` using
  `--update-secrets`. The key never appears in GitHub logs.

### High‑level deployment steps

0. **Create the Artifact Registry repo via Cloud Shell**

```bash
# Make sure you are on the right project
gcloud config set project studio-1697788595-a34f5

# Create a Docker Artifact Registry repo named "markettech" in us-central1
gcloud artifacts repositories create markettech \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repo for MarketTech demo"
```

1. **In Google Cloud**
   - Create or choose a project (e.g. `studio-1697788595-a34f5`).
   - Enable Artifact Registry, Cloud Run, and Cloud Build.
   - Create a Docker repo (e.g. `markettech` in `us-central1`).
   - Create a Secret Manager secret `openai-api-key` and add your key as
     version 1.
   - Grant the Cloud Run service account `roles/secretmanager.secretAccessor`.

2. **Set up Workload Identity Federation (once)**
   - Create a Workload Identity Pool and OIDC provider for GitHub Actions.
   - Allow that pool to impersonate a deployer service account
     (e.g. `github-actions-deployer@...`).

3. **In GitHub repo settings**
   - Add repository **secrets/variables** for:
     - `GCP_PROJECT_ID`, `GCP_REGION`, `CLOUD_RUN_SERVICE`, `ARTIFACT_REPO`
     - `GCP_SERVICE_ACCOUNT`, `WORKLOAD_IDENTITY_PROVIDER`.

4. **Let Actions do the rest**
   - On push to `main`, `.github/workflows/ci.yml` will:
     - authenticate to GCP via OIDC (no long-lived keys)
     - run tests
     - build and push the image
     - deploy to Cloud Run with `OPENAI_API_KEY` wired from Secret Manager.

If you prefer a one-off manual deploy instead of CI/CD, you can still use
`deploy_cloud_run.sh`, `cleanup_cloud_run.sh`, and `set_openai_key_cloud_run.sh`
from Cloud Shell, but the recommended path is the GitHub Actions pipeline.

## Appendix B: Teardown / cleanup (for instructors)

When you are done with the workshop environment, you can clean up GCP
resources from Cloud Shell:

```bash
# Delete the Cloud Run service (replace with your actual service name if different)
gcloud run services delete markettech-truth-engine \
  --region=us-central1 \
  --quiet

# Delete the Artifact Registry repo used by this demo (irreversible)
gcloud artifacts repositories delete markettech \
  --location=us-central1 \
  --quiet

# (Optional) Delete the OpenAI API key from Secret Manager
gcloud secrets delete openai-api-key --quiet
```

---

## Beyond this workshop

After you finish the exercises, it is worth stepping back and noticing what
you have actually done:

- You started from raw event data and used SQL to define a clear, repeatable
  metric contract.
- You added simple but powerful quality checks so that silent data problems
  do not turn into silent business problems.
- You wrapped that logic in a real app (Streamlit) that non-technical
  stakeholders can use.
- You saw that the same app can run locally on your laptop or behind a
  Cloud Run URL.

In production teams, people layer on more engineering practices (CI/CD,
RBAC, multiple databases, etc.), but the **core ideas do not change**:

- metrics are defined in code,
- contracts and checks protect “truth”,
- and products are just user-friendly ways to surface that logic.

The point is not to turn every analyst into a platform engineer. The point is
to show that you can participate in building robust analytics systems, not
just consume dashboards, and that the tools you already know (SQL, basic
Python) scale surprisingly far.