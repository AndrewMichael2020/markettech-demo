## NorthPeak Retail / MarketTech demo

This repo contains a **teaching demo** for data analytics students.It was developed with assistance from ChatGPT, GitHub Copilot, GitHub Workspaces, GitHub Actions, Google Cloud Run, and Google Cloud Build. 

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

## Status badges

CI and Docker workflows run on GitHub Actions:

![CI](https://github.com/AndrewMichael2020/markettech-demo/actions/workflows/ci.yml/badge.svg)
![Docker build](https://github.com/AndrewMichael2020/markettech-demo/actions/workflows/docker.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

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
   An OpenAI-based “cleaning agent” proposes how to fix bad rows, and an
   AI judge checks the result.

The goal is to move from an **Excel mindset** (“numbers just appear”) to a
**systems mindset** (“numbers come from code, contracts, and checks”).

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

If you want to explore the AI part, you need an OpenAI API key. **Never
commit this key to Git or share it in screenshots.**

### 1. Set your key in the terminal

```bash
export OPENAI_API_KEY="YOUR_REAL_KEY_HERE"
```

On Windows PowerShell, use:

```powershell
$env:OPENAI_API_KEY = "YOUR_REAL_KEY_HERE"
```

### 2. Use the AI cleaner

- In the notebook (`markettech_workshop.py` / `.ipynb`), find the AI cleaning
  phase and run those cells.
- In the Streamlit app, turn on **Run AI cleaner + judge** in the sidebar.

Behind the scenes:

- The **planner model** suggests a small set of allowed operations
  (e.g. “drop rows with negative revenue”).
- Your Python code applies those steps deterministically using DuckDB.
- The **judge model** checks before/after quality checks and decides
  whether the cleaning is acceptable.

If `OPENAI_API_KEY` is not set, the app will simply skip the AI part and
explain that AI is optional.

---

## 120‑minute teaching flow (for instructors)

This is a suggested lesson plan if you are teaching this workshop.

### 0–10: Setup and premise

- Show two conflicting numbers for “conversions” from different dashboards.
- Explain that both are “correct” for their own definition, but confusing
  together.
- Open the Streamlit app URL (local or Cloud Run). This is the **product**
  students are trying to understand.

### 10–35: Notebook – the raw reality

- Generate deterministic `session_start` and `purchase` events.
- Register them as `raw_sessions` and `raw_conversions` in DuckDB.
- Discuss why event data is messy for business users (multiple timestamps,
  missing links, strange edge cases).

### 35–60: Notebook – the metric contract

- Build `f_attribution` as a semantic view on top of the raw tables.
- Define the contract: a purchase only counts if it happens within **N days**
  of a session.
- Compare:
  - naive conversions (every purchase)
  - trusted conversions (within the contract window)
  - out-of-window conversions.
- Flip the window from 7 → 30 days and see how “truth” changes.

### 60–75: Notebook – quality gates

- Add and run simple checks:
  - negative revenue
  - orphan conversions (no matching session)
  - invalid or future timestamps.
- Message: if checks fail, we don’t ship the metric.

### 75–95: Product – Streamlit app

- Open `app.py` and point out it uses the same generator and SQL view.
- Use the filters to change the attribution window and history window live.
- Toggle “Inject demo anomalies” and watch quality metrics react.

### 95–115: (Optional) AI loop and/or Cloud Run

- Show how the AI cleaner proposes a plan and how the judge reviews it.
- Briefly show the Dockerfile and explain that Cloud Run just wraps this
  app in a container behind a URL.

### 115–120: Close

- One-line summary: **Metrics are code, contracts are governance, and
  products are URLs.**

---

## Cloud Run and CI/CD (for instructors / DevOps)

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
