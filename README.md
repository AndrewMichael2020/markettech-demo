# MarketTech: The Truth Engine

Engineering data products for growth strategy, in 120 minutes.

This workshop kit moves a mixed audience from an **Excel mindset** to a **systems engineering mindset** by showing the full lifecycle:
Notebook (R&D) → App (Product) → Cloud Run (Deployment).

## Badges
Replace `<OWNER>/<REPO>` after you create the GitHub repo.

- CI: `https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml`
- Docker build: `https://github.com/<OWNER>/<REPO>/actions/workflows/docker.yml`

![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)
![Docker build](https://github.com/<OWNER>/<REPO>/actions/workflows/docker.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## What you will demo
1. **Event stream ingestion** (sessions + purchases)
2. **DuckDB “lakehouse”** (raw tables → SQL)
3. **Metric contract** (semantic view with an attribution window)
4. **Quality gates** (stop bad data from shipping)
5. **Streamlit product** (the same engine behind a URL)
6. **Optional AI loop** (AI cleaner proposes a plan, AI judge verifies)

## Repository structure
- `markettech_workshop.py`  
  Notebook-as-code with `# %%` cells (VS Code friendly)
- `markettech_workshop.ipynb`  
  Same content as a Jupyter notebook
- `app.py`  
  Streamlit app using the same generator + SQL engine
- `ai_cleaning_agent.py`  
  Optional: OpenAI planner + judge for cleaning, with deterministic execution
- `tests/`  
  Minimal tests that prove determinism + contract behavior
- `.github/workflows/`  
  CI workflow (pytest) and Docker build workflow
- `terraform/`  
  IaC for Artifact Registry + Cloud Run
- `scripts/`  
  Deploy, cleanup, and set-key helpers

---

# MarketTech Workshop Curriculum (Runbook)

## Audience
- Marketing and growth stakeholders: care about the decisions and the story
- Technical stakeholders: care about logic, repeatability, and correctness

## Learning outcomes
By the end, students can:
- Explain why “dashboard truth” depends on definitions
- Write a metric contract as SQL
- Add quality gates that prevent shipping bad data
- Wrap an analysis engine into a product endpoint
- Describe an agentic loop: plan → execute → judge

## 120-minute flow (recommended timing)

### 0–10 Setup and premise
- Show two conflicting numbers for a key metric.
- State the root cause: different definitions, hidden assumptions.
- Show the Streamlit URL early. This is the “product.”

### 10–35 Notebook: the raw reality
- Generate deterministic session_start and purchase events.
- Register them in DuckDB as `raw_sessions` and `raw_conversions`.
- Show what makes event data “unfriendly” for dashboards.

### 35–60 Notebook: the metric contract
- Create `f_attribution` as the semantic view.
- Contract: a purchase counts only if it occurs within **N days** of the session.
- Run summary: naive vs trusted vs out-of-window.
- Flip the attribution window (7 → 30 days). Re-run. Discuss how the “truth” changes.

### 60–75 Notebook: quality gates
- Run three assertions:
  - negative revenue
  - orphan conversions
  - future timestamps
- Message: if checks fail, we do not publish results.

### 75–95 Product: Streamlit app
- Open `app.py` and point to the same generator + SQL.
- Use the slider to change the attribution window live.
- Optional: toggle “Inject bad data” and show quality failing or shifting.

### 95–115 Deployment: Cloud Run
- Show Dockerfile: Streamlit bound to `$PORT`.
- Show Terraform: Cloud Run service with public access and min instances 0.
- If you want, run the deploy script once (manual).

### 115–120 Close
- One-line wrap: metrics are code, contracts are governance, products are URLs.

---

## Local quickstart
### 1) Create venv
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install deps
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3) Run tests
```bash
pytest -q
```

### 4) Run the Streamlit app
```bash
streamlit run app.py
```

---

## Optional: AI cleaning agent + AI judge
This shows a minimal agentic loop:
- planner model proposes a constrained cleaning plan (JSON)
- your code executes it deterministically in DuckDB
- judge model returns PASS or FAIL

Set:
```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Then:
- In the notebook, run “Phase 5A”
- In the app, toggle “Use AI cleaning agent”

---

## Cloud Run deploy (low cost, scales down when idle)
### Required tools
- `gcloud`
- `terraform`

### Deploy
```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export REPO="markettech"
export SERVICE="markettech-truth-engine"
export TAG="v1"

# build + push image via Cloud Build, then apply Terraform
./scripts/deploy_cloud_run.sh
```

### Set OpenAI key on Cloud Run (server-side)
```bash
export OPENAI_API_KEY="YOUR_KEY"
./scripts/set_openai_key_cloud_run.sh
```

---

## Stop costs fast
### Option A: destroy Terraform resources
```bash
export PROJECT_ID="your-project-id"
./scripts/cleanup_cloud_run.sh
```

### Option B: delete the entire project (one action)
```bash
gcloud projects delete "${PROJECT_ID}"
```

---

## CI and best practice defaults
This repo includes:
- `CI` workflow that runs pytest on push/PR
- `Docker build` workflow that builds the container on push/PR (no push by default)

To show CI in the workshop:
- open the Actions tab
- show a green run from a small change (README or comment) plus tests
