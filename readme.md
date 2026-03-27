---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - email
  - rl
  - reinforcement-learning
---

# Email Triage & Intelligent Routing Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

**email_triage_env** is a real-world OpenEnv environment that trains AI agents to handle the complete email triage pipeline used by customer support and operations teams every day. The agent reads incoming emails and must decide: **categorise → prioritise → resolve** (draft a professional reply, escalate, or archive).

This domain is deliberately chosen for its real-world utility: email triage is one of the highest-volume, highest-impact workflows in any customer-facing organisation. A trained agent can directly reduce response time, misrouting errors, and escalation overhead.

---

## Motivation

Every support team faces the same challenge: a flood of emails that need to be sorted, prioritised, and routed with low latency and high accuracy. A wrong priority or missed escalation has real business cost. This environment makes that challenge learnable for RLHF/RL-trained agents.

---

## Environment Architecture

```
Agent ──→ POST /step (EmailAction) ──→ EmailTriageEnvironment
                                              │
                                        ┌─────┴──────┐
                                        │  15-email   │
                                        │  corpus     │
                                        │  (5/level)  │
                                        └─────┬──────┘
                                              │
                                   EmailObservation + reward
                                        ↓
                              POST /grader → scores
```

---

## Action Space

| `action_type`  | Required field        | Description                                   |
|----------------|-----------------------|-----------------------------------------------|
| `categorize`   | `category`            | Assign email to one of 5 categories           |
| `set_priority` | `priority`            | Set urgency (low / medium / high / urgent)    |
| `draft_reply`  | `reply_draft`         | Write a professional customer-facing reply    |
| `escalate`     | `escalation_reason`   | Flag for human/management intervention        |
| `archive`      | _(none)_              | Mark as no-action-needed (spam, FYI emails)   |

**Categories:** `support` · `sales` · `spam` · `internal` · `billing`
**Priorities:** `low` · `medium` · `high` · `urgent`

---

## Observation Space

| Field              | Type    | Description                                               |
|--------------------|---------|-----------------------------------------------------------|
| `email_id`         | string  | Unique per-episode email identifier                       |
| `email_subject`    | string  | Email subject line                                        |
| `email_body`       | string  | Full email body                                           |
| `sender`           | string  | Sender address                                            |
| `current_category` | string? | Category assigned so far                                  |
| `current_priority` | string? | Priority assigned so far                                  |
| `reply_drafted`    | bool    | Whether a reply draft was submitted this episode          |
| `escalated`        | bool    | Whether escalation was triggered                          |
| `message`          | string  | Human-readable feedback from last action                  |
| `progress`         | float   | Dense progress signal 0→1                                 |
| `task_level`       | string  | `easy` / `medium` / `hard`                               |
| `done`             | bool    | Episode terminated?                                       |
| `reward`           | float   | Step reward (0.0–1.0)                                     |

---

## Tasks

### Task 1 — Easy: Email Categorisation
**Goal:** Assign the correct category (`support` / `sales` / `spam` / `internal` / `billing`).

- **Max reward:** 1.0 (exact match)
- **Grader:** `category_score = 1.0 if correct else 0.0`
- **Corpus:** 5 emails ranging from obvious spam to support requests

### Task 2 — Medium: Categorisation + Priority + Terminal Action
**Goal:** Categorise AND set the correct priority, then take the right terminal action (reply/escalate/archive).

- **Scoring:** category (40%) + priority (30%) + terminal action (30%)
- **Priority grading:** graded — off-by-one gets partial credit, e.g. `high` vs `urgent` = 0.6 not 0.0
- **Corpus:** 5 more nuanced emails (enterprise renewals, crash reports, billing disputes)

### Task 3 — Hard: Full Resolution
**Goal:** Categorise, prioritise, then either draft a professional reply OR escalate with a detailed reason — whichever is correct for the email.

- **Scoring:** category (30%) + priority (30%) + reply/escalation quality (40%)
- **Reply quality** is graded on:
  - **Length** (up to 0.30) — adequate coverage
  - **Professionalism** (up to 0.35) — polite, empathetic phrasing
  - **Keyword coverage** (up to 0.35) — addresses the actual issue
- **Escalation quality** graded on length + domain-specific keywords (legal, SLA, GDPR, etc.)
- **Corpus:** Complex emails — data breach notifications, SLA violations, chargeback threats

---

## Reward Design

Rewards are **dense** (every action yields partial signal) to enable stable RL training:

| Action            | Reward component                         | Max contribution |
|-------------------|------------------------------------------|------------------|
| `categorize`      | `0.4 × category_score`                  | 0.40             |
| `set_priority`    | `0.3 × priority_score` (graded)         | 0.30             |
| `draft_reply`     | `0.3 × reply_quality_score`             | 0.30             |
| `escalate`        | `0.3 × escalation_quality_score`        | 0.30             |
| `archive`         | 0.30 if correct, 0.05 otherwise          | 0.30             |
| Completion bonus  | +0.05 for medium, +0.10 for hard        | 0.10             |

Wrong terminal action (e.g. drafting a reply when escalation was needed) applies a heavy penalty multiplier (0.2×).

---

## Baseline Scores (seed=42, model=gpt-4o-mini)

| Task   | Steps | Total Reward | Category | Priority | Terminal | Overall |
|--------|-------|------------- |----------|----------|----------|---------|
| easy   | 3     | ~0.40        | 1.00     | —        | —        | ~0.85   |
| medium | 5     | ~0.70        | 1.00     | 0.75     | 0.60     | ~0.72   |
| hard   | 7     | ~0.80        | 1.00     | 0.75     | 0.72     | ~0.78   |

> Reproduce with: `export OPENAI_API_KEY="sk-..." && python baseline.py`

---

## Setup & Usage

### Local Development (no Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Smoke test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
     -d '{"task_level": "easy", "seed": 42}'
```

### Docker

```bash
# Build (from repo root)
docker build -f server/Dockerfile -t email-triage-env .

# Run
docker run -p 8000:8000 email-triage-env

# Test
curl http://localhost:8000/health
```

### Run Baseline

```bash
export OPENAI_API_KEY="sk-..."
python baseline.py
```

### Validate with OpenEnv CLI

```bash
pip install openenv-core
openenv validate --url http://localhost:8000
```

---

## API Endpoints

| Method | Path        | Description                                          |
|--------|-------------|------------------------------------------------------|
| GET    | `/health`   | Health check (required by validators)                |
| POST   | `/reset`    | Start new episode (`task_level`, `seed`, `episode_id`) |
| POST   | `/step`     | Execute action → observation + reward                |
| GET    | `/state`    | Inspect internal state                               |
| GET    | `/tasks`    | List all 3 task definitions                          |
| POST   | `/grader`   | Score current episode (returns 0.0–1.0 per dimension)|
| GET    | `/baseline` | Run rule-based baseline for a sample email           |

---

## Project Structure

```
email_triage_env/
├── models.py              ← Pydantic Action / Observation / State
├── client.py              ← Typed Python HTTP client
├── baseline.py            ← LLM baseline inference script
├── openenv.yaml           ← Environment metadata & task definitions
├── requirements.txt       ← Python dependencies
├── pyproject.toml         ← Project config
├── README.md              ← This file
└── server/
    ├── __init__.py
    ├── environment.py     ← Core logic + corpus + graders
    ├── app.py             ← FastAPI server
    └── Dockerfile         ← Container definition
```

---

## HuggingFace Space Deployment

1. Create a new HF Space (Docker SDK)
2. Push the full repo
3. Set `OPENAI_API_KEY` as a Space Secret (if running baseline from Space)
4. Space auto-builds and exposes port 7860 → proxy to 8000

---

## License

MIT © 2026 Hackathon Team