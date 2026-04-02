---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - email
  - triage
  - customer-support
  - reinforcement-learning
---

# Email Triage & Intelligent Routing Environment

**A real-world OpenEnv environment** that trains AI agents to triage, categorize, prioritize, and resolve incoming customer support emails.

## Why This Matters
Email triage is one of the highest-volume and highest-impact workflows in customer support teams. A well-trained agent can reduce response time, prevent misrouting, and minimize costly escalations. This environment provides a realistic, reproducible, and challenging testbed for RL/RLHF agents.

**Live Space**: [https://huggingface.co/spaces/Sachin903/email-triage-env](https://huggingface.co/spaces/Sachin903/email-triage-env)

---

## Features

- 15 realistic customer emails (5 per difficulty level)
- 3 progressive difficulty levels (Easy → Medium → Hard)
- Dense partial rewards with clear progress signals
- Programmatic graders with meaningful variance
- Full OpenEnv compliance (`reset`, `step`, `state`, `grader`)
- Docker-based deployment on Hugging Face Spaces

---

## Tasks

| Task   | Description                                           | Max Steps | Key Grading Focus                          |
|--------|-------------------------------------------------------|-----------|--------------------------------------------|
| Easy   | Correct category classification                       | 5         | Category accuracy                          |
| Medium | Category + Priority + Terminal action                 | 8         | Category (40%) + Priority (30%) + Terminal (30%) |
| Hard   | Full resolution (Reply draft or Escalation)           | 12        | Category (30%) + Priority (30%) + Quality (40%) |

---

## Baseline Results (Offline Deterministic, seed=42)

| Task   | Grader Score | Total Reward |
|--------|--------------|--------------|
| Easy   | 1.0000       | 0.8950       |
| Medium | 0.9150       | 0.9300       |
| Hard   | 0.8750       | 0.9800       |

**Reproduce locally:**
```bash
python baseline.py
```

### Quick Start (Local)
Bash
```
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Test health
curl http://localhost:8000/health
```

Run baseline:
```
Bashpython baseline.py
```

## API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| **GET** | `/health` | Health check |
| **POST** | `/reset` | Start new episode (`task_level`) |
| **POST** | `/step` | Execute action → observation + reward |
| **GET** | `/tasks` | List all 3 task definitions |
| **POST** | `/grader` | Get final score for current episode |

## Baseline Results (Offline Deterministic, seed=42)

| Task | Grader Score | Total Reward |
| :--- | :--- | :--- |
| Easy | 1.0000 | 0.8950 |
| Medium | 0.9150 | 0.9300 |
| Hard | 0.8750 | 0.9800 |

for current episode

## Project Structure

```text
textemail_triage_env/
├── Dockerfile          # For HF Spaces deployment
├── requirements.txt
├── openenv.yaml         Environment metadata
├── baseline.py          Reproducible baseline
├── models.py
├── server/
│   ├── app.py
│   ├── environment.py  # Core logic + graders
│   └── __init__.py
├── README.md
└── .env                 (gitignored)
```

## Deployment

- Platform: Hugging Face Spaces (Docker SDK)
- Port: 7860
- Auto-deploy: On every git push to main branch

---------------

## Built for Meta PyTorch × Scaler OpenEnv Hackathon 2026 — Round 1
## Author: Sachin Mahajan

## License: MIT