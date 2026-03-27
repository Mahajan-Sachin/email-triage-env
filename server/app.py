"""
Email Triage OpenEnv — server/app.py
Clean, production-ready version for hackathon submission.
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Simple path setup
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import environment and models
try:
    from server.environment import EmailTriageEnvironment
except ImportError:
    from environment import EmailTriageEnvironment  # type: ignore

try:
    from models import EmailAction
except ImportError:
    from email_triage_env.models import EmailAction  # type: ignore


app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment for training AI agents.",
    version="1.0.0",
)

# Session registry: episode_id → Environment instance
_sessions: Dict[str, EmailTriageEnvironment] = {}


class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=None)
    task_level: str = Field(default="easy")


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(...)
    episode_id: Optional[str] = Field(default=None)


class GraderRequest(BaseModel):
    episode_id: Optional[str] = Field(default=None)


@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "email_triage_env"}


@app.post("/reset")
async def reset(req: ResetRequest):
    episode_id = str(uuid.uuid4())
    _sessions[episode_id] = EmailTriageEnvironment()

    env = _sessions[episode_id]
    obs = env.reset(seed=req.seed, episode_id=episode_id, task_level=req.task_level)

    return JSONResponse(content={
        "observation": obs.model_dump(),
        "episode_id": episode_id,
        "reward": None,
        "done": False,
    })


@app.post("/step")
async def step(req: StepRequest):
    if not req.episode_id or req.episode_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="Valid episode_id is required. Call /reset first."
        )

    env = _sessions[req.episode_id]
    try:
        action_obj = EmailAction(**req.action)
        obs = env.step(action_obj)
        return JSONResponse(content={
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
            "episode_id": req.episode_id,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state(episode_id: Optional[str] = None):
    if not episode_id or episode_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return JSONResponse(content=_sessions[episode_id].state.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def get_tasks():
    return JSONResponse(content={"tasks": EmailTriageEnvironment.get_tasks()})


@app.post("/grader")
async def grader(req: GraderRequest = GraderRequest()):
    if not req.episode_id or req.episode_id not in _sessions:
        raise HTTPException(status_code=400, detail="Valid episode_id is required")
    try:
        scores = _sessions[req.episode_id].grade()
        return JSONResponse(content=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/baseline")
async def baseline(task_level: str = "easy", seed: Optional[int] = 42):
    """Self-contained baseline run"""
    env = EmailTriageEnvironment()
    obs = env.reset(seed=seed, task_level=task_level)
    action_seq = EmailTriageEnvironment.baseline_action(obs.model_dump())

    rewards = []
    for action_data in action_seq.get("steps", []):
        try:
            obs2 = env.step(EmailAction(**action_data))
            rewards.append(obs2.reward or 0.0)
        except Exception:
            rewards.append(0.0)

    return JSONResponse(content={
        "task_level": task_level,
        "baseline_steps": action_seq.get("steps", []),
        "total_reward": round(sum(rewards), 4),
        "grader_scores": env.grade(),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)