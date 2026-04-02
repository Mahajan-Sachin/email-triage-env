# server/app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uuid

# Import environment
try:
    from environment import EmailTriageEnvironment
except ImportError:
    from .environment import EmailTriageEnvironment

try:
    from models import EmailAction
except ImportError:
    from ..models import EmailAction

app = FastAPI(title="Email Triage OpenEnv")

# Session registry to support multiple concurrent episodes
_sessions: Dict[str, EmailTriageEnvironment] = {}
_session_lock = None  # Will be initialized on first use

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_level: str = "easy"

class StepRequest(BaseModel):
    action: Dict[str, Any]
    episode_id: Optional[str] = None

@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "email_triage_env"}

@app.post("/reset")
async def reset(req: ResetRequest):
    global _session_lock
    if _session_lock is None:
        import threading
        _session_lock = threading.Lock()

    episode_id = str(uuid.uuid4())

    with _session_lock:
        env = EmailTriageEnvironment()
        obs = env.reset(seed=req.seed, episode_id=episode_id, task_level=req.task_level)

        _sessions[episode_id] = env

    return {
        "episode_id": episode_id,
        "observation": obs.model_dump(),
        "done": False,
        "reward": None
    }

@app.post("/step")
async def step(req: StepRequest):
    if not req.episode_id or req.episode_id not in _sessions:
        raise HTTPException(status_code=400, detail="Valid episode_id is required")

    env = _sessions[req.episode_id]
    try:
        action = EmailAction(**req.action)
        obs = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_state(episode_id: str):
    if episode_id not in _sessions:
        raise HTTPException(status_code=400, detail="Valid episode_id is required")
    env = _sessions[episode_id]
    return env.state.model_dump()

@app.get("/tasks")
async def get_tasks():
    return {"tasks": EmailTriageEnvironment.get_tasks()}

@app.post("/grader")
async def grader(req: Dict[str, Any]):
    episode_id = req.get("episode_id")
    if not episode_id or episode_id not in _sessions:
        raise HTTPException(status_code=400, detail="Valid episode_id is required")

    env = _sessions[episode_id]
    return env.grade()

@app.get("/baseline")
async def baseline(task_level: str = "easy"):
    env = EmailTriageEnvironment()
    obs = env.reset(task_level=task_level)
    action_sequence = env.baseline_action(obs.model_dump())

    return {
        "task_level": task_level,
        "baseline_actions": action_sequence["steps"],
        "grader_scores": env.grade()
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()