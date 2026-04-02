from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uuid

try:
    from environment import EmailTriageEnvironment
except ImportError:
    from .environment import EmailTriageEnvironment

try:
    from models import EmailAction
except ImportError:
    from ..models import EmailAction

app = FastAPI(title="Email Triage OpenEnv")

_sessions: Dict[str, EmailTriageEnvironment] = {}

@app.get("/")
async def root():
    return {"message": "Email Triage OpenEnv is running", "status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "email_triage_env"}

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_level: str = "easy"

class StepRequest(BaseModel):
    action: Dict[str, Any]
    episode_id: Optional[str] = None

@app.post("/reset")
async def reset(req: ResetRequest):
    episode_id = str(uuid.uuid4())
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

@app.get("/tasks")
async def get_tasks():
    return {"tasks": EmailTriageEnvironment.get_tasks()}

@app.post("/grader")
async def grader(data: Dict[str, Any] = None):
    episode_id = data.get("episode_id") if data else None
    if not episode_id or episode_id not in _sessions:
        raise HTTPException(status_code=400, detail="Valid episode_id is required")
    env = _sessions[episode_id]
    return env.grade()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)