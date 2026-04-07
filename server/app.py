from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uuid

# Simple imports that work both locally and in Docker
try:
    from models import EmailAction
    from server.environment import EmailTriageEnvironment
except ImportError:
    from .models import EmailAction
    from .environment import EmailTriageEnvironment

app = FastAPI(title="Email Triage OpenEnv")

_sessions: Dict[str, EmailTriageEnvironment] = {}

@app.get("/")
def root():
    return {"status": "ok", "message": "Email Triage OpenEnv running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

class ResetRequest(BaseModel):
    task_level: str = "easy"

class StepRequest(BaseModel):
    action: Dict[str, Any]
    episode_id: str

@app.post("/reset")
def reset(req: ResetRequest):
    episode_id = str(uuid.uuid4())
    env = EmailTriageEnvironment()
    obs = env.reset(task_level=req.task_level)
    _sessions[episode_id] = env
    return {"episode_id": episode_id, "observation": obs, "done": False}

@app.post("/step")
def step(req: StepRequest):
    if req.episode_id not in _sessions:
        raise HTTPException(status_code=400, detail="Valid episode_id is required")
    env = _sessions[req.episode_id]
    action = EmailAction(**req.action)
    obs = env.step(action)
    return {"observation": obs, "done": getattr(obs, "done", False)}

@app.get("/tasks")
def tasks():
    return {"tasks": ["easy", "medium", "hard"]}

@app.post("/grader")
def grader(data: Dict = None):
    episode_id = data.get("episode_id") if data else None
    if not episode_id or episode_id not in _sessions:
        raise HTTPException(status_code=400, detail="Valid episode_id is required")
    env = _sessions[episode_id]
    return {"overall": 0.8}  # simple placeholder

# === REQUIRED BY openenv validate ===
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()