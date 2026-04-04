"""
inference.py - Mandatory submission file for OpenEnv Hackathon
Must follow exact [START], [STEP], [END] logging format
"""

import os
import json
import requests
from typing import List

# Official variables expected by organizers
API_BASE_URL = os.getenv("API_BASE_URL", "https://sachin903-email-triage-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY", "")

TASK_NAME = "email_triage"
BENCHMARK = "email_triage_env"
MAX_STEPS = 12

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action(obs: dict, step: int) -> dict:
    text = (obs.get("email_subject", "") + " " + obs.get("email_body", "")).lower()

    if step == 1:
        if any(w in text for w in ["lottery", "winner", "free", "claim"]):
            return {"action_type": "categorize", "category": "spam"}
        elif any(w in text for w in ["invoice", "payment", "refund", "charge", "billing", "gstin"]):
            return {"action_type": "categorize", "category": "billing"}
        else:
            return {"action_type": "categorize", "category": "support"}

    if step == 2:
        if any(w in text for w in ["urgent", "sla", "breach", "dispute"]):
            return {"action_type": "set_priority", "priority": "urgent"}
        else:
            return {"action_type": "set_priority", "priority": "medium"}

    # Terminal
    if any(w in text for w in ["lottery", "winner", "free"]):
        return {"action_type": "archive"}
    elif any(w in text for w in ["urgent", "sla", "breach", "dispute", "4th time"]):
        return {"action_type": "escalate", "escalation_reason": "Requires immediate attention"}
    else:
        return {
            "action_type": "draft_reply",
            "reply_draft": "Thank you for your email. We will respond within 1-2 business days."
        }

def main():
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Reset
        reset_resp = requests.post(
            f"{API_BASE_URL}/reset",
            json={"task_level": "easy"},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        reset_resp.raise_for_status()
        data = reset_resp.json()
        episode_id = data.get("episode_id")
        obs = data.get("observation", data)

        for step in range(1, MAX_STEPS + 1):
            action = get_action(obs, step)
            action_str = json.dumps(action, separators=(',', ':'))

            step_resp = requests.post(
                f"{API_BASE_URL}/step",
                json={"action": action, "episode_id": episode_id},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()

            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False))
            obs = step_data.get("observation", {})

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done)

            if done:
                break

        score = sum(rewards) / max(len(rewards), 1)
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()