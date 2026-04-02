"""
Final Offline Baseline for Email Triage Env - Optimized for Scores
"""

import json
import os
import time
from typing import Any, Dict

import requests

BASE_URL = os.environ.get("OPENENV_BASE_URL", "http://localhost:8000").rstrip("/")
TASK_LEVELS = ["easy", "medium", "hard"]

def call_post(path: str, json_data: Dict = None):
    url = f"{BASE_URL}{path}"
    try:
        r = requests.post(url, json=json_data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [ERROR] {path} → {e}")
        return None


def get_action(obs: Dict[str, Any], step: int, task_level: str) -> Dict[str, Any]:
    text = (obs.get("email_subject", "") + " " + obs.get("email_body", "")).lower()

    if step == 1:
        if any(w in text for w in ["lottery", "winner", "free money", "claim"]):
            return {"action_type": "categorize", "category": "spam"}
        elif any(w in text for w in ["invoice", "payment", "refund", "charge", "billing", "gstin"]):
            return {"action_type": "categorize", "category": "billing"}
        elif any(w in text for w in ["pricing", "plan", "enterprise", "upgrade", "partnership"]):
            return {"action_type": "categorize", "category": "sales"}
        else:
            return {"action_type": "categorize", "category": "support"}

    if step == 2:
        if any(w in text for w in ["urgent", "asap", "sla", "breach", "dispute", "legal"]):
            return {"action_type": "set_priority", "priority": "urgent"}
        elif any(w in text for w in ["enterprise", "2000", "high"]):
            return {"action_type": "set_priority", "priority": "high"}
        else:
            return {"action_type": "set_priority", "priority": "medium"}

    # Terminal action
    if any(w in text for w in ["lottery", "winner", "free money", "spam"]):
        return {"action_type": "archive"}
    elif any(w in text for w in ["sla", "breach", "dispute", "urgent", "legal", "4th time", "contacting my bank"]):
        return {
            "action_type": "escalate",
            "escalation_reason": "This email requires immediate management or legal attention due to urgency or regulatory issue."
        }
    else:
        return {
            "action_type": "draft_reply",
            "reply_draft": "Dear Customer,\n\nThank you for reaching out. We have received your email and are reviewing it. We apologize for any inconvenience and will provide a resolution within 1-2 business days.\n\nBest regards,\nCustomer Support Team"
        }


def run_episode(task_level: str):
    print(f"\n=== {task_level.upper()} Task ===")

    reset_resp = call_post("/reset", {"task_level": task_level})
    if not reset_resp:
        return {"task_level": task_level, "success": False}

    episode_id = reset_resp.get("episode_id")
    if not episode_id and isinstance(reset_resp.get("observation"), dict):
        episode_id = reset_resp["observation"].get("episode_id")

    if not episode_id:
        print("  [ERROR] Could not get episode_id")
        return {"task_level": task_level, "success": False}

    obs = reset_resp.get("observation", reset_resp)

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < 12:
        steps += 1
        action = get_action(obs, steps, task_level)

        step_resp = call_post("/step", {"action": action, "episode_id": episode_id})
        if not step_resp:
            break

        obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward") or 0.0
        done = step_resp.get("done", False)
        total_reward += reward

        print(f"Step {steps}: {action.get('action_type')} → reward = {reward:.3f} done={done}")

    grade_resp = call_post("/grader", {"episode_id": episode_id})
    grader_score = grade_resp.get("overall", 0.0) if grade_resp else 0.0

    print(f"Grader Score: {grader_score:.4f} | Total Reward: {total_reward:.4f}\n")

    return {
        "task_level": task_level,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "grader_score": round(grader_score, 4)
    }


def main():
    print("=== Email Triage Baseline Runner (Offline) ===\n")

    results = []
    for level in TASK_LEVELS:
        results.append(run_episode(level))
        time.sleep(1)

    print("=== SUMMARY ===")
    for r in results:
        print(f"{r['task_level']:8} | Score: {r.get('grader_score', 0):.4f} | Reward: {r.get('total_reward', 0):.4f}")

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()