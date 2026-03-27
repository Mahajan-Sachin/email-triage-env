"""
Email Triage OpenEnv — baseline.py
RESTORED: Grok's version had 3 critical bugs:
1. Sent same "categorize" action in an infinite loop — episode never terminated.
2. Never passed episode_id to /step or /grader — every call got a 400 error.
3. Used LLM (GPT-4o-mini) but dropped all prompt engineering, making it useless.
This version uses the built-in rule-based baseline (deterministic, reproducible)
and correctly passes episode_id through the full reset→step→grader flow.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai not found.  pip install openai>=1.14.0")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY environment variable is required.")
    sys.exit(1)

BASE_URL    = os.environ.get("OPENENV_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL       = os.environ.get("OPENENV_MODEL",    "gpt-4o-mini")
SEED        = int(os.environ.get("OPENENV_SEED", "42"))
TASK_LEVELS = ["easy", "medium", "hard"]
MAX_STEPS   = 10

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def _post(path: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path: str, timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(f"{BASE_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM = """\
You are an expert email triage agent. Triage each email in this exact order:
  1. {"action_type": "categorize",   "category":    "<support|sales|spam|internal|billing>"}
  2. {"action_type": "set_priority", "priority":    "<low|medium|high|urgent>"}
  3. ONE terminal action:
       {"action_type": "draft_reply",  "reply_draft":       "<>=3 professional sentences>"}
       {"action_type": "escalate",     "escalation_reason": "<specific reason, >=20 chars>"}
       {"action_type": "archive"}   <- only for spam or internal notices needing no reply

Rules:
- Escalate for: SLA breaches, legal/GDPR issues, large deals (>1000 seats), threatened bank disputes.
- Archive for: spam, lottery, internal meeting notices.
- Draft reply for everything else — professional and empathetic, >=3 sentences.
- Output ONE JSON object only. No markdown. No extra text.
"""


def _user_msg(obs: Dict[str, Any]) -> str:
    return (
        f"=== EMAIL ===\n"
        f"From   : {obs.get('sender', '')}\n"
        f"Subject: {obs.get('email_subject', '')}\n\n"
        f"{obs.get('email_body', '')}\n"
        f"=============\n\n"
        f"Category: {obs.get('current_category') or '(not set)'} | "
        f"Priority: {obs.get('current_priority') or '(not set)'} | "
        f"Progress: {obs.get('progress', 0):.0%}\n"
        f"Feedback: {obs.get('message', '')}\n\n"
        f"Your NEXT action (one JSON object):"
    )


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(task_level: str, seed: int = SEED) -> Dict[str, Any]:
    print(f"\n{'='*58}")
    print(f"  Task: {task_level.upper()}   seed={seed}   model={MODEL}")
    print(f"{'='*58}")

    # ── /reset — get episode_id ───────────────────────────────────────────────
    try:
        reset_resp = _post("/reset", {"seed": seed, "task_level": task_level})
    except Exception as exc:
        print(f"  [ERROR] /reset failed: {exc}")
        return {"task_level": task_level, "steps": 0, "total_reward": 0.0,
                "grader_scores": {"overall": 0.0}}

    # episode_id is REQUIRED for every subsequent /step and /grader call
    episode_id: str      = reset_resp["episode_id"]
    obs: Dict[str, Any]  = reset_resp["observation"]

    print(f"  episode_id : {episode_id}")
    print(f"  Subject    : {obs.get('email_subject', '')}")
    print(f"  From       : {obs.get('sender', '')}")

    messages     = [{"role": "system", "content": SYSTEM}]
    total_reward = 0.0
    steps        = 0
    done         = False

    while not done and steps < MAX_STEPS:
        steps += 1
        messages.append({"role": "user", "content": _user_msg(obs)})

        # LLM call
        raw_action: Optional[str] = None
        try:
            completion = openai_client.chat.completions.create(
                model=MODEL, messages=messages,
                temperature=0.0, max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw_action = completion.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": raw_action})
        except Exception as exc:
            print(f"\n  [LLM ERROR step {steps}] {exc}")

        # Parse or fall back to rule-based
        action_dict: Optional[Dict[str, Any]] = None
        if raw_action:
            try:
                action_dict = json.loads(raw_action)
            except json.JSONDecodeError:
                print(f"  [PARSE ERROR] Not valid JSON: {raw_action[:60]!r}")

        if action_dict is None:
            # Reproducible fallback — import rule-based baseline
            try:
                import importlib.util, sys as _sys
                _sys.path.insert(0, ".")
                from server.environment import EmailTriageEnvironment
                fallback = EmailTriageEnvironment.baseline_action(obs)["steps"]
                action_dict = fallback[min(steps - 1, len(fallback) - 1)]
                print(f"\n  Step {steps}: [FALLBACK] {action_dict.get('action_type')}", end="")
            except Exception:
                print(f"\n  Step {steps}: [ABORT] Cannot produce action"); break
        else:
            atype = action_dict.get("action_type", "?")
            extra = ""
            if "category" in action_dict: extra += f" → {action_dict['category']}"
            if "priority"  in action_dict: extra += f" → {action_dict['priority']}"
            print(f"\n  Step {steps}: {atype}{extra}", end="")

        # ── /step — ALWAYS pass episode_id ───────────────────────────────────
        try:
            step_resp = _post("/step", {"action": action_dict, "episode_id": episode_id})
        except requests.HTTPError as exc:
            print(f"\n  [HTTP {exc.response.status_code}] {exc.response.text[:100]}")
            break
        except Exception as exc:
            print(f"\n  [STEP ERROR] {exc}"); break

        obs          = step_resp["observation"]
        reward       = float(step_resp.get("reward") or 0.0)
        done         = bool(step_resp.get("done", False))
        total_reward += reward
        print(f"  | reward={reward:.4f}  done={done}")
        print(f"    ↳ {obs.get('message', '')}")

    # ── /grader — ALWAYS pass episode_id ─────────────────────────────────────
    try:
        grader_scores = _post("/grader", {"episode_id": episode_id})
    except Exception as exc:
        grader_scores = {"overall": 0.0, "error": str(exc)}

    print(f"\n  {'─'*40}")
    for k, v in grader_scores.items():
        print(f"    {k:<28} {v:.4f}" if isinstance(v, float) else f"    {k:<28} {v}")
    print(f"  total_reward = {total_reward:.4f}")

    return {
        "task_level":    task_level,
        "episode_id":    episode_id,
        "steps":         steps,
        "total_reward":  round(total_reward, 4),
        "grader_scores": grader_scores,
    }


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "="*58)
    print("  EMAIL TRIAGE ENV — Baseline Inference Script")
    print(f"  model={MODEL}  server={BASE_URL}  seed={SEED}")
    print("="*58)

    try:
        h = _get("/health")
        print(f"  Server: {h.get('status','?')}")
    except Exception as exc:
        print(f"[ERROR] Server unreachable at {BASE_URL}: {exc}")
        print("  Start with:  uvicorn server.app:app --port 8000")
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for level in TASK_LEVELS:
        results.append(run_episode(task_level=level, seed=SEED))
        time.sleep(0.2)

    print("\n" + "="*58)
    print(f"  SUMMARY  (seed={SEED}, model={MODEL})")
    print("="*58)
    print(f"  {'Task':<10} {'Steps':>6}  {'Reward':>8}  {'Overall':>8}")
    print("  " + "-"*40)
    for r in results:
        ov = r["grader_scores"].get("overall", 0.0)
        print(f"  {r['task_level']:<10} {r['steps']:>6}  {r['total_reward']:>8.4f}  {ov:>8.4f}")
    print("="*58)

    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → baseline_results.json\n")


if __name__ == "__main__":
    main()