"""
inference.py — Email Triage OpenEnv
====================================
Hackathon-required inference script (must be named 'inference.py').

Environment variables (set these before running):
  API_BASE_URL   OpenAI-compatible API base URL
                 e.g. "https://api.openai.com/v1"
  MODEL_NAME     Model identifier, e.g. "gpt-4o-mini"
  HF_TOKEN       API key (Hugging Face token or OpenAI key)

Usage:
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o-mini"
  export HF_TOKEN="sk-..."
  python inference.py

Optional overrides:
  OPENENV_BASE_URL   Server URL (default: http://localhost:8000)
  OPENENV_SEED       Random seed   (default: 42)
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Required environment variables (competition spec) ─────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "")
HF_TOKEN:     str = os.environ.get("HF_TOKEN",     "")

_missing = [k for k, v in [("API_BASE_URL", API_BASE_URL),
                             ("MODEL_NAME",   MODEL_NAME),
                             ("HF_TOKEN",     HF_TOKEN)] if not v]
if _missing:
    print(f"[ERROR] Missing required environment variables: {', '.join(_missing)}")
    print("  Set API_BASE_URL, MODEL_NAME, and HF_TOKEN before running this script.")
    sys.exit(1)

# ── Optional config ────────────────────────────────────────────────────────────
SERVER_URL:  str = os.environ.get("OPENENV_BASE_URL", "http://localhost:8000").rstrip("/")
SEED:        int = int(os.environ.get("OPENENV_SEED", "42"))
TASK_LEVELS: List[str] = ["easy", "medium", "hard"]
MAX_STEPS:   int = 10

# ── OpenAI client (using competition-standard variables) ──────────────────────
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def _post(path: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    r = requests.post(f"{SERVER_URL}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _get(path: str, timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(f"{SERVER_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert email triage agent. Triage each email in EXACTLY this order:
  1. {"action_type": "categorize",   "category":    "<support|sales|spam|internal|billing>"}
  2. {"action_type": "set_priority", "priority":    "<low|medium|high|urgent>"}
  3. ONE terminal action (choose best fit):
       {"action_type": "draft_reply",  "reply_draft":       "<3+ professional sentences>"}
       {"action_type": "escalate",     "escalation_reason": "<detailed reason, 20+ chars>"}
       {"action_type": "archive"}   <- ONLY for spam or notices that need no reply

Guidelines:
- Escalate: SLA breach, legal/GDPR issue, large enterprise deals, threatened chargebacks.
- Archive: spam, lottery, internal meeting notices with no action needed.
- Draft reply: everything else — empathetic and professional, 3+ sentences minimum.
- Output EXACTLY ONE JSON object. No markdown fences. No extra text.
"""


def _user_message(obs: Dict[str, Any]) -> str:
    return (
        f"EMAIL FROM  : {obs.get('sender', '')}\n"
        f"SUBJECT     : {obs.get('email_subject', '')}\n\n"
        f"{obs.get('email_body', '')}\n\n"
        f"--- TRIAGE STATE ---\n"
        f"Category : {obs.get('current_category') or '(not set)'}\n"
        f"Priority : {obs.get('current_priority') or '(not set)'}\n"
        f"Progress : {obs.get('progress', 0):.0%}\n"
        f"Feedback : {obs.get('message', '')}\n\n"
        f"Output your NEXT triage action as a single JSON object:"
    )


# ── Fallback: deterministic rule-based baseline ────────────────────────────────
def _rule_based_action(obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Used when LLM call fails — ensures episode always completes."""
    subject = (obs.get("email_subject") or "").lower()
    body    = (obs.get("email_body")    or "").lower()
    text    = subject + " " + body

    # Step 1 fallback
    if step == 1:
        if any(k in text for k in ["lottery","winner","free money","congratulations","selected for"]):
            return {"action_type": "categorize", "category": "spam"}
        if any(k in text for k in ["invoice","refund","charge","billing","gstin","payment"]):
            return {"action_type": "categorize", "category": "billing"}
        if any(k in text for k in ["pricing","plan","enterprise","partnership","seats"]):
            return {"action_type": "categorize", "category": "sales"}
        if any(k in text for k in ["all-hands","agenda","infosec@","hr@"]):
            return {"action_type": "categorize", "category": "internal"}
        return {"action_type": "categorize", "category": "support"}

    # Step 2 fallback
    if step == 2:
        if any(k in text for k in ["urgent","sla","breach","immediately","today","lawsuit"]):
            return {"action_type": "set_priority", "priority": "urgent"}
        if any(k in text for k in ["enterprise","2000","partnership","series a","incorrect charge"]):
            return {"action_type": "set_priority", "priority": "high"}
        if any(k in text for k in ["spam","lottery","all-hands"]):
            return {"action_type": "set_priority", "priority": "low"}
        return {"action_type": "set_priority", "priority": "medium"}

    # Step 3+ fallback (terminal action)
    cat = obs.get("current_category", "")
    if cat == "spam" or any(k in text for k in ["lottery","all-hands","congratulations","free money"]):
        return {"action_type": "archive"}
    if any(k in text for k in ["sla","legal","breach","gdpr","dispute","bank","executive","4th time"]):
        return {
            "action_type": "escalate",
            "escalation_reason": (
                f"Urgent issue from {obs.get('sender','unknown')}: requires immediate "
                "management/legal/engineering attention due to SLA breach, regulatory "
                "implications, or high-value business risk."
            ),
        }
    return {
        "action_type": "draft_reply",
        "reply_draft": (
            f"Dear Customer,\n\n"
            f"Thank you for reaching out about '{obs.get('email_subject', 'your inquiry')}'. "
            f"We have received your message and our team is actively reviewing it. "
            f"We sincerely apologise for any inconvenience and will provide a resolution "
            f"within 1–2 business days.\n\n"
            f"Please don't hesitate to contact us if you need further assistance.\n\n"
            f"Best regards,\nCustomer Support Team"
        ),
    }


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(task_level: str, seed: int = SEED) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_level.upper()}   seed={seed}   model={MODEL_NAME}")
    print(f"{'='*60}")

    # /reset — always returns episode_id
    try:
        reset_resp = _post("/reset", {"seed": seed, "task_level": task_level})
    except Exception as exc:
        print(f"  [ERROR] /reset failed: {exc}")
        return {"task_level": task_level, "steps": 0, "total_reward": 0.0,
                "grader_scores": {"overall": 0.0, "error": str(exc)}}

    episode_id: str      = reset_resp["episode_id"]
    obs: Dict[str, Any]  = reset_resp["observation"]

    print(f"  episode_id : {episode_id}")
    print(f"  Subject    : {obs.get('email_subject', '')}")
    print(f"  From       : {obs.get('sender', '')}")

    messages     = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    steps        = 0
    done         = False

    while not done and steps < MAX_STEPS:
        steps += 1
        messages.append({"role": "user", "content": _user_message(obs)})

        # LLM call
        action_dict: Optional[Dict[str, Any]] = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": raw})
            action_dict = json.loads(raw)
        except Exception as exc:
            print(f"\n  [LLM step {steps}] fallback ({exc})")

        # Fallback if LLM failed
        if action_dict is None:
            action_dict = _rule_based_action(obs, steps)
            print(f"\n  Step {steps}: [RULE] {action_dict.get('action_type')}", end="")
        else:
            atype = action_dict.get("action_type", "?")
            extra = ""
            if "category" in action_dict: extra += f" → {action_dict['category']}"
            if "priority"  in action_dict: extra += f" → {action_dict['priority']}"
            print(f"\n  Step {steps}: {atype}{extra}", end="")

        # /step — always pass episode_id
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

    # /grader — always pass episode_id
    try:
        grader_scores = _post("/grader", {"episode_id": episode_id})
    except Exception as exc:
        grader_scores = {"overall": 0.0, "error": str(exc)}

    print(f"\n  {'─'*42}")
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "="*60)
    print("  Email Triage OpenEnv — Inference Script")
    print(f"  API_BASE_URL : {API_BASE_URL}")
    print(f"  MODEL_NAME   : {MODEL_NAME}")
    print(f"  SERVER       : {SERVER_URL}")
    print(f"  SEED         : {SEED}")
    print("="*60)

    # Health check
    try:
        h = _get("/health")
        print(f"  Server health: {h.get('status', '?')}")
    except Exception as exc:
        print(f"[ERROR] Cannot reach server at {SERVER_URL}: {exc}")
        print("  Make sure the server is running:")
        print("    uvicorn server.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for level in TASK_LEVELS:
        results.append(run_episode(task_level=level, seed=SEED))
        time.sleep(0.3)

    # Summary table
    print("\n" + "="*60)
    print(f"  RESULTS  (seed={SEED}, model={MODEL_NAME})")
    print("="*60)
    print(f"  {'Task':<10} {'Steps':>6}  {'Reward':>8}  {'Overall':>8}")
    print("  " + "-"*40)
    for r in results:
        ov = r["grader_scores"].get("overall", 0.0)
        if not isinstance(ov, float):
            ov = 0.0
        print(f"  {r['task_level']:<10} {r['steps']:>6}  {r['total_reward']:>8.4f}  {ov:>8.4f}")
    print("="*60)

    # Save for reproducibility & validator
    out = "inference_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out}\n")


if __name__ == "__main__":
    main()
