"""
Email Triage OpenEnv — server/environment.py
RESTORED: Grok's edit replaced the entire email corpus (15 emails), all grader
helper functions, step() logic, grade(), get_tasks(), baseline_action() and
_compute_progress()/_completion_bonus() with placeholder comments. This is the
complete, working version.
"""
from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Import the real OpenEnv Environment base class.
# Fallback stub is minimal but signature-accurate.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import Environment
except ImportError:
    try:
        from openenv.core.env_server.interfaces import Environment
    except ImportError:
        class Environment(ABC):  # type: ignore[no-redef]
            """Minimal local stub — used only when openenv package is not installed."""
            SUPPORTS_CONCURRENT_SESSIONS: bool = False

            def __init__(self, transform: Any = None, rubric: Any = None) -> None:
                self.transform = transform
                self.rubric = rubric

            @abstractmethod
            def reset(self, seed: Optional[int] = None,
                      episode_id: Optional[str] = None, **kwargs: Any) -> Any: ...

            @abstractmethod
            def step(self, action: Any, timeout_s: Optional[float] = None,
                     **kwargs: Any) -> Any: ...

            @property
            @abstractmethod
            def state(self) -> Any: ...

# ---------------------------------------------------------------------------
# Models import (PYTHONPATH=/app set by Dockerfile)
# ---------------------------------------------------------------------------
try:
    from models import EmailAction, EmailObservation, EmailState
except ImportError:
    from email_triage_env.models import EmailAction, EmailObservation, EmailState  # type: ignore


# ===========================================================================
# Email corpus — 5 emails × 3 difficulty levels = 15 total
# ===========================================================================

EASY_EMAILS: List[Dict[str, Any]] = [
    {
        "subject": "Cannot reset my password",
        "body": (
            "Hi support team,\n\nI have been trying to reset my password for two days "
            "but the reset link never arrives. I checked spam too. "
            "Please help me regain access.\n\nRegards, Alex"
        ),
        "sender": "alex.johnson@gmail.com",
        "correct_category": "support",
        "correct_priority": "medium",
        "expected_action": "draft_reply",
        "requires_reply": True,
        "requires_escalation": False,
        "reply_keywords": ["password", "reset", "link", "account"],
    },
    {
        "subject": "AMAZING DEAL – Claim your prize NOW",
        "body": (
            "CONGRATULATIONS! You have been selected. "
            "Click to claim: http://totally-legit-deals.xyz/free-money"
        ),
        "sender": "deals@spam-mailer.net",
        "correct_category": "spam",
        "correct_priority": "low",
        "expected_action": "archive",
        "requires_reply": False,
        "requires_escalation": False,
        "reply_keywords": [],
    },
    {
        "subject": "Monthly all-hands meeting – agenda attached",
        "body": "Team, agenda for Friday's all-hands is attached. Cheers, HR",
        "sender": "hr@ourcompany.com",
        "correct_category": "internal",
        "correct_priority": "low",
        "expected_action": "archive",
        "requires_reply": False,
        "requires_escalation": False,
        "reply_keywords": [],
    },
    {
        "subject": "Interested in your Pro plan",
        "body": (
            "Hello, I run a small agency and I'm interested in the Pro plan. "
            "Could you send pricing details?\n\nBest, Priya"
        ),
        "sender": "priya@creativestudio.in",
        "correct_category": "sales",
        "correct_priority": "medium",
        "expected_action": "draft_reply",
        "requires_reply": True,
        "requires_escalation": False,
        "reply_keywords": ["pricing", "pro", "plan"],
    },
    {
        "subject": "Invoice INV-001 appears overdue",
        "body": "Dear Accounts, Invoice INV-001 from last month seems unpaid. Please advise.",
        "sender": "accounts@vendor.com",
        "correct_category": "billing",
        "correct_priority": "medium",
        "expected_action": "draft_reply",
        "requires_reply": True,
        "requires_escalation": False,
        "reply_keywords": ["invoice", "payment", "overdue"],
    },
]

MEDIUM_EMAILS: List[Dict[str, Any]] = [
    {
        "subject": "App crashes on every launch – deadline tomorrow",
        "body": (
            "Your app crashes every time on iOS 17.4 / iPhone 15. "
            "I have a deadline tomorrow and cannot work. Completely unacceptable. – David"
        ),
        "sender": "david.t@consulting.com",
        "correct_category": "support",
        "correct_priority": "urgent",
        "expected_action": "escalate",
        "requires_reply": False,
        "requires_escalation": True,
        "reply_keywords": ["crash", "ios", "engineering", "escalate"],
    },
    {
        "subject": "Enterprise contract renewal – 2,000 seats",
        "body": (
            "Our contract (2,000 seats) is up for renewal next month. "
            "We're evaluating alternatives — we need a meeting to discuss pricing and SLAs.\n"
            "Thomas Reed, VP Operations, MegaCorp Inc."
        ),
        "sender": "t.reed@megacorp.com",
        "correct_category": "sales",
        "correct_priority": "urgent",
        "expected_action": "escalate",
        "requires_reply": False,
        "requires_escalation": True,
        "reply_keywords": ["enterprise", "renewal", "meeting"],
    },
    {
        "subject": "Incorrect charge of $849 on my card",
        "body": (
            "I was billed $849 for annual but only signed up for monthly at $79. "
            "I need an immediate refund."
        ),
        "sender": "maria.g@hotmail.com",
        "correct_category": "billing",
        "correct_priority": "high",
        "expected_action": "escalate",
        "requires_reply": False,
        "requires_escalation": True,
        "reply_keywords": ["refund", "charge", "billing"],
    },
    {
        "subject": "Winner – Nigeria Lottery 2026",
        "body": "You have won $5,000,000. Send bank details to claim. Confidential.",
        "sender": "lottery-nigeria@free-mail.tk",
        "correct_category": "spam",
        "correct_priority": "low",
        "expected_action": "archive",
        "requires_reply": False,
        "requires_escalation": False,
        "reply_keywords": [],
    },
    {
        "subject": "API rate limit docs contradictory",
        "body": (
            "Page 12 says 1000 req/min but /events returns 429 after 500. "
            "Could you clarify the rate limits?"
        ),
        "sender": "dev@techstartup.io",
        "correct_category": "support",
        "correct_priority": "medium",
        "expected_action": "draft_reply",
        "requires_reply": True,
        "requires_escalation": False,
        "reply_keywords": ["api", "rate", "limit", "documentation"],
    },
]

HARD_EMAILS: List[Dict[str, Any]] = [
    {
        "subject": "URGENT: Data breach – customer PII exposed",
        "body": (
            "Security monitoring detected unauthorised access to our production DB. "
            "~12,000 customer records may be exfiltrated. GDPR Article 33 applies. "
            "Needs immediate executive and legal attention. – InfoSec"
        ),
        "sender": "infosec@ourcompany.com",
        "correct_category": "internal",
        "correct_priority": "urgent",
        "expected_action": "escalate",
        "requires_reply": False,
        "requires_escalation": True,
        "reply_keywords": ["breach", "gdpr", "legal", "security", "executive"],
    },
    {
        "subject": "Refund denied 4th time – contacting my bank",
        "body": (
            "I contacted support 4 times re: refund for #ORD-55821. Nothing happened. "
            "I will dispute the charge and report you to consumer protection TODAY."
        ),
        "sender": "angry.customer@gmail.com",
        "correct_category": "billing",
        "correct_priority": "urgent",
        "expected_action": "escalate",
        "requires_reply": False,
        "requires_escalation": True,
        "reply_keywords": ["refund", "order", "dispute", "escalate"],
    },
    {
        "subject": "Partnership opportunity – AI startup (Series A)",
        "body": (
            "We are an AI startup ($8M Series A) building on your platform. "
            "We'd like to explore a partnership and reseller agreement. "
            "CEO would like a call with your BD team. – Ananya Shah, IntelliRoute AI"
        ),
        "sender": "ananya@intelliroute.ai",
        "correct_category": "sales",
        "correct_priority": "high",
        "expected_action": "draft_reply",
        "requires_reply": True,
        "requires_escalation": False,
        "reply_keywords": ["partnership", "business development", "meeting", "call"],
    },
    {
        "subject": "Platform down 2 hours – SLA breach",
        "body": (
            "150 users can't access the platform since 09:00 IST. "
            "This breaches clause 4.2 (99.9% uptime). We demand an incident report "
            "and compensation credit within 24 hours."
        ),
        "sender": "cto@enterprise-client.com",
        "correct_category": "support",
        "correct_priority": "urgent",
        "expected_action": "escalate",
        "requires_reply": False,
        "requires_escalation": True,
        "reply_keywords": ["sla", "incident", "credit", "engineering"],
    },
    {
        "subject": "GSTIN error on invoice – cannot claim tax credit",
        "body": (
            "Invoice #INV-2026-0445 has wrong GSTIN. "
            "We cannot claim input tax credit. Please issue a credit note urgently. "
            "Finance Team, Sunrise Exports"
        ),
        "sender": "finance@sunriseexports.in",
        "correct_category": "billing",
        "correct_priority": "high",
        "expected_action": "draft_reply",
        "requires_reply": True,
        "requires_escalation": False,
        "reply_keywords": ["gstin", "invoice", "credit note", "correction"],
    },
]

CORPUS: Dict[str, List[Dict[str, Any]]] = {
    "easy":   EASY_EMAILS,
    "medium": MEDIUM_EMAILS,
    "hard":   HARD_EMAILS,
}

# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------
_PRIORITY_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2, "urgent": 3}


def _category_score(submitted: Optional[str], target: str) -> float:
    return 1.0 if submitted == target else 0.0


def _priority_score(submitted: Optional[str], target: str) -> float:
    if submitted is None:
        return 0.0
    delta = abs(_PRIORITY_RANK.get(submitted, -9) - _PRIORITY_RANK.get(target, -9))
    return round(max(0.0, 1.0 - delta * 0.4), 4)


def _reply_quality_score(reply: Optional[str], keywords: List[str]) -> float:
    """Multi-component: length (0.30) + professionalism (0.35) + keyword coverage (0.35)."""
    if not reply or len(reply.strip()) < 20:
        return 0.0
    text = reply.lower()
    length_score = min(1.0, len(reply) / 200) * 0.30
    pro_phrases  = ["dear","hello","thank you","apolog","understand","assist",
                    "resolve","please","regard","sincerely","team"]
    pro_score    = min(0.35, sum(1 for p in pro_phrases if p in text) * 0.07)
    if keywords:
        kw_score = (sum(1 for kw in keywords if kw.lower() in text) / len(keywords)) * 0.35
    else:
        kw_score = 0.35
    return round(min(1.0, length_score + pro_score + kw_score), 4)


def _escalation_quality_score(reason: Optional[str]) -> float:
    """Score by length + domain-relevant keywords."""
    if not reason or len(reason.strip()) < 10:
        return 0.0
    length_score = min(0.5, len(reason) / 100)
    keywords = ["urgent","legal","executive","engineering","refund","sla",
                "security","breach","contract","gdpr","dispute"]
    kw_score = sum(0.1 for kw in keywords if kw.lower() in reason.lower())
    return round(min(1.0, length_score + kw_score), 4)


# ===========================================================================
# Environment
# ===========================================================================
class EmailTriageEnvironment(Environment):
    """
    Email Triage & Intelligent Routing — OpenEnv Environment.
    Inherits from openenv.core.env_server.interfaces.Environment.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, transform: Any = None, rubric: Any = None) -> None:
        super().__init__(transform=transform, rubric=rubric)
        self._state: Optional[EmailState] = None
        self._email: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        task_level: str           = "easy",
        **kwargs:   Any,
    ) -> EmailObservation:
        if seed is not None:
            random.seed(seed)

        task_level = task_level if task_level in CORPUS else "easy"
        self._email = random.choice(CORPUS[task_level]).copy()
        email_id    = str(uuid.uuid4())[:8]

        self._state = EmailState(
            episode_id                  = episode_id or str(uuid.uuid4()),
            step_count                  = 0,
            email_id                    = email_id,
            task_level                  = task_level,
            target_category             = self._email["correct_category"],
            target_priority             = self._email["correct_priority"],
            expected_action             = self._email["expected_action"],
            requires_reply              = self._email["requires_reply"],
            requires_escalation         = self._email["requires_escalation"],
            reply_keywords              = list(self._email["reply_keywords"]),
            category_submitted          = None,
            priority_submitted          = None,
            reply_submitted             = None,
            escalation_reason_submitted = None,
            reply_drafted               = False,
            escalated                   = False,
            archived                    = False,
            category_reward_given       = False,
            priority_reward_given       = False,
            reply_reward_given          = False,
            escalation_reward_given     = False,
            history                     = [],
        )

        return EmailObservation(
            done              = False,
            reward            = 0.0,
            email_id          = email_id,
            email_subject     = self._email["subject"],
            email_body        = self._email["body"],
            sender            = self._email["sender"],
            thread_length     = 1,
            current_category  = None,
            current_priority  = None,
            reply_drafted     = False,
            escalated         = False,
            message           = (
                "New email received. Triage: categorise → set priority → "
                f"reply / escalate / archive.  Task: {task_level}."
            ),
            available_actions = ["categorize","set_priority","draft_reply","escalate","archive"],
            progress          = 0.0,
            task_level        = task_level,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self,
        action:    EmailAction,
        timeout_s: Optional[float] = None,
        **kwargs:  Any,
    ) -> EmailObservation:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        s       = self._state
        reward  = 0.0
        done    = False
        message = ""

        if action.action_type == "categorize":
            s.category_submitted = action.category
            if not s.category_reward_given and action.category:
                r = _category_score(action.category, s.target_category)
                reward += r * 0.4
                s.category_reward_given = True
                message = (f"✓ Correct category: {action.category}" if r == 1.0
                           else f"✗ Category '{action.category}' incorrect.")

        elif action.action_type == "set_priority":
            s.priority_submitted = action.priority
            if not s.priority_reward_given and action.priority:
                r = _priority_score(action.priority, s.target_priority)
                reward += r * 0.3
                s.priority_reward_given = True
                message = (f"✓ Correct priority: {action.priority}"
                           if action.priority == s.target_priority
                           else f"Priority '{action.priority}' set — partial credit.")

        elif action.action_type == "draft_reply":
            s.reply_submitted = action.reply_draft
            s.reply_drafted   = True
            if not s.reply_reward_given:
                r = _reply_quality_score(action.reply_draft, s.reply_keywords)
                if s.requires_escalation:
                    reward += r * 0.06
                    message = "Reply drafted, but escalation was required."
                elif s.requires_reply:
                    reward += r * 0.3
                    message = f"Reply drafted. Quality: {r:.2f}"
                else:
                    reward += r * 0.1
                    message = "Reply drafted (archive was optimal)."
                s.reply_reward_given = True
            done = True

        elif action.action_type == "escalate":
            s.escalation_reason_submitted = action.escalation_reason
            s.escalated = True
            if not s.escalation_reward_given:
                r = _escalation_quality_score(action.escalation_reason)
                reward += r * 0.3 if s.requires_escalation else r * 0.1
                s.escalation_reward_given = True
                message = (f"Escalated. Reason quality: {r:.2f}"
                           if s.requires_escalation else "Escalated (sub-optimal).")
            done = True

        elif action.action_type == "archive":
            s.archived = True
            if s.expected_action == "archive":
                reward += 0.30
                message = "✓ Correctly archived."
            else:
                reward += 0.05
                message = "Archived (reply/escalate was needed)."
            done = True

        if done:
            reward += self._completion_bonus()

        reward = round(min(1.0, max(0.0, reward)), 4)
        s.history.append({"step": s.step_count, "action": action.action_type,
                          "reward": reward, "done": done})

        return EmailObservation(
            done              = done,
            reward            = reward,
            email_id          = s.email_id,
            email_subject     = self._email["subject"],
            email_body        = self._email["body"],
            sender            = self._email["sender"],
            thread_length     = 1,
            current_category  = s.category_submitted,
            current_priority  = s.priority_submitted,
            reply_drafted     = s.reply_drafted,
            escalated         = s.escalated,
            message           = message or "Action recorded.",
            available_actions = ([] if done
                                 else ["categorize","set_priority","draft_reply","escalate","archive"]),
            progress          = self._compute_progress(),
            task_level        = s.task_level,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------
    @property
    def state(self) -> EmailState:
        if self._state is None:
            raise RuntimeError("Not initialised — call reset() first.")
        return self._state

    # ------------------------------------------------------------------
    # Task list
    # ------------------------------------------------------------------
    @staticmethod
    def get_tasks() -> List[Dict[str, Any]]:
        return [
            {
                "name": "easy",
                "description": "Categorise the email (support/sales/spam/internal/billing).",
                "difficulty": "easy", "max_steps": 5,
                "grader": {"score_range": [0.0, 1.0], "dimensions": ["category_score"]},
            },
            {
                "name": "medium",
                "description": "Categorise + set priority + correct terminal action.",
                "difficulty": "medium", "max_steps": 8,
                "grader": {"score_range": [0.0, 1.0],
                           "weights": {"category": 0.4, "priority": 0.3, "terminal": 0.3}},
            },
            {
                "name": "hard",
                "description": "Full resolution: categorise, prioritise, draft quality reply or detailed escalation.",
                "difficulty": "hard", "max_steps": 12,
                "grader": {"score_range": [0.0, 1.0],
                           "weights": {"category": 0.3, "priority": 0.3, "terminal": 0.4}},
            },
        ]

    # ------------------------------------------------------------------
    # Programmatic grader — consistent dict shape always
    # ------------------------------------------------------------------
    def grade(self) -> Dict[str, Any]:
        if self._state is None:
            return {"category_score": 0.0, "priority_score": 0.0,
                    "terminal_score": 0.0, "overall": 0.0,
                    "task_level": "unknown", "steps_taken": 0,
                    "error": "not_initialised"}

        s = self._state
        cat_score = _category_score(s.category_submitted, s.target_category)
        pri_score = _priority_score(s.priority_submitted, s.target_priority)

        if s.requires_reply:
            term_score = _reply_quality_score(s.reply_submitted, s.reply_keywords)
        elif s.requires_escalation:
            term_score = _escalation_quality_score(s.escalation_reason_submitted)
        else:
            term_score = 1.0 if s.archived else (0.3 if (s.reply_drafted or s.escalated) else 0.0)

        if s.task_level == "easy":
            overall = cat_score
        elif s.task_level == "medium":
            overall = cat_score * 0.40 + pri_score * 0.30 + term_score * 0.30
        else:
            overall = cat_score * 0.30 + pri_score * 0.30 + term_score * 0.40

        efficiency = max(0.0, 0.05 - s.step_count * 0.005)
        overall    = round(min(1.0, overall + efficiency), 4)

        return {
            "category_score": round(cat_score, 4),
            "priority_score": round(pri_score, 4),
            "terminal_score": round(term_score, 4),
            "overall":        overall,
            "task_level":     s.task_level,
            "steps_taken":    s.step_count,
        }

    # ------------------------------------------------------------------
    # Deterministic rule-based baseline
    # ------------------------------------------------------------------
    @staticmethod
    def baseline_action(observation: Dict[str, Any]) -> Dict[str, Any]:
        subject = (observation.get("email_subject") or "").lower()
        body    = (observation.get("email_body")    or "").lower()
        text    = subject + " " + body

        if any(s in text for s in ["lottery","winner","free money","claim","congratulations","selected for"]):
            category = "spam"
        elif any(s in text for s in ["invoice","payment","refund","charge","billed","billing","gstin","tax"]):
            category = "billing"
        elif any(s in text for s in ["pricing","plan","enterprise","upgrade","partnership","seats","reseller"]):
            category = "sales"
        elif any(s in text for s in ["all-hands","agenda","infosec@","hr@","ourcompany"]):
            category = "internal"
        else:
            category = "support"

        if any(s in text for s in ["urgent","asap","immediately","sla breach","today","dispute","lawsuit","contacting my bank"]):
            priority = "urgent"
        elif any(s in text for s in ["enterprise contract","2000 seats","partnership","incorrect charge","series a"]):
            priority = "high"
        elif any(s in text for s in ["spam","all-hands","lottery"]):
            priority = "low"
        else:
            priority = "medium"

        if category == "spam" or any(s in text for s in ["lottery","all-hands","free money","congratulations"]):
            terminal: Dict[str, Any] = {"action_type": "archive"}
        elif any(s in text for s in ["sla","legal","breach","gdpr","dispute","bank","executive","infosec","contacting my bank","4th time"]):
            terminal = {
                "action_type": "escalate",
                "escalation_reason": (
                    f"Email from {observation.get('sender','unknown')} requires urgent "
                    "management/legal/engineering attention due to SLA, regulatory, "
                    "or high-value business implications."
                ),
            }
        else:
            terminal = {
                "action_type": "draft_reply",
                "reply_draft": (
                    f"Dear Customer,\n\n"
                    f"Thank you for reaching out regarding '{observation.get('email_subject','')}'. "
                    f"We have received your message and our team is reviewing it. "
                    f"We apologise for any inconvenience and will provide a resolution "
                    f"within 1-2 business days.\n\n"
                    f"Please do not hesitate to contact us if you need further assistance.\n\n"
                    f"Best regards,\nCustomer Support Team"
                ),
            }

        return {
            "steps": [
                {"action_type": "categorize", "category": category},
                {"action_type": "set_priority", "priority": priority},
                terminal,
            ]
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _compute_progress(self) -> float:
        if self._state is None:
            return 0.0
        s = self._state
        return round(sum([
            s.category_reward_given,
            s.priority_reward_given,
            s.reply_reward_given or s.escalation_reward_given,
        ]) / 3.0, 4)

    def _completion_bonus(self) -> float:
        if self._state is None:
            return 0.0
        s = self._state
        if s.task_level in ("medium", "hard") and s.category_reward_given and s.priority_reward_given:
            return 0.05 if s.task_level == "medium" else 0.10
        return 0.0