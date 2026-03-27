# models.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Base types with fallback
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    class Action(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: Optional[str] = None
        step_count: int = 0


ActionType = Literal["categorize", "set_priority", "draft_reply", "escalate", "archive"]
CategoryType = Literal["support", "sales", "spam", "internal", "billing"]
PriorityType = Literal["low", "medium", "high", "urgent"]


class EmailAction(Action):
    action_type: ActionType = Field(..., description="Type of action")
    category: Optional[CategoryType] = Field(default=None)
    priority: Optional[PriorityType] = Field(default=None)
    reply_draft: Optional[str] = Field(default=None, min_length=10)
    escalation_reason: Optional[str] = Field(default=None, min_length=5)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailObservation(Observation):
    # Email content
    email_id: str = Field(...)
    email_subject: str = Field(...)
    email_body: str = Field(...)
    sender: str = Field(...)
    thread_length: int = Field(default=1, ge=1)

    # Triage state
    current_category: Optional[CategoryType] = Field(default=None)
    current_priority: Optional[PriorityType] = Field(default=None)
    reply_drafted: bool = Field(default=False)
    escalated: bool = Field(default=False)

    # Feedback
    message: str = Field(...)
    available_actions: List[ActionType] = Field(default_factory=list)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    task_level: str = Field(default="easy")

    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailState(State):
    """All fields used in environment.py are explicitly declared here."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: Optional[str] = None
    step_count: int = 0

    email_id: str = ""
    task_level: str = "easy"

    # Ground truth
    target_category: CategoryType = "support"
    target_priority: PriorityType = "medium"
    expected_action: str = "draft_reply"
    requires_reply: bool = False
    requires_escalation: bool = False
    reply_keywords: List[str] = Field(default_factory=list)

    # Agent decisions
    category_submitted: Optional[str] = None
    priority_submitted: Optional[str] = None
    reply_submitted: Optional[str] = None
    escalation_reason_submitted: Optional[str] = None
    reply_drafted: bool = False
    escalated: bool = False
    archived: bool = False

    # Reward flags
    category_reward_given: bool = False
    priority_reward_given: bool = False
    reply_reward_given: bool = False
    escalation_reward_given: bool = False

    history: List[Dict[str, Any]] = Field(default_factory=list)