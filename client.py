"""
Typed Python client for the Email Triage OpenEnv environment.

Usage:
    from client import EmailTriageEnvClient

    env = EmailTriageEnvClient(base_url="http://localhost:8000")
    obs = env.reset(task_level="hard", seed=42)
    print(obs.email_subject)

    obs2, reward, done = env.step({"action_type": "categorize", "category": "billing"})
    scores = env.grade()
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests

try:
    from models import EmailAction, EmailObservation, EmailState
except ImportError:
    from email_triage_env.models import EmailAction, EmailObservation, EmailState  # type: ignore


class EmailTriageEnvClient:
    """
    HTTP client for the Email Triage OpenEnv server.

    Wraps all REST endpoints with typed Python methods.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, str]:
        """Check server health."""
        return self._get("/health")

    def reset(
        self,
        task_level: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> EmailObservation:
        """
        Start a new episode.

        Args:
            task_level: ``"easy"``, ``"medium"``, or ``"hard"``.
            seed: Optional RNG seed for reproducibility.
            episode_id: Optional custom episode identifier.

        Returns:
            The initial :class:`EmailObservation`.
        """
        payload: Dict[str, Any] = {"task_level": task_level}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        data = self._post("/reset", payload)
        return EmailObservation(**data["observation"])

    def step(
        self,
        action: Dict[str, Any] | EmailAction,
        timeout_s: Optional[float] = None,
    ) -> Tuple[EmailObservation, float, bool]:
        """
        Execute one action.

        Args:
            action: Either an :class:`EmailAction` instance or a plain dict.
            timeout_s: Optional server-side timeout.

        Returns:
            ``(observation, reward, done)`` tuple.
        """
        if isinstance(action, EmailAction):
            action_dict = action.model_dump()
        else:
            action_dict = action

        payload: Dict[str, Any] = {"action": action_dict}
        if timeout_s is not None:
            payload["timeout_s"] = timeout_s

        data = self._post("/step", payload)
        obs = EmailObservation(**data["observation"])
        reward = float(data.get("reward") or 0.0)
        done = bool(data.get("done", False))
        return obs, reward, done

    def state(self) -> EmailState:
        """Return the internal environment state."""
        data = self._get("/state")
        return EmailState(**data)

    def tasks(self) -> list:
        """Return all task definitions."""
        data = self._get("/tasks")
        return data.get("tasks", [])

    def grade(self) -> Dict[str, Any]:
        """
        Grade the current episode.

        Returns a dict with ``category_score``, ``priority_score``,
        ``terminal_score``, and ``overall`` (all in [0, 1]).
        """
        return self._post("/grader", {})

    def baseline(
        self,
        task_level: str = "easy",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return baseline action sequence and scores for a sample email."""
        params: Dict[str, Any] = {"task_level": task_level}
        if seed is not None:
            params["seed"] = seed
        return self._get("/baseline", params=params)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def run_episode(
        self,
        task_level: str = "easy",
        actions: list[Dict[str, Any]] | None = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete episode with a pre-defined action list.

        Args:
            task_level: Task difficulty.
            actions: List of action dicts to execute in order.
                     If ``None``, only reset is performed.
            seed: RNG seed.

        Returns:
            ``{"observations": [...], "rewards": [...], "done": bool, "grader": {...}}``
        """
        obs = self.reset(task_level=task_level, seed=seed)
        observations = [obs]
        rewards: list[float] = []
        done = False

        for action in (actions or []):
            if done:
                break
            obs, reward, done = self.step(action)
            observations.append(obs)
            rewards.append(reward)

        grader_scores = self.grade()
        return {
            "observations": [o.model_dump() for o in observations],
            "rewards": rewards,
            "done": done,
            "grader": grader_scores,
        }

    # ------------------------------------------------------------------
    # Private HTTP methods
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}{path}", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()