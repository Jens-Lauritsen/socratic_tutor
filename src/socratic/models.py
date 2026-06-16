"""
Core data models for the SocraticCode Tutor.

Defines the canonical types used across all agents: skill profiles,
hint levels, conversation state, exercises, and evaluation results.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HintLevel(int, Enum):
    """
    Structured hint progression used by the Teaching Agent.

    Level 1 – Question   : make the user think.
    Level 2 – Direction  : point toward the correct area.
    Level 3 – Concept    : explain the underlying concept.
    Level 4 – Partial    : partial solution (only when necessary).
    """

    QUESTION = 1
    DIRECTION = 2
    CONCEPT = 3
    PARTIAL = 4


class ExerciseDifficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


# ---------------------------------------------------------------------------
# Skill tracking
# ---------------------------------------------------------------------------

class SkillEntry(BaseModel):
    """A single skill with a proficiency score (0.0 – 1.0)."""
    name: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_practiced: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserProfile(BaseModel):
    """Persistent learning profile stored locally as JSON."""
    user_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    skills: dict[str, float] = Field(default_factory=dict)
    common_mistakes: list[str] = Field(default_factory=list)
    completed_exercises: list[str] = Field(default_factory=list)
    current_topic: str | None = None
    preferred_difficulty: ExerciseDifficulty = ExerciseDifficulty.BEGINNER
    total_sessions: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_skill(self, name: str) -> float:
        return self.skills.get(name, 0.0)

    def update_skill(self, name: str, score: float) -> None:
        self.skills[name] = max(0.0, min(1.0, score))
        self.updated_at = datetime.now(timezone.utc)

    def record_mistake(self, mistake: str) -> None:
        if mistake not in self.common_mistakes:
            self.common_mistakes.append(mistake)

    def known_skills(self) -> list[str]:
        """Return skills the user has demonstrated proficiency in (>=0.6)."""
        return [k for k, v in self.skills.items() if v >= 0.6]


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str  # "user" | "tutor"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationState(BaseModel):
    """Tracks the state of an ongoing tutoring session."""
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    messages: list[Message] = Field(default_factory=list)
    current_hint_level: HintLevel = HintLevel.QUESTION
    hint_count: int = 0
    max_hints_before_reveal: int = 5
    topic: str | None = None
    user_attempts: int = 0
    revealed_solution: bool = False

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
        self.user_attempts += 1

    def add_tutor_message(self, content: str) -> None:
        self.messages.append(Message(role="tutor", content=content))

    def advance_hint(self) -> HintLevel:
        self.hint_count += 1
        if self.hint_count >= self.max_hints_before_reveal:
            self.current_hint_level = HintLevel.PARTIAL
        elif self.hint_count >= 3:
            self.current_hint_level = HintLevel.CONCEPT
        elif self.hint_count >= 2:
            self.current_hint_level = HintLevel.DIRECTION
        else:
            self.current_hint_level = HintLevel.QUESTION
        return self.current_hint_level

    def should_reveal(self) -> bool:
        return self.hint_count >= self.max_hints_before_reveal


# ---------------------------------------------------------------------------
# Exercise
# ---------------------------------------------------------------------------

class Exercise(BaseModel):
    """A programming exercise generated for the user."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    description: str
    difficulty: ExerciseDifficulty
    target_skills: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    evaluation_criteria: list[str] = Field(default_factory=list)


class ExerciseSubmission(BaseModel):
    """User's solution to an exercise."""
    exercise_id: str
    code: str
    explanation: str | None = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Result of evaluating a user's understanding or solution."""
    concept: str
    understanding: float = Field(ge=0.0, le=1.0)
    needs_practice: bool = False
    feedback: str = ""
    missing_concepts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    provider: str  # "openai" | "anthropic" | "ollama"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
