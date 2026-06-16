"""
High-level SocraticTutor API — the primary interface for applications.

This module ties together the LLM provider, tutor engine, and learning
profile into a simple, cohesive interface that CLI apps (and eventually
web apps or IDE extensions) can use.
"""

from __future__ import annotations

import os
from pathlib import Path

from socratic.engine import TutorEngine
from socratic.models import (
    EvaluationResult,
    Exercise,
    ExerciseDifficulty,
    ExerciseSubmission,
    HintLevel,
    ProviderConfig,
    UserProfile,
)
from socratic.progress import LearningProfile
from socratic.providers import LLMProvider, create_provider


class SocraticTutor:
    """
    The main entry point for the SocraticCode tutor.

    Usage:
        tutor = SocraticTutor(provider="openai", model="gpt-4o-mini")
        tutor.start(topic="Python classes")

        response = await tutor.ask("How do I create a class?")
        print(response)
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        profile_path: str | Path | None = None,
        working_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the Socratic Tutor.

        Parameters
        ----------
        provider : str
            One of "openai", "anthropic", "ollama".
            Defaults to SOCRATIC_LLM_PROVIDER env var, then "openai".
        model : str, optional
            Override the default model for the chosen provider.
        api_key : str, optional
            API key. If not provided, reads from the provider's standard
            environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
        base_url : str, optional
            Base URL for the API (useful for proxies or Ollama).
        profile_path : str or Path, optional
            Custom path for the learning profile JSON file.
        working_dir : str or Path, optional
            Working directory for code context analysis.
        """
        provider_name = provider or os.getenv("SOCRATIC_LLM_PROVIDER", "openai")

        # Resolve API key by provider convention
        key_env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": None,  # Ollama doesn't need an API key
            "opencode": "OPENCODE_API_KEY",
        }
        resolved_key = api_key
        if resolved_key is None and provider_name in key_env_map:
            env_var = key_env_map[provider_name]
            if env_var:
                resolved_key = os.getenv(env_var)

        # Resolve base URL for Ollama
        resolved_base = base_url
        if resolved_base is None and provider_name == "ollama":
            resolved_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        config = ProviderConfig(
            provider=provider_name,
            model=model,
            api_key=resolved_key,
            base_url=resolved_base,
        )

        self._provider: LLMProvider = create_provider(config)
        self._learning_profile = LearningProfile(profile_path)
        self._profile: UserProfile = self._learning_profile.load()
        self._engine = TutorEngine(
            self._provider,
            self._profile,
            working_dir=str(working_dir) if working_dir else None,
        )

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def start(self, topic: str | None = None) -> str:
        """Start a new tutoring session."""
        state = self._engine.start_session(topic)
        self._profile.current_topic = topic
        self._save_profile()

        topic_str = topic or "programming"
        return (
            f"Welcome! I'm your Socratic programming tutor.\n\n"
            f"You're learning: {topic_str}\n\n"
            f"I'll guide you with questions and hints rather than giving "
            f"you direct answers. Ready? What would you like to work on?"
        )

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    async def ask(
        self,
        message: str,
        *,
        code_snippet: str | None = None,
        file_path: str | None = None,
    ) -> str:
        """
        Send a message to the tutor and get a response.

        The tutor will use the Socratic method: starting with questions,
        progressing through hints, and only revealing information when
        the user has made genuine effort.
        """
        return await self._engine.handle_message(
            message,
            code_snippet=code_snippet,
            file_path=file_path,
        )

    async def ask_stream(
        self,
        message: str,
        *,
        code_snippet: str | None = None,
        file_path: str | None = None,
    ):
        """Stream the tutor's response (async generator)."""
        async for chunk in self._engine.handle_message_stream(
            message,
            code_snippet=code_snippet,
            file_path=file_path,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # Hints
    # ------------------------------------------------------------------

    async def hint(self, level: HintLevel | None = None, context: str | None = None) -> str:
        """Request a hint at a specific level."""
        if level is None:
            level = self._engine.current_hint_level or HintLevel.QUESTION
        return await self._engine.get_hint(level, context)

    # ------------------------------------------------------------------
    # Exercises
    # ------------------------------------------------------------------

    async def generate_exercise(
        self,
        topic: str | None = None,
        difficulty: str | ExerciseDifficulty | None = None,
    ) -> Exercise:
        """Generate a programming exercise."""
        if isinstance(difficulty, str):
            difficulty = ExerciseDifficulty(difficulty)
        return await self._engine.generate_exercise(topic, difficulty)

    async def submit_solution(
        self,
        exercise: Exercise,
        code: str,
        explanation: str | None = None,
    ) -> list[EvaluationResult]:
        """Submit a solution to an exercise and get evaluation."""
        submission = ExerciseSubmission(
            exercise_id=exercise.id,
            code=code,
            explanation=explanation,
        )
        results = await self._engine.evaluate_submission(exercise, submission)
        self._save_profile()
        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        concept: str,
        user_explanation: str,
    ) -> EvaluationResult:
        """Evaluate the user's understanding of a concept."""
        result = await self._engine.evaluate_understanding(concept, user_explanation)
        self._save_profile()
        return result

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    @property
    def profile(self) -> UserProfile:
        return self._profile

    def get_progress_summary(self) -> str:
        """Return a human-readable summary of the user's learning progress."""
        return self._learning_profile.get_summary(self._profile)

    def save_profile(self) -> None:
        """Persist the learning profile to disk."""
        self._save_profile()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict]:
        """Return the current conversation history."""
        return self._engine.get_conversation_history()

    @property
    def hint_level(self) -> HintLevel | None:
        return self._engine.current_hint_level

    @property
    def provider_name(self) -> str:
        return self._provider.config.provider

    @property
    def model_name(self) -> str:
        return self._provider._model

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_profile(self) -> None:
        self._learning_profile.save(self._profile)
