"""
Tutor Engine — the central orchestrator.

The TutorEngine ties together the ContextAgent, TeachingAgent, and
EvaluationAgent, managing the full Socratic conversation flow.

It enforces teaching rules (e.g. never reveal a solution too early)
and maintains session state.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import AsyncIterator

from socratic.agents.context import ContextAgent
from socratic.agents.evaluation import EvaluationAgent
from socratic.agents.teaching import TeachingAgent
from socratic.models import (
    ConversationState,
    EvaluationResult,
    Exercise,
    ExerciseDifficulty,
    ExerciseSubmission,
    HintLevel,
    UserProfile,
)
from socratic.prompts import (
    SOCRATIC_SYSTEM_PROMPT,
    build_exercise_prompt,
    get_system_prompt_for_mode,
)
from socratic.providers import LLMProvider


class TutorEngine:
    """
    The central engine that orchestrates Socratic tutoring sessions.

    It owns the conversation loop, manages hint progression, delegates
    to specialist agents, and talks to the LLM provider.
    """

    def __init__(
        self,
        provider: LLMProvider,
        profile: UserProfile,
        *,
        working_dir: str | None = None,
    ) -> None:
        self.provider = provider
        self.profile = profile
        self.state: ConversationState | None = None
        self.context_agent = ContextAgent(working_dir)
        self.teaching_agent = TeachingAgent()
        self.evaluation_agent = EvaluationAgent()

    # ==================================================================
    # Session management
    # ==================================================================

    def start_session(self, topic: str | None = None) -> ConversationState:
        """Begin a new tutoring session."""
        self.state = ConversationState(topic=topic)
        self.profile.total_sessions += 1
        return self.state

    def end_session(self) -> None:
        """Clean up the current session."""
        self.state = None

    # ==================================================================
    # Main conversation handler
    # ==================================================================

    async def handle_message(
        self,
        user_message: str,
        *,
        code_snippet: str | None = None,
        file_path: str | None = None,
    ) -> str:
        """
        Process a user message and return the tutor's response.

        This is the main entry point for the Socratic dialogue.
        """
        if self.state is None:
            self.start_session()

        assert self.state is not None

        # Record the user's message
        self.state.add_user_message(user_message)

        # Gather context
        context = self.context_agent.build_context(
            user_message,
            self.state,
            code_snippet=code_snippet,
            file_path=file_path,
        )

        # Decide teaching action
        action = self.teaching_agent.decide_action(
            user_message,
            self.state,
            self.profile,
        )

        # Build and send prompt to LLM
        prompt = self.teaching_agent.build_teaching_prompt(
            self.state,
            self.profile,
            hint_level=action["hint_level"],
            context=context,
        )

        response = await self.provider.generate(
            prompt,
            system_prompt=get_system_prompt_for_mode("tutor"),
        )

        # Record tutor's response
        self.state.add_tutor_message(response)

        # Mark revealed if partial solution was given
        if action["should_reveal"]:
            self.state.revealed_solution = True

        return response

    async def handle_message_stream(
        self,
        user_message: str,
        *,
        code_snippet: str | None = None,
        file_path: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream the tutor's response token by token (when the provider
        supports streaming). Falls back to non-streaming if not.
        """
        response = await self.handle_message(
            user_message,
            code_snippet=code_snippet,
            file_path=file_path,
        )
        # For now yield the whole response; true streaming is a future enhancement
        yield response

    # ==================================================================
    # Exercise mode
    # ==================================================================

    async def generate_exercise(
        self,
        topic: str | None = None,
        difficulty: ExerciseDifficulty | None = None,
    ) -> Exercise:
        """Generate a programming exercise tailored to the user."""
        prompt = build_exercise_prompt(
            self.profile,
            topic=topic,
            difficulty=difficulty,
        )

        response = await self.provider.generate(
            prompt,
            system_prompt=get_system_prompt_for_mode("exercise"),
            temperature=0.8,
            max_tokens=8192,  # Reasoning models burn tokens thinking first
        )

        return self._parse_exercise_response(response)

    async def evaluate_submission(
        self,
        exercise: Exercise,
        submission: ExerciseSubmission,
    ) -> list[EvaluationResult]:
        """Evaluate a user's exercise submission."""
        return self.evaluation_agent.evaluate_exercise_submission(
            exercise,
            submission,
            self.profile,
        )

    # ==================================================================
    # Evaluation mode
    # ==================================================================

    async def evaluate_understanding(
        self,
        concept: str,
        user_explanation: str,
    ) -> EvaluationResult:
        """Use the LLM to evaluate the user's understanding of a concept."""
        prompt = self.evaluation_agent.build_evaluation_prompt(
            concept,
            user_explanation,
            self.profile,
        )

        response = await self.provider.generate(
            prompt,
            system_prompt=get_system_prompt_for_mode("evaluate"),
            temperature=0.3,  # Lower temperature for evaluation
        )

        result = self.evaluation_agent.parse_evaluation_response(response, concept)

        # Update profile
        self.profile.update_skill(concept, result.understanding)
        for gap in result.missing_concepts:
            if gap not in self.profile.common_mistakes:
                self.profile.common_mistakes.append(gap)

        return result

    # ==================================================================
    # Direct hint request (bypasses teaching agent logic)
    # ==================================================================

    async def get_hint(
        self,
        level: HintLevel,
        context: str | None = None,
    ) -> str:
        """Request a hint at a specific level (for programmatic use)."""
        if self.state is None:
            self.start_session()
        assert self.state is not None

        prompt = self.teaching_agent.build_teaching_prompt(
            self.state,
            self.profile,
            hint_level=level,
            context={"problem_summary": context} if context else None,
        )

        return await self.provider.generate(
            prompt,
            system_prompt=get_system_prompt_for_mode("tutor"),
        )

    # ==================================================================
    # Convenience
    # ==================================================================

    def get_conversation_history(self) -> list[dict]:
        """Return the conversation history as a list of dicts."""
        if self.state is None:
            return []
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
            for m in self.state.messages
        ]

    @property
    def current_hint_level(self) -> HintLevel | None:
        if self.state is None:
            return None
        return self.state.current_hint_level

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _parse_exercise_response(response: str) -> Exercise:
        """Parse the LLM's JSON exercise response into an Exercise object."""
        # Reasoning models often output their thinking before the JSON.
        # Try to extract just the JSON portion.
        
        # Strip code fences if present
        clean = response.strip()
        if "```json" in clean:
            match = re.search(r"```json\s*(.*?)\s*```", clean, re.DOTALL)
            if match:
                clean = match.group(1).strip()
        elif "```" in clean:
            match = re.search(r"```\s*(.*?)\s*```", clean, re.DOTALL)
            if match:
                clean = match.group(1).strip()

        # Try to find a JSON object — look for the outermost braces
        # First try: find a complete JSON object
        for attempt in range(3):
            try:
                start = clean.find("{")
                end = clean.rfind("}")
                if start >= 0 and end > start:
                    json_str = clean[start:end + 1]
                    data = json.loads(json_str)
                    if "title" in data or "description" in data:
                        return Exercise(
                            title=data.get("title", "Untitled Exercise"),
                            description=data.get("description", ""),
                            difficulty=ExerciseDifficulty(
                                data.get("difficulty", "beginner")
                            ),
                            target_skills=data.get("target_skills", []),
                            hints=data.get("hints", []),
                            evaluation_criteria=data.get("evaluation_criteria", []),
                        )
                break
            except json.JSONDecodeError:
                # Try to fix truncated JSON by adding missing closing braces
                if attempt == 0:
                    # Count braces and add missing ones
                    open_count = clean.count("{") - clean.count("}")
                    if open_count > 0:
                        clean = clean.rstrip() + ("}" * open_count)
                elif attempt == 1:
                    # Remove trailing incomplete text after last valid brace
                    last_valid = clean.rfind('"')
                    if last_valid > 0:
                        clean = clean[:last_valid + 1] + "}"
                else:
                    break

        # Fallback: if the response looks like reasoning, try to pull out
        # the meaningful parts as a plain-text exercise
        # Strip obvious reasoning prefixes
        clean = response.strip()
        for prefix in ["We are asked", "The user wants", "I need to", "Let me"]:
            if clean.startswith(prefix):
                # Find where the actual exercise starts (look for "title" or JSON)
                json_start = clean.find('{"title"')
                if json_start < 0:
                    json_start = clean.find('{\n  "title"')
                if json_start >= 0:
                    json_str = clean[json_start:]
                    # Find matching closing brace
                    end = json_str.rfind("}")
                    if end > 0:
                        try:
                            data = json.loads(json_str[:end + 1])
                            return Exercise(
                                title=data.get("title", "Untitled Exercise"),
                                description=data.get("description", ""),
                                difficulty=ExerciseDifficulty(data.get("difficulty", "beginner")),
                                target_skills=data.get("target_skills", []),
                                hints=data.get("hints", []),
                                evaluation_criteria=data.get("evaluation_criteria", []),
                            )
                        except json.JSONDecodeError:
                            pass
                break

        return Exercise(
            title="Practice Exercise",
            description=clean[:500],
            difficulty=ExerciseDifficulty.BEGINNER,
            target_skills=["general_programming"],
        )
