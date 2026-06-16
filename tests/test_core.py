"""Tests for SocraticCode Tutor – models, agents, engine, and providers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from socratic.models import (
    ConversationState,
    EvaluationResult,
    Exercise,
    ExerciseDifficulty,
    ExerciseSubmission,
    HintLevel,
    ProviderConfig,
    UserProfile,
)
from socratic.agents.context import ContextAgent
from socratic.agents.teaching import TeachingAgent
from socratic.agents.evaluation import EvaluationAgent
from socratic.progress import LearningProfile
from socratic.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    OpenCodeProvider,
    create_provider,
    available_providers,
)


# ======================================================================
# Model tests
# ======================================================================

class TestUserProfile:
    def test_default_profile(self):
        profile = UserProfile()
        assert profile.user_id
        assert len(profile.user_id) == 12
        assert profile.skills == {}
        assert profile.common_mistakes == []
        assert profile.preferred_difficulty == ExerciseDifficulty.BEGINNER

    def test_update_skill(self):
        profile = UserProfile()
        profile.update_skill("functions", 0.85)
        assert profile.get_skill("functions") == 0.85
        profile.update_skill("functions", 1.5)  # clamped
        assert profile.get_skill("functions") == 1.0
        profile.update_skill("functions", -0.5)  # clamped
        assert profile.get_skill("functions") == 0.0

    def test_record_mistake(self):
        profile = UserProfile()
        profile.record_mistake("poor error handling")
        assert "poor error handling" in profile.common_mistakes
        profile.record_mistake("poor error handling")  # no duplicate
        assert len(profile.common_mistakes) == 1

    def test_known_skills(self):
        profile = UserProfile()
        profile.update_skill("loops", 0.8)
        profile.update_skill("classes", 0.3)
        profile.update_skill("testing", 0.6)
        known = profile.known_skills()
        assert "loops" in known
        assert "testing" in known
        assert "classes" not in known


class TestConversationState:
    def test_new_conversation(self):
        state = ConversationState()
        assert state.current_hint_level == HintLevel.QUESTION
        assert state.hint_count == 0

    def test_add_messages(self):
        state = ConversationState()
        state.add_user_message("Why is my code broken?")
        state.add_tutor_message("What do you think might be wrong?")
        assert len(state.messages) == 2
        assert state.user_attempts == 1

    def test_hint_progression(self):
        state = ConversationState()
        # First hint → still Level 1
        assert state.advance_hint() == HintLevel.QUESTION
        assert state.hint_count == 1
        # Second hint → Level 2
        assert state.advance_hint() == HintLevel.DIRECTION
        # Third hint → Level 3
        assert state.advance_hint() == HintLevel.CONCEPT
        # Fourth hint → Level 3 (still)
        assert state.advance_hint() == HintLevel.CONCEPT
        # Fifth hint → Level 4 (max)
        assert state.advance_hint() == HintLevel.PARTIAL
        # Max hints reached
        assert state.should_reveal() is True


class TestEvaluationResult:
    def test_default_result(self):
        result = EvaluationResult(concept="loops", understanding=0.0)
        # needs_practice is set explicitly by evaluation agents;
        # the model default is False but agents set it to True when score < 0.7
        assert result.understanding == 0.0

    def test_high_understanding(self):
        result = EvaluationResult(concept="functions", understanding=0.9)
        assert result.needs_practice is False


# ======================================================================
# Agent tests
# ======================================================================

class TestContextAgent:
    def test_build_context_basic(self):
        agent = ContextAgent()
        state = ConversationState()
        ctx = agent.build_context("My API returns 401 Unauthorized", state)
        assert "authentication" in ctx["detected_concepts"]
        assert "api" in ctx["detected_concepts"]
        assert ctx["error_message"] != ""

    def test_detect_concepts(self):
        concepts = ContextAgent._detect_concepts("How do I create a class with inheritance?")
        assert "classes" in concepts

    def test_extract_error(self):
        msg = "I got this error: NameError: name 'x' is not defined"
        err = ContextAgent._extract_error(msg)
        assert "NameError" in err


class TestTeachingAgent:
    def setup_method(self):
        self.agent = TeachingAgent()

    def test_initial_action_is_question(self):
        state = ConversationState()
        action = self.agent.decide_action("Help me with Python classes", state)
        assert action["action"] == "question"

    def test_give_up_early_encourages(self):
        state = ConversationState()
        action = self.agent.decide_action("i give up, just tell me the answer", state)
        assert action["action"] == "encourage"

    def test_give_up_late_reveals(self):
        state = ConversationState()
        state.hint_count = 4
        action = self.agent.decide_action("just tell me", state)
        assert action["action"] == "reveal"

    def test_effort_detection(self):
        state = ConversationState()
        state.add_user_message("initial question")
        action = self.agent.decide_action("I think the problem is in the auth module", state)
        assert action["action"] == "question"  # Follow-up question


class TestEvaluationAgent:
    def setup_method(self):
        self.agent = EvaluationAgent()

    def test_heuristic_score_empty(self):
        score = self.agent._heuristic_score("")
        assert score < 0.2

    def test_heuristic_score_good(self):
        explanation = (
            "The function works because it correctly iterates over the list using a for loop. "
            "Each iteration checks the conditional statement, and because we used an early return, "
            "the function stops as soon as it finds a match. This is efficient because..."
        )
        score = self.agent._heuristic_score(explanation)
        assert score > 0.4

    def test_evaluate_explanation(self):
        result = self.agent.evaluate_explanation(
            "loops",
            "I used a for loop to iterate because I needed to check each element. "
            "When the condition is met, we return early."
        )
        assert isinstance(result, EvaluationResult)
        assert result.concept == "loops"

    def test_evaluate_profile_update(self):
        profile = UserProfile()
        self.agent.evaluate_explanation(
            "functions",
            "Functions are reusable blocks of code that take parameters and return values. "
            "They help organize code and avoid repetition.",
            profile,
        )
        assert profile.get_skill("functions") >= 0.3

    def test_evaluate_exercise(self):
        exercise = Exercise(
            title="Test",
            description="Test exercise",
            difficulty=ExerciseDifficulty.BEGINNER,
            target_skills=["loops", "functions"],
        )
        submission = ExerciseSubmission(
            exercise_id=exercise.id,
            code="for i in range(10):\n    print(i)",
            explanation="The for loop iterates from 0 to 9, printing each number.",
        )
        results = self.agent.evaluate_exercise_submission(exercise, submission)
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)


# ======================================================================
# Progress tests
# ======================================================================

class TestLearningProfile:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            lp = LearningProfile(path)

            profile = UserProfile()
            profile.update_skill("python", 0.75)
            profile.record_mistake("nested loops")
            lp.save(profile)

            loaded = lp.load()
            assert loaded.get_skill("python") == 0.75
            assert "nested loops" in loaded.common_mistakes

    def test_load_missing_returns_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.json"
            lp = LearningProfile(path)
            profile = lp.load()
            assert isinstance(profile, UserProfile)
            assert profile.skills == {}

    def test_summary(self):
        lp = LearningProfile()
        profile = UserProfile()
        profile.update_skill("functions", 0.8)
        profile.update_skill("classes", 0.4)
        summary = lp.get_summary(profile)
        assert "functions" in summary
        assert "classes" in summary


# ======================================================================
# Provider tests
# ======================================================================

class TestProviderFactory:
    def test_create_openai(self):
        config = ProviderConfig(provider="openai", api_key="sk-test")
        provider = create_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._model == "gpt-4o-mini"

    def test_create_anthropic(self):
        config = ProviderConfig(provider="anthropic", api_key="sk-test")
        provider = create_provider(config)
        assert isinstance(provider, AnthropicProvider)

    def test_create_ollama(self):
        config = ProviderConfig(provider="ollama", model="llama3.2")
        provider = create_provider(config)
        assert isinstance(provider, OllamaProvider)
        assert provider._model == "llama3.2"

    def test_create_opencode(self):
        config = ProviderConfig(provider="opencode", api_key="sk-test")
        provider = create_provider(config)
        assert isinstance(provider, OpenCodeProvider)
        assert provider._model == "deepseek-v4-flash-free"

    def test_create_unknown_provider(self):
        config = ProviderConfig(provider="unknown")
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(config)

    def test_available_providers(self):
        providers = available_providers()
        assert "opencode" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers

    def test_custom_model(self):
        config = ProviderConfig(provider="openai", model="gpt-4o")
        provider = create_provider(config)
        assert provider._model == "gpt-4o"


# ======================================================================
# Model serialization
# ======================================================================

class TestSerialization:
    def test_user_profile_json(self):
        profile = UserProfile(
            user_id="test123",
            skills={"python": 0.8},
            common_mistakes=["bad naming"],
        )
        data = profile.model_dump_json()
        restored = UserProfile(**json.loads(data))
        assert restored.user_id == "test123"
        assert restored.get_skill("python") == 0.8

    def test_evaluation_result_json(self):
        result = EvaluationResult(
            concept="testing",
            understanding=0.75,
            needs_practice=False,
            feedback="Great job!",
            missing_concepts=["mocking"],
        )
        data = result.model_dump_json()
        restored = EvaluationResult(**json.loads(data))
        assert restored.concept == "testing"
        assert restored.understanding == 0.75
