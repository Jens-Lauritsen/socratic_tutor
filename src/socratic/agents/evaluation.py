"""
Evaluation Agent

Responsible for:
  - Checking user understanding after they solve a problem
  - Scoring explanations of solutions
  - Updating the learning profile with new skill assessments
  - Identifying knowledge gaps and suggesting next steps
"""

from __future__ import annotations

import json
import re

from socratic.models import (
    EvaluationResult,
    Exercise,
    ExerciseSubmission,
    HintLevel,
    UserProfile,
)
from socratic.prompts import build_evaluation_prompt


class EvaluationAgent:
    """
    Assesses the user's understanding and maintains the learning profile.

    After the user solves a problem or submits an exercise, this agent
    evaluates their explanation and updates their skill scores.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_explanation(
        self,
        concept: str,
        user_explanation: str,
        profile: UserProfile | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a user's verbal explanation of a concept or solution.

        This performs a local heuristic evaluation for MVP. In production,
        this would call the LLM via build_evaluation_prompt().
        """
        score = self._heuristic_score(user_explanation)

        result = EvaluationResult(
            concept=concept,
            understanding=score,
            needs_practice=score < 0.7,
            feedback=self._generate_feedback(concept, score),
            missing_concepts=self._identify_gaps(concept, user_explanation),
        )

        # Update the profile if provided
        if profile:
            profile.update_skill(concept, score)
            for gap in result.missing_concepts:
                if gap not in profile.common_mistakes:
                    profile.common_mistakes.append(gap)

        return result

    def evaluate_exercise_submission(
        self,
        exercise: Exercise,
        submission: ExerciseSubmission,
        profile: UserProfile | None = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate a complete exercise submission against the exercise's
        evaluation criteria. Returns one EvaluationResult per target skill.
        """
        results: list[EvaluationResult] = []

        for skill in exercise.target_skills:
            # Use the user's explanation as the primary evaluation input
            explanation = submission.explanation or ""
            code_quality = self._assess_code_quality(submission.code, skill)

            # Combine explanation score (60%) and code quality score (40%)
            explanation_score = self._heuristic_score(explanation)
            combined = (explanation_score * 0.6) + (code_quality * 0.4)

            result = EvaluationResult(
                concept=skill,
                understanding=round(combined, 2),
                needs_practice=combined < 0.7,
                feedback=self._generate_feedback(skill, combined),
            )
            results.append(result)

            if profile:
                profile.update_skill(skill, combined)

        # Record completed exercise
        if profile and submission.exercise_id not in profile.completed_exercises:
            profile.completed_exercises.append(submission.exercise_id)

        return results

    def build_evaluation_prompt(
        self,
        concept: str,
        user_explanation: str,
        profile: UserProfile | None = None,
    ) -> str:
        """Build the LLM prompt for evaluating understanding."""
        return build_evaluation_prompt(concept, user_explanation, profile)

    @staticmethod
    def parse_evaluation_response(response: str, concept: str) -> EvaluationResult:
        """Parse JSON evaluation response from the LLM."""
        try:
            # Try extracting JSON from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return EvaluationResult(
                    concept=concept,
                    understanding=float(data.get("understanding", 0.5)),
                    needs_practice=bool(data.get("needs_practice", True)),
                    feedback=str(data.get("feedback", "")),
                    missing_concepts=list(data.get("missing_concepts", [])),
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: return a neutral result
        return EvaluationResult(
            concept=concept,
            understanding=0.5,
            needs_practice=True,
            feedback="Could not parse evaluation. Please try explaining again.",
        )

    # ------------------------------------------------------------------
    # Heuristic scoring (MVP – no LLM call needed for basic evaluation)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_score(explanation: str) -> float:
        """
        Score an explanation based on simple heuristics:
          - Length (longer explanations tend to show more understanding)
          - Keyword presence (technical terms suggest deeper knowledge)
          - Structure (presence of reasoning words like "because", "therefore")
        """
        if not explanation or len(explanation.strip()) < 10:
            return 0.1

        text = explanation.lower()
        score = 0.0

        # Length factor (up to 0.3)
        length = len(text.split())
        if length > 100:
            score += 0.3
        elif length > 50:
            score += 0.2
        elif length > 20:
            score += 0.1

        # Reasoning indicators (up to 0.3)
        reasoning_words = [
            "because", "therefore", "since", "so that", "as a result",
            "due to", "this means", "which causes", "the reason",
            "consequently", "thus", "hence",
        ]
        reasoning_count = sum(1 for w in reasoning_words if w in text)
        if reasoning_count >= 3:
            score += 0.3
        elif reasoning_count >= 1:
            score += 0.15

        # Technical depth (up to 0.4)
        technical_terms = [
            "function", "class", "variable", "loop", "array", "dictionary",
            "object", "method", "parameter", "return", "import", "module",
            "api", "endpoint", "database", "query", "async", "await",
            "error", "exception", "type", "instance", "inheritance",
            "algorithm", "complexity", "recursion", "iteration",
            "scope", "closure", "callback", "promise", "state",
            "struct", "trait", "ownership", "borrow", "enum", "match",
            "macro", "mut", "parse", "stdin", "unwrap",
        ]
        tech_count = sum(1 for t in technical_terms if t in text)
        if tech_count >= 5:
            score += 0.4
        elif tech_count >= 3:
            score += 0.3
        elif tech_count >= 1:
            score += 0.15

        return min(1.0, score)

    @staticmethod
    def _assess_code_quality(code: str, skill: str) -> float:
        """
        Basic heuristic for code quality assessment.
        Checks: non-empty, has structure, uses relevant constructs.
        """
        if not code or len(code.strip()) < 5:
            return 0.0

        score = 0.3  # base score for submitting something

        lines = code.strip().splitlines()
        if len(lines) >= 5:
            score += 0.1
        if len(lines) >= 15:
            score += 0.1

        # Check for comments (good practice)
        if "#" in code or "//" in code:
            score += 0.1

        # Check for relevant constructs based on skill (fuzzy match)
        skill_checks = {
            "function": ["def ", "fn ", "function ", "func ", "=>", "->"],
            "class": ["class ", "struct ", "impl "],
            "loop": ["for ", "while ", "loop "],
            "error": ["try", "except", "catch", "raise", "throw", "expect(", "unwrap(", "?"],
            "test": ["assert", "test", "#[test]", "expect("],
            "async": ["async ", "await ", ".await", "tokio::"],
            "variable": ["let ", "const ", "var ", "mut "],
            "input": ["stdin", "read_line", "input(", "readline"],
            "string": ["trim(", "parse(", ".to_string()", "String::"],
            "print": ["print!", "println!", "console.log", "print("],
            "arithmetic": [" + ", " - ", " * ", " / ", " % "],
        }

        for key, keywords in skill_checks.items():
            if key in skill.lower():
                if any(kw in code for kw in keywords):
                    score += 0.2
                    break

        # Check for reasonable naming (no single-char variables in long code)
        if len(lines) > 10:
            single_char_vars = sum(
                1 for line in lines
                for word in line.split()
                if len(word) == 1 and word.isalpha() and word not in ("a", "i")
            )
            if single_char_vars <= 2:
                score += 0.1

        return min(1.0, score)

    @staticmethod
    def _generate_feedback(concept: str, score: float) -> str:
        """Generate encouraging feedback based on score."""
        if score >= 0.9:
            return (
                f"Excellent understanding of {concept}! You clearly grasp the "
                "fundamentals and can apply them effectively."
            )
        elif score >= 0.7:
            return (
                f"Good grasp of {concept}. You understand the core ideas – "
                "with a bit more practice you'll master the nuances."
            )
        elif score >= 0.4:
            return (
                f"You have a basic understanding of {concept}, but there are "
                "some gaps. Try practicing more examples and reviewing the "
                "underlying principles."
            )
        else:
            return (
                f"You're just getting started with {concept}. That's okay! "
                "Try explaining how your code works in your own words — "
                "even a simple 'I used a loop to...' helps."
            )

    @staticmethod
    def _identify_gaps(concept: str, explanation: str) -> list[str]:
        """
        Identify related concepts the user might be missing based on
        what's absent from their explanation.
        """
        text = explanation.lower()
        gaps: list[str] = []

        # Concept-specific gap detection
        gap_map = {
            "functions": ["scope", "parameters", "return values"],
            "classes": ["inheritance", "encapsulation", "polymorphism"],
            "loops": ["iteration", "conditions", "break/continue"],
            "error_handling": ["try/except", "exceptions", "logging"],
            "testing": ["assertions", "test cases", "edge cases"],
            "async": ["await", "coroutines", "event loop"],
            "api": ["http methods", "status codes", "headers"],
            "database": ["queries", "transactions", "indexes"],
        }

        if concept in gap_map:
            for related in gap_map[concept]:
                if related.replace("/", " ") not in text and related not in text:
                    gaps.append(related)

        return gaps[:3]  # Limit to top 3 gaps
