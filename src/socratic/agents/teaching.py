"""
Teaching Agent

Responsible for:
  - Deciding the next hint level and content
  - Adapting teaching style based on user profile
  - Deciding when to reveal information vs. keep prompting
  - Orchestrating the Socratic dialogue flow

This is the "brain" of the tutor – it makes pedagogical decisions.
"""

from __future__ import annotations

from socratic.models import ConversationState, HintLevel, UserProfile
from socratic.prompts import build_hint_prompt


class TeachingAgent:
    """
    Makes teaching decisions: when to ask, when to hint, when to explain.

    The TeachingAgent uses the conversation state, user profile, and
    context to decide the next pedagogical action.
    """

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    # How many user attempts before advancing hint level
    STUCK_THRESHOLD = 2

    # Keywords that suggest the user is giving up / wants the answer
    GIVE_UP_SIGNALS = [
        "just tell me",
        "give me the answer",
        "i give up",
        "show me the solution",
        "what's the answer",
        "i don't know",
        "idk",
    ]

    # Keywords that suggest the user is making genuine effort
    EFFORT_SIGNALS = [
        "i think",
        "maybe",
        "i tried",
        "let me try",
        "i believe",
        "my guess is",
        "could it be",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide_action(
        self,
        user_message: str,
        state: ConversationState,
        profile: UserProfile | None = None,
    ) -> dict:
        """
        Decide what teaching action to take next.

        Returns a dict:
          - action        : "question" | "hint" | "explain" | "reveal" | "encourage"
          - hint_level    : HintLevel (if action is hint/reveal)
          - should_reveal : bool
          - reasoning     : str  (why this action was chosen)
        """
        norm = user_message.strip().lower()

        # 1. User is giving up → consider revealing if enough attempts
        if self._is_giving_up(norm):
            if state.hint_count >= 3:
                return self._action("reveal", HintLevel.PARTIAL, True,
                                    "User is frustrated and has received multiple hints.")
            else:
                return self._action("encourage", state.current_hint_level, False,
                                    "User seems discouraged; offer encouragement before escalating.")

        # 2. User made a genuine attempt → evaluate and possibly advance
        if self._is_making_effort(norm):
            # If they seem on the right track, ask a follow-up question
            return self._action("question", state.current_hint_level, False,
                                "User is reasoning; ask a follow-up to deepen thinking.")

        # 3. User asked a direct question → first response should be a question
        if state.user_attempts <= 1 or state.hint_count == 0:
            return self._action("question", HintLevel.QUESTION, False,
                                "First interaction – start with a question to make them think.")

        # 4. User is stuck (multiple attempts, no progress) → advance hint
        if state.user_attempts >= self.STUCK_THRESHOLD and state.hint_count < state.max_hints_before_reveal:
            new_level = state.advance_hint()
            return self._action("hint", new_level, False,
                                f"User appears stuck after {state.user_attempts} attempts; advancing to Level {new_level.value}.")

        # 5. Max hints reached → reveal partial solution
        if state.should_reveal() and not state.revealed_solution:
            return self._action("reveal", HintLevel.PARTIAL, True,
                                "Maximum hints reached; providing partial solution.")

        # 6. Default: continue at current level
        return self._action("hint", state.current_hint_level, False,
                            "Continuing at current hint level.")

    def build_teaching_prompt(
        self,
        state: ConversationState,
        profile: UserProfile | None = None,
        *,
        hint_level: HintLevel | None = None,
        context: dict | None = None,
    ) -> str:
        """
        Build the full prompt that will be sent to the LLM to generate
        the tutor's next response.
        """
        level = hint_level or state.current_hint_level

        prompt = build_hint_prompt(state, profile, hint_level=level)

        if context:
            ctx_parts = []
            if context.get("problem_summary"):
                ctx_parts.append(f"Problem summary: {context['problem_summary']}")
            if context.get("detected_concepts"):
                ctx_parts.append(f"Relevant concepts: {', '.join(context['detected_concepts'])}")
            if context.get("error_message"):
                ctx_parts.append(f"Error details: {context['error_message']}")
            if context.get("code_context"):
                ctx_parts.append(f"User's code:\n```\n{context['code_context']}\n```")
            if ctx_parts:
                prompt += "\n\nAdditional context:\n" + "\n".join(ctx_parts)

        return prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_giving_up(self, message: str) -> bool:
        return any(signal in message for signal in self.GIVE_UP_SIGNALS)

    def _is_making_effort(self, message: str) -> bool:
        return any(signal in message for signal in self.EFFORT_SIGNALS)

    @staticmethod
    def _action(
        action: str,
        hint_level: HintLevel,
        should_reveal: bool,
        reasoning: str,
    ) -> dict:
        return {
            "action": action,
            "hint_level": hint_level,
            "should_reveal": should_reveal,
            "reasoning": reasoning,
        }
