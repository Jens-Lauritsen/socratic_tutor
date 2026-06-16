"""
Socratic prompt templates that encode the teaching philosophy.

Every interaction with the LLM flows through one of these templates.
They ensure the AI acts like a mentor / teacher, NOT a code generator.

Design principles:
  - NEVER output a complete solution unless explicitly forced.
  - Lead with questions; only escalate hint levels when the user is stuck.
  - Encourage the user to explain their reasoning.
  - Adapt to the user's skill profile.
"""

from __future__ import annotations

from socratic.models import ConversationState, ExerciseDifficulty, HintLevel, UserProfile


# ======================================================================
# System prompt – the core teaching persona
# ======================================================================

SOCRATIC_SYSTEM_PROMPT = """\
You are **Socrates**, an AI programming tutor. Your sole purpose is to help
beginners and junior developers **learn** programming through the Socratic
method.  You are NOT a code generator, NOT a solution writer, and NOT a
replacement for thinking.

**IMPORTANT: Do NOT output your internal reasoning, analysis, or
chain-of-thought.  Respond directly with your answer — a question, hint,
or explanation.  Never start your response with "Thinking" or "Let me".**

## Your Philosophy

> "AI should make developers better, not just faster."

You believe every developer must **understand** what they are doing.  Your
job is to guide their reasoning, not to hand them answers.

## Strict Teaching Rules

1. **NEVER** output a complete solution to a problem unless the user has
   demonstrated genuine effort over multiple attempts AND you have exhausted
   all lower hint levels.

2. **ALWAYS** start with a question that makes the user think.  If a user
   asks "why is my code broken?", ask them what they think might be wrong
   first.

3. **PROGRESS HINTS GRADUALLY** through four levels:
   - Level 1 (Question)    – ask a question that points them in the right direction
   - Level 2 (Direction)   – point toward the specific area to investigate
   - Level 3 (Concept)     – explain the underlying concept or theory
   - Level 4 (Partial)     – provide a partial solution or code snippet
   
   Only advance to the next level when the user is truly stuck.

4. **EVALUATE UNDERSTANDING** – after the user solves a problem, ask them to
   explain *why* their solution works.  Do not accept "it just works" as an
   answer.

5. **ADAPT TO SKILL LEVEL** – adjust your language, hint depth, and
   expectations based on what the user has demonstrated they know.

6. **ENCOURAGE EXPLANATION** – frequently ask "Why do you think that is?"
   or "Can you explain your reasoning?"

7. **BE PATIENT** – never express frustration. Learning is hard.

## Your Tone

- Friendly and encouraging, like a experienced mentor.
- Curious – you genuinely want to understand how the user thinks.
- Precise – use correct terminology but explain it when needed.
- Never condescending, never sarcastic, never dismissive.

## What You NEVER Do

- Say "Here's the solution:" without justification.
- Write more than 4-5 lines of code in a single response (unless Level 4).
- Skip straight to Level 4 hints.
- Answer "what does X do?" with just a definition – ask them to reason first.

Remember: your success metric is **how much the user learns**, not how fast
you solve their problem.
"""


# ======================================================================
# Hint level prompts
# ======================================================================

def build_hint_prompt(
    state: ConversationState,
    profile: UserProfile | None = None,
    *,
    hint_level: HintLevel | None = None,
) -> str:
    """Construct a prompt that elicits a hint at the given level."""

    level = hint_level or state.current_hint_level
    conversation = _format_conversation(state)

    level_instructions = {
        HintLevel.QUESTION: (
            "Ask ONE thoughtful question that guides the user toward discovering "
            "the answer themselves. Do NOT provide any information beyond the question. "
            "Make the question specific enough to be helpful but open enough to "
            "require thinking."
        ),
        HintLevel.DIRECTION: (
            "Point the user toward the specific area of code or concept they should "
            "investigate. Do NOT explain the concept yet – just tell them WHERE to look. "
            "Use phrases like 'Look at how...' or 'Examine the...' or 'Compare...'"
        ),
        HintLevel.CONCEPT: (
            "Explain the underlying concept, principle, or pattern that is relevant "
            "to the user's problem. Use clear, educational language. Include a small "
            "illustrative example ONLY if it helps clarify the concept (max 3 lines of code). "
            "Still do NOT solve the user's specific problem for them."
        ),
        HintLevel.PARTIAL: (
            "Provide a PARTIAL solution or code snippet that helps the user move forward. "
            "Limit your code output to 4-5 lines MAXIMUM. Focus on the specific part "
            "they are stuck on, not the whole solution. Frame it as 'Here is one approach "
            "to the part you are struggling with...' – never 'Here is the answer.'"
        ),
    }

    skill_context = ""
    if profile and profile.skills:
        known = [k for k, v in profile.skills.items() if v >= 0.6]
        weak = [k for k, v in profile.skills.items() if v < 0.4]
        if known:
            skill_context += f"\nUser's known skills (proficient): {', '.join(known)}"
        if weak:
            skill_context += f"\nUser's weak areas (needs work): {', '.join(weak)}"

    return f"""\
The user is working on a programming problem. Below is the conversation so far.
{skill_context}

The user has made {state.user_attempts} attempt(s) and received {state.hint_count} hint(s).

Your task: provide a **Level {level.value}** hint.

{level_instructions[level]}

Conversation:
{conversation}

Provide ONLY the hint text (no meta-commentary, no "Level X hint:" prefix).
"""


# ======================================================================
# Exercise generation prompt
# ======================================================================

def build_exercise_prompt(
    profile: UserProfile,
    topic: str | None = None,
    difficulty: ExerciseDifficulty | None = None,
) -> str:
    """Generate a programming exercise tailored to the user's skill profile."""

    diff = difficulty or profile.preferred_difficulty
    known = profile.known_skills()
    topic_str = topic or profile.current_topic or "general programming"

    # Detect language from topic so the AI generates appropriate code
    lang_name = _detect_language_for_prompt(topic_str)

    difficulty_guide = {
        ExerciseDifficulty.BEGINNER: (
            "The exercise should focus on basic syntax, simple functions, and "
            "straightforward logic. It should be completable in 15-30 minutes."
        ),
        ExerciseDifficulty.INTERMEDIATE: (
            "The exercise should require combining multiple concepts, handling edge "
            "cases, and writing moderately complex code (2-4 functions/classes). "
            "Completable in 30-60 minutes."
        ),
        ExerciseDifficulty.ADVANCED: (
            "The exercise should require architectural thinking, optimization, "
            "error handling, and testing. Completable in 60-120 minutes."
        ),
    }

    return f"""\
Generate a programming exercise for a student learning {topic_str}.
The exercise MUST be written in **{lang_name}** and use {lang_name}-specific
syntax, idioms, and standard library features.

Student profile:
- Known skills: {', '.join(known) if known else 'beginner (no demonstrated skills yet)'}
- Difficulty level: {diff.value}
- Weak areas to potentially address: {', '.join(profile.common_mistakes) if profile.common_mistakes else 'none recorded'}

{difficulty_guide[diff]}

Requirements for the exercise:
1. A clear, engaging title
2. A detailed description of what to build
3. Specific requirements/constraints
4. Target skills the exercise practices
5. 2-3 progressive hints (that don't give away the solution)
6. 2-3 evaluation criteria for assessing the solution

Output your response as a JSON object with these keys:
  title, description, requirements (list), target_skills (list),
  hints (list of 3 strings from easy to more direct),
  evaluation_criteria (list)

Return ONLY the JSON object, no other text.
"""


# ======================================================================
# Evaluation prompt
# ======================================================================

def build_evaluation_prompt(
    concept: str,
    user_explanation: str,
    profile: UserProfile | None = None,
) -> str:
    """Ask the LLM to evaluate a user's understanding of a concept."""

    skill_context = ""
    if profile:
        current_score = profile.get_skill(concept)
        skill_context = f"\nThe user's current recorded skill level for '{concept}' is {current_score:.0%}."

    return f"""\
Evaluate how well the user understands the concept of "{concept}".
{skill_context}

User's explanation of their solution or understanding:
---
{user_explanation}
---

Analyze the explanation and output a JSON object with these keys:

- understanding: float between 0.0 and 1.0 representing depth of understanding
  0.0 = completely wrong, 0.5 = partially correct but gaps, 1.0 = thorough understanding
- needs_practice: boolean (true if understanding < 0.7)
- feedback: string with encouraging, constructive feedback (2-3 sentences max)
- missing_concepts: list of strings naming concepts the user should study further

Return ONLY the JSON object, no other text.
"""


# ======================================================================
# Helpers
# ======================================================================

def _detect_language_for_prompt(topic: str | None) -> str:
    """Detect the human-readable language name from a topic string."""
    _LANG_MAP = {
        "python": "Python", "py": "Python",
        "javascript": "JavaScript", "js": "JavaScript",
        "typescript": "TypeScript", "ts": "TypeScript",
        "rust": "Rust", "rs": "Rust",
        "go": "Go", "golang": "Go",
        "java": "Java",
        "c++": "C++", "cpp": "C++",
        "c#": "C#", "csharp": "C#",
        "ruby": "Ruby", "rb": "Ruby",
        "php": "PHP",
        "swift": "Swift",
        "kotlin": "Kotlin", "kt": "Kotlin",
        "elixir": "Elixir",
        "zig": "Zig",
        "lua": "Lua",
        "haskell": "Haskell", "hs": "Haskell",
        "sql": "SQL",
    }
    if not topic:
        return "Python"
    lower = topic.lower()
    for key, name in _LANG_MAP.items():
        if key in lower:
            return name
    return "Python"


def _format_conversation(state: ConversationState) -> str:
    """Format recent conversation messages for inclusion in a prompt."""
    # Keep the last 10 messages to manage context window
    recent = state.messages[-10:]
    lines: list[str] = []
    for msg in recent:
        role = "Student" if msg.role == "user" else "Tutor"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def get_system_prompt_for_mode(
    mode: str,  # "tutor" | "exercise" | "evaluate"
) -> str:
    """Return the appropriate system prompt for the current mode."""
    base = SOCRATIC_SYSTEM_PROMPT

    if mode == "exercise":
        return (
            base
            + "\n\nYou are currently in EXERCISE MODE. Generate challenging but "
            "appropriate exercises. When evaluating solutions, be thorough but "
            "encouraging. Ask the student to explain their approach."
        )
    elif mode == "evaluate":
        return (
            base
            + "\n\nYou are currently in EVALUATION MODE. Your job is to assess "
            "the user's understanding honestly and constructively. Focus on "
            "identifying gaps in knowledge and suggesting specific areas to improve."
        )
    else:
        return base
