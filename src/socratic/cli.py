"""
Command-line interface for the SocraticCode Tutor.

Usage:
    socratic start              Start an interactive tutoring session
    socratic exercise [topic]   Generate a programming exercise
    socratic progress           View your learning progress
    socratic config             Show current configuration
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from socratic.models import ExerciseDifficulty, HintLevel
from socratic.tutor import SocraticTutor


# ======================================================================
# Color helpers (no external dependency needed for basic ANSI)
# ======================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


# ======================================================================
# Exercise file saving
# ======================================================================

# Map of language names/aliases to file extensions (longer keys first
# so "c++" matches before "c", "javascript" before "java", etc.)
_LANGUAGE_EXTENSIONS: dict[str, str] = {
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "c++": ".cpp", "cpp": ".cpp",
    "c#": ".cs", "csharp": ".cs",
    "python": ".py", "py": ".py",
    "golang": ".go", "go": ".go",
    "rust": ".rs", "rs": ".rs",
    "java": ".java",
    "ruby": ".rb", "rb": ".rb",
    "swift": ".swift",
    "kotlin": ".kt", "kt": ".kt",
    "scala": ".scala",
    "elixir": ".exs", "ex": ".exs",
    "haskell": ".hs", "hs": ".hs",
    "clojure": ".clj",
    "c": ".c",
    "php": ".php",
    "r": ".r",
    "zig": ".zig",
    "lua": ".lua",
    "sql": ".sql",
    "bash": ".sh", "shell": ".sh",
    "html": ".html",
    "css": ".css",
}


def _detect_language(topic: str | None) -> str:
    """Detect the programming language from a topic string.

    Returns the language name and file extension.
    """
    if not topic:
        return "python", ".py"

    lower = topic.lower()
    # Check for explicit language mentions
    for lang, ext in _LANGUAGE_EXTENSIONS.items():
        if lang in lower:
            return lang, ext

    return "python", ".py"


def _save_exercise_to_file(exercise, topic: str | None = None) -> Path | None:
    """Save an exercise to exercises/<topic>/<title>.<ext> and return the path."""
    import re

    lang_name, ext = _detect_language(topic)

    topic_slug = (
        re.sub(r"[^a-z0-9]", "_", (topic or "general").lower())
        .strip("_")
    ) or "general"
    title_slug = (
        re.sub(r"[^a-z0-9]", "_", exercise.title.lower())
        .strip("_")
    ) or "exercise"

    folder = Path("exercises") / topic_slug
    folder.mkdir(parents=True, exist_ok=True)
    filepath = folder / f"{title_slug}{ext}"

    # Build file content: description as comments, then a stub
    # Use the correct comment syntax for the language
    comment = "//" if ext in (".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".cs", ".kt", ".swift", ".scala", ".zig") else "#"

    lines: list[str] = []
    lines.append(f"{comment} {exercise.title}")
    lines.append(f"{comment} Difficulty: {exercise.difficulty.value}")
    if exercise.target_skills:
        lines.append(f"{comment} Skills: {', '.join(exercise.target_skills)}")
    lines.append(f"{comment}")
    for line in exercise.description.splitlines():
        lines.append(f"{comment} {line}")
    lines.append("")
    if exercise.hints:
        lines.append(f"{comment} Hints:")
        for i, h in enumerate(exercise.hints, 1):
            lines.append(f"{comment}   {i}. {h}")
        lines.append("")
    if exercise.evaluation_criteria:
        lines.append(f"{comment} Evaluation criteria:")
        for c in exercise.evaluation_criteria:
            lines.append(f"{comment}   • {c}")
        lines.append("")
    lines.append("")
    lines.append(f"{comment} Write your solution below:")
    lines.append("")

    filepath.write_text("\n".join(lines))
    return filepath


def _strip_exercise_header(code: str) -> str:
    """Remove the comment header from an exercise file, leaving user code."""
    lines = code.splitlines()
    result: list[str] = []
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            if not result:
                continue  # Skip leading comments/blank lines
        result.append(line)
    return "\n".join(result)


# ======================================================================
# Interactive session
# ======================================================================

async def cmd_start(args: argparse.Namespace) -> None:
    """Run an interactive Socratic tutoring session."""
    tutor = SocraticTutor(
        provider=args.provider,
        model=args.model,
        working_dir=args.dir or os.getcwd(),
    )

    print()
    print(_c("╔══════════════════════════════════════════════╗", Colors.CYAN))
    print(_c("║        SocraticCode Programming Tutor        ║", Colors.CYAN))
    print(_c("╚══════════════════════════════════════════════╝", Colors.CYAN))
    print()

    # Show provider info
    print(_c(f"Provider: {tutor.provider_name} ({tutor.model_name})", Colors.DIM))
    print(_c(f"Profile:  {tutor.profile.user_id} ({tutor.profile.total_sessions} sessions)", Colors.DIM))
    print()

    # Start session
    topic = args.topic
    if not topic:
        print("What are you learning today?")
        print("Examples: Python classes, JavaScript async, SQL queries, etc.")
        topic = input(_c("> ", Colors.GREEN)).strip()
        print()

    welcome = tutor.start(topic)
    print(_c(welcome, Colors.CYAN))
    print()

    # Main conversation loop
    print(_c("Commands:", Colors.DIM))
    print(_c("  /hint [level]  — request a hint (level 1-4)", Colors.DIM))
    print(_c("  /file <path>  — have the tutor look at a file in your project", Colors.DIM))
    print(_c("  /exercise     — generate a practice exercise", Colors.DIM))
    print(_c("  /evaluate     — evaluate your understanding of the current topic", Colors.DIM))
    print(_c("  /progress     — view your learning progress", Colors.DIM))
    print(_c("  /help         — show this help", Colors.DIM))
    print(_c("  /quit         — end the session", Colors.DIM))
    print()

    while True:
        try:
            user_input = input(_c("You: ", Colors.GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            cmd_arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                print(_c("\nGoodbye! Keep learning. 📚", Colors.CYAN))
                break

            elif cmd == "/help":
                print(_c("Commands:", Colors.DIM))
                print(_c("  /hint [1-4]  — request a hint at a specific level", Colors.DIM))
                print(_c("  /file <path> — have the tutor look at a file in your project", Colors.DIM))
                print(_c("  /exercise    — generate a programming exercise", Colors.DIM))
                print(_c("  /evaluate    — explain what you learned for evaluation", Colors.DIM))
                print(_c("  /progress    — view your learning profile", Colors.DIM))
                print(_c("  /quit        — end the session", Colors.DIM))
                print()
                continue

            elif cmd == "/progress":
                print()
                print(_c(tutor.get_progress_summary(), Colors.BLUE))
                print()
                continue

            elif cmd == "/file":
                if not cmd_arg:
                    print(_c("Usage: /file <path> — e.g., /file src/main.py", Colors.RED))
                    continue
                print(_c(f"\nReading {cmd_arg}...", Colors.DIM))
                # Read the file and send it along with the user's next message
                try:
                    file_content = Path(cmd_arg).read_text()
                    lines = file_content.splitlines()
                    preview = "\n".join(lines[:30])
                    line_count = len(lines)
                    print(_c(f"  {line_count} lines loaded from {cmd_arg}", Colors.DIM))
                    print(_c("  What would you like to know about this code?", Colors.DIM))
                    question = input(_c("> ", Colors.GREEN)).strip()
                    if not question:
                        continue
                    print(_c("\nTutor (thinking...)", Colors.DIM))
                    response = await tutor.ask(question, file_path=cmd_arg)
                    print(_c(f"\nTutor: {response}", Colors.CYAN))
                except FileNotFoundError:
                    print(_c(f"  File not found: {cmd_arg}", Colors.RED))
                except Exception as e:
                    print(_c(f"  Error reading file: {e}", Colors.RED))
                print()
                continue

            elif cmd == "/hint":
                try:
                    level = int(cmd_arg) if cmd_arg else None
                except ValueError:
                    level = None

                if level is not None and level not in (1, 2, 3, 4):
                    print(_c("Hint level must be 1-4.", Colors.RED))
                    continue

                hint_level = HintLevel(level) if level else None
                print(_c("\nTutor (generating hint...)", Colors.DIM))
                response = await tutor.hint(hint_level)
                print(_c(f"\nTutor: {response}", Colors.CYAN))
                print()
                continue

            elif cmd == "/exercise":
                print(_c("\nGenerating an exercise for you...", Colors.DIM))
                current_topic = tutor.profile.current_topic or args.topic
                exercise = await tutor.generate_exercise(current_topic)
                print()
                print(_c(f"📝 Exercise: {exercise.title}", Colors.BOLD))
                print(_c(f"   Difficulty: {exercise.difficulty.value}", Colors.YELLOW))
                print(_c(f"   Skills: {', '.join(exercise.target_skills)}", Colors.DIM))
                print()
                print(exercise.description)
                print()
                if exercise.hints:
                    print(_c("Hints (use if stuck):", Colors.DIM))
                    for i, h in enumerate(exercise.hints, 1):
                        print(_c(f"  {i}. {h}", Colors.DIM))
                print()

                # Save exercise to file
                saved_path = _save_exercise_to_file(exercise, current_topic)
                if saved_path:
                    print(_c(f"📁 Saved to: {saved_path}", Colors.GREEN))
                    print(_c("   Open it in VS Code and write your solution!", Colors.DIM))
                print()

                # Offer to submit solution
                print(_c("When you're ready to submit, you can:", Colors.DIM))
                print(_c("  • Paste your code below (type /done to finish)", Colors.DIM))
                if saved_path:
                    print(_c(f"  • OR just type /done to read from {saved_path}", Colors.DIM))
                print(_c("  • Press Enter to skip for now", Colors.DIM))
                code_lines: list[str] = []
                while True:
                    line = input()
                    if line.strip() == "/done":
                        break
                    code_lines.append(line)

                # If user typed /done with no pasted code, read from the saved file
                if not code_lines and saved_path and saved_path.exists():
                    code = saved_path.read_text()
                    # Strip the comment header to get just the user's code
                    code = _strip_exercise_header(code)
                    if code.strip():
                        print(_c(f"\n📄 Read {len(code.splitlines())} lines from {saved_path}", Colors.DIM))
                    else:
                        print(_c("\n⚠️  File is empty — write some code first!", Colors.YELLOW))
                        code = ""
                elif code_lines:
                    code = "\n".join(code_lines)
                else:
                    code = ""

                if code.strip():
                    print(_c("\nNow explain why your solution works (2-3 sentences):", Colors.DIM))
                    explanation = input(_c("> ", Colors.GREEN)).strip()

                    print(_c("\nEvaluating your solution...", Colors.DIM))
                    results = await tutor.submit_solution(exercise, code, explanation)
                    print()
                    for r in results:
                        status = "✅" if not r.needs_practice else "⚠️"
                        print(_c(f"  {status} {r.concept}: {r.understanding:.0%} understanding", Colors.CYAN))
                        print(_c(f"     {r.feedback}", Colors.DIM))
                        if r.missing_concepts:
                            print(_c(f"     Study: {', '.join(r.missing_concepts)}", Colors.YELLOW))
                    print()
                continue

            elif cmd == "/evaluate":
                concept = args.topic or tutor.profile.current_topic or "general programming"
                print(_c(f"\nExplain what you understand about '{concept}':", Colors.DIM))
                explanation = input(_c("> ", Colors.GREEN)).strip()

                if explanation:
                    print(_c("\nEvaluating...", Colors.DIM))
                    result = await tutor.evaluate(concept, explanation)
                    print()
                    status = "✅" if not result.needs_practice else "⚠️"
                    print(_c(f"  {status} Understanding: {result.understanding:.0%}", Colors.CYAN))
                    print(_c(f"  {result.feedback}", Colors.DIM))
                    if result.missing_concepts:
                        print(_c(f"  Suggested topics: {', '.join(result.missing_concepts)}", Colors.YELLOW))
                    print()
                continue

            else:
                print(_c(f"Unknown command: {cmd}. Type /help for available commands.", Colors.RED))
                continue

        # Normal message — send to tutor
        print(_c("\nTutor (thinking...)", Colors.DIM))
        try:
            response = await tutor.ask(user_input)
            print(_c(f"\nTutor: {response}", Colors.CYAN))
        except Exception as e:
            print(_c(f"\nError: {e}", Colors.RED))
            print(_c("Make sure your API key is set and the provider is accessible.", Colors.DIM))
        print()

    # Save profile on exit
    tutor.save_profile()
    print(_c(f"Progress saved to {tutor.profile.user_id}", Colors.DIM))


# ======================================================================
# Exercise generation (standalone)
# ======================================================================

async def cmd_exercise(args: argparse.Namespace) -> None:
    """Generate a programming exercise."""
    tutor = SocraticTutor(
        provider=args.provider,
        model=args.model,
    )

    print(_c("\nGenerating exercise...", Colors.DIM))
    exercise = await tutor.generate_exercise(
        topic=args.topic,
        difficulty=args.difficulty,
    )

    print()
    print(_c(f"📝 {exercise.title}", Colors.BOLD))
    print(_c(f"   Difficulty: {exercise.difficulty.value}", Colors.YELLOW))
    print(_c(f"   Skills: {', '.join(exercise.target_skills)}", Colors.DIM))
    print()
    print(exercise.description)
    print()

    if exercise.hints:
        print(_c("Hints:", Colors.DIM))
        for i, h in enumerate(exercise.hints, 1):
            print(_c(f"  {i}. {h}", Colors.DIM))
        print()

    if exercise.evaluation_criteria:
        print(_c("Evaluation criteria:", Colors.DIM))
        for c in exercise.evaluation_criteria:
            print(_c(f"  • {c}", Colors.DIM))
        print()

    # Save exercise to file
    saved_path = _save_exercise_to_file(exercise, args.topic)
    if saved_path:
        print(_c(f"📁 Saved to: {saved_path}", Colors.GREEN))
        print()

    tutor.save_profile()


# ======================================================================
# Progress view
# ======================================================================

def cmd_progress(args: argparse.Namespace) -> None:
    """Display the user's learning progress."""
    tutor = SocraticTutor(
        provider=args.provider,
        model=args.model,
    )
    print()
    print(_c(tutor.get_progress_summary(), Colors.BLUE))
    print()
    print(_c(f"Profile stored at: {tutor._learning_profile.path}", Colors.DIM))


# ======================================================================
# Config view
# ======================================================================

def cmd_config(args: argparse.Namespace) -> None:
    """Show current configuration."""
    print()
    print(_c("SocraticCode Configuration", Colors.BOLD))
    print()

    provider = args.provider or os.getenv("SOCRATIC_LLM_PROVIDER", "openai")
    print(f"  Provider:        {provider}")

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "(not set)")
        print(f"  OPENAI_API_KEY:  {'***' + key[-4:] if key != '(not set)' and len(key) > 4 else key}")
    elif provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "(not set)")
        print(f"  ANTHROPIC_API_KEY: {'***' + key[-4:] if key != '(not set)' and len(key) > 4 else key}")
    elif provider == "opencode":
        key = os.getenv("OPENCODE_API_KEY", "(not set)")
        print(f"  OPENCODE_API_KEY: {'***' + key[-4:] if key != '(not set)' and len(key) > 4 else key}")
        base = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
        print(f"  OPENCODE_BASE_URL: {base}")
    elif provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"  OLLAMA_BASE_URL:  {base}")

    if args.model:
        print(f"  Model:           {args.model}")

    # Show profile path
    from socratic.progress import _default_profile_path
    print(f"  Profile path:    {_default_profile_path()}")
    print()

    print(_c("Set environment variables:", Colors.DIM))
    print(_c("  export SOCRATIC_LLM_PROVIDER=openai", Colors.DIM))
    print(_c("  export OPENAI_API_KEY=sk-...", Colors.DIM))
    print(_c("  # or for local models:", Colors.DIM))
    print(_c("  export SOCRATIC_LLM_PROVIDER=ollama", Colors.DIM))
    print(_c("  export OLLAMA_BASE_URL=http://localhost:11434", Colors.DIM))
    print()


# ======================================================================
# CLI entry point
# ======================================================================

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="socratic",
        description="SocraticCode — an AI programming tutor that teaches using the Socratic method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  socratic start                     Start an interactive tutoring session
  socratic start --topic "Python"    Start with a specific topic
  socratic exercise loops            Generate a loop exercise
  socratic progress                  View your learning profile
  socratic config                    Show configuration

Environment variables:
  SOCRATIC_LLM_PROVIDER    Provider: openai, anthropic, ollama, or opencode
  OPENAI_API_KEY           OpenAI API key
  ANTHROPIC_API_KEY        Anthropic API key
  OLLAMA_BASE_URL          Ollama server URL (default: http://localhost:11434)
  OPENCODE_API_KEY         OpenCode API key
  OPENCODE_BASE_URL        OpenCode base URL (default: https://api.opencode.ai/v1)
        """,
    )

    parser.add_argument(
        "--provider", "-p",
        default=os.getenv("SOCRATIC_LLM_PROVIDER", "openai"),
        help="LLM provider: openai, anthropic, or ollama",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (uses provider default if not specified)",
    )
    parser.add_argument(
        "--dir", "-d",
        default=None,
        help="Working directory for code context analysis",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # socratic start
    start_parser = subparsers.add_parser("start", help="Start an interactive tutoring session")
    start_parser.add_argument("--topic", "-t", default=None, help="Learning topic")

    # socratic exercise
    exercise_parser = subparsers.add_parser("exercise", help="Generate a programming exercise")
    exercise_parser.add_argument("topic", nargs="?", default=None, help="Topic for the exercise")
    exercise_parser.add_argument(
        "--difficulty", "-d",
        choices=["beginner", "intermediate", "advanced"],
        default=None,
        help="Exercise difficulty",
    )

    # socratic progress
    subparsers.add_parser("progress", help="View your learning progress")

    # socratic config
    subparsers.add_parser("config", help="Show current configuration")

    args = parser.parse_args()

    if args.command == "start":
        asyncio.run(cmd_start(args))
    elif args.command == "exercise":
        asyncio.run(cmd_exercise(args))
    elif args.command == "progress":
        cmd_progress(args)
    elif args.command == "config":
        cmd_config(args)
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
