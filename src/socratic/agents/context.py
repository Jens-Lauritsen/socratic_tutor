"""
Context Agent

Responsible for:
  - Understanding the user's code and problem description
  - Finding relevant files and collecting debugging context
  - Summarising the technical situation for the Teaching Agent

This agent does NOT generate hints; it gathers information so the
Teaching Agent can make better decisions.
"""

from __future__ import annotations

import os
from pathlib import Path

from socratic.models import ConversationState


class ContextAgent:
    """
    Analyses the user's code and problem to build a structured context
    that the Teaching Agent can use to craft precise, helpful hints.
    """

    def __init__(self, working_dir: str | Path | None = None) -> None:
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(
        self,
        user_message: str,
        state: ConversationState,
        *,
        code_snippet: str | None = None,
        file_path: str | None = None,
    ) -> dict:
        """
        Build a context dictionary from the user's latest message and
        any supplementary information they provide (code snippet, file path).

        Returns a dict with keys:
          - problem_summary   : str  – one-sentence summary of what's wrong
          - detected_concepts : list – programming concepts involved
          - code_context      : str  – relevant code excerpt (if any)
          - file_info         : str  – file path and language (if any)
          - error_message     : str  – extracted error text (if any)
        """
        context: dict = {
            "problem_summary": self._summarise_problem(user_message),
            "detected_concepts": self._detect_concepts(user_message),
            "code_context": "",
            "file_info": "",
            "error_message": self._extract_error(user_message),
        }

        if code_snippet:
            context["code_context"] = code_snippet.strip()

        if file_path:
            context["file_info"] = self._inspect_file(file_path)

        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_problem(message: str) -> str:
        """Extract a one-line summary from the user's message."""
        # Simple heuristic: take the first sentence-like chunk
        for delim in (". ", "?\n", "!\n", "\n", ".", "?", "!"):
            if delim in message:
                return message.split(delim)[0].strip()
        return message.strip()[:120]

    @staticmethod
    def _detect_concepts(message: str) -> list[str]:
        """
        Naive keyword-based concept detection.
        In production this would be an LLM call, but for MVP we use
        keyword matching.
        """
        lower = message.lower()
        concepts: dict[str, list[str]] = {
            "authentication": ["401", "403", "auth", "token", "jwt", "login", "session"],
            "api": ["api", "endpoint", "route", "rest", "http", "request", "response"],
            "database": ["database", "db", "sql", "query", "orm", "migration"],
            "functions": ["function", "def ", "return", "argument", "parameter"],
            "variables": ["variable", "undefined", "nameerror", "scope"],
            "classes": ["class", "self", "object", "instance", "inheritance"],
            "error_handling": ["error", "exception", "try", "catch", "except", "raise"],
            "loops": ["loop", "for ", "while", "iterate", "iteration"],
            "testing": ["test", "assert", "pytest", "unittest", "mock"],
            "async": ["async", "await", "asyncio", "coroutine"],
        }

        detected: list[str] = []
        for concept, keywords in concepts.items():
            if any(kw in lower for kw in keywords):
                detected.append(concept)
        return detected or ["general_programming"]

    @staticmethod
    def _extract_error(message: str) -> str:
        """Try to extract an error message from the user's text."""
        indicators = ["error:", "traceback", "exception:", "failed", "status code"]
        lower = message.lower()
        for indicator in indicators:
            idx = lower.find(indicator)
            if idx >= 0:
                # Return the line containing the indicator
                start = message.rfind("\n", 0, idx) + 1
                end = message.find("\n", idx)
                if end == -1:
                    end = len(message)
                return message[start:end].strip()
        # Also check for HTTP status code patterns (e.g. "401", "500")
        import re
        http_match = re.search(r'\b([1-5]\d{2})\b', message)
        if http_match:
            # Extract surrounding sentence
            code = http_match.group(1)
            return f"HTTP status code: {code}"
        return ""

    def _inspect_file(self, file_path: str) -> str:
        """Read a file and return basic info about it."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.working_dir / path

        info_parts = [f"File: {path.name}"]

        # Determine language from extension
        ext_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "React JSX",
            ".tsx": "React TSX",
            ".java": "Java",
            ".go": "Go",
            ".rs": "Rust",
            ".cpp": "C++",
            ".c": "C",
            ".rb": "Ruby",
            ".php": "PHP",
        }
        ext = path.suffix
        if ext in ext_map:
            info_parts.append(f"Language: {ext_map[ext]}")

        if path.exists():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                info_parts.append(f"Lines: {len(lines)}")
                # Include first 20 lines as preview
                preview = "\n".join(lines[:20])
                info_parts.append(f"Preview:\n{preview}")
            except Exception:
                info_parts.append("(could not read file)")
        else:
            info_parts.append("(file not found)")

        return "\n".join(info_parts)
