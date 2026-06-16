"""
Learning progress — local storage and management of the user's skill profile.

Stores user progress as a JSON file in the user's config directory.
No cloud dependency. 100% local. 100% private.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from socratic.models import UserProfile


# Default location: ~/.config/socratic-code/profile.json
def _default_profile_path() -> Path:
    """Return the default path for the user's learning profile JSON file."""
    xdg_config = os.getenv("XDG_CONFIG_HOME")
    if xdg_config:
        base = Path(xdg_config)
    else:
        base = Path.home() / ".config"
    return base / "socratic-code" / "profile.json"


class LearningProfile:
    """
    Manages the persistent learning profile.

    Usage:
        lp = LearningProfile()
        profile = lp.load()
        profile.update_skill("functions", 0.85)
        lp.save(profile)
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _default_profile_path()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> UserProfile:
        """Load the user profile from disk, creating a default if missing."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                return UserProfile(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Corrupted file – start fresh
                pass
        return UserProfile()

    def save(self, profile: UserProfile) -> None:
        """Persist the user profile to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def get_summary(self, profile: UserProfile) -> str:
        """Return a human-readable summary of the user's progress."""
        lines: list[str] = []
        lines.append(f"User ID: {profile.user_id}")
        lines.append(f"Total sessions: {profile.total_sessions}")
        lines.append(f"Completed exercises: {len(profile.completed_exercises)}")
        lines.append(f"Current topic: {profile.current_topic or 'Not set'}")
        lines.append(f"Preferred difficulty: {profile.preferred_difficulty.value}")
        lines.append("")

        if profile.skills:
            lines.append("Skills:")
            for name, score in sorted(profile.skills.items(), key=lambda x: -x[1]):
                bar = self._skill_bar(score)
                lines.append(f"  {name:<20} {bar} {score:.0%}")
        else:
            lines.append("No skills recorded yet. Start practicing!")

        if profile.common_mistakes:
            lines.append("")
            lines.append("Common mistakes:")
            for m in profile.common_mistakes:
                lines.append(f"  • {m}")

        return "\n".join(lines)

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skill_bar(score: float, width: int = 10) -> str:
        """Draw a simple ASCII progress bar."""
        filled = int(score * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"
