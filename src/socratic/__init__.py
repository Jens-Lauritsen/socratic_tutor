"""
SocraticCode — An AI programming tutor that teaches using the Socratic method.

The goal is NOT to create another AI coding assistant that generates solutions.
The goal is to create a system that helps beginners and junior developers
learn programming by forcing them to reason, debug, and understand concepts.

Core philosophy: "AI should make developers better, not just faster."
"""

from socratic.tutor import SocraticTutor
from socratic.engine import TutorEngine
from socratic.models import (
    UserProfile,
    ConversationState,
    HintLevel,
    Exercise,
    ExerciseDifficulty,
    ExerciseSubmission,
    EvaluationResult,
    ProviderConfig,
)
from socratic.providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    create_provider,
    available_providers,
)

__version__ = "0.1.0"
__all__ = [
    "SocraticTutor",
    "TutorEngine",
    "UserProfile",
    "ConversationState",
    "HintLevel",
    "Exercise",
    "ExerciseDifficulty",
    "ExerciseSubmission",
    "EvaluationResult",
    "ProviderConfig",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "create_provider",
    "available_providers",
]
