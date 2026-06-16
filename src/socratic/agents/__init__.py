"""Agent modules for the Socratic Tutor Engine.

Each agent handles a distinct responsibility:
  - ContextAgent   – understands user code and collects debugging context
  - TeachingAgent  – manages hint progression and teaching decisions
  - EvaluationAgent – assesses understanding and updates the learning profile
"""

from socratic.agents.context import ContextAgent
from socratic.agents.teaching import TeachingAgent
from socratic.agents.evaluation import EvaluationAgent

__all__ = ["ContextAgent", "TeachingAgent", "EvaluationAgent"]
