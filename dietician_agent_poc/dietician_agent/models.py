from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class UserProfile:
    age: Optional[int] = None
    sex: str = ""
    goals: str = ""
    allergies: str = ""
    conditions: str = ""
    dietary_preferences: str = ""
    activity_level: str = ""
    meals_per_day: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalHit:
    source: str
    title: str
    chunk: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    answer_markdown: str
    safety_flags: List[str]
    retrieved_sources: List[RetrievalHit]
    tool_trace: List[str]
    raw_model_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_markdown": self.answer_markdown,
            "safety_flags": list(self.safety_flags),
            "retrieved_sources": [hit.to_dict() for hit in self.retrieved_sources],
            "tool_trace": list(self.tool_trace),
            "raw_model_output": self.raw_model_output,
        }
