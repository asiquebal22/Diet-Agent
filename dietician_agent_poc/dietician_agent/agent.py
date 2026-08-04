from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from .models import AgentResult, RetrievalHit, UserProfile
from .rag import SimpleRAG
from .safety import safety_check

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"

SYSTEM_PROMPT = """You are a careful dietician assistant for demo purposes.
You must:
- keep advice general and non-diagnostic,
- avoid extreme diets and unsafe medical claims,
- respect allergies, conditions, and preferences,
- cite retrieved context when used,
- output concise, structured markdown.

Return JSON with keys:
summary, meal_plan, rationale, safety_notes, follow_up_questions.
"""


class DieticianAgent:
    def __init__(self, kb_dir: Path | str = KB_DIR):
        self.rag = SimpleRAG(kb_dir)
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-5.5")

    def _build_prompt(self, profile: UserProfile, retrieved: List[RetrievalHit], safety_flags: List[str]) -> str:
        evidence_text = "\n\n".join(
            f"Source: {hit.source}\nTitle: {hit.title}\nScore: {hit.score:.3f}\nText: {hit.chunk}"
            for hit in retrieved
        ) or "No relevant documents retrieved."

        return f"""{SYSTEM_PROMPT}

USER PROFILE
{json.dumps(profile.to_dict(), indent=2)}

SAFETY FLAGS
{json.dumps(safety_flags, indent=2)}

RETRIEVED CONTEXT
{evidence_text}

TASK
Create a one-day practical meal-plan suggestion and explain the reasoning.
Use a friendly clinical tone and keep the answer company-demo friendly.
"""

    def _call_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        return getattr(response, "output_text", "") or ""

    def _fallback_response(self, profile: UserProfile, retrieved: List[RetrievalHit], safety_flags: List[str]) -> Dict[str, object]:
        preferences = profile.dietary_preferences or "balanced"
        goals = profile.goals or "general wellness"
        allergies = profile.allergies or "none stated"
        conditions = profile.conditions or "none stated"

        meal_plan = [
            "Breakfast: Greek yogurt or fortified soy yogurt, berries, and oats.",
            "Lunch: Protein bowl with chicken, tofu, or beans; whole grains; vegetables; olive-oil based dressing.",
            "Snack: Fruit plus nuts or hummus with vegetables.",
            "Dinner: Salmon, lentils, or tofu with roasted vegetables and a high-fiber carb.",
        ]

        if "vegan" in preferences.lower():
            meal_plan = [
                "Breakfast: Overnight oats with chia, soy milk, and berries.",
                "Lunch: Lentil quinoa bowl with mixed greens and tahini.",
                "Snack: Fruit plus nuts or hummus with vegetables.",
                "Dinner: Tofu, beans, or tempeh with roasted vegetables and brown rice.",
            ]

        rationale = (
            f"This plan supports the stated goal of {goals} while staying general, balanced, and flexible. "
            f"It avoids making clinical claims and keeps allergy/condition constraints in view: {allergies}; {conditions}."
        )

        summary = "A simple, balanced one-day plan tailored to the intake form and supported by the retrieved guidance."
        follow_up_questions = [
            "What portion sizes and calorie target should the plan aim for?",
            "Are there foods you dislike or cannot prepare at home?",
        ]

        if retrieved:
            rationale += " Retrieved context was used to ground the recommendation."

        return {
            "summary": summary,
            "meal_plan": meal_plan,
            "rationale": rationale,
            "safety_notes": safety_flags or ["No major safety flags detected."],
            "follow_up_questions": follow_up_questions,
        }

    def generate(self, profile: UserProfile) -> AgentResult:
        safety_flags = safety_check(profile.to_dict())
        query = " ".join(
            filter(
                None,
                [
                    profile.goals,
                    profile.allergies,
                    profile.conditions,
                    profile.dietary_preferences,
                    profile.activity_level,
                ],
            )
        ) or "general nutrition guidance"
        retrieved = self.rag.retrieve(query, top_k=3)

        tool_trace = [
            f"safety_check -> {len(safety_flags)} flag(s)",
            f"retrieve -> {len(retrieved)} chunk(s)",
        ]

        prompt = self._build_prompt(profile, retrieved, safety_flags)
        raw = ""
        parsed: Dict[str, object] | None = None

        try:
            raw = self._call_openai(prompt)
            tool_trace.append(f"openai_responses -> model={self.model_name}")
            parsed = self._extract_json(raw)
        except Exception as exc:
            tool_trace.append(f"fallback_template -> {exc.__class__.__name__}")
            parsed = self._fallback_response(profile, retrieved, safety_flags)

        if not parsed:
            parsed = self._fallback_response(profile, retrieved, safety_flags)

        markdown = self._format_markdown(parsed, retrieved)
        return AgentResult(
            answer_markdown=markdown,
            safety_flags=safety_flags,
            retrieved_sources=retrieved,
            tool_trace=tool_trace,
            raw_model_output=raw,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, object]]:
        text = text.strip()
        if not text:
            return None
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return None

    @staticmethod
    def _format_markdown(parsed: Dict[str, object], retrieved: List[RetrievalHit]) -> str:
        summary = parsed.get("summary", "")
        meal_plan = parsed.get("meal_plan", [])
        rationale = parsed.get("rationale", "")
        safety_notes = parsed.get("safety_notes", [])
        follow_up = parsed.get("follow_up_questions", [])

        lines: List[str] = []
        lines.append(f"## Summary\n{summary}")
        lines.append("## Meal plan")
        if isinstance(meal_plan, list):
            for item in meal_plan:
                lines.append(f"- {item}")
        else:
            lines.append(str(meal_plan))
        lines.append(f"## Rationale\n{rationale}")
        lines.append("## Safety notes")
        if isinstance(safety_notes, list):
            for item in safety_notes:
                lines.append(f"- {item}")
        else:
            lines.append(str(safety_notes))
        lines.append("## Follow-up questions")
        if isinstance(follow_up, list):
            for item in follow_up:
                lines.append(f"- {item}")
        else:
            lines.append(str(follow_up))
        if retrieved:
            lines.append("## Retrieved evidence")
            for hit in retrieved:
                lines.append(f"- **{hit.title}** ({hit.source}, score={hit.score:.3f}): {hit.chunk}")
        return "\n\n".join(lines)


def run_agent(profile: UserProfile) -> AgentResult:
    return DieticianAgent().generate(profile)
