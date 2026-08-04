from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dietician_agent.agent import DieticianAgent
from dietician_agent.models import UserProfile

CASES_PATH = BASE_DIR / "evals" / "cases.json"


def main() -> None:
    agent = DieticianAgent()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        profile = UserProfile(**case["profile"])
        result = agent.generate(profile)
        text = result.answer_markdown.lower()
        rows.append(
            {
                "name": case["name"],
                "flags": len(result.safety_flags),
                "retrieval_hits": len(result.retrieved_sources),
                "has_meal_plan": "## meal plan" in text,
                "has_safety": "## safety notes" in text,
                "mentions_allergy": "allerg" in text or "peanut" in text,
                "mentions_condition": "diabetes" in text or "medical" in text,
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
