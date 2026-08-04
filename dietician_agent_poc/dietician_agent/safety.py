from __future__ import annotations

from typing import List


def safety_check(profile_dict: dict) -> List[str]:
    flags: List[str] = []

    conditions = (profile_dict.get("conditions") or "").lower()
    goals = (profile_dict.get("goals") or "").lower()
    allergies = (profile_dict.get("allergies") or "").lower()
    prefs = (profile_dict.get("dietary_preferences") or "").lower()

    if any(term in conditions for term in ["diabetes", "kidney", "renal", "pregnant", "eating disorder", "bariatric"]):
        flags.append("Medical condition present; keep recommendations general and advise clinician review.")

    if any(term in goals for term in ["lose 10", "lose 20", "fast", "detox", "cleanse", "crash"]):
        flags.append("Potentially unsafe weight-loss language detected; avoid extreme calorie restriction.")

    if "allerg" in allergies:
        flags.append("Allergy noted; avoid foods with uncertain ingredients and call out label checking.")

    if any(term in prefs for term in ["keto", "vegan", "vegetarian", "halal", "kosher"]):
        flags.append("Diet preference detected; preserve the requested dietary pattern.")

    return flags
