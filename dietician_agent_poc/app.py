from __future__ import annotations

import streamlit as st

from dietician_agent.agent import run_agent
from dietician_agent.models import UserProfile

st.set_page_config(page_title="Dietician Agent PoC", page_icon="🥗", layout="wide")

st.title("Dietician Agent PoC")
st.caption("Demo of intake -> retrieval -> safety checks -> meal plan generation")

with st.sidebar:
    st.header("Intake")
    age = st.number_input("Age", min_value=0, max_value=120, value=32, step=1)
    sex = st.selectbox("Sex", ["", "female", "male", "other"], index=0)
    goals = st.text_input("Goals", value="eat healthier and improve energy")
    allergies = st.text_input("Allergies", value="none")
    conditions = st.text_input("Medical conditions", value="none")
    dietary_preferences = st.text_input("Dietary preferences", value="balanced")
    activity_level = st.selectbox("Activity level", ["", "sedentary", "light", "moderate", "high"], index=3)
    meals_per_day = st.slider("Meals per day", min_value=2, max_value=6, value=3)

    submit = st.button("Generate plan", type="primary")

col1, col2 = st.columns([1, 1])

if submit:
    profile = UserProfile(
        age=int(age) if age else None,
        sex=sex,
        goals=goals,
        allergies=allergies,
        conditions=conditions,
        dietary_preferences=dietary_preferences,
        activity_level=activity_level,
        meals_per_day=int(meals_per_day),
    )

    result = run_agent(profile)

    with col1:
        st.subheader("Output")
        st.markdown(result.answer_markdown)

    with col2:
        st.subheader("Behind the scenes")
        st.write("Safety flags")
        st.write(result.safety_flags or ["None"])
        st.write("Tool trace")
        st.code("\n".join(result.tool_trace), language="text")
        st.write("Retrieved evidence")
        for hit in result.retrieved_sources:
            st.markdown(f"**{hit.title}** ({hit.source}, score={hit.score:.3f})")
            st.write(hit.chunk)
else:
    with col1:
        st.info("Set the intake values in the sidebar and click Generate plan.")
    with col2:
        st.write("The demo shows:")
        st.markdown("- intake form\n- retrieval over local docs\n- guardrails\n- deterministic fallback when no API key is configured")
