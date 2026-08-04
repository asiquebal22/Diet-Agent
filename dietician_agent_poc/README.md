# Dietician Agent PoC

A small, demo-ready proof of concept for a dietician assistant built around:

- a simple agent workflow,
- retrieval over local knowledge-base documents,
- lightweight evaluation scripts,
- an optional OpenAI-backed generation path,
- a Streamlit front end.

## What this demonstrates

- intake collection for goals, allergies, conditions, and diet preferences
- retrieval of relevant diet guidance from a local knowledge base
- guardrails for unsafe or ambiguous requests
- evaluation cases for basic quality checks

## Project layout

- `app.py` - Streamlit demo UI
- `dietician_agent/agent.py` - orchestration, prompt assembly, and output synthesis
- `dietician_agent/rag.py` - simple TF-IDF retrieval over local docs
- `dietician_agent/safety.py` - basic health and diet safety checks
- `dietician_agent/models.py` - shared data models
- `knowledge_base/` - markdown reference docs used for retrieval
- `evals/` - sample evaluation cases and runner

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional OpenAI setup

If `OPENAI_API_KEY` is set, the agent will try to use the OpenAI API for response generation. Without a key, it falls back to a deterministic template-based response so the demo still runs.

Set environment variables:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-5.5"
```

## Run evaluations

```bash
python evals/run_evals.py
```

## Push to GitHub

After cloning this folder locally:

```bash
git init
git add .
git commit -m "Initial dietician agent PoC"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
