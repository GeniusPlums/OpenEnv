---
title: LeadQualEnv
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - rl-environment
  - sales
---

# 🏠 LeadQualEnv

> **A deterministic OpenEnv benchmark for outbound real-estate lead qualification.**

An AI agent acts as a Sales Development Representative (SDR) who must uncover a prospective buyer's **budget**, **timeline**, and **decision authority** through natural conversation before routing the lead as `qualified`, `nurture`, or `unqualified`. The environment features personality-driven buyer responses, adversarial hard-mode with misleading surface signals, lead temperature decay, and deterministic scoring — making it ideal for RL training and evaluation.

---

## ✨ Key Features

- 🎯 **4 difficulty levels** — easy → medium → hard → requalification with increasing deception and ambiguity
- 🔍 **Verification mechanics** — hard-mode forces agents to verify misleading surface signals before routing
- 🌡️ **Lead temperature** — buyer engagement decays over time, penalizing over-probing
- 🧠 **5 buyer personalities** — direct, evasive, verbose, terse, friendly — each with unique response patterns
- 💰 **Dense reward shaping** — per-turn rewards for probe quality plus terminal decision bonuses
- 📊 **Deterministic graders** — normalized 0.0–1.0 task scores with component breakdowns
- 🔁 **Requalification task** — re-engaging nurture leads with CRM history and motivation shift detection
- ⚡ **Optional LLM paraphrasing** — Groq-powered buyer responses for richer, more varied conversation
- 🏗️ **Procedural generation** — infinite profile generation beyond the 34 curated profiles

---

## 🚀 Quickstart

### Install

```bash
python -m pip install -e .[dev]
```

### Run Tests

```bash
python -m pytest
```

### Run Baseline Inference

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
set HF_TOKEN=your_hugging_face_token

python inference.py
```

### Validate OpenEnv Compliance

```bash
openenv validate
```

Expected output:

```text
Validation successful
Environment metadata loaded from openenv.yaml
Entrypoint import succeeded: server.leadqualenv_environment:LeadQualOpenEnv
```

---

## 🎮 Hugging Face Space Demo

The Hugging Face Space provides an interactive demo of the full environment loop:

- 🔄 **Reset** — choose difficulty (easy / medium / hard / requalification) and seed
- 💬 **Chat** — type questions as the SDR agent and receive simulated buyer responses
- ✅ **Decide** — route the lead as qualified, nurture, or unqualified
- 📊 **Score** — see the full scoring breakdown (decision, coverage, quality, verification, efficiency)
- 🔓 **Reveal** — after the episode, see the hidden buyer profile including surface traps

### Demo Features

| Feature | Description |
|---------|-------------|
| **Preset Scenarios** | One-click presets for easy, medium, hard, and requalification |
| **Run Baseline Agent** | Runs the deterministic policy end-to-end automatically |
| **Scoring Breakdown** | Component-level task score with weights per difficulty |
| **Profile Reveal** | Hidden buyer profile exposed after episode — shows surface traps vs. true signals |
| **Live Signals** | Real-time signal coverage, probe log, temperature, and confidence |

> [!NOTE]
> The Space is a demo of the **OpenEnv LeadQualEnv environment**. It demonstrates the conversation loop, hidden-profile mechanics, scoring, and reproducibility of the benchmark.

---

## 🌍 Why This Environment Matters

Lead qualification is expensive, repetitive, and high-leverage. In real estate, SDRs spend hours daily assessing whether prospects have the authority, urgency, and budget for follow-ups. Poor qualification wastes $100–200 per bad lead or misses hot ones, reducing conversion rates by 20–30%.

This benchmark captures realistic failure modes from real estate sales pipelines:

- **Premature decisions** — routing based on gut feel without verifying budget/timeline/decision-maker
- **Vague probing** — asking "tell me about yourself" instead of targeted questions
- **Surface signal traps** — over-relying on stated budgets without probing deeper
- **Verification gaps** — not circling back on suspicious signals
- **Engagement decay** — over-probing cools leads, mirroring real-world buyer disengagement

### LLM Paraphrasing (Optional)

Buyer responses use tailored hardcoded templates by default. For richer conversation, enable LLM-based paraphrasing:

```bash
set LEADQUALENV_USE_LLM=1
set GROQ_API_KEY=your_groq_api_key_here
```

---

## 📐 Environment API

The core environment is implemented in [`leadqualenv/environment/env.py`](leadqualenv/environment/env.py):

| Method | Description |
|--------|-------------|
| `reset(seed)` | Start a clean episode, returns initial observation |
| `step(action)` | Send a message or make a terminal decision |
| `state()` | Inspect full internal environment state |
| `snapshot()` / `restore()` | Serialize/deserialize for batched RL |

The HTTP OpenEnv wrapper is in [`server/leadqualenv_environment.py`](server/leadqualenv_environment.py).

---

## 🎯 Action & Observation Space

### Actions

Each step accepts exactly one of:

```python
{"message": "<question to ask the buyer>", "decision": null}
{"message": null, "decision": "qualified" | "nurture" | "unqualified"}
```

Rules: `message` and `decision` are mutually exclusive. A `decision` ends the episode. Deciding before `budget`, `timeline`, and `decision_maker` are known incurs a penalty. Vague, irrelevant, or repeated probes are penalized.

### Observations

```python
{
    "conversation_history": [{"role": str, "content": str}, ...],
    "known_signals": {
        "budget": "low" | "medium" | "high" | None,
        "timeline": "immediate" | "3-6 months" | "6+ months" | None,
        "decision_maker": bool | None,
        "motivation": "self_use" | "investment" | "exploring" | None,
    },
    "probe_log": [[signal, quality], ...],
    "turn_number": int,
    "max_turns": int,
    "lead_temperature": float,          # 1.0 → ~0.4, decays across the episode
    "qualification_confidence": float,   # 0.0 → 1.0
    "property_context": str,             # e.g. "apartment in downtown"
}
```

- **Lead temperature** simulates real-world lead cooling — the longer you take, the less engaged the buyer becomes.
- **Qualification confidence** gives the agent a running estimate of information coverage + verification status.

---

## 📊 Task Ladder

| Task | Profiles | Key Challenge | Correct Decision |
|------|----------|---------------|------------------|
| **Easy** | 12 curated | Straightforward buyers, clear signals | Mostly `qualified`, some `unqualified` |
| **Medium** | 10 curated | Attractive budget but delayed timeline | `nurture` |
| **Hard** | 12 curated | Misleading surface signals | Varies — requires verification |
| **Requalification** | 5 curated | Returning leads, motivation may shift | Varies by current signals |

### Easy

Straightforward buyers with direct, truthful answers. A small number are intentionally `unqualified` to test basic negative routing.

### Medium

Leads that look attractive on budget but should be `nurture` because the timeline is 3–6 months. Some have evasive personalities.

### Hard

Adversarial profiles where direct probes reveal **misleading surface values**. The agent must use verification probes to uncover true signals. Additional mechanics:

- **Competitor pressure** — some leads mention shopping with other agents
- **Objection handling** — some leads push back on certain questions
- **Surface signal traps** — some surface values match reality (requiring the agent to verify anyway)

### Requalification

The agent begins with `previous_crm` data from the last interaction. Rather than starting from scratch, the agent must selectively re-verify — some values may have shifted, especially `motivation`.

Task profiles: [`leadqualenv/environment/profiles.py`](leadqualenv/environment/profiles.py) — 34 curated profiles plus infinite procedural generation.

---

## 💰 Reward Design

Reward shaping is implemented in [`leadqualenv/environment/reward.py`](leadqualenv/environment/reward.py).

**Trajectory rewards:**

| Signal | Reward |
|--------|--------|
| Direct, useful probe | `+0.05` |
| Verification probe | `+0.06` |
| Motivation discovery | `+0.02` |
| Vague probing | `-0.03` |
| Irrelevant probing | `-0.05` |
| Re-asking known signal (tight-loop) | `-0.05` |
| Re-asking known signal (non-loop) | `-0.025` |
| Cold lead penalty (per turn > 6) | `-0.015/turn` |
| No decision at turn limit | `-0.20` |
| Decision with insufficient signals | `-0.30` |

**Terminal rewards:**

| Outcome | Base | Timing Bonus (turns 3–5) | Decay Bonus |
|---------|------|--------------------------|-------------|
| Correct decision | `+0.50` | `+0.15` | up to `+0.10` |
| Incorrect decision | `-0.40` | — | — |

---

## 📝 Task Graders

Deterministic graders in [`leadqualenv/environment/task_graders.py`](leadqualenv/environment/task_graders.py) return normalized scores in `[0.0, 1.0]`.

| Component | Description |
|-----------|-------------|
| `correct_decision` | 1.0 if correct, 0.0 if wrong |
| `signal_coverage` | Proportion of signals uncovered |
| `probe_quality` | Average probe quality (irrelevant=0, vague=0.2, direct=0.8, verified=1.0) |
| `verification` | Proportion of required signals verified |
| `efficiency` | Penalizes excessive probing |
| `misleading_detection` | Hard-mode credit for verifying traps |
| `motivation_shift` | Requalification credit for uncovering motivation changes |

**Per-task weights:**

| Task | Decision | Coverage | Quality | Verification | Efficiency | Misleading | Motiv. Shift |
|------|----------|----------|---------|--------------|------------|------------|--------------|
| Easy | 0.45 | 0.25 | 0.15 | — | 0.15 | — | — |
| Medium | 0.45 | 0.20 | 0.15 | 0.10 | 0.10 | — | — |
| Hard | 0.20 | 0.10 | 0.10 | **0.45** | 0.05 | 0.10 | — |
| Requal. | 0.35 | 0.15 | 0.10 | 0.25 | 0.05 | — | 0.10 |

Hard mode heavily weights verification (**45%**), forcing precise verification behavior over random guessing.

---

## 🤖 Baseline Inference

The inference entrypoint is [`inference.py`](inference.py). It uses the OpenAI Python client, reads credentials from `HF_TOKEN` or `OPENAI_API_KEY`, emits `[START]`/`[STEP]`/`[END]` logs, and falls back to a deterministic policy if the LLM is unavailable.

### Reproducible Baseline Scores (seed=0, deterministic fallback)

| Task | Steps | Score | Rewards |
|------|-------|-------|---------|
| Easy | 5 | **0.989** | `0.05, 0.05, 0.05, 0.07, 0.73` |
| Medium | 5 | **0.922** | `0.05, 0.05, 0.05, 0.07, 0.73` |
| Hard | 7 | **0.675** | `0.05, 0.05, 0.05, 0.07, 0.06, 0.04, -0.35` |
| Requal. | 8 | **0.905** | `0.05, 0.06, 0.05, 0.06, 0.07, 0.06, 0.04, 0.42` |

### LLM Testing (Groq Llama 3.3 70B Versatile)

| Task | Score | Notes |
|------|-------|-------|
| Easy | 0.989 | Matches baseline |
| Medium | 0.445 | Struggles with nurture routing |
| Hard | 0.195 | Challenges verification mechanics |
| Requal. | 0.730 | Strong performance |

Demonstrates the environment's challenge gradient for frontier models.

---

## 💬 Example Conversation (Easy Task)

```
[system] You are qualifying a prospective real estate buyer.
[buyer]  Hi, I saw your apartment listing and wanted to learn more.

[agent]  Are you the person who can make the purchase decision yourself?
[buyer]  I can make the purchase decision myself.
         → signal: decision_maker = True, reward: +0.05

[agent]  When are you planning to move or buy, specifically?
[buyer]  I want to move pretty quickly if something fits.
         → signal: timeline = immediate, reward: +0.05

[agent]  What budget range are you looking at exactly?
[buyer]  I have a mid-range budget in mind.
         → signal: budget = medium, reward: +0.05

[agent]  What is the main purpose for this purchase — own use or investment?
[buyer]  This would be for my own use.
         → signal: motivation = self_use, reward: +0.07

[agent]  Decision: qualified
         → correct! reward: +0.73, task_score: 0.989
```

---

## 📁 Project Structure

```text
leadqualenv/
├── environment/
│   ├── env.py            # Core LeadQualEnv with step/reset/state
│   ├── grader.py         # Probe classification and lead classification
│   ├── models.py         # Typed dataclass models (Action, Observation, etc.)
│   ├── profiles.py       # 34 lead profiles across 4 difficulty levels
│   ├── reward.py         # Dense reward shaping with timing and decay
│   ├── simulator.py      # Personality-aware buyer response generation
│   └── task_graders.py   # Deterministic 0.0–1.0 graders per task
├── py.typed              # PEP 561 type marker
└── __init__.py
server/
├── app.py                # FastAPI/uvicorn HTTP server + Gradio demo mount
├── demo.py               # Gradio demo UI for Hugging Face Spaces
├── leadqualenv_environment.py  # OpenEnv Environment wrapper
└── models.py             # Pydantic API models
tests/
├── test_classifier.py    # Lead classification tests
├── test_episodes.py      # Full episode integration tests
├── test_grader.py        # Probe classification tests
├── test_server_environment.py  # HTTP wrapper tests
├── test_simulator.py     # Response generation tests
└── test_task_graders.py  # Grader scoring tests
inference.py              # Baseline inference script
openenv.yaml              # OpenEnv metadata
Dockerfile                # Docker container (local + HF Spaces)
pyproject.toml            # Project configuration
```

---

## 🐳 Docker & Deployment

The root [`Dockerfile`](Dockerfile) is used for both local container runs and Hugging Face Spaces deployment.

```bash
docker build -t leadqualenv .
docker run --rm -p 7860:7860 leadqualenv
```

The HF Space is tagged with `openenv` and exposes both the HTTP environment API and the Gradio demo UI at `/`.

---

## ⚙️ Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | For LLM inference | — | OpenAI-compatible API endpoint |
| `MODEL_NAME` | For LLM inference | — | Model identifier |
| `HF_TOKEN` | For LLM inference | — | Hugging Face API token |
| `OPENAI_API_KEY` | Alternative to HF_TOKEN | — | OpenAI-compatible API key |
| `LEADQUALENV_TASK` | No | `easy` | Task level: easy, medium, hard, requalification |
| `LEADQUALENV_SEED` | No | `0` | Random seed for reproducibility |
| `LEADQUALENV_MAX_STEPS` | No | `10` | Maximum turns per episode |
| `LEADQUALENV_USE_LLM` | No | `0` | Enable LLM buyer paraphrasing (`1`/`true`) |
| `GROQ_API_KEY` | For LLM paraphrasing | — | Groq API key |
| `LEADQUALENV_MAX_CONCURRENT` | No | `4` | Max concurrent environment instances |

---

## ⚠️ Limitations

- The probe classifier uses keyword matching — sophisticated rephrasing may not be detected correctly
- Buyer responses are template-based by default; enable LLM paraphrasing with `LEADQUALENV_USE_LLM=1` for more variety
- Hard-mode surface traps follow fixed patterns per profile — a memorizing agent could exploit this (mitigated by procedural generation)
