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

# LeadQualEnv

**A deterministic OpenEnv benchmark for outbound real-estate lead qualification.**

An AI agent acts as a Sales Development Representative (SDR) who must uncover a prospective buyer's budget, timeline, and decision authority before routing the lead as `qualified`, `nurture`, or `unqualified`.

---

## Why This Environment Matters

Lead qualification is expensive, repetitive, and high-leverage. Human SDRs routinely assess whether a buyer has the authority, urgency, and budget to justify a sales follow-up. Agents that can do this reliably are immediately useful in real businesses.

This benchmark captures several realistic failure modes:

- Deciding too early with incomplete information
- Asking vague or repetitive questions
- Over-weighting exciting surface signals (e.g. a lead claiming high budget when they actually have low)
- Failing to verify contradictory cues before routing
- Losing engagement by asking too many questions (lead temperature decay)

The environment is designed for both RL training and agent evaluation with dense trajectory rewards, personality-driven buyer responses, adversarial hard-mode profiles, and deterministic graders.

### LLM Paraphrasing (Optional)
Buyer responses use tailored hardcoded templates by default depending on the personality. For richer, more varied conversation, you can enable LLM-based paraphrasing. The system uses Groq to rewrite the buyer's answers in real time without altering the underlying ground-truth facts.

```bash
export LEADQUALENV_USE_LLM=1
export GROQ_API_KEY="your_groq_api_key_here"
```

> [!IMPORTANT]
> `GROQ_API_KEY` must be set before server start. Late injection will not enable LLM mode.

---

## Environment API

The core environment is implemented in [`leadqualenv/environment/env.py`](leadqualenv/environment/env.py) and supports:

- `reset(seed)` — start a clean episode, returns initial observation
- `step(action)` — send a message or make a terminal decision, returns observation + reward + done + info
- `state()` — inspect full internal environment state

The HTTP OpenEnv wrapper is in [`server/leadqualenv_environment.py`](server/leadqualenv_environment.py) using Pydantic models from [`server/models.py`](server/models.py).

---

## Action Space

Each step accepts exactly one of:

```python
{"message": "<question to ask the buyer>", "decision": null}
```

```python
{"message": null, "decision": "qualified" | "nurture" | "unqualified"}
```

Rules:

- `message` and `decision` are mutually exclusive
- A `decision` ends the episode
- Deciding before `budget`, `timeline`, and `decision_maker` are known ends with a penalty
- Vague or irrelevant probing is penalized
- Re-asking about known signals without verification language is penalized
- Generic openers ("tell me about yourself") are penalized

---

## Observation Space

Each observation contains:

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
    "qualification_confidence": float,   # 0.0 → 1.0, reaches 1.0 when coverage + verification are complete
    "property_context": str,             # e.g. "apartment in downtown"
}
```

**Lead temperature** simulates real-world lead cooling — the longer you take, the less engaged the buyer becomes. It now decays across the configured episode length instead of collapsing near the end by default.

**Qualification confidence** gives the agent a running estimate of how much information it has gathered, combining signal coverage with verification of required signals.

---

## Task Ladder

LeadQualEnv includes three deterministic tasks with increasing difficulty:

### Easy
Mostly straightforward buyers with direct, truthful answers and clear qualifying signals (immediate timeline, medium/high budget, decision maker). A small number of easy profiles are intentionally `unqualified` to test basic negative routing without introducing deceptive hard-mode behavior.

### Medium
Leads that look attractive on budget or motivation but should be routed to `nurture` because the timeline is 3–6 months. Tests whether the agent avoids over-qualifying. Some leads have evasive personalities.

### Hard
Adversarial profiles where direct probes reveal **misleading surface values** for budget and/or timeline. The agent must use verification probes to uncover true signals before routing. Correct answers vary across `qualified`, `nurture`, and `unqualified` depending on the profile. Additional mechanics:

- **Competitor pressure** — Some leads mention shopping with other agents on timeline or budget probes, creating urgency
- **Objection handling** — Some leads push back on certain questions, blocking the first probe
- **Surface signal traps** — Some surface signals match reality (requiring the agent to verify anyway), while others are partially misleading (only budget or only timeline is fake)

### Requalification
A real-world CRM requalification task. The agent begins the conversation equipped with `previous_crm` information representing what was known (or surfaced) during the last interaction with the lead months ago. Rather than starting from scratch, the agent must selectively re-verify this prior context (some values may have shifted, especially `motivation`), simulating an SDR circling back on a cooled lead.

Task profiles live in [`leadqualenv/environment/profiles.py`](leadqualenv/environment/profiles.py) — curated profiles across the difficulty levels, with diverse personalities, property types, and locations. The `reset` method also supports infinite procedural profile generation using the `generated_profiles` count.

---

## Reward Design

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

This produces meaningful learning signal over the full trajectory while aligning with the final objective. The cold lead penalty and timing curve ensure agents learn to be efficient without being reckless.

---

## Task Graders

Task-level graders in [`leadqualenv/environment/task_graders.py`](leadqualenv/environment/task_graders.py) are deterministic and return normalized scores in `[0.0, 1.0]`.

Scoring components:

| Component | Description |
|-----------|-------------|
| `correct_decision` | 1.0 if correct, 0.0 if wrong |
| `signal_coverage` | Proportion of signals uncovered (required + motivation bonus) |
| `probe_quality` | Average quality of probes (irrelevant=0, vague=0.2, direct=0.8, verified=1.0) |
| `verification` | Proportion of required signals that were verified |
| `efficiency` | Penalizes excessive probing beyond expected count |
| `misleading_detection` | Hard-mode credit for verifying signals that were actually misleading |
| `motivation_shift` | Requalification credit for uncovering changes in buyer motivation |

Per-task weights:

| Task | Decision | Coverage | Quality | Verification | Efficiency | Misleading | Motiv. Shift |
|------|----------|----------|---------|--------------|------------|------------|--------------|
| Easy | 0.45 | 0.25 | 0.15 | — | 0.15 | — | — |
| Medium | 0.45 | 0.20 | 0.15 | 0.10 | 0.10 | — | — |
| Hard | 0.20 | 0.10 | 0.10 | **0.45** | 0.05 | 0.10 | — |
| Requal. | 0.35 | 0.15 | 0.10 | 0.25 | 0.05 | — | 0.10 |

Hard mode heavily weights verification (**45%**) and drastically reduces base decision points, forcing the agent to exhibit highly precise verification behavior instead of randomly guessing correct decisions.

---

## Baseline Inference

The inference entrypoint is [`inference.py`](inference.py). It:

- Uses the OpenAI Python client for all model calls
- Reads credentials from `HF_TOKEN` or `OPENAI_API_KEY`
- Reads endpoint configuration from `API_BASE_URL` and `MODEL_NAME`
- Emits required `[START]`, `[STEP]`, and `[END]` logs with `score=` field
- Falls back to a deterministic policy if the model output is invalid or unavailable
- Strips markdown-wrapped JSON from LLM responses
- Applies a 15-second timeout on API calls and a 15-minute global runtime cap
- Uses the same lead-classification logic as the grader for its deterministic fallback
- Preserves partial `task_score` on incomplete episodes such as timeout or no-decision endings

### Required Environment Variables

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
set HF_TOKEN=your_hugging_face_or_router_token
```

### Optional Environment Variables

```bash
set OPENAI_API_KEY=your_openai_compatible_token
set LEADQUALENV_TASK=easy
set LEADQUALENV_SEED=0
set LEADQUALENV_MAX_STEPS=10
set LEADQUALENV_BENCHMARK=leadqualenv
set LEADQUALENV_USE_LLM=1
set GROQ_API_KEY=your_groq_api_key
```

### Run

```bash
python inference.py
```

### Reproducible Baseline Scores (seed=0, deterministic fallback)

| Task | Steps | Score | Rewards |
|------|-------|-------|---------|
| Easy | 5 | **0.989** | `0.05, 0.05, 0.05, 0.07, 0.73` |
| Medium | 5 | **0.922** | `0.05, 0.05, 0.05, 0.07, 0.73` |
| Hard | 7 | **0.675** | `0.05, 0.05, 0.05, 0.07, 0.06, 0.04, -0.35` |
| Requal. | 8 | **0.905** | `0.05, 0.06, 0.05, 0.06, 0.07, 0.06, 0.04, 0.42` |

---

## Example Conversation (Easy Task)

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

## Setup

### Install Locally

```bash
python -m pip install -e .[dev]
python -m pytest
python inference.py
```

### OpenEnv Validation

```bash
openenv validate
```

Example passing output:

```text
Validation successful
Environment metadata loaded from openenv.yaml
Entrypoint import succeeded: server.leadqualenv_environment:LeadQualOpenEnv
```

---

## Docker and Hugging Face Spaces

The repository uses the root [`Dockerfile`](Dockerfile) for both local container runs and Hugging Face Spaces deployment.

### Build and Run Locally

```bash
docker build -t leadqualenv .
docker run --rm -p 7860:7860 leadqualenv
```

The HF Space should be tagged with `openenv` and exposes the HTTP environment from [`server/app.py`](server/app.py).

---

## Project Structure

```text
leadqualenv/
├── environment/
│   ├── env.py            # Core LeadQualEnv with step/reset/state
│   ├── grader.py         # Probe classification and lead classification
│   ├── models.py         # Typed dataclass models (Action, Observation, etc.)
│   ├── profiles.py       # 34 lead profiles across 3 difficulty levels
│   ├── reward.py         # Dense reward shaping with timing and decay
│   ├── simulator.py      # Personality-aware buyer response generation
│   └── task_graders.py   # Deterministic 0.0–1.0 graders per task
├── py.typed              # PEP 561 type marker
└── __init__.py
server/
├── app.py                # FastAPI/uvicorn HTTP server
├── leadqualenv_environment.py  # OpenEnv Environment wrapper
├── models.py             # Pydantic API models
└── Dockerfile            # Local + HF Spaces deployment
tests/
├── test_classifier.py    # Lead classification tests
├── test_episodes.py      # Full episode integration tests
├── test_grader.py        # Probe classification tests
├── test_server_environment.py  # HTTP wrapper tests
├── test_simulator.py     # Response generation tests
└── test_task_graders.py  # Grader scoring tests
inference.py              # Baseline inference script
openenv.yaml              # OpenEnv metadata
Dockerfile                # Root container build
pyproject.toml            # Project configuration
```

---

## Limitations

- The probe classifier uses keyword matching — sophisticated rephrasing may not be detected correctly
- Buyer responses are template-based by default; optional LLM paraphrasing can be enabled with `LEADQUALENV_USE_LLM=1` and `GROQ_API_KEY`
