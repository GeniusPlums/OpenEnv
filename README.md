# LeadQualEnv

LeadQualEnv is an OpenEnv benchmark for outbound real-estate lead qualification. The agent acts like an SDR who has to uncover budget, timeline, and decision authority before deciding whether a lead should be marked `qualified`, `nurture`, or `unqualified`.

The benchmark is designed to be useful for both RL training and agent evaluation:

- it models a real sales workflow that teams already run today
- it gives dense trajectory rewards instead of only terminal pass/fail
- it includes adversarial hard-mode behavior where surface answers can be misleading until the agent verifies them
- it exposes deterministic, task-level graders that score every run in the `0.0` to `1.0` range

## Why This Environment Matters

Lead qualification is expensive, repetitive, and high leverage. Human SDRs routinely need to identify whether a buyer has the authority, urgency, and budget to justify a sales follow-up. Agents that can do this reliably are useful in real businesses, and the benchmark captures several realistic failure modes:

- deciding too early with incomplete information
- asking vague or repetitive questions
- over-weighting exciting surface signals
- failing to verify contradictory cues before routing the lead

## Environment API

The core environment is implemented in [`leadqualenv/environment/env.py`](c:/Users/anish/OpenEnv/leadqualenv/environment/env.py) and supports:

- `reset(seed)` to start a clean episode
- `step(action)` to send a message or make a terminal decision
- `state()` to inspect current internal environment state

The HTTP OpenEnv wrapper is implemented in [`server/leadqualenv_environment.py`](c:/Users/anish/OpenEnv/server/leadqualenv_environment.py) using Pydantic request and response models from [`server/models.py`](c:/Users/anish/OpenEnv/server/models.py).

## Action Space

Each step accepts exactly one of the following actions:

```python
{"message": str, "decision": None}
```

```python
{"message": None, "decision": "qualified" | "nurture" | "unqualified"}
```

Rules:

- `message` and `decision` are mutually exclusive
- a `decision` ends the episode
- deciding before `budget`, `timeline`, and `decision_maker` are known ends the episode with a penalty
- invalid or low-value probing is penalized

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
}
```

The environment also exposes `state()` for full internal state inspection during local evaluation and debugging.

## Task Ladder

LeadQualEnv includes three deterministic tasks with increasing difficulty:

1. `easy`
Straightforward buyers with direct, truthful answers. This tests basic signal extraction and correct qualification decisions.

2. `medium`
Leads that look attractive on budget or motivation, but should still be routed to `nurture` because the timeline is not immediate. This tests whether the agent avoids over-qualifying.

3. `hard`
Adversarial profiles where direct probes can reveal misleading surface values for budget and timeline. The agent has to verify suspicious answers before making the final routing decision.

Task profiles live in [`leadqualenv/environment/profiles.py`](c:/Users/anish/OpenEnv/leadqualenv/environment/profiles.py).

## Reward Design

Reward shaping is implemented in [`leadqualenv/environment/reward.py`](c:/Users/anish/OpenEnv/leadqualenv/environment/reward.py).

Trajectory rewards:

- `+0.05` for a direct, useful probe
- `+0.06` for a verification probe
- `-0.03` for vague probing
- `-0.05` for irrelevant probing
- `-0.05` for repeated direct probing of the same signal
- `-0.30` for making a decision before the required signals are known
- `-0.20` for hitting the turn limit without deciding

Terminal rewards:

- correct decision reward with timing bonus
- incorrect decision penalty

This produces meaningful learning signal over the full trajectory while still aligning with the final objective.

## Task Graders

Task-level graders are implemented in [`leadqualenv/environment/task_graders.py`](c:/Users/anish/OpenEnv/leadqualenv/environment/task_graders.py). All graders are deterministic and return normalized scores in `[0.0, 1.0]`.

Scoring components used across tasks:

- `correct_decision`
- `signal_coverage`
- `probe_quality`
- `efficiency`
- `verification`

Per-task weights:

- `easy`: decision `0.50`, coverage `0.30`, probe quality `0.10`, efficiency `0.10`
- `medium`: decision `0.55`, coverage `0.20`, probe quality `0.15`, efficiency `0.10`
- `hard`: decision `0.45`, coverage `0.15`, probe quality `0.20`, verification `0.20`

This lets the environment reward partial progress while still clearly separating weak, decent, and strong trajectories.

## Baseline Inference

The required inference entrypoint is [`inference.py`](c:/Users/anish/OpenEnv/inference.py). It:

- uses the OpenAI Python client for all model calls
- reads credentials from `HF_TOKEN` or `OPENAI_API_KEY`
- reads endpoint configuration from `API_BASE_URL`
- reads the model id from `MODEL_NAME`
- emits required `[START]`, `[STEP]`, and `[END]` logs
- falls back to a deterministic local policy if the model output is invalid or unavailable

Required environment variables:

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
set HF_TOKEN=your_hugging_face_or_router_token
```

Optional environment variables:

```bash
set OPENAI_API_KEY=your_openai_compatible_token
set LEADQUALENV_TASK=easy
set LEADQUALENV_SEED=0
set LEADQUALENV_MAX_STEPS=10
set LEADQUALENV_BENCHMARK=leadqualenv
```

Run:

```bash
python inference.py
```

Current reproducible baseline with seed `0`:

- `easy`: success in `4` steps, score `0.980`, rewards `0.05,0.05,0.05,0.70`
- `medium`: success in `4` steps, score `0.970`, rewards `0.05,0.05,0.05,0.70`
- `hard`: success in `6` steps, score `0.909`, rewards `0.05,0.05,0.05,0.06,0.06,0.70`

## Setup

Install locally:

```bash
python -m pip install -e .[dev]
python -m pytest
python inference.py
```

OpenEnv validation:

```bash
openenv validate
```

Validator helper script:

```bash
bash scripts/validate-submission.sh <space-url>
```

## Docker and Hugging Face Spaces

The repository includes both a root [`Dockerfile`](c:/Users/anish/OpenEnv/Dockerfile) and a mirrored [`server/Dockerfile`](c:/Users/anish/OpenEnv/server/Dockerfile). This supports both repo-root and `server/`-based validation flows.

Build and run locally:

```bash
docker build -t leadqualenv .
docker run --rm -p 7860:7860 leadqualenv
```

The Space should be tagged with `openenv` and expose the HTTP environment from [`server/app.py`](c:/Users/anish/OpenEnv/server/app.py).

## Project Structure

```text
leadqualenv/
|-- environment/
|   |-- env.py
|   |-- grader.py
|   |-- models.py
|   |-- profiles.py
|   |-- reward.py
|   |-- simulator.py
|   `-- task_graders.py
server/
|-- app.py
|-- leadqualenv_environment.py
`-- models.py
tests/
|-- test_classifier.py
|-- test_episodes.py
|-- test_grader.py
|-- test_simulator.py
`-- test_task_graders.py
```
