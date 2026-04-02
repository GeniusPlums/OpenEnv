# LeadQualEnv v2

LeadQualEnv is an OpenEnv environment for outbound lead qualification in real estate sales. The agent acts like an SDR, probes for key signals, and decides whether the lead is `qualified`, `nurture`, or `unqualified`.

## Design goals

- Deterministic grading with a single source of truth for lead classification.
- Dense reward shaping around probe quality, not just final accuracy.
- Adversarial hard-mode behavior where surface signals can be misleading until verified.

## Why This Is Real

Real estate teams spend expensive human SDR time qualifying leads that often never convert. This environment models a real qualification workflow with budget, timeline, decision authority, and motivation signals, then rewards agents for reaching the right sales decision efficiently and safely.

## Action Space

Each step accepts exactly one of:

```python
{
  "message": str,
  "decision": None
}
```

or

```python
{
  "message": None,
  "decision": "qualified" | "nurture" | "unqualified"
}
```

Rules:

- `message` and `decision` are mutually exclusive
- a `decision` ends the episode
- deciding before the required signals are known triggers a penalty

## Observation Space

The environment returns:

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

## Tasks

- `easy`: straightforward qualified lead, tests basic extraction and correct qualification
- `medium`: nurture trap, tests whether the agent avoids over-weighting budget and motivation
- `hard`: adversarial lead with misleading surface signals, tests verification behavior

All three tasks have deterministic grading and produce scores in the `0.0` to `1.0` range through the agent grader plus shaped trajectory reward.

## Agent Graders

Task-level normalized graders are implemented in [task_graders.py](c:/Users/anish/OpenEnv/leadqualenv/environment/task_graders.py). Each task uses fixed weights:

- `easy`: decision `0.50`, signal coverage `0.30`, probe quality `0.20`
- `medium`: decision `0.65`, signal coverage `0.20`, probe quality `0.15`
- `hard`: decision `0.45`, signal coverage `0.15`, probe quality `0.40`

This keeps task scores in the `0.0–1.0` range while the shaped reward continues to provide dense step-by-step learning signal.

## Project structure

```text
leadqualenv/
├── environment/
│   ├── env.py
│   ├── grader.py
│   ├── models.py
│   ├── profiles.py
│   ├── reward.py
│   └── simulator.py
└── __init__.py
tests/
├── test_classifier.py
├── test_episodes.py
├── test_grader.py
└── test_simulator.py
```

## Quickstart

```bash
python -m pip install -e .[dev]
pytest
python inference.py
```

## Required Inference Variables

The competition runner expects these variables:

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
set HF_TOKEN=your_hugging_face_or_router_token
```

Optional:

```bash
set LEADQUALENV_TASK=easy
set LEADQUALENV_SEED=0
set LEADQUALENV_MAX_STEPS=10
set LEADQUALENV_BENCHMARK=leadqualenv
```

`inference.py` uses the OpenAI Python client with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`, and emits the required `[START]`, `[STEP]`, and `[END]` stdout lines.

## Meta Llama paraphrase mode

The environment keeps grading deterministic even when the response surface is LLM-backed. Only the canonical rule-layer value is written to `known_signals`; the LLM output is display-only.

Set these environment variables to use Groq-hosted Meta Llama for the paraphrase layer:

```bash
set GROQ_API_KEY=your_key_here
set LEADQUALENV_USE_LLM=1
set GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
python inference.py
```

If the API call fails or the variables are unset, the simulator falls back to deterministic local templates.

## Docker

```bash
docker build -t leadqualenv .
docker run --rm -p 7860:7860 leadqualenv
docker run --rm -e GROQ_API_KEY=your_key_here -e LEADQUALENV_USE_LLM=1 leadqualenv
```

The repository includes both a root [Dockerfile](c:/Users/anish/OpenEnv/Dockerfile) and a mirrored [server/Dockerfile](c:/Users/anish/OpenEnv/server/Dockerfile). This matches the OpenEnv validator's expected layout and ensures either repo-root or `server/`-based builds can succeed.

## Baseline Inference

Run:

```bash
python inference.py
```

Current reproducible baseline with seed `0`:

- `easy`: success in 4 steps, rewards `0.05,0.05,0.05,0.70`
- `medium`: success in 4 steps, rewards `0.05,0.05,0.05,0.70`
- `hard`: success in 6 steps, rewards `0.05,0.05,0.05,-0.02,-0.02,0.70`

The script uses the model response when it is valid JSON, and otherwise falls back to the deterministic baseline policy. That keeps the submission reproducible while still satisfying the OpenAI-client requirement in the competition instructions.

## Environment contract

- Max turns: `10`
- Required signals before decision: `budget`, `timeline`, `decision_maker`
- Invalid actions:
  - both `message` and `decision`
  - neither `message` nor `decision`
- Hard mode:
  - direct probes may return misleading surface values
  - verified probes always return the true values

## Notes

The paraphrase layer is implemented as deterministic templates so the repo remains fully reproducible out of the box. If you later swap in an LLM wrapper for richer utterances, keep the current invariant: the grader must only read canonical rule-layer values from `known_signals`.
