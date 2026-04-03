from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from leadqualenv.environment import Action, Decision, LeadQualEnv, SignalKey, TaskLevel

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct")
_hf = os.getenv("HF_TOKEN", "").strip()
_oai = os.getenv("OPENAI_API_KEY", "").strip()
HF_TOKEN: str | None = _hf or _oai or None
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("LEADQUALENV_BENCHMARK", "leadqualenv")
MAX_STEPS = int(os.getenv("LEADQUALENV_MAX_STEPS", "10"))
GLOBAL_TIMEOUT = int(os.getenv("LEADQUALENV_TIMEOUT", "900"))  # 15-minute safety cap
SEED = int(os.getenv("LEADQUALENV_SEED", "0"))
TASKS = {
    "easy": TaskLevel.EASY,
    "medium": TaskLevel.MEDIUM,
    "hard": TaskLevel.HARD,
}


@dataclass
class EpisodeResult:
    task_name: str
    success: bool
    steps: int
    rewards: list[float]
    total_reward: float
    score: float


def deterministic_fallback(env: LeadQualEnv) -> Action:
    known = env.known_signals
    task = env.task

    if known[SignalKey.DECISION_MAKER] is None:
        return Action(message="Are you the person who can make the purchase decision yourself?")

    if known[SignalKey.TIMELINE] is None:
        return Action(message="When are you planning to move or buy, specifically?")

    if known[SignalKey.BUDGET] is None:
        return Action(message="What budget range are you looking at exactly?")

    if known[SignalKey.MOTIVATION] is None:
        return Action(message="What is the main purpose for this purchase — own use or investment?")

    if task == TaskLevel.HARD:
        if known[SignalKey.TIMELINE] == "immediate":
            return Action(message="Just to verify, you mentioned moving quickly earlier. What is your actual timeline?")
        if known[SignalKey.BUDGET] in ("high", "medium"):
            budget_verified = any(
                s == SignalKey.BUDGET and q.value == "verified"
                for s, q in env.probe_log
            )
            if not budget_verified:
                return Action(
                    message="To confirm, you said you could stretch. "
                    "What budget level are you really targeting?"
                )

    if known[SignalKey.DECISION_MAKER] is False:
        return Action(decision=Decision.UNQUALIFIED)

    return Action(decision=_classify_from_known(known))


def _classify_from_known(known: dict[SignalKey, Any]) -> Decision:
    """Classify lead from known signals without needing a full profile object."""
    budget = known[SignalKey.BUDGET]
    timeline = known[SignalKey.TIMELINE]
    decision_maker = known[SignalKey.DECISION_MAKER]

    if not decision_maker:
        return Decision.UNQUALIFIED
    if budget == "low":
        return Decision.UNQUALIFIED
    if timeline == "immediate":
        return Decision.QUALIFIED if budget in ("medium", "high") else Decision.UNQUALIFIED
    if timeline == "3-6 months":
        return Decision.NURTURE
    return Decision.UNQUALIFIED


def create_client() -> OpenAI | None:
    if not HF_TOKEN:
        return None
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def strip_markdown_json(text: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def ask_model_for_action(client: OpenAI | None, env: LeadQualEnv) -> tuple[Action, bool]:
    """Returns (action, used_fallback)."""
    fallback = deterministic_fallback(env)
    if client is None:
        return fallback, True

    observation = {
        "turn_number": env.turn_number,
        "max_turns": env.max_turns,
        "known_signals": {key.value: value for key, value in env.known_signals.items()},
        "probe_log": [(signal.value, quality.value) for signal, quality in env.probe_log],
        "last_messages": env.conversation_history[-4:],
        "task": env.task.value,
        "lead_temperature": env._lead_temperature,
        "qualification_confidence": env._qual_confidence,
    }
    prompt = (
        "You are choosing the next action for a real-estate lead qualification environment. "
        "Return strict JSON with exactly one of:\n"
        '{"message":"<question>"}\n'
        'or {"decision":"qualified|nurture|unqualified"}.\n'
        "Prioritize extracting decision_maker, timeline, and budget before deciding. "
        "Also try to uncover motivation for bonus points. "
        "In hard mode, verify suspiciously strong surface signals before deciding.\n\n"
        f"Observation:\n{json.dumps(observation, separators=(',', ':'))}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Return only strict JSON with no markdown and no explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            timeout=15,
        )
        content = (response.choices[0].message.content or "").strip()
        content = strip_markdown_json(content)
        parsed = json.loads(content)
        if "message" in parsed and parsed["message"]:
            return Action(message=str(parsed["message"])), False
        if "decision" in parsed and parsed["decision"]:
            return Action(decision=Decision(str(parsed["decision"]))), False
    except Exception:
        return fallback, True

    return fallback, True


def format_action(action: Action) -> str:
    if action.message is not None:
        return action.message.replace("\n", " ").strip()
    assert action.decision is not None
    return f"decision:{action.decision.value}"


def format_reward(value: float) -> str:
    return f"{value:.2f}"


def run_task(task_name: str, task: TaskLevel, client: OpenAI | None, start_time: float) -> EpisodeResult:
    env = LeadQualEnv(task=task, max_turns=MAX_STEPS)
    env.reset(seed=SEED)

    rewards: list[float] = []
    success = False
    steps = 0
    score = 0.0

    print(
        f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME} api_base={API_BASE_URL}",
        flush=True,
    )
    try:
        while not env.done and steps < MAX_STEPS:
            elapsed = time.time() - start_time
            if elapsed > GLOBAL_TIMEOUT:
                print(f"[STEP] step={steps + 1} action=TIMEOUT reward=0.00 done=true error=global_timeout", flush=True)
                break
            action, used_fallback = ask_model_for_action(client, env)
            result = env.step(action)
            steps += 1
            rewards.append(float(result.reward))
            error = result.info.get("error")
            error_text = str(error) if error is not None else "null"
            fallback_text = " fallback=true" if used_fallback else ""
            print(
                f"[STEP] step={steps} action={format_action(action)} "
                f"reward={format_reward(float(result.reward))} "
                f"done={'true' if result.done else 'false'} error={error_text}{fallback_text}",
                flush=True,
            )
            if result.done:
                success = bool(result.info.get("correct_decision", False))
                raw_score = result.info.get("task_score", 0.0)
                score = float(raw_score) if isinstance(raw_score, int | float) else 0.0
                break
    finally:
        rewards_text = ",".join(format_reward(value) for value in rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={steps} score={score:.3f} rewards={rewards_text}",
            flush=True,
        )

    return EpisodeResult(
        task_name=task_name,
        success=success,
        steps=steps,
        rewards=rewards,
        total_reward=sum(rewards),
        score=score,
    )


def main() -> list[EpisodeResult]:
    client = create_client()
    requested_task = os.getenv("LEADQUALENV_TASK")
    start_time = time.time()

    if requested_task:
        task = TASKS.get(requested_task)
        if task is None:
            raise ValueError(f"Unknown task {requested_task!r}. Expected one of: {', '.join(TASKS)}")
        items = [(requested_task, task)]
    else:
        items = list(TASKS.items())

    return [run_task(task_name, task, client, start_time) for task_name, task in items]


if __name__ == "__main__":
    main()
