from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from leadqualenv.environment import Action, Decision, LeadQualEnv, SignalKey, TaskLevel
from leadqualenv.environment.grader import classify_lead


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("LEADQUALENV_BENCHMARK", "leadqualenv")
MAX_STEPS = int(os.getenv("LEADQUALENV_MAX_STEPS", "10"))
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


def deterministic_fallback(env: LeadQualEnv) -> Action:
    known = env.known_signals
    task = env.task

    if known[SignalKey.DECISION_MAKER] is None:
        return Action(message="Are you the person who can make the purchase decision yourself?")

    if known[SignalKey.TIMELINE] is None:
        return Action(message="When are you planning to move or buy, specifically?")

    if known[SignalKey.BUDGET] is None:
        return Action(message="What budget range are you looking at exactly?")

    if task == TaskLevel.HARD:
        if known[SignalKey.TIMELINE] == "immediate":
            return Action(message="Just to verify, you mentioned moving quickly earlier. What is your actual timeline?")
        if known[SignalKey.BUDGET] == "high":
            return Action(message="To confirm, you said you could stretch. What budget level are you really targeting?")

    if known[SignalKey.DECISION_MAKER] is False:
        return Action(decision=Decision.UNQUALIFIED)

    profile_like = type(
        "ProfileLike",
        (),
        {
            "budget": known[SignalKey.BUDGET],
            "timeline": known[SignalKey.TIMELINE],
            "decision_maker": known[SignalKey.DECISION_MAKER],
            "motivation": known[SignalKey.MOTIVATION] or "exploring",
        },
    )()
    return Action(decision=classify_lead(profile_like))


def create_client() -> OpenAI | None:
    if not HF_TOKEN:
        return None
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def ask_model_for_action(client: OpenAI | None, env: LeadQualEnv) -> Action:
    fallback = deterministic_fallback(env)
    if client is None:
        return fallback

    observation = {
        "turn_number": env.turn_number,
        "max_turns": env.max_turns,
        "known_signals": {key.value: value for key, value in env.known_signals.items()},
        "probe_log": [(signal.value, quality.value) for signal, quality in env.probe_log],
        "last_messages": env.conversation_history[-4:],
        "task": env.task.value,
    }
    prompt = (
        "You are choosing the next action for a real-estate lead qualification environment. "
        "Return strict JSON with exactly one of:\n"
        '{"message":"<question>"}\n'
        'or {"decision":"qualified|nurture|unqualified"}.\n'
        "Prioritize extracting decision_maker, timeline, and budget before deciding. "
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
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        if "message" in parsed and parsed["message"]:
            return Action(message=str(parsed["message"]))
        if "decision" in parsed and parsed["decision"]:
            return Action(decision=Decision(str(parsed["decision"])))
    except Exception:
        return fallback

    return fallback


def format_action(action: Action) -> str:
    if action.message is not None:
        return action.message.replace("\n", " ").strip()
    return f"decision:{action.decision.value}"


def format_reward(value: float) -> str:
    return f"{value:.2f}"


def run_task(task_name: str, task: TaskLevel, client: OpenAI | None) -> EpisodeResult:
    env = LeadQualEnv(task=task, max_turns=MAX_STEPS)
    env.reset(seed=SEED)

    rewards: list[float] = []
    success = False
    steps = 0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
    try:
        while not env.done and steps < MAX_STEPS:
            action = ask_model_for_action(client, env)
            result = env.step(action)
            steps += 1
            rewards.append(float(result.reward))
            error = result.info.get("error")
            error_text = str(error) if error is not None else "null"
            print(
                f"[STEP] step={steps} action={format_action(action)} "
                f"reward={format_reward(float(result.reward))} "
                f"done={'true' if result.done else 'false'} error={error_text}"
            )
            if result.done:
                success = bool(result.info.get("correct_decision", False))
                break
    finally:
        rewards_text = ",".join(format_reward(value) for value in rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={steps} rewards={rewards_text}"
        )

    return EpisodeResult(
        task_name=task_name,
        success=success,
        steps=steps,
        rewards=rewards,
        total_reward=sum(rewards),
    )


def main() -> list[EpisodeResult]:
    client = create_client()
    requested_task = os.getenv("LEADQUALENV_TASK")

    if requested_task:
        items = [(requested_task, TASKS[requested_task])]
    else:
        items = list(TASKS.items())

    return [run_task(task_name, task, client) for task_name, task in items]


if __name__ == "__main__":
    main()
