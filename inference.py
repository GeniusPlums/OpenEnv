from __future__ import annotations

import httpx
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from leadqualenv.environment import Action, Decision, LeadProfile, LeadQualEnv, Personality, SignalKey, TaskLevel
from leadqualenv.environment.grader import classify_lead
from leadqualenv.environment.models import ProbeQuality

class RemoteEnv:
    """HTTP client that wraps the OpenEnv server endpoints."""
    
    def __init__(self, base_url: str, task: TaskLevel, seed: int, max_turns: int):
        self.base_url = base_url.rstrip("/")
        self.task = task
        self.seed = seed
        self.max_turns = max_turns
        self._obs: dict = {}
        self.done = False
        self.turn_number = 0
        self.known_signals: dict = {}
        self.probe_log: list = []
        self._lead_temperature = 1.0
        self._qual_confidence = 0.0
        self.conversation_history: list = []
        self.profile = None  # not available from HTTP
    
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.seed = seed
        r = httpx.post(
            f"{self.base_url}/reset",
            json={"seed": self.seed, "task": self.task.value},
            timeout=30,
        )
        r.raise_for_status()
        self._obs = r.json()
        self.done = False
        self._sync_from_obs(self._obs)
        return self._obs
    
    def step(self, action: Action) -> object:
        payload = {}
        if action.message:
            payload["message"] = action.message
        else:
            payload["decision"] = action.decision.value
        
        r = httpx.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        self.done = data.get("done", False)
        self._sync_from_obs(data)
        
        # Wrap in StepResult-compatible object
        return _RemoteStepResult(data)
    
    def _sync_from_obs(self, obs: dict) -> None:
        self.turn_number = obs.get("turn_number", 0)
        self.known_signals = {
            SignalKey(k): v 
            for k, v in obs.get("known_signals", {}).items()
        }
        self.probe_log = [
            (SignalKey(s), ProbeQuality(q)) 
            for s, q in obs.get("probe_log", [])
        ]
        self._lead_temperature = obs.get("lead_temperature", 1.0)
        self._qual_confidence = obs.get("qualification_confidence", 0.0)
        self.conversation_history = obs.get("conversation_history", [])
        self.previous_crm = obs.get("previous_crm")
    
    @property
    def max_turns(self):
        return self._max_turns
    
    @max_turns.setter
    def max_turns(self, v):
        self._max_turns = v
        
    def _partial_grade(self):
        return {"task_score": 0.0}

@dataclass  
class _RemoteStepResult:
    """Adapter to make HTTP response look like StepResult."""
    _data: dict
    
    @property
    def reward(self) -> float:
        return float(self._data.get("reward", 0.0))
    
    @property
    def done(self) -> bool:
        return bool(self._data.get("done", False))
    
    @property
    def info(self) -> dict:
        return self._data.get("info", {})

TASKS = {
    "easy": TaskLevel.EASY,
    "medium": TaskLevel.MEDIUM,
    "hard": TaskLevel.HARD,
    "requalification": TaskLevel.REQUALIFICATION,
}


@dataclass(frozen=True)
class RuntimeConfig:
    api_base_url: str
    model_name: str
    hf_token: str | None
    benchmark: str
    max_steps: int
    global_timeout: int
    seed: int


def load_config() -> RuntimeConfig:
    hf_token = os.getenv("HF_TOKEN", "").strip() or os.getenv("OPENAI_API_KEY", "").strip() or None
    return RuntimeConfig(
        api_base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        model_name=os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
        hf_token=hf_token,
        benchmark=os.getenv("LEADQUALENV_BENCHMARK", "leadqualenv"),
        max_steps=int(os.getenv("LEADQUALENV_MAX_STEPS", "10")),
        global_timeout=int(os.getenv("LEADQUALENV_TIMEOUT", "900")),
        seed=int(os.getenv("LEADQUALENV_SEED", "0")),
    )


@dataclass
class EpisodeResult:
    task_name: str
    success: bool
    steps: int
    rewards: list[float]
    total_reward: float
    score: float


def deterministic_fallback(env: LeadQualEnv | RemoteEnv) -> Action:
    known = env.known_signals
    task = env.task

    def _is_verified(signal: SignalKey) -> bool:
        return any(s == signal and q == ProbeQuality.VERIFIED for s, q in env.probe_log)

    if task == TaskLevel.REQUALIFICATION:
        prev_crm = env.previous_crm if hasattr(env, "previous_crm") else (env._observation().previous_crm if hasattr(env, "_observation") else None)
        if prev_crm and "known_at_time" in prev_crm:
            if not _is_verified(SignalKey.BUDGET):
                return Action(message="Just to double check, is your budget still what you mentioned before?")
            if not _is_verified(SignalKey.TIMELINE):
                return Action(message="To confirm what we discussed previously, what is your timeline looking like now?")
            if not _is_verified(SignalKey.MOTIVATION):
                return Action(message="Circling back to your previous plans, what is your main reason for this purchase now?")

    if known[SignalKey.DECISION_MAKER] is None:
        return Action(message="Are you the person who can make the purchase decision yourself?")

    if known[SignalKey.TIMELINE] is None:
        return Action(message="When are you planning to move or buy, specifically?")

    if known[SignalKey.BUDGET] is None:
        return Action(message="What budget range are you looking at exactly?")

    if known[SignalKey.MOTIVATION] is None:
        return Action(message="What is the main purpose for this purchase — own use or investment?")

    if task == TaskLevel.HARD:
        if known[SignalKey.TIMELINE] in ("immediate", "3-6 months") and not _is_verified(SignalKey.TIMELINE):
            return Action(message="Just to verify our earlier discussion, what is your actual timeline?")
        if known[SignalKey.BUDGET] in ("high", "medium") and not _is_verified(SignalKey.BUDGET):
            return Action(message="To confirm, you said you could stretch. What budget level are you really targeting?")

    if known[SignalKey.DECISION_MAKER] is False:
        return Action(decision=Decision.UNQUALIFIED)

    return Action(decision=_classify_from_known(known))


def _classify_from_known(known: dict[SignalKey, Any]) -> Decision:
    """Classify from discovered signals using the same logic as the environment grader."""
    synthetic_profile = LeadProfile(
        budget=str(known[SignalKey.BUDGET]),
        timeline=str(known[SignalKey.TIMELINE]),
        decision_maker=bool(known[SignalKey.DECISION_MAKER]),
        motivation=str(known.get(SignalKey.MOTIVATION) or "self_use"),
        personality=Personality.DIRECT,
    )
    return classify_lead(synthetic_profile)


def create_client(config: RuntimeConfig) -> OpenAI | None:
    if not config.hf_token:
        return None
    return OpenAI(api_key=config.hf_token, base_url=config.api_base_url)


def strip_markdown_json(text: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def ask_model_for_action(client: OpenAI | None, env: LeadQualEnv, config: RuntimeConfig) -> tuple[Action, bool]:
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
    prev_crm = env.previous_crm if hasattr(env, "previous_crm") else (env._observation().previous_crm if hasattr(env, "_observation") else None)
    if prev_crm is not None:
        observation["previous_crm"] = prev_crm
    
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
            model=config.model_name,
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


def run_task(task_name: str, task: TaskLevel, client: OpenAI | None, start_time: float, config: RuntimeConfig) -> EpisodeResult:
    server_url = os.getenv("LEADQUALENV_SERVER_URL", "")
    if server_url:
        env = RemoteEnv(server_url, task, config.seed, config.max_steps)
    else:
        env = LeadQualEnv(task=task, max_turns=config.max_steps)
    env.reset(seed=config.seed)

    rewards: list[float] = []
    success = False
    steps = 0
    score = 0.0

    print(
        f"[START] task={task_name} env={config.benchmark} model={config.model_name} api_base={config.api_base_url}",
        flush=True,
    )
    try:
        while not env.done and steps < config.max_steps:
            elapsed = time.time() - start_time
            if elapsed > config.global_timeout:
                partial_score = env._partial_grade()["task_score"] if env.profile is not None else 0.0
                score = float(partial_score)
                print(
                    f"[STEP] step={steps + 1} action=TIMEOUT reward=0.00 done=true error=global_timeout partial_score={score:.3f}",
                    flush=True,
                )
                break
            action, used_fallback = ask_model_for_action(client, env, config)
            result = env.step(action)
            steps += 1
            
            buyer_response = ""
            try:
                hist = result.observation.conversation_history if hasattr(result, "observation") else getattr(env, "conversation_history", [])
                if hist and hist[-1]["role"] == "user":
                    buyer_response = hist[-1]["content"].replace("\n", " ").strip()
            except AttributeError:
                pass
            
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
            if action.message is not None and buyer_response:
                print(f"[BUYER] step={steps} response={buyer_response}", flush=True)

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
    os.environ["LEADQUALENV_USE_LLM"] = "0"
    config = load_config()
    client = create_client(config)
    requested_task = os.getenv("LEADQUALENV_TASK")
    start_time = time.time()

    if requested_task:
        task = TASKS.get(requested_task)
        if task is None:
            raise ValueError(f"Unknown task {requested_task!r}. Expected one of: {', '.join(TASKS)}")
        items = [(requested_task, task)]
    else:
        items = list(TASKS.items())

    return [run_task(task_name, task, client, start_time, config) for task_name, task in items]


if __name__ == "__main__":
    main()
