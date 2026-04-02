from __future__ import annotations

from dataclasses import asdict

from .grader import classify_lead, classify_probe, is_generic_opener
from .models import (
    Action,
    Decision,
    InsufficientSignalsError,
    InvalidActionError,
    Observation,
    ProbeQuality,
    SignalKey,
    StepResult,
    TaskLevel,
)
from .profiles import sample_profile
from .reward import (
    INSUFFICIENT_SIGNALS_PENALTY,
    NO_DECISION_PENALTY,
    REPEATED_QUESTION_PENALTY,
    episode_end_reward,
    turn_reward,
)
from .simulator import generate_response


class LeadQualEnv:
    def __init__(self, task: TaskLevel, max_turns: int = 10):
        self.task = task
        self.max_turns = max_turns
        self.profile = None
        self.turn_number = 0
        self.known_signals: dict[SignalKey, str | bool | None] = {}
        self.probe_log: list[tuple[SignalKey, ProbeQuality]] = []
        self.conversation_history: list[dict[str, str]] = []
        self.done = False

    def reset(self, seed: int | None = None) -> Observation:
        self.profile = sample_profile(self.task, seed)
        self.turn_number = 0
        self.done = False
        self.known_signals = {
            SignalKey.BUDGET: None,
            SignalKey.TIMELINE: None,
            SignalKey.DECISION_MAKER: None,
            SignalKey.MOTIVATION: None,
        }
        self.probe_log = []
        self.conversation_history = [
            {"role": "system", "content": "You are qualifying a prospective real estate buyer."},
            {"role": "user", "content": "Hi, I saw your property listing and wanted to learn more."},
        ]
        return self._observation()

    def _observation(self) -> Observation:
        return Observation(
            conversation_history=list(self.conversation_history),
            known_signals=dict(self.known_signals),
            probe_log=list(self.probe_log),
            turn_number=self.turn_number,
            max_turns=self.max_turns,
        )

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise InvalidActionError("Episode already completed.")
        if self.profile is None:
            raise InvalidActionError("Environment must be reset before stepping.")

        self.turn_number += 1
        if action.message:
            return self._handle_message(action.message)
        return self._handle_decision(action.decision)

    def _handle_message(self, message: str) -> StepResult:
        probe = classify_probe(message, self.known_signals)
        reward = turn_reward(probe.quality)

        if is_generic_opener(message):
            reward = min(reward, -0.05)

        if probe.signal is not None:
            recent_signals = [signal for signal, _ in self.probe_log[-2:]]
            if probe.signal in recent_signals:
                reward += REPEATED_QUESTION_PENALTY

        self.conversation_history.append({"role": "assistant", "content": message})

        response_text, value = generate_response(self.profile, probe.signal, probe.quality, self.task)
        self.conversation_history.append({"role": "user", "content": response_text})

        if probe.signal is not None:
            self.probe_log.append((probe.signal, probe.quality))
            if probe.quality in (ProbeQuality.DIRECT, ProbeQuality.VERIFIED):
                self.known_signals[probe.signal] = value

        done = self.turn_number >= self.max_turns
        if done:
            self.done = True
            reward += NO_DECISION_PENALTY

        return StepResult(
            observation=self._observation(),
            reward=reward,
            done=done,
            info={"probe_quality": probe.quality.value, "signal": probe.signal.value if probe.signal else None},
        )

    def _handle_decision(self, decision: Decision | None) -> StepResult:
        assert decision is not None
        self.conversation_history.append({"role": "assistant", "content": f"Decision: {decision.value}"})

        required = [
            self.known_signals[SignalKey.BUDGET],
            self.known_signals[SignalKey.TIMELINE],
            self.known_signals[SignalKey.DECISION_MAKER],
        ]
        known_required_count = sum(value is not None for value in required)
        if self.turn_number == 1 or known_required_count < 3:
            self.done = True
            error = InsufficientSignalsError("At least budget, timeline, and decision_maker are required.")
            return StepResult(
                observation=self._observation(),
                reward=INSUFFICIENT_SIGNALS_PENALTY,
                done=True,
                info={"error": type(error).__name__, "message": str(error)},
            )

        correct = decision == classify_lead(self.profile)
        reward = episode_end_reward(correct, self.turn_number)
        self.done = True
        return StepResult(
            observation=self._observation(),
            reward=reward,
            done=True,
            info={
                "correct_decision": correct,
                "expected_decision": classify_lead(self.profile).value,
                "profile": asdict(self.profile),
            },
        )
