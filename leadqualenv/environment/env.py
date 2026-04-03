from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .grader import classify_lead, classify_probe, is_generic_opener
from .models import (
    Action,
    Decision,
    EnvironmentState,
    InsufficientSignalsError,
    InvalidActionError,
    LeadProfile,
    Observation,
    ProbeQuality,
    SignalKey,
    StepResult,
    TaskLevel,
)
from .profiles import sample_opener, sample_profile
from .reward import (
    INSUFFICIENT_SIGNALS_PENALTY,
    MOTIVATION_DISCOVERY_BONUS,
    NO_DECISION_PENALTY,
    REPEATED_QUESTION_PENALTY,
    cold_lead_penalty,
    episode_end_reward,
    rapport_bonus,
    signal_order_penalty,
    turn_reward,
)
from .simulator import generate_response
from .task_graders import grade_episode


class LeadQualEnv:
    def __init__(self, task: TaskLevel, max_turns: int = 10):
        self.task = task
        self.max_turns = max_turns
        self.profile: LeadProfile | None = None
        self.turn_number = 0
        self.known_signals: dict[SignalKey, str | bool | None] = {}
        self.probe_log: list[tuple[SignalKey, ProbeQuality]] = []
        self.conversation_history: list[dict[str, str]] = []
        self.done = False
        self._lead_temperature = 1.0
        self._qual_confidence = 0.0

    def reset(self, seed: int | None = None) -> Observation:
        profile = sample_profile(self.task, seed)
        self.profile = profile
        self.turn_number = 0
        self.done = False
        self.known_signals = {
            SignalKey.BUDGET: None,
            SignalKey.TIMELINE: None,
            SignalKey.DECISION_MAKER: None,
            SignalKey.MOTIVATION: None,
        }
        self.probe_log = []
        self._lead_temperature = 1.0
        self._qual_confidence = 0.0

        opener = sample_opener(profile, seed)
        property_ctx = f"{profile.property_type} in {profile.location}"

        self.conversation_history = [
            {"role": "system", "content": "You are qualifying a prospective real estate buyer."},
            {"role": "user", "content": opener},
        ]
        return self._observation(property_context=property_ctx)

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task=self.task,
            turn_number=self.turn_number,
            max_turns=self.max_turns,
            done=self.done,
            conversation_history=list(self.conversation_history),
            known_signals=dict(self.known_signals),
            probe_log=list(self.probe_log),
            lead_temperature=self._lead_temperature,
            qualification_confidence=self._qual_confidence,
        )

    def _observation(self, property_context: str | None = None) -> Observation:
        ctx = property_context
        if ctx is None and self.profile is not None:
            ctx = f"{self.profile.property_type} in {self.profile.location}"
        return Observation(
            conversation_history=list(self.conversation_history),
            known_signals=dict(self.known_signals),
            probe_log=list(self.probe_log),
            turn_number=self.turn_number,
            max_turns=self.max_turns,
            lead_temperature=self._lead_temperature,
            qualification_confidence=self._qual_confidence,
            property_context=ctx,
        )

    def _crm_card(self, status: str) -> dict[str, Any]:
        assert self.profile is not None

        next_action = {
            Decision.QUALIFIED.value: "handoff_to_sales_agent",
            Decision.NURTURE.value: "schedule_follow_up",
            Decision.UNQUALIFIED.value: "archive_or_recycle",
            "no_decision": "reengage_or_close_lost",
        }.get(status, "review")

        notes = [
            f"budget={self.known_signals.get(SignalKey.BUDGET)!r}",
            f"timeline={self.known_signals.get(SignalKey.TIMELINE)!r}",
            f"decision_maker={self.known_signals.get(SignalKey.DECISION_MAKER)!r}",
            f"motivation={self.known_signals.get(SignalKey.MOTIVATION)!r}",
        ]
        return {
            "property_interest": f"{self.profile.property_type} in {self.profile.location}",
            "qualification_status": status,
            "lead_personality": self.profile.personality.value,
            "lead_temperature": round(self._lead_temperature, 2),
            "qualification_confidence": round(self._qual_confidence, 2),
            "notes": "; ".join(notes),
            "next_action": next_action,
        }

    def _update_temperature(self) -> None:
        """Lead temperature decays each turn — simulates lead cooling."""
        self._lead_temperature = max(0.0, 1.0 - 0.08 * self.turn_number)

    def _update_confidence(self) -> None:
        """Compute running qualification confidence from known signals."""
        known_count = sum(1 for v in self.known_signals.values() if v is not None)
        verified_count = sum(1 for _, q in self.probe_log if q == ProbeQuality.VERIFIED)
        self._qual_confidence = min(1.0, (known_count * 0.2) + (verified_count * 0.1))

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise InvalidActionError("Episode already completed.")
        if self.profile is None:
            raise InvalidActionError("Environment must be reset before stepping.")

        self.turn_number += 1
        self._update_temperature()

        if action.message:
            return self._handle_message(action.message)
        return self._handle_decision(action.decision)

    def _handle_message(self, message: str) -> StepResult:
        assert self.profile is not None
        probe = classify_probe(message, self.known_signals)
        reward = turn_reward(probe.quality)

        if is_generic_opener(message):
            reward = min(reward, -0.05)

        # Rapport bonus for early property-context engagement
        reward += rapport_bonus(message, self.turn_number)

        # Signal order penalty for asking motivation before critical signals
        if probe.signal is not None:
            signal_name = probe.signal.value
            known_as_str = {k.value: v for k, v in self.known_signals.items()}
            reward += signal_order_penalty(signal_name, self.turn_number, known_as_str)

        # Repeated question penalty — only applies once per step (fixed stacking bug)
        if probe.signal is not None and self.known_signals.get(probe.signal) is not None:
            if probe.quality != ProbeQuality.VERIFIED:
                recent_signals = [signal for signal, _ in self.probe_log[-3:]]
                if probe.signal in recent_signals:
                    # Tight-loop repetition: harder penalty
                    reward += REPEATED_QUESTION_PENALTY
                else:
                    # Re-asking without verification but not tight-loop
                    reward += REPEATED_QUESTION_PENALTY * 0.5

        self.conversation_history.append({"role": "assistant", "content": message})

        response_text, value = generate_response(
            self.profile, probe.signal, probe.quality, self.task,
            lead_temperature=self._lead_temperature,
        )
        self.conversation_history.append({"role": "user", "content": response_text})

        if probe.signal is not None:
            self.probe_log.append((probe.signal, probe.quality))
            if probe.quality in (ProbeQuality.DIRECT, ProbeQuality.VERIFIED):
                previous_value = self.known_signals.get(probe.signal)
                self.known_signals[probe.signal] = value

                # Motivation discovery bonus
                if probe.signal == SignalKey.MOTIVATION and previous_value is None and value is not None:
                    reward += MOTIVATION_DISCOVERY_BONUS

        self._update_confidence()

        # Cold lead penalty for dawdling — harder in hard mode
        reward += cold_lead_penalty(self.turn_number, task=self.task)

        done = self.turn_number >= self.max_turns
        if done:
            self.done = True
            reward += NO_DECISION_PENALTY

        return StepResult(
            observation=self._observation(),
            reward=round(reward, 4),
            done=done,
            info={
                "probe_quality": probe.quality.value,
                "signal": probe.signal.value if probe.signal else None,
                "lead_temperature": self._lead_temperature,
                "crm_card": self._crm_card("no_decision") if done else None,
            },
        )

    def _handle_decision(self, decision: Decision | None) -> StepResult:
        assert decision is not None
        assert self.profile is not None
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
        grade = grade_episode(
            task=self.task,
            known_signals=self.known_signals,
            probe_log=self.probe_log,
            correct_decision=correct,
            personality=self.profile.personality,
        )
        self.done = True
        return StepResult(
            observation=self._observation(),
            reward=round(reward, 4),
            done=True,
            info={
                "correct_decision": correct,
                "expected_decision": classify_lead(self.profile).value,
                "profile": asdict(self.profile),
                "task_score": grade.score,
                "task_score_components": grade.components,
                "crm_card": self._crm_card(decision.value),
            },
        )
