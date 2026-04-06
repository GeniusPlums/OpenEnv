from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .grader import classify_lead, classify_probe, is_generic_opener
from .models import (
    Action,
    Decision,
    EnvironmentState,
    EnvironmentSnapshot,
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
        self._objections_seen: set[SignalKey] = set()
        self._verification_evasions: set[SignalKey] = set()
        self._competitor_mentioned = False

    def reset(self, seed: int | None = None, generated_profiles: int = 0) -> Observation:
        profile = sample_profile(self.task, seed, generated_count=generated_profiles)
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
        self._objections_seen = set()
        self._verification_evasions = set()
        self._competitor_mentioned = False

        opener = sample_opener(profile, seed)
        property_ctx = f"{profile.property_type} in {profile.location}"

        self.conversation_history = [
            {"role": "system", "content": "You are qualifying a prospective real estate buyer."},
            {"role": "user", "content": opener},
        ]
        return self._observation(property_context=property_ctx)

    def restore(self, snapshot: EnvironmentSnapshot) -> Observation:
        self.task = snapshot.task
        self.max_turns = snapshot.max_turns
        self.profile = snapshot.profile
        self.turn_number = snapshot.turn_number
        self.done = snapshot.done
        self.conversation_history = list(snapshot.conversation_history)
        self.known_signals = dict(snapshot.known_signals)
        self.probe_log = list(snapshot.probe_log)
        self._lead_temperature = snapshot.lead_temperature
        self._qual_confidence = snapshot.qualification_confidence
        self._objections_seen = set(snapshot.objections_seen)
        self._verification_evasions = set(snapshot.verification_evasions)
        self._competitor_mentioned = snapshot.competitor_mentioned
        return self._observation()

    def snapshot(self) -> EnvironmentSnapshot:
        assert self.profile is not None
        return EnvironmentSnapshot(
            task=self.task,
            max_turns=self.max_turns,
            profile=self.profile,
            turn_number=self.turn_number,
            done=self.done,
            conversation_history=list(self.conversation_history),
            known_signals=dict(self.known_signals),
            probe_log=list(self.probe_log),
            lead_temperature=self._lead_temperature,
            qualification_confidence=self._qual_confidence,
            objections_seen=sorted(self._objections_seen, key=lambda item: item.value),
            competitor_mentioned=self._competitor_mentioned,
            verification_evasions=sorted(self._verification_evasions, key=lambda item: item.value),
        )

    def observation(self) -> Observation:
        return self._observation()

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
            verification_evasions=sorted(self._verification_evasions, key=lambda item: item.value),
        )

    def _observation(self, property_context: str | None = None) -> Observation:
        ctx = property_context
        if ctx is None and self.profile is not None:
            ctx = f"{self.profile.property_type} in {self.profile.location}"

        previous_crm = None
        if self.task == TaskLevel.REQUALIFICATION and self.profile is not None:
            previous_crm = self.profile.previous_crm

        return Observation(
            conversation_history=list(self.conversation_history),
            known_signals=dict(self.known_signals),
            probe_log=list(self.probe_log),
            turn_number=self.turn_number,
            max_turns=self.max_turns,
            lead_temperature=self._lead_temperature,
            qualification_confidence=self._qual_confidence,
            property_context=ctx,
            competitor_mentioned=self._competitor_mentioned,
            previous_crm=previous_crm,
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
        """Lead temperature decays gradually without auto-collapsing near the turn cap."""
        if self.max_turns <= 1:
            self._lead_temperature = 0.4
            return
        progress = min(1.0, max(0.0, self.turn_number / self.max_turns))
        self._lead_temperature = round(max(0.4, 1.0 - 0.6 * progress), 4)

    def _update_confidence(self) -> None:
        """Compute running qualification confidence from known signals."""
        known_count = sum(1 for v in self.known_signals.values() if v is not None)
        verified_required = len({
            signal for signal, quality in self.probe_log
            if quality == ProbeQuality.VERIFIED and signal in {
                SignalKey.BUDGET,
                SignalKey.TIMELINE,
                SignalKey.DECISION_MAKER,
            }
        })
        coverage_score = known_count / len(self.known_signals)
        verification_score = verified_required / 3
        self._qual_confidence = round(min(1.0, 0.75 * coverage_score + 0.25 * verification_score), 4)

    def _partial_grade(self) -> dict[str, object]:
        assert self.profile is not None
        grade = grade_episode(
            task=self.task,
            known_signals=self.known_signals,
            probe_log=self.probe_log,
            correct_decision=False,
            personality=self.profile.personality,
            misleading_signals=set(self.profile.surface_signals),
            profile=self.profile,
        )
        return {
            "task_score": grade.score,
            "task_score_components": grade.components,
        }

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

        verification_already_evaded = (probe.signal in self._verification_evasions) if probe.signal else False

        response_text, value = generate_response(
            self.profile, probe.signal, probe.quality, self.task,
            lead_temperature=self._lead_temperature,
            objection_already_triggered=probe.signal in self._objections_seen if probe.signal else False,
            verification_already_evaded=verification_already_evaded,
        )
        self.conversation_history.append({"role": "user", "content": response_text})

        if self.profile.competitor_mention and probe.signal in {SignalKey.TIMELINE, SignalKey.BUDGET}:
            self._competitor_mentioned = True

        objection_triggered = (
            probe.signal is not None
            and self.profile.objection_on == probe.signal
            and probe.quality == ProbeQuality.DIRECT
            and probe.signal not in self._objections_seen
        )
        if objection_triggered:
            self._objections_seen.add(probe.signal)

        verification_evasion_triggered = (
            probe.signal is not None
            and probe.quality == ProbeQuality.VERIFIED
            and probe.signal in self.profile.verification_evasion_signals
            and not verification_already_evaded
        )
        if verification_evasion_triggered:
            self._verification_evasions.add(probe.signal)

        if probe.signal is not None:
            if not objection_triggered:
                self.probe_log.append((probe.signal, probe.quality))
            if not objection_triggered and not verification_evasion_triggered and probe.quality in (ProbeQuality.DIRECT, ProbeQuality.VERIFIED):
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
            partial_grade = self._partial_grade()

        return StepResult(
            observation=self._observation(),
            reward=round(reward, 4),
            done=done,
            info={
                "probe_quality": probe.quality.value,
                "signal": probe.signal.value if probe.signal else None,
                "lead_temperature": self._lead_temperature,
                "termination_reason": "max_turns_reached" if done else None,
                **(partial_grade if done else {}),
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
            partial_grade = self._partial_grade()
            return StepResult(
                observation=self._observation(),
                reward=INSUFFICIENT_SIGNALS_PENALTY,
                done=True,
                info={
                    "error": type(error).__name__,
                    "message": str(error),
                    "termination_reason": "insufficient_signals",
                    **partial_grade,
                },
            )

        correct = decision == classify_lead(self.profile)
        reward = episode_end_reward(correct, self.turn_number)
        grade = grade_episode(
            task=self.task,
            known_signals=self.known_signals,
            probe_log=self.probe_log,
            correct_decision=correct,
            personality=self.profile.personality,
            misleading_signals=set(self.profile.surface_signals),
            profile=self.profile,
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
