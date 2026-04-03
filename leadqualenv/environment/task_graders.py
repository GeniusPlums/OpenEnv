from __future__ import annotations

from dataclasses import dataclass

from .models import ProbeQuality, SignalKey, TaskLevel


TASK_WEIGHTS: dict[TaskLevel, dict[str, float]] = {
    TaskLevel.EASY: {
        "correct_decision": 0.50,
        "signal_coverage": 0.30,
        "probe_quality": 0.10,
        "efficiency": 0.10,
    },
    TaskLevel.MEDIUM: {
        "correct_decision": 0.55,
        "signal_coverage": 0.20,
        "probe_quality": 0.15,
        "efficiency": 0.10,
    },
    TaskLevel.HARD: {
        "correct_decision": 0.45,
        "signal_coverage": 0.15,
        "probe_quality": 0.20,
        "verification": 0.20,
    },
}

REQUIRED_SIGNALS = {
    SignalKey.BUDGET,
    SignalKey.TIMELINE,
    SignalKey.DECISION_MAKER,
}

PROBE_QUALITY_SCORES = {
    ProbeQuality.IRRELEVANT: 0.0,
    ProbeQuality.VAGUE: 0.2,
    ProbeQuality.DIRECT: 0.8,
    ProbeQuality.VERIFIED: 1.0,
}


@dataclass(frozen=True)
class TaskGrade:
    task: str
    score: float
    components: dict[str, float]


def grade_episode(
    task: TaskLevel,
    known_signals: dict[SignalKey, str | bool | None],
    probe_log: list[tuple[SignalKey, ProbeQuality]],
    correct_decision: bool,
) -> TaskGrade:
    weights = TASK_WEIGHTS[task]

    coverage = sum(1 for signal in REQUIRED_SIGNALS if known_signals.get(signal) is not None) / len(REQUIRED_SIGNALS)

    if probe_log:
        probe_quality = sum(PROBE_QUALITY_SCORES[quality] for _, quality in probe_log) / len(probe_log)
    else:
        probe_quality = 0.0

    verified_signals = {signal for signal, quality in probe_log if quality == ProbeQuality.VERIFIED}
    verification = len(verified_signals & REQUIRED_SIGNALS) / len(REQUIRED_SIGNALS)

    max_expected_probes = 3 if task != TaskLevel.HARD else 5
    efficiency = max(0.0, 1.0 - max(0, len(probe_log) - max_expected_probes) * 0.2)

    components = {
        "correct_decision": 1.0 if correct_decision else 0.0,
        "signal_coverage": round(coverage, 4),
        "probe_quality": round(probe_quality, 4),
        "verification": round(verification, 4),
        "efficiency": round(efficiency, 4),
    }
    score = sum(weights[key] * components[key] for key in weights)
    return TaskGrade(task=task.value, score=round(min(max(score, 0.0), 1.0), 4), components=components)
