from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Personality, ProbeQuality, SignalKey, TaskLevel


def _efficiency_score(probe_count: int, max_expected: int) -> float:
    if probe_count <= max_expected:
        return 1.0
    excess = probe_count - max_expected
    # Decays to ~0.22 at 2x expected, ~0.05 at 3x — still differentiates
    return round(math.exp(-0.15 * excess), 4)

# Personality difficulty multiplier: harder personalities earn slightly more credit
PERSONALITY_DIFFICULTY: dict[Personality, float] = {
    Personality.DIRECT: 1.0,
    Personality.FRIENDLY: 1.0,
    Personality.VERBOSE: 1.02,
    Personality.TERSE: 1.04,
    Personality.EVASIVE: 1.06,
}


TASK_WEIGHTS: dict[TaskLevel, dict[str, float]] = {
    TaskLevel.EASY: {
        "correct_decision": 0.45,
        "signal_coverage": 0.25,
        "probe_quality": 0.15,
        "efficiency": 0.15,
    },
    TaskLevel.MEDIUM: {
        "correct_decision": 0.45,
        "signal_coverage": 0.20,
        "probe_quality": 0.15,
        "efficiency": 0.10,
        "verification": 0.10,
    },
    TaskLevel.HARD: {
        "correct_decision": 0.20,
        "signal_coverage": 0.10,
        "probe_quality": 0.10,
        "verification": 0.45,
        "efficiency": 0.05,
        "misleading_detection": 0.10,
    },
    TaskLevel.REQUALIFICATION: {
        "correct_decision": 0.35,
        "signal_coverage": 0.15,
        "probe_quality": 0.10,
        "verification": 0.25,
        "efficiency": 0.05,
        "motivation_shift": 0.10,
    },
}

REQUIRED_SIGNALS = {
    SignalKey.BUDGET,
    SignalKey.TIMELINE,
    SignalKey.DECISION_MAKER,
}

ALL_SIGNALS = {
    SignalKey.BUDGET,
    SignalKey.TIMELINE,
    SignalKey.DECISION_MAKER,
    SignalKey.MOTIVATION,
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
    *,
    personality: Personality | None = None,
    misleading_signals: set[SignalKey] | None = None,
    profile: getattr(__import__("leadqualenv.environment.models", fromlist=["LeadProfile"]), "LeadProfile") | None = None, # type: ignore
) -> TaskGrade:
    weights = TASK_WEIGHTS[task]

    # Signal coverage: required signals + bonus for motivation
    required_known = sum(1 for s in REQUIRED_SIGNALS if known_signals.get(s) is not None)
    all_known = sum(1 for s in ALL_SIGNALS if known_signals.get(s) is not None)
    coverage = (required_known / len(REQUIRED_SIGNALS)) * 0.8 + (all_known / len(ALL_SIGNALS)) * 0.2

    # Probe quality: average quality across all probes
    if probe_log:
        probe_quality = sum(PROBE_QUALITY_SCORES[quality] for _, quality in probe_log) / len(probe_log)
    else:
        probe_quality = 0.0

    # Verification: proportion of required signals that were verified
    verified_signals = {signal for signal, quality in probe_log if quality == ProbeQuality.VERIFIED}
    verification = len(verified_signals & REQUIRED_SIGNALS) / len(REQUIRED_SIGNALS)

    # Efficiency: penalize excessive probing
    max_expected_probes = 4 if task == TaskLevel.EASY else 5 if task == TaskLevel.MEDIUM else 7
    efficiency = _efficiency_score(len(probe_log), max_expected_probes)

    misleading_detection = 0.0
    if task == TaskLevel.HARD:
        direct_signals = {signal for signal, quality in probe_log if quality == ProbeQuality.DIRECT}

        # The traps we expect the agent to detect
        if misleading_signals is not None and len(misleading_signals) > 0:
            target_traps = misleading_signals & REQUIRED_SIGNALS
            if target_traps:
                verified_traps = {
                    signal for signal in target_traps
                    if signal in verified_signals and signal in direct_signals
                }
                misleading_detection = len(verified_traps) / len(target_traps)
            else:
                misleading_detection = 1.0
        else:
            # Full credit if there are no misleading signals in this profile
            misleading_detection = 1.0

    motivation_shift = 0.0
    if task == TaskLevel.REQUALIFICATION:
        if profile is not None and profile.motivation_shift and known_signals.get(SignalKey.MOTIVATION) is not None:
            motivation_shift = 1.0

    components = {
        "correct_decision": 1.0 if correct_decision else 0.0,
        "signal_coverage": round(coverage, 4),
        "probe_quality": round(probe_quality, 4),
        "verification": round(verification, 4),
        "efficiency": round(efficiency, 4),
        "misleading_detection": round(misleading_detection, 4),
        "motivation_shift": round(motivation_shift, 4),
    }
    raw_score = sum(weights.get(key, 0.0) * components[key] for key in components)

    # Apply personality difficulty multiplier (rewards harder conversations)
    if personality is not None:
        raw_score *= PERSONALITY_DIFFICULTY.get(personality, 1.0)

    score = round(min(max(raw_score, 0.0), 1.0), 4)
    return TaskGrade(task=task.value, score=score, components=components)
