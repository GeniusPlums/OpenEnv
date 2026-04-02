from __future__ import annotations

from .models import Decision, ProbeQuality


TURN_REWARDS = {
    ProbeQuality.DIRECT: 0.05,
    ProbeQuality.VERIFIED: 0.03,
    ProbeQuality.VAGUE: -0.03,
    ProbeQuality.IRRELEVANT: -0.05,
}

INSUFFICIENT_SIGNALS_PENALTY = -0.30
REPEATED_QUESTION_PENALTY = -0.05
CORRECT_DECISION_REWARD = 0.50
INCORRECT_DECISION_PENALTY = -0.40
NO_DECISION_PENALTY = -0.20


def turn_reward(probe_quality: ProbeQuality) -> float:
    return TURN_REWARDS[probe_quality]


def decision_timing_reward(turn_number: int) -> float:
    if 4 <= turn_number <= 6:
        return 0.10
    if 7 <= turn_number <= 8:
        return 0.00
    if turn_number > 8:
        return -0.10
    return 0.00


def lead_decay_bonus(turn_number: int, correct_decision: bool) -> float:
    if not correct_decision:
        return 0.0
    return max(0.0, 0.10 - 0.05 * max(0, turn_number - 6))


def episode_end_reward(correct_decision: bool, turn_number: int) -> float:
    base = CORRECT_DECISION_REWARD if correct_decision else INCORRECT_DECISION_PENALTY
    return base + decision_timing_reward(turn_number) + lead_decay_bonus(turn_number, correct_decision)
