from __future__ import annotations

from .models import ProbeQuality, TaskLevel

TURN_REWARDS = {
    ProbeQuality.DIRECT: 0.05,
    ProbeQuality.VERIFIED: 0.06,
    ProbeQuality.VAGUE: -0.03,
    ProbeQuality.IRRELEVANT: -0.05,
}

INSUFFICIENT_SIGNALS_PENALTY = -0.30
REPEATED_QUESTION_PENALTY = -0.05
CORRECT_DECISION_REWARD = 0.50
INCORRECT_DECISION_PENALTY = -0.40
NO_DECISION_PENALTY = -0.20
MOTIVATION_DISCOVERY_BONUS = 0.02
COLD_LEAD_PENALTY_PER_TURN = 0.015
RAPPORT_BONUS = 0.03
SIGNAL_ORDER_PENALTY = -0.02


def turn_reward(probe_quality: ProbeQuality) -> float:
    return TURN_REWARDS[probe_quality]


def rapport_bonus(message: str, turn_number: int) -> float:
    """Reward agents that engage with property context before diving into qualification.

    Mirrors real sales best-practice: build rapport first, then qualify.
    Only applies in the first 2 turns.
    """
    if turn_number > 2:
        return 0.0
    lowered = message.lower()
    rapport_markers = [
        "tell me more about",
        "what stood out",
        "what do you like about",
        "how does",
        "what caught your eye",
        "what are you hoping for",
        "what kind of",
        "which part of",
    ]
    property_keywords = [
        "property", "apartment", "villa", "condo", "townhouse", "penthouse",
        "studio", "duplex", "bungalow", "neighborhood", "area", "location",
        "listing", "home", "house", "place",
    ]
    has_property_context = any(keyword in lowered for keyword in property_keywords)
    if has_property_context and any(marker in lowered for marker in rapport_markers):
        return RAPPORT_BONUS
    return 0.0


def signal_order_penalty(
    signal_name: str | None,
    turn_number: int,
    known_signals: dict[str, str | bool | None],
) -> float:
    """Slight penalty for asking about motivation before critical signals.

    Real SDRs prioritize budget/timeline/DM over motivation.
    """
    if signal_name != "motivation":
        return 0.0
    critical_known = sum(
        1 for key in ("budget", "timeline", "decision_maker")
        if known_signals.get(key) is not None
    )
    if critical_known < 2:
        return SIGNAL_ORDER_PENALTY
    return 0.0


def decision_timing_reward(turn_number: int) -> float:
    if 3 <= turn_number <= 5:
        return 0.15
    if 6 <= turn_number <= 7:
        return 0.05
    if turn_number > 7:
        return -0.10
    return 0.00


def lead_decay_bonus(turn_number: int, correct_decision: bool) -> float:
    if not correct_decision:
        return 0.0
    return max(0.0, 0.10 - 0.02 * max(0, turn_number - 4))


def cold_lead_penalty(turn_number: int, task: TaskLevel | None = None) -> float:
    """Leads cool down over time — penalizes dawdling.

    Hard mode has earlier and steeper decay to force tighter play.
    """
    threshold = 5 if task == TaskLevel.HARD else 6
    rate = COLD_LEAD_PENALTY_PER_TURN * (1.5 if task == TaskLevel.HARD else 1.0)
    if turn_number <= threshold:
        return 0.0
    return -rate * (turn_number - threshold)


def episode_end_reward(correct_decision: bool, turn_number: int) -> float:
    base = CORRECT_DECISION_REWARD if correct_decision else INCORRECT_DECISION_PENALTY
    return base + decision_timing_reward(turn_number) + lead_decay_bonus(turn_number, correct_decision)
