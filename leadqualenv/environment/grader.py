from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import Decision, ProbeQuality, SignalKey

if TYPE_CHECKING:
    from .models import LeadProfile


SIGNAL_KEYWORDS: dict[SignalKey, list[str]] = {
    SignalKey.BUDGET: [
        "budget", "afford", "price", "cost", "range", "spend",
        "lakh", "crore", "financial", "money", "funds", "financing",
        "down payment", "mortgage", "loan", "paying", "invest amount",
    ],
    SignalKey.TIMELINE: [
        "timeline", "when", "soon", "months", "years", "urgency",
        "move", "timeframe", "deadline", "schedule", "date", "ready",
        "how quickly", "planning to buy", "looking to close",
    ],
    SignalKey.DECISION_MAKER: [
        "decision", "decide", "authority", "yourself", "spouse",
        "family", "alone", "partner", "co-buyer", "jointly",
        "sign", "approval", "consult", "permission",
    ],
    SignalKey.MOTIVATION: [
        "purpose", "investment", "own use", "why", "reason",
        "looking for", "goal", "plan", "intend", "objective",
        "rental", "flip", "primary residence", "second home",
    ],
}

# Ordered by priority: budget keywords should not accidentally match motivation
_PRIORITY_ORDER = [
    SignalKey.BUDGET,
    SignalKey.TIMELINE,
    SignalKey.DECISION_MAKER,
    SignalKey.MOTIVATION,
]

DIRECT_MARKERS = [
    "above", "below", "between", "more than", "less than",
    "confirm", "specifically", "exactly", "is it", "are you",
    "do you", "can you", "would you", "how much", "what is",
    "tell me about your", "what's your",
]

VERIFIED_MARKERS = [
    "you mentioned", "you said", "earlier you", "to confirm",
    "just to verify", "you told me", "so you're saying",
    "you indicated", "you stated", "based on what you said",
    "circling back", "revisiting", "double check", "want to make sure",
]

GENERIC_OPENERS = [
    "tell me about yourself",
    "tell me more about yourself",
    "how are you",
    "what's up",
    "nice to meet you",
    "hello there",
    "good morning",
    "good afternoon",
]

QUESTION_HINTS = [
    "what", "when", "which", "who", "how much",
    "how soon", "could you", "can you", "would you",
    "do you", "are you", "is there",
]


@dataclass(frozen=True)
class ProbeResult:
    quality: ProbeQuality
    signal: SignalKey | None


def classify_lead(profile: LeadProfile) -> Decision:
    if not profile.decision_maker:
        return Decision.UNQUALIFIED

    if profile.budget == "low":
        return Decision.UNQUALIFIED

    if profile.timeline == "immediate":
        return Decision.QUALIFIED if profile.budget in ("medium", "high") else Decision.UNQUALIFIED

    if profile.timeline == "3-6 months":
        return Decision.NURTURE

    if profile.timeline == "6+ months":
        return Decision.NURTURE if profile.budget in ("medium", "high") else Decision.UNQUALIFIED

    return Decision.UNQUALIFIED


def detect_signal_keyword(message: str) -> SignalKey | None:
    """Detect which signal the message is probing, respecting priority order."""
    lowered = message.lower()
    for signal in _PRIORITY_ORDER:
        keywords = SIGNAL_KEYWORDS[signal]
        if any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in keywords):
            return signal
    return None


def classify_probe(message: str, known_signals: Mapping[SignalKey, object | None]) -> ProbeResult:
    if is_generic_opener(message):
        return ProbeResult(quality=ProbeQuality.IRRELEVANT, signal=None)

    signal = detect_signal_keyword(message)
    if signal is None:
        return ProbeResult(quality=ProbeQuality.IRRELEVANT, signal=None)

    lowered = message.lower()
    already_known = known_signals.get(signal) is not None

    # Verification: must reference earlier conversation AND signal must already be known
    if already_known and any(marker in lowered for marker in VERIFIED_MARKERS):
        return ProbeResult(quality=ProbeQuality.VERIFIED, signal=signal)

    # Direct probe with question structure or direct markers
    if any(marker in lowered for marker in DIRECT_MARKERS):
        quality = ProbeQuality.VAGUE if already_known else ProbeQuality.DIRECT
        return ProbeResult(quality=quality, signal=signal)

    has_question_shape = "?" in message or any(hint in lowered for hint in QUESTION_HINTS)
    if has_question_shape:
        quality = ProbeQuality.VAGUE if already_known else ProbeQuality.DIRECT
        return ProbeResult(quality=quality, signal=signal)

    # Bare keyword mention with no question structure
    if already_known:
        return ProbeResult(quality=ProbeQuality.IRRELEVANT, signal=signal)

    core_triggers = {"range", "budget", "timeline", "authority", "investment", "decision"}
    if any(keyword in lowered for keyword in core_triggers):
        return ProbeResult(quality=ProbeQuality.DIRECT, signal=signal)
    return ProbeResult(quality=ProbeQuality.VAGUE, signal=signal)


def is_generic_opener(message: str) -> bool:
    lowered = message.lower().strip()
    return any(opener in lowered for opener in GENERIC_OPENERS)
