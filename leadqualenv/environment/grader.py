from __future__ import annotations

from dataclasses import dataclass

from .models import Decision, ProbeQuality, SignalKey


SIGNAL_KEYWORDS: dict[SignalKey, list[str]] = {
    SignalKey.BUDGET: ["budget", "afford", "price", "cost", "range", "spend", "lakh", "crore"],
    SignalKey.TIMELINE: ["timeline", "when", "soon", "months", "years", "urgency", "move"],
    SignalKey.DECISION_MAKER: ["decision", "decide", "authority", "yourself", "spouse", "family", "alone"],
    SignalKey.MOTIVATION: ["looking for", "purpose", "investment", "own use", "why"],
}

DIRECT_MARKERS = [
    "above",
    "below",
    "between",
    "more than",
    "less than",
    "confirm",
    "specifically",
    "exactly",
    "is it",
    "are you",
    "do you",
]

VERIFIED_MARKERS = [
    "you mentioned",
    "you said",
    "earlier you",
    "to confirm",
    "just to verify",
    "you told me",
    "so you're saying",
]

GENERIC_OPENERS = [
    "tell me about yourself",
    "tell me more about yourself",
    "how are you",
    "what's up",
    "nice to meet you",
]


@dataclass(frozen=True)
class ProbeResult:
    quality: ProbeQuality
    signal: SignalKey | None


def classify_lead(profile) -> Decision:
    if not profile.decision_maker:
        return Decision.UNQUALIFIED

    if profile.timeline == "immediate":
        return Decision.QUALIFIED if profile.budget in ("medium", "high") else Decision.UNQUALIFIED

    if profile.timeline == "3-6 months":
        return Decision.NURTURE

    return Decision.UNQUALIFIED


def detect_signal_keyword(message: str) -> SignalKey | None:
    lowered = message.lower()
    for signal, keywords in SIGNAL_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return signal
    return None


def classify_probe(message: str, known_signals: dict[SignalKey, object | None]) -> ProbeResult:
    del known_signals
    if is_generic_opener(message):
        return ProbeResult(ProbeQuality.IRRELEVANT, None)

    signal = detect_signal_keyword(message)
    if signal is None:
        return ProbeResult(ProbeQuality.IRRELEVANT, None)

    lowered = message.lower()
    if any(marker in lowered for marker in VERIFIED_MARKERS):
        return ProbeResult(ProbeQuality.VERIFIED, signal)
    if any(marker in lowered for marker in DIRECT_MARKERS):
        return ProbeResult(ProbeQuality.DIRECT, signal)
    return ProbeResult(ProbeQuality.VAGUE, signal)


def is_generic_opener(message: str) -> bool:
    lowered = message.lower().strip()
    return any(opener in lowered for opener in GENERIC_OPENERS)
