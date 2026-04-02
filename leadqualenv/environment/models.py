from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ProbeQuality(str, Enum):
    IRRELEVANT = "irrelevant"
    VAGUE = "vague"
    DIRECT = "direct"
    VERIFIED = "verified"


class SignalKey(str, Enum):
    BUDGET = "budget"
    TIMELINE = "timeline"
    DECISION_MAKER = "decision_maker"
    MOTIVATION = "motivation"


class Decision(str, Enum):
    QUALIFIED = "qualified"
    NURTURE = "nurture"
    UNQUALIFIED = "unqualified"


class LeadQualEnvError(Exception):
    """Base environment error."""


class InvalidActionError(LeadQualEnvError):
    """Raised when the agent sends an invalid action."""


class InsufficientSignalsError(LeadQualEnvError):
    """Raised when the agent decides before collecting required signals."""


@dataclass(frozen=True)
class LeadProfile:
    budget: str
    timeline: str
    decision_maker: bool
    motivation: str
    surface_budget: Optional[str] = None
    surface_timeline: Optional[str] = None

    @property
    def true_signals(self) -> dict[SignalKey, str | bool]:
        return {
            SignalKey.BUDGET: self.budget,
            SignalKey.TIMELINE: self.timeline,
            SignalKey.DECISION_MAKER: self.decision_maker,
            SignalKey.MOTIVATION: self.motivation,
        }

    @property
    def surface_signals(self) -> dict[SignalKey, str | bool]:
        signals: dict[SignalKey, str | bool] = {}
        if self.surface_budget is not None:
            signals[SignalKey.BUDGET] = self.surface_budget
        if self.surface_timeline is not None:
            signals[SignalKey.TIMELINE] = self.surface_timeline
        return signals


@dataclass(frozen=True)
class Action:
    message: Optional[str] = None
    decision: Optional[Decision] = None

    def __post_init__(self) -> None:
        if bool(self.message) == bool(self.decision):
            raise InvalidActionError("Exactly one of message or decision must be set.")


@dataclass
class Observation:
    conversation_history: list[dict[str, str]]
    known_signals: dict[SignalKey, Optional[str | bool]]
    probe_log: list[tuple[SignalKey, ProbeQuality]]
    turn_number: int
    max_turns: int


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, object] = field(default_factory=dict)
