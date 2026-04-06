from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    REQUALIFICATION = "requalification"


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


class Personality(str, Enum):
    DIRECT = "direct"
    EVASIVE = "evasive"
    VERBOSE = "verbose"
    TERSE = "terse"
    FRIENDLY = "friendly"


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
    personality: Personality = Personality.DIRECT
    property_type: str = "apartment"
    location: str = "downtown"
    surface_budget: Optional[str] = None
    surface_timeline: Optional[str] = None
    competitor_mention: bool = False
    objection_on: Optional[SignalKey] = None
    verification_evasion_signals: frozenset[SignalKey] = field(default_factory=frozenset)
    previous_qualification: Optional[str] = None
    motivation_shift: bool = False
    previous_crm: Optional[dict[str, Any]] = field(default=None, hash=False)

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

    @property
    def has_misleading_signals(self) -> bool:
        return bool(self.surface_signals)


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
    lead_temperature: float = 1.0
    qualification_confidence: float = 0.0
    property_context: Optional[str] = None
    competitor_mentioned: bool = False
    previous_crm: Optional[dict[str, Any]] = None


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict[str, object] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    task: TaskLevel
    turn_number: int
    max_turns: int
    done: bool
    conversation_history: list[dict[str, str]]
    known_signals: dict[SignalKey, Optional[str | bool]]
    probe_log: list[tuple[SignalKey, ProbeQuality]]
    lead_temperature: float = 1.0
    qualification_confidence: float = 0.0
    verification_evasions: list[SignalKey] = field(default_factory=list)


@dataclass
class EnvironmentSnapshot:
    task: TaskLevel
    max_turns: int
    profile: LeadProfile
    turn_number: int
    done: bool
    conversation_history: list[dict[str, str]]
    known_signals: dict[SignalKey, Optional[str | bool]]
    probe_log: list[tuple[SignalKey, ProbeQuality]]
    lead_temperature: float = 1.0
    qualification_confidence: float = 0.0
    objections_seen: list[SignalKey] = field(default_factory=list)
    competitor_mentioned: bool = False
    verification_evasions: list[SignalKey] = field(default_factory=list)
