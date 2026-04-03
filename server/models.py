from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, model_validator


class LeadQualActionModel(Action):
    message: str | None = Field(default=None, description="Agent message to send to the buyer")
    decision: str | None = Field(
        default=None,
        description="Optional final decision: qualified, nurture, or unqualified",
    )

    @model_validator(mode="after")
    def validate_exclusive_fields(self) -> "LeadQualActionModel":
        if bool(self.message) == bool(self.decision):
            raise ValueError("Exactly one of message or decision must be provided.")
        return self


class LeadQualObservationModel(Observation):
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    known_signals: dict[str, str | bool | None] = Field(default_factory=dict)
    probe_log: list[tuple[str, str]] = Field(default_factory=list)
    turn_number: int = 0
    max_turns: int = 10
    info: dict[str, Any] = Field(default_factory=dict)


class LeadQualStateModel(State):
    task: str = Field(default="easy")
    max_turns: int = 10
    done: bool = False
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    known_signals: dict[str, str | bool | None] = Field(default_factory=dict)
    probe_log: list[tuple[str, str]] = Field(default_factory=list)
