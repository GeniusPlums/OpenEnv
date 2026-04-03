from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from leadqualenv.environment import Action, Decision, LeadQualEnv, TaskLevel

from .models import LeadQualActionModel, LeadQualObservationModel, LeadQualRewardModel, LeadQualStateModel

TASK_NAME_MAP = {
    "easy": TaskLevel.EASY,
    "medium": TaskLevel.MEDIUM,
    "hard": TaskLevel.HARD,
}


class LeadQualOpenEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._task = TaskLevel.EASY
        self._env = LeadQualEnv(task=self._task)
        self._state = LeadQualStateModel(episode_id=str(uuid4()), step_count=0, task=self._task.value)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> LeadQualObservationModel:
        task_name = kwargs.get("task", "easy")
        self._task = TASK_NAME_MAP.get(task_name, TaskLevel.EASY)
        self._env = LeadQualEnv(task=self._task)
        obs = self._env.reset(seed=seed)
        self._sync_state(episode_id=episode_id or str(uuid4()))
        return self._convert_observation(
            obs,
            reward=None,
            done=False,
            info={
                "task": self._task.value,
                "available_actions": {
                    "message": "Ask the buyer a follow-up question.",
                    "decision": ["qualified", "nurture", "unqualified"],
                },
            },
        )

    def step(
        self,
        action: LeadQualActionModel,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> LeadQualObservationModel:
        del timeout_s, kwargs
        env_action = Action(
            message=action.message,
            decision=Decision(action.decision) if action.decision is not None else None,
        )
        result = self._env.step(env_action)
        self._sync_state(episode_id=self._state.episode_id)
        return self._convert_observation(
            result.observation,
            reward=result.reward,
            done=result.done,
            info=result.info,
        )

    @property
    def state(self) -> LeadQualStateModel:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="LeadQualEnv",
            description="Outbound real-estate lead qualification benchmark with deterministic grading.",
            version="2.2.0",
        )

    def _sync_state(self, episode_id: str | None) -> None:
        internal_state = self._env.state()
        self._state = LeadQualStateModel(
            episode_id=episode_id,
            step_count=internal_state.turn_number,
            task=internal_state.task.value,
            max_turns=internal_state.max_turns,
            done=internal_state.done,
            conversation_history=internal_state.conversation_history,
            known_signals={key.value: value for key, value in internal_state.known_signals.items()},
            probe_log=[(signal.value, quality.value) for signal, quality in internal_state.probe_log],
            lead_temperature=internal_state.lead_temperature,
            qualification_confidence=internal_state.qualification_confidence,
        )

    def _convert_observation(
        self,
        obs: Any,
        reward: float | None,
        done: bool,
        info: dict[str, Any],
    ) -> LeadQualObservationModel:
        return LeadQualObservationModel(
            conversation_history=obs.conversation_history,
            known_signals={key.value: value for key, value in obs.known_signals.items()},
            probe_log=[(signal.value, quality.value) for signal, quality in obs.probe_log],
            turn_number=obs.turn_number,
            max_turns=obs.max_turns,
            lead_temperature=obs.lead_temperature,
            qualification_confidence=obs.qualification_confidence,
            property_context=obs.property_context,
            reward=reward,
            reward_detail=LeadQualRewardModel(
                value=reward,
                components={
                    "task_score": float(info["task_score"]),
                } if reward is not None and "task_score" in info else {},
                description=str(info.get("termination_reason") or info.get("probe_quality") or ""),
            ) if reward is not None else None,
            done=done,
            info=info,
            metadata={"task": self._task.value},
        )
