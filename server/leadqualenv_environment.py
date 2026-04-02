from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

from leadqualenv.environment import Action, Decision, LeadQualEnv, TaskLevel

from .models import LeadQualActionModel, LeadQualObservationModel


TASK_NAME_MAP = {
    "easy": TaskLevel.EASY,
    "medium": TaskLevel.MEDIUM,
    "hard": TaskLevel.HARD,
}


class LeadQualOpenEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        self._task = TaskLevel.EASY
        self._env = LeadQualEnv(task=self._task)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> LeadQualObservationModel:
        task_name = kwargs.get("task", "easy")
        self._task = TASK_NAME_MAP.get(task_name, TaskLevel.EASY)
        self._env = LeadQualEnv(task=self._task)
        obs = self._env.reset(seed=seed)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return self._convert_observation(obs, reward=None, done=False, info={"task": self._task.value})

    def step(
        self,
        action: LeadQualActionModel,
        timeout_s: float | None = None,
        **kwargs,
    ) -> LeadQualObservationModel:
        del timeout_s, kwargs
        env_action = Action(
            message=action.message,
            decision=Decision(action.decision) if action.decision is not None else None,
        )
        result = self._env.step(env_action)
        self._state.step_count = self._env.turn_number
        return self._convert_observation(
            result.observation,
            reward=result.reward,
            done=result.done,
            info=result.info,
        )

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="LeadQualEnv",
            description="Outbound real-estate lead qualification benchmark with deterministic grading.",
            version="2.0.0",
        )

    def _convert_observation(self, obs, reward, done: bool, info: dict) -> LeadQualObservationModel:
        return LeadQualObservationModel(
            conversation_history=obs.conversation_history,
            known_signals={key.value: value for key, value in obs.known_signals.items()},
            probe_log=[(signal.value, quality.value) for signal, quality in obs.probe_log],
            turn_number=obs.turn_number,
            max_turns=obs.max_turns,
            reward=reward,
            done=done,
            info=info,
            metadata={"task": self._task.value},
        )
