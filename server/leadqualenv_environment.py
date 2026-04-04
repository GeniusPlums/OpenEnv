from __future__ import annotations

from leadqualenv.environment import Action, LeadQualEnv, TaskLevel
from server.models import (
    LeadQualActionModel,
    LeadQualObservationModel,
    LeadQualRewardModel,
    LeadQualStateModel,
)


class LeadQualOpenEnv:
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._env: LeadQualEnv | None = None
        self._step_count = 0

    def reset(
        self,
        seed: int | None = None,
        task: str = "easy",
        **kwargs: object,
    ) -> LeadQualObservationModel:
        task_level = TaskLevel(task)
        self._env = LeadQualEnv(task=task_level, max_turns=10)
        
        obs = self._env.reset(seed=seed)
        self._step_count = 0

        return LeadQualObservationModel(
            conversation_history=obs.conversation_history,
            known_signals={k.value: v for k, v in obs.known_signals.items()},
            probe_log=[(sig.value, qual.value) for sig, qual in obs.probe_log],
            turn_number=obs.turn_number,
            max_turns=obs.max_turns,
            lead_temperature=obs.lead_temperature,
            qualification_confidence=obs.qualification_confidence,
            property_context=obs.property_context,
            done=False,
            reward=None,
            reward_detail=None,
            metadata={"task": task_level.value},
            info={},
        )

    def step(self, action: LeadQualActionModel, **kwargs: object) -> LeadQualObservationModel:
        if self._env is None:
            raise RuntimeError("Environment must be reset before stepping.")

        from leadqualenv.environment import Decision
        env_action = Action(
            message=action.message,
            decision=Decision(action.decision) if action.decision else None,
        )

        result = self._env.step(env_action)
        self._step_count += 1
        
        # Populate reward detail
        reward_detail = LeadQualRewardModel(
            value=result.reward,
            components={},
            description="Reward for current step based on probe quality and engagement",
        )

        return LeadQualObservationModel(
            conversation_history=result.observation.conversation_history,
            known_signals={k.value: v for k, v in result.observation.known_signals.items()},
            probe_log=[(sig.value, qual.value) for sig, qual in result.observation.probe_log],
            turn_number=result.observation.turn_number,
            max_turns=result.observation.max_turns,
            lead_temperature=result.observation.lead_temperature,
            qualification_confidence=result.observation.qualification_confidence,
            property_context=result.observation.property_context,
            done=result.done,
            reward=result.reward,
            reward_detail=reward_detail,
            metadata={"task": self._env.task.value},
            info=result.info,
        )

    @property
    def state(self) -> LeadQualStateModel:
        if self._env is None:
            return LeadQualStateModel()

        return LeadQualStateModel(
            task=self._env.task.value,
            max_turns=self._env.max_turns,
            done=self._env.done,
            conversation_history=self._env.conversation_history,
            known_signals={k.value: v for k, v in self._env.known_signals.items()},
            probe_log=[(sig.value, qual.value) for sig, qual in self._env.probe_log],
            lead_temperature=self._env._lead_temperature,
            qualification_confidence=self._env._qual_confidence,
            step_count=self._step_count,
        )
