from leadqualenv.environment import Action, Decision, LeadQualEnv, SignalKey, TaskLevel


def test_easy_episode_can_end_correctly():
    env = LeadQualEnv(TaskLevel.EASY)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    env.step(Action(message="When are you planning to move, specifically?"))
    env.step(Action(message="What budget range are you looking at exactly?"))
    result = env.step(Action(decision=Decision.QUALIFIED))

    assert result.done is True
    assert result.info["correct_decision"] is True


def test_decision_without_required_signals_is_penalized():
    env = LeadQualEnv(TaskLevel.MEDIUM)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    result = env.step(Action(decision=Decision.QUALIFIED))

    assert result.done is True
    assert result.reward == -0.30
    assert result.info["error"] == "InsufficientSignalsError"


def test_hard_episode_requires_verification_to_overwrite_surface_signal():
    env = LeadQualEnv(TaskLevel.HARD)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    env.step(Action(message="When are you planning to move, specifically?"))
    env.step(Action(message="What budget range are you looking at exactly?"))

    assert env.known_signals[SignalKey.TIMELINE] == "immediate"
    assert env.known_signals[SignalKey.BUDGET] == "high"

    env.step(Action(message="To confirm, you said you could stretch. What budget level are you really targeting?"))
    env.step(Action(message="Just to verify, you mentioned moving quickly earlier. What is your actual timeline?"))

    assert env.known_signals[SignalKey.BUDGET] == "low"
    assert env.known_signals[SignalKey.TIMELINE] == "6+ months"
