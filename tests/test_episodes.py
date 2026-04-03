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

    assert env.known_signals[SignalKey.TIMELINE] == env.profile.surface_timeline
    assert env.known_signals[SignalKey.BUDGET] == env.profile.surface_budget

    env.step(Action(message="To confirm, you said you could stretch. What budget level are you really targeting?"))
    env.step(Action(message="Just to verify, you mentioned moving quickly earlier. What is your actual timeline?"))

    assert env.known_signals[SignalKey.BUDGET] == env.profile.budget
    assert env.known_signals[SignalKey.TIMELINE] == env.profile.timeline


def test_state_reports_current_progress():
    env = LeadQualEnv(TaskLevel.EASY)
    env.reset(seed=0)

    state = env.state()
    assert state.task == TaskLevel.EASY
    assert state.turn_number == 0
    assert state.done is False

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    state = env.state()
    assert state.turn_number == 1
    assert state.known_signals[SignalKey.DECISION_MAKER] is True


def test_hard_verification_probe_is_rewarded_without_repeat_penalty():
    env = LeadQualEnv(TaskLevel.HARD)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    env.step(Action(message="When are you planning to move, specifically?"))
    env.step(Action(message="What budget range are you looking at exactly?"))
    result = env.step(Action(message="Just to verify, you mentioned moving quickly earlier. What is your actual timeline?"))

    assert result.reward == 0.06


def test_terminal_decision_returns_task_score():
    env = LeadQualEnv(TaskLevel.EASY)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    env.step(Action(message="When are you planning to move, specifically?"))
    env.step(Action(message="What budget range are you looking at exactly?"))
    result = env.step(Action(decision=Decision.QUALIFIED))

    assert result.info["task_score"] == 0.98
    assert "task_score_components" in result.info
