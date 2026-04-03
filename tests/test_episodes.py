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

    # In hard mode, direct probes may return surface (misleading) values
    profile = env.profile
    if profile.surface_timeline is not None:
        assert env.known_signals[SignalKey.TIMELINE] == profile.surface_timeline
    if profile.surface_budget is not None:
        assert env.known_signals[SignalKey.BUDGET] == profile.surface_budget

    env.step(Action(message="To confirm, you said you could stretch. What budget level are you really targeting?"))
    env.step(Action(message="Just to verify, you mentioned moving quickly earlier. What is your actual timeline?"))

    assert env.known_signals[SignalKey.BUDGET] == profile.budget
    assert env.known_signals[SignalKey.TIMELINE] == profile.timeline


def test_state_reports_current_progress():
    env = LeadQualEnv(TaskLevel.EASY)
    env.reset(seed=0)

    state = env.state()
    assert state.task == TaskLevel.EASY
    assert state.turn_number == 0
    assert state.done is False
    assert state.lead_temperature == 1.0

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    state = env.state()
    assert state.turn_number == 1
    assert state.known_signals[SignalKey.DECISION_MAKER] is not None
    assert state.lead_temperature < 1.0


def test_hard_verification_probe_is_rewarded_without_repeat_penalty():
    env = LeadQualEnv(TaskLevel.HARD)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    env.step(Action(message="When are you planning to move, specifically?"))
    env.step(Action(message="What budget range are you looking at exactly?"))
    result = env.step(Action(message="Just to verify, you mentioned moving quickly earlier. What is your actual timeline?"))

    # Verification probes should get positive reward (0.06) without repeat penalty
    assert result.reward > 0


def test_terminal_decision_returns_task_score():
    env = LeadQualEnv(TaskLevel.EASY)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    env.step(Action(message="When are you planning to move, specifically?"))
    env.step(Action(message="What budget range are you looking at exactly?"))
    result = env.step(Action(decision=Decision.QUALIFIED))

    assert 0.0 <= result.info["task_score"] <= 1.0
    assert "task_score_components" in result.info


def test_lead_temperature_decays_over_turns():
    env = LeadQualEnv(TaskLevel.EASY)
    env.reset(seed=0)

    initial_temp = env._lead_temperature
    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    after_one = env._lead_temperature
    env.step(Action(message="When are you planning to move, specifically?"))
    after_two = env._lead_temperature

    assert initial_temp > after_one > after_two


def test_qualification_confidence_increases_with_probes():
    env = LeadQualEnv(TaskLevel.EASY)
    env.reset(seed=0)

    assert env._qual_confidence == 0.0

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    after_one = env._qual_confidence
    env.step(Action(message="When are you planning to move, specifically?"))
    after_two = env._qual_confidence

    assert after_one > 0.0
    assert after_two > after_one


def test_property_context_in_observation():
    env = LeadQualEnv(TaskLevel.EASY)
    obs = env.reset(seed=0)

    assert obs.property_context is not None
    assert len(obs.property_context) > 0


def test_motivation_discovery_gives_bonus():
    env = LeadQualEnv(TaskLevel.EASY)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    env.step(Action(message="When are you planning to move, specifically?"))
    env.step(Action(message="What budget range are you looking at exactly?"))
    result = env.step(Action(message="What is the main purpose for this purchase — own use or investment?"))

    # Motivation discovery should include a small bonus
    assert result.reward > 0


def test_max_turns_triggers_no_decision_penalty():
    env = LeadQualEnv(TaskLevel.EASY, max_turns=2)
    env.reset(seed=0)

    env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
    result = env.step(Action(message="When are you planning to move, specifically?"))

    assert result.done is True
    assert result.reward < 0  # NO_DECISION_PENALTY
    assert "task_score" in result.info
    assert result.info["termination_reason"] == "max_turns_reached"


def test_objection_does_not_pollute_probe_log_and_allows_retry():
    env = LeadQualEnv(TaskLevel.HARD)
    env.reset(seed=17)

    result = env.step(Action(message="What budget range are you looking at exactly?"))

    assert env.probe_log == []
    assert env.known_signals[SignalKey.BUDGET] is None
    assert "rather not discuss exact numbers" in result.observation.conversation_history[-1]["content"]

    env.step(Action(message="To confirm, what budget level are you really targeting?"))
    assert env.known_signals[SignalKey.BUDGET] == env.profile.budget


def test_hard_profiles_have_varied_correct_decisions():
    """Hard profiles should not all have the same correct answer."""
    from leadqualenv.environment.grader import classify_lead
    from leadqualenv.environment.profiles import TASK_PROFILES

    decisions = set()
    for profile in TASK_PROFILES[TaskLevel.HARD]:
        decisions.add(classify_lead(profile))

    assert len(decisions) >= 2, "Hard profiles should have at least 2 different correct decisions"


def test_episode_on_all_difficulty_levels():
    """Smoke test: run a full episode on each difficulty level."""
    for task_level in TaskLevel:
        env = LeadQualEnv(task_level)
        env.reset(seed=42)

        env.step(Action(message="Are you the person who can make the purchase decision yourself?"))
        env.step(Action(message="When are you planning to move, specifically?"))
        env.step(Action(message="What budget range are you looking at exactly?"))

        if task_level == TaskLevel.HARD:
            env.step(Action(message="To confirm, you said you could stretch. What budget level are you really targeting?"))
            env.step(Action(message="Just to verify, you mentioned moving quickly earlier. What is your actual timeline?"))

        # Make a decision based on known signals
        from leadqualenv.environment.grader import classify_lead
        decision = classify_lead(env.profile)
        result = env.step(Action(decision=decision))

        assert result.done is True
        assert result.info["correct_decision"] is True
        assert 0.0 <= result.info["task_score"] <= 1.0
