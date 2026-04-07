from leadqualenv.environment import ProbeQuality, SignalKey, TaskLevel, grade_episode


def test_easy_grade_is_normalized():
    grade = grade_episode(
        task=TaskLevel.EASY,
        known_signals={
            SignalKey.BUDGET: "medium",
            SignalKey.TIMELINE: "immediate",
            SignalKey.DECISION_MAKER: True,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
        ],
        correct_decision=True,
    )
    assert 0.0 <= grade.score <= 1.0
    assert grade.components["efficiency"] == 1.0


def test_easy_grade_with_motivation_is_higher():
    without_motivation = grade_episode(
        task=TaskLevel.EASY,
        known_signals={
            SignalKey.BUDGET: "medium",
            SignalKey.TIMELINE: "immediate",
            SignalKey.DECISION_MAKER: True,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
        ],
        correct_decision=True,
    )
    with_motivation = grade_episode(
        task=TaskLevel.EASY,
        known_signals={
            SignalKey.BUDGET: "medium",
            SignalKey.TIMELINE: "immediate",
            SignalKey.DECISION_MAKER: True,
            SignalKey.MOTIVATION: "self_use",
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
            (SignalKey.MOTIVATION, ProbeQuality.DIRECT),
        ],
        correct_decision=True,
    )
    assert with_motivation.score >= without_motivation.score


def test_hard_grade_rewards_verification():
    low_verification = grade_episode(
        task=TaskLevel.HARD,
        known_signals={
            SignalKey.BUDGET: "high",
            SignalKey.TIMELINE: "immediate",
            SignalKey.DECISION_MAKER: False,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
        ],
        correct_decision=True,
    )
    high_verification = grade_episode(
        task=TaskLevel.HARD,
        known_signals={
            SignalKey.BUDGET: "low",
            SignalKey.TIMELINE: "6+ months",
            SignalKey.DECISION_MAKER: False,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.VERIFIED),
            (SignalKey.BUDGET, ProbeQuality.VERIFIED),
        ],
        correct_decision=True,
    )
    assert high_verification.score > low_verification.score
    assert high_verification.components["verification"] > low_verification.components["verification"]


def test_hard_verification_weight_is_significant():
    """Hard task should weight verification heavily — a major scoring component."""
    grade = grade_episode(
        task=TaskLevel.HARD,
        known_signals={
            SignalKey.BUDGET: "low",
            SignalKey.TIMELINE: "6+ months",
            SignalKey.DECISION_MAKER: False,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
        ],
        correct_decision=True,
    )
    # With no verification, score should be lower despite correct decision
    assert grade.components["verification"] == 0.0
    assert grade.score < 0.85


def test_hard_misleading_detection_credits_only_actual_surface_traps():
    grade = grade_episode(
        task=TaskLevel.HARD,
        known_signals={
            SignalKey.BUDGET: "high",
            SignalKey.TIMELINE: "3-6 months",
            SignalKey.DECISION_MAKER: True,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.VERIFIED),
        ],
        correct_decision=True,
        misleading_signals={SignalKey.TIMELINE},
    )
    assert grade.components["misleading_detection"] == 1.0


def test_medium_grade_penalizes_inefficient_extra_probes():
    efficient = grade_episode(
        task=TaskLevel.MEDIUM,
        known_signals={
            SignalKey.BUDGET: "high",
            SignalKey.TIMELINE: "3-6 months",
            SignalKey.DECISION_MAKER: True,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
        ],
        correct_decision=True,
    )
    inefficient = grade_episode(
        task=TaskLevel.MEDIUM,
        known_signals={
            SignalKey.BUDGET: "high",
            SignalKey.TIMELINE: "3-6 months",
            SignalKey.DECISION_MAKER: True,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.VAGUE),
            (SignalKey.BUDGET, ProbeQuality.VAGUE),
            (SignalKey.TIMELINE, ProbeQuality.VAGUE),
        ],
        correct_decision=True,
    )
    assert efficient.score > inefficient.score


def test_wrong_decision_scores_low():
    grade = grade_episode(
        task=TaskLevel.EASY,
        known_signals={
            SignalKey.BUDGET: "medium",
            SignalKey.TIMELINE: "immediate",
            SignalKey.DECISION_MAKER: True,
            SignalKey.MOTIVATION: None,
        },
        probe_log=[
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
        ],
        correct_decision=False,
    )
    assert grade.score < 0.55  # correct_decision component is 0.0


def test_all_tasks_produce_valid_scores():
    for task in TaskLevel:
        grade = grade_episode(
            task=task,
            known_signals={
                SignalKey.BUDGET: "medium",
                SignalKey.TIMELINE: "immediate",
                SignalKey.DECISION_MAKER: True,
                SignalKey.MOTIVATION: "self_use",
            },
            probe_log=[
                (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
                (SignalKey.TIMELINE, ProbeQuality.DIRECT),
                (SignalKey.BUDGET, ProbeQuality.DIRECT),
            ],
            correct_decision=True,
        )
        assert 0.0 <= grade.score <= 1.0, f"Score out of range for {task}"


def test_naive_agent_scores_low_on_hard():
    """A naive agent (no verification) should score < 0.5 on hard mode."""
    grade = grade_episode(
        task=TaskLevel.HARD,
        known_signals={
            SignalKey.BUDGET: "high",
            SignalKey.TIMELINE: "immediate",
            SignalKey.DECISION_MAKER: True,
            SignalKey.MOTIVATION: "self_use",
        },
        probe_log=[
            (SignalKey.BUDGET, ProbeQuality.DIRECT),
            (SignalKey.TIMELINE, ProbeQuality.DIRECT),
            (SignalKey.DECISION_MAKER, ProbeQuality.DIRECT),
            (SignalKey.MOTIVATION, ProbeQuality.DIRECT),
        ],
        correct_decision=True,  # Even with correct decision
        misleading_signals={SignalKey.BUDGET, SignalKey.TIMELINE},
    )
    assert grade.score < 0.5, f"Naive agent scored {grade.score} on hard — environment may be gameable"
