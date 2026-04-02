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
    assert grade.score == 0.96


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
