
from leadqualenv.environment import Action, LeadQualEnv, SignalKey, TaskLevel
from leadqualenv.environment.models import LeadProfile, ProbeQuality
from leadqualenv.environment.profiles import TASK_PROFILES
from leadqualenv.environment.simulator import generate_response


def test_verification_evasion_fires_once_per_signal():
    env = LeadQualEnv(task=TaskLevel.HARD)

    # Create profile with known evasion
    env.reset()
    env.profile = LeadProfile(
        budget="low",
        timeline="6+ months",
        decision_maker=True,
        motivation="self_use",
        verification_evasion_signals=frozenset({SignalKey.BUDGET, SignalKey.TIMELINE})
    )

    # 1. Direct probe for budget
    env.step(Action(message="What is your exact budget?"))
    # Agent hasn't verified, just vaguely evaded if the probe parser classifies it as VERIFIED or DIRECT
    # Let's bypass parser and test generation directly
    response, val = generate_response(
        env.profile, SignalKey.BUDGET, ProbeQuality.VERIFIED, TaskLevel.HARD,
        verification_already_evaded=False
    )
    assert val is None, "Should evade and return None on first verified probe"

def test_verification_evasion_does_not_fire_when_not_configured():
    profile = LeadProfile(
        budget="low",
        timeline="6+ months",
        decision_maker=True,
        motivation="self_use",
        verification_evasion_signals=frozenset()
    )
    response, val = generate_response(
        profile, SignalKey.BUDGET, ProbeQuality.VERIFIED, TaskLevel.HARD,
        verification_already_evaded=False
    )
    assert val == "low", "Should not evade when verification_evasion_signals is empty"

def test_second_verification_returns_true_value():
    profile = LeadProfile(
        budget="low",
        timeline="6+ months",
        decision_maker=True,
        motivation="self_use",
        verification_evasion_signals=frozenset({SignalKey.BUDGET})
    )
    response, val = generate_response(
        profile, SignalKey.BUDGET, ProbeQuality.VERIFIED, TaskLevel.HARD,
        verification_already_evaded=True
    )
    assert val == "low", "Should bypass evasion on second verified attempt"

def test_requalification_observation_contains_previous_crm():
    env = LeadQualEnv(task=TaskLevel.REQUALIFICATION)
    obs = env.reset(seed=42)
    assert obs.previous_crm is not None, "Requalification should initialize previous_crm"
    assert "known_at_time" in obs.previous_crm
    assert "status" in obs.previous_crm

def test_requalification_grader_rewards_motivation_shift():
    from leadqualenv.environment.task_graders import grade_episode
    profile = LeadProfile(
        budget="medium",
        timeline="immediate",
        decision_maker=True,
        motivation="self_use",
        motivation_shift=True
    )

    # Positive case: motivation was extracted
    grade1 = grade_episode(
        task=TaskLevel.REQUALIFICATION,
        known_signals={SignalKey.MOTIVATION: "self_use"},
        probe_log=[(SignalKey.MOTIVATION, ProbeQuality.DIRECT)],
        correct_decision=True,
        profile=profile,
    )
    assert grade1.components["motivation_shift"] > 0, "Motivation shift should be rewarded if detected"

    # Negative case: motivation not extracted
    grade2 = grade_episode(
        task=TaskLevel.REQUALIFICATION,
        known_signals={},
        probe_log=[],
        correct_decision=True,
        profile=profile,
    )
    assert grade2.components["motivation_shift"] == 0.0, "Motivation shift should NOT be rewarded if not detected"

def test_hard_score_drops_with_evasion_enabled():
    from leadqualenv.environment.task_graders import grade_episode
    profile = TASK_PROFILES[TaskLevel.HARD][0]
    grade = grade_episode(
        task=TaskLevel.HARD,
        known_signals={SignalKey.BUDGET: "low", SignalKey.TIMELINE: "6+ months", SignalKey.DECISION_MAKER: False},
        probe_log=[(SignalKey.BUDGET, ProbeQuality.DIRECT)],
        correct_decision=True,
        profile=profile,
        misleading_signals=set(profile.surface_signals)
    )
    # Testing that it doesn't just error out, specific scores shouldn't hit 1.0 without verification
    assert grade.score < 1.0, "Hard mode score should not be perfect without full verified probing"
