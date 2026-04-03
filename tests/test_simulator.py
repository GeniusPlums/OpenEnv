from leadqualenv.environment.models import LeadProfile, Personality, ProbeQuality, SignalKey, TaskLevel
from leadqualenv.environment.simulator import generate_response, resolve_signal


def test_hard_mode_direct_probe_returns_surface_signal():
    profile = LeadProfile(
        budget="low",
        timeline="6+ months",
        decision_maker=False,
        motivation="exploring",
        surface_budget="high",
        surface_timeline="immediate",
    )
    value = resolve_signal(profile, SignalKey.BUDGET, ProbeQuality.DIRECT, TaskLevel.HARD)
    assert value == "high"


def test_hard_mode_verified_probe_returns_true_signal():
    profile = LeadProfile(
        budget="low",
        timeline="6+ months",
        decision_maker=False,
        motivation="exploring",
        surface_budget="high",
        surface_timeline="immediate",
    )
    value = resolve_signal(profile, SignalKey.BUDGET, ProbeQuality.VERIFIED, TaskLevel.HARD)
    assert value == "low"


def test_irrelevant_message_returns_filler():
    profile = LeadProfile("medium", "immediate", True, "self_use")
    response, value = generate_response(profile, None, ProbeQuality.IRRELEVANT, TaskLevel.EASY)
    assert "understand" in response.lower()
    assert value is None


def test_personality_affects_response():
    """Different personalities should produce different response text."""
    direct_profile = LeadProfile("medium", "immediate", True, "self_use", personality=Personality.DIRECT)
    verbose_profile = LeadProfile("medium", "immediate", True, "self_use", personality=Personality.VERBOSE)

    response_direct, _ = generate_response(direct_profile, SignalKey.BUDGET, ProbeQuality.DIRECT, TaskLevel.EASY)
    response_verbose, _ = generate_response(verbose_profile, SignalKey.BUDGET, ProbeQuality.DIRECT, TaskLevel.EASY)

    assert response_direct != response_verbose
    assert len(response_verbose) > len(response_direct)


def test_vague_response_varies_by_personality():
    direct_profile = LeadProfile("medium", "immediate", True, "self_use", personality=Personality.DIRECT)
    evasive_profile = LeadProfile("medium", "immediate", True, "self_use", personality=Personality.EVASIVE)

    value_direct = resolve_signal(direct_profile, SignalKey.BUDGET, ProbeQuality.VAGUE, TaskLevel.EASY)
    value_evasive = resolve_signal(evasive_profile, SignalKey.BUDGET, ProbeQuality.VAGUE, TaskLevel.EASY)

    assert isinstance(value_direct, str)
    assert isinstance(value_evasive, str)
    assert value_direct != value_evasive


def test_objection_mechanic_blocks_first_probe():
    profile = LeadProfile(
        "high", "3-6 months", True, "investment",
        objection_on=SignalKey.BUDGET,
    )
    response, value = generate_response(profile, SignalKey.BUDGET, ProbeQuality.DIRECT, TaskLevel.HARD)
    assert value is None  # Objection blocks the signal
    assert "rather not" in response.lower() or "matter" in response.lower()


def test_easy_mode_direct_probe_returns_true_signal():
    """Easy mode should always return true signals, no surface trickery."""
    profile = LeadProfile("medium", "immediate", True, "self_use")
    value = resolve_signal(profile, SignalKey.BUDGET, ProbeQuality.DIRECT, TaskLevel.EASY)
    assert value == "medium"
