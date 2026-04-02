from leadqualenv.environment.models import LeadProfile, ProbeQuality, SignalKey, TaskLevel
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
