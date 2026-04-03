from leadqualenv.environment.grader import ProbeResult, classify_probe, is_generic_opener
from leadqualenv.environment.models import ProbeQuality, SignalKey


def test_direct_probe_detects_timeline():
    result = classify_probe("When are you planning to move, specifically?", {})
    assert result == ProbeResult(ProbeQuality.DIRECT, SignalKey.TIMELINE)


def test_verified_probe_detects_budget():
    result = classify_probe("Just to verify, you mentioned your budget earlier.", {SignalKey.BUDGET: "medium"})
    assert result == ProbeResult(ProbeQuality.VERIFIED, SignalKey.BUDGET)


def test_irrelevant_probe_has_no_signal():
    result = classify_probe("Tell me more about yourself.", {})
    assert result == ProbeResult(ProbeQuality.IRRELEVANT, None)


def test_generic_opener_detection():
    assert is_generic_opener("Tell me about yourself")


def test_repeat_non_verification_probe_gets_downgraded_when_signal_known():
    result = classify_probe(
        "What budget range are you looking at exactly?",
        {SignalKey.BUDGET: "medium"},
    )
    assert result == ProbeResult(ProbeQuality.VAGUE, SignalKey.BUDGET)


def test_verification_probe_requires_known_signal():
    result = classify_probe(
        "Just to verify, you mentioned your budget earlier.",
        {SignalKey.BUDGET: None},
    )
    assert result == ProbeResult(ProbeQuality.DIRECT, SignalKey.BUDGET)
