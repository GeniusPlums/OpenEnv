from leadqualenv.environment.grader import ProbeResult, classify_probe, is_generic_opener
from leadqualenv.environment.models import ProbeQuality, SignalKey


def test_direct_probe_detects_timeline():
    result = classify_probe("When are you planning to move, specifically?", {})
    assert result == ProbeResult(quality=ProbeQuality.DIRECT, signal=SignalKey.TIMELINE)


def test_verified_probe_detects_budget():
    result = classify_probe(
        "Just to verify, you mentioned your budget earlier.",
        {SignalKey.BUDGET: "medium"},
    )
    assert result == ProbeResult(quality=ProbeQuality.VERIFIED, signal=SignalKey.BUDGET)


def test_irrelevant_probe_has_no_signal():
    result = classify_probe("Tell me more about yourself.", {})
    assert result == ProbeResult(quality=ProbeQuality.IRRELEVANT, signal=None)


def test_generic_opener_detection():
    assert is_generic_opener("Tell me about yourself")
    assert is_generic_opener("How are you")
    assert not is_generic_opener("What is your budget range?")


def test_repeat_non_verification_probe_gets_downgraded_when_signal_known():
    result = classify_probe(
        "What budget range are you looking at exactly?",
        {SignalKey.BUDGET: "medium"},
    )
    assert result == ProbeResult(quality=ProbeQuality.VAGUE, signal=SignalKey.BUDGET)


def test_verification_probe_requires_known_signal():
    result = classify_probe(
        "Just to verify, you mentioned your budget earlier.",
        {SignalKey.BUDGET: None},
    )
    # Signal not yet known, so verification markers fall through to direct
    assert result == ProbeResult(quality=ProbeQuality.DIRECT, signal=SignalKey.BUDGET)


def test_financial_keywords_detect_budget():
    result = classify_probe("What kind of financing are you looking at?", {})
    assert result.signal == SignalKey.BUDGET


def test_schedule_keywords_detect_timeline():
    result = classify_probe("How quickly are you looking to close?", {})
    assert result.signal == SignalKey.TIMELINE


def test_priority_order_budget_over_motivation():
    """'invest amount' should match budget, not motivation via 'invest'."""
    result = classify_probe("What invest amount are you considering for the down payment?", {})
    assert result.signal == SignalKey.BUDGET


def test_multiple_verified_markers():
    """Various verification phrases should all work."""
    markers = [
        "You mentioned your budget is high, want to make sure that's right.",
        "Earlier you said the budget was high, is that accurate?",
        "Circling back on the budget you stated earlier.",
    ]
    for msg in markers:
        result = classify_probe(msg, {SignalKey.BUDGET: "high"})
        assert result.quality == ProbeQuality.VERIFIED, f"Failed for: {msg}"
        assert result.signal == SignalKey.BUDGET
