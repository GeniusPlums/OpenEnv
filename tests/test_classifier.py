from leadqualenv.environment.grader import classify_lead
from leadqualenv.environment.models import Decision, LeadProfile


def test_classify_immediate_medium_budget_is_qualified():
    profile = LeadProfile("medium", "immediate", True, "self_use")
    assert classify_lead(profile) == Decision.QUALIFIED


def test_classify_three_to_six_months_is_nurture():
    profile = LeadProfile("high", "3-6 months", True, "investment")
    assert classify_lead(profile) == Decision.NURTURE


def test_non_decision_maker_is_unqualified():
    profile = LeadProfile("high", "immediate", False, "self_use")
    assert classify_lead(profile) == Decision.UNQUALIFIED


def test_low_budget_decision_maker_is_unqualified():
    profile = LeadProfile("low", "immediate", True, "self_use")
    assert classify_lead(profile) == Decision.UNQUALIFIED


def test_high_budget_immediate_dm_is_qualified():
    profile = LeadProfile("high", "immediate", True, "investment")
    assert classify_lead(profile) == Decision.QUALIFIED


def test_six_plus_months_high_budget_dm_is_nurture():
    profile = LeadProfile("high", "6+ months", True, "exploring")
    assert classify_lead(profile) == Decision.NURTURE


def test_low_budget_non_dm_is_unqualified():
    profile = LeadProfile("low", "6+ months", False, "exploring")
    assert classify_lead(profile) == Decision.UNQUALIFIED
