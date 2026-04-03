from __future__ import annotations

import random

from .models import LeadProfile, Personality, SignalKey, TaskLevel

TASK_PROFILES: dict[TaskLevel, list[LeadProfile]] = {
    TaskLevel.EASY: [
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="apartment", location="downtown",
        ),
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=True,
            motivation="investment", personality=Personality.FRIENDLY,
            property_type="villa", location="suburbs",
        ),
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=True,
            motivation="investment", personality=Personality.VERBOSE,
            property_type="condo", location="waterfront",
        ),
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.TERSE,
            property_type="townhouse", location="midtown",
        ),
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="apartment", location="east side",
        ),
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=True,
            motivation="investment", personality=Personality.FRIENDLY,
            property_type="penthouse", location="financial district",
        ),
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.VERBOSE,
            property_type="studio", location="university area",
        ),
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=True,
            motivation="investment", personality=Personality.DIRECT,
            property_type="duplex", location="north end",
        ),
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.TERSE,
            property_type="apartment", location="lakeside",
        ),
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.FRIENDLY,
            property_type="bungalow", location="hillside",
        ),
    ],
    TaskLevel.MEDIUM: [
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="investment", personality=Personality.DIRECT,
            property_type="apartment", location="downtown",
        ),
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=True,
            motivation="self_use", personality=Personality.EVASIVE,
            property_type="townhouse", location="suburbs",
        ),
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="self_use", personality=Personality.VERBOSE,
            property_type="villa", location="countryside",
        ),
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=True,
            motivation="investment", personality=Personality.TERSE,
            property_type="condo", location="waterfront",
        ),
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="exploring", personality=Personality.FRIENDLY,
            property_type="apartment", location="tech park",
        ),
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=True,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="studio", location="arts district",
        ),
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="investment", personality=Personality.EVASIVE,
            property_type="duplex", location="east side",
        ),
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=True,
            motivation="self_use", personality=Personality.VERBOSE,
            property_type="apartment", location="old town",
        ),
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="exploring", personality=Personality.TERSE,
            property_type="penthouse", location="city center",
        ),
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=True,
            motivation="self_use", personality=Personality.FRIENDLY,
            property_type="townhouse", location="garden district",
        ),
    ],
    TaskLevel.HARD: [
        # 1. Surface: high/immediate → True: low/6+, not DM → unqualified
        LeadProfile(
            budget="low", timeline="6+ months", decision_maker=False,
            motivation="exploring", personality=Personality.EVASIVE,
            property_type="apartment", location="downtown",
            surface_budget="high", surface_timeline="immediate",
        ),
        # 2. Surface: high/immediate → True: medium/3-6, DM → nurture
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=True,
            motivation="investment", personality=Personality.FRIENDLY,
            property_type="villa", location="suburbs",
            surface_budget="high", surface_timeline="immediate",
        ),
        # 3. Surface: medium/immediate → True: low/3-6, not DM → unqualified
        LeadProfile(
            budget="low", timeline="3-6 months", decision_maker=False,
            motivation="self_use", personality=Personality.VERBOSE,
            property_type="condo", location="waterfront",
            surface_budget="medium", surface_timeline="immediate",
        ),
        # 4. Surface: high/immediate → True: high/3-6, DM → nurture (budget is real, timeline is fake)
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="townhouse", location="midtown",
            surface_timeline="immediate",
        ),
        # 5. Surface: high/immediate → True: low/immediate, DM → unqualified (timeline real, budget fake)
        LeadProfile(
            budget="low", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.TERSE,
            property_type="apartment", location="east side",
            surface_budget="high",
        ),
        # 6. Surface: high/immediate → True: high/immediate, DM → qualified (surface matches reality!)
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=True,
            motivation="investment", personality=Personality.FRIENDLY,
            property_type="penthouse", location="financial district",
            surface_budget="high", surface_timeline="immediate",
        ),
        # 7. Surface: medium/immediate → True: medium/6+, DM → unqualified (budget real, timeline fake)
        LeadProfile(
            budget="medium", timeline="6+ months", decision_maker=True,
            motivation="exploring", personality=Personality.EVASIVE,
            property_type="apartment", location="university area",
            surface_timeline="immediate",
        ),
        # 8. Surface: high/immediate → True: medium/3-6, not DM → unqualified
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=False,
            motivation="investment", personality=Personality.VERBOSE,
            property_type="duplex", location="north end",
            surface_budget="high", surface_timeline="immediate",
            competitor_mention=True,
        ),
        # 9. Surface: high → True: high, but timeline=3-6, DM, objection on budget → nurture
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="investment", personality=Personality.EVASIVE,
            property_type="villa", location="hillside",
            surface_timeline="immediate",
            objection_on=SignalKey.BUDGET,
        ),
        # 10. Surface: medium/immediate → True: medium/immediate, not DM → unqualified
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=False,
            motivation="self_use", personality=Personality.TERSE,
            property_type="apartment", location="lakeside",
            surface_budget="medium", surface_timeline="immediate",
        ),
        # 11. Surface: high/immediate → True: low/6+, DM → unqualified
        LeadProfile(
            budget="low", timeline="6+ months", decision_maker=True,
            motivation="exploring", personality=Personality.FRIENDLY,
            property_type="studio", location="arts district",
            surface_budget="high", surface_timeline="immediate",
            competitor_mention=True,
        ),
        # 12. Surface: high/immediate → True: medium/immediate, DM → qualified (partially misleading)
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="townhouse", location="garden district",
            surface_budget="high",
        ),
    ],
}

# Pre-computed correct decisions per profile for deterministic verification
_OPENER_TEMPLATES: dict[str, list[str]] = {
    "apartment": [
        "Hi, I saw your apartment listing and wanted to learn more.",
        "Hello, I'm interested in the apartment you have listed.",
        "Hey there, I came across your apartment listing online.",
    ],
    "villa": [
        "Hi, I noticed your villa listing and it caught my eye.",
        "Hello, I'm looking at villas and yours stood out.",
    ],
    "condo": [
        "Hi, I saw your condo listing and wanted to get some details.",
        "Hello, the condo you have listed looks interesting.",
    ],
    "townhouse": [
        "Hi, I'm interested in the townhouse you have on the market.",
        "Hello, I saw your townhouse listing and wanted to chat.",
    ],
    "penthouse": [
        "Hi, I noticed your penthouse listing. Can you tell me more?",
    ],
    "studio": [
        "Hi, I saw your studio listing and wanted to learn more about it.",
    ],
    "duplex": [
        "Hi, I came across your duplex listing. Is it still available?",
    ],
    "bungalow": [
        "Hello, I saw your bungalow listing and I'm quite interested.",
    ],
}

DEFAULT_OPENER = "Hi, I saw your property listing and wanted to learn more."


def sample_profile(task: TaskLevel, seed: int | None = None) -> LeadProfile:
    rng = random.Random(seed)
    return rng.choice(TASK_PROFILES[task])


def sample_opener(profile: LeadProfile, seed: int | None = None) -> str:
    rng = random.Random(seed)
    templates = _OPENER_TEMPLATES.get(profile.property_type, [DEFAULT_OPENER])
    return rng.choice(templates)
