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
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=False,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="apartment", location="riverfront",
        ),
        LeadProfile(
            budget="low", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.FRIENDLY,
            property_type="studio", location="market district",
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
            verification_evasion_signals=frozenset({SignalKey.BUDGET}),
        ),
        # 2. Surface: high/immediate → True: medium/3-6, DM → nurture
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=True,
            motivation="investment", personality=Personality.FRIENDLY,
            property_type="villa", location="suburbs",
            surface_budget="high", surface_timeline="immediate",
            verification_evasion_signals=frozenset({SignalKey.TIMELINE}),
        ),
        # 3. Surface: medium/immediate → True: low/3-6, not DM → unqualified
        LeadProfile(
            budget="low", timeline="3-6 months", decision_maker=False,
            motivation="self_use", personality=Personality.VERBOSE,
            property_type="condo", location="waterfront",
            surface_budget="medium", surface_timeline="immediate",
            verification_evasion_signals=frozenset({SignalKey.BUDGET, SignalKey.TIMELINE}),
        ),
        # 4. Surface: high/immediate → True: high/3-6, DM → nurture (budget is real, timeline is fake)
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="townhouse", location="midtown",
            surface_timeline="immediate",
            verification_evasion_signals=frozenset({SignalKey.TIMELINE}),
        ),
        # 5. Surface: high/immediate → True: low/immediate, DM → unqualified (timeline real, budget fake)
        LeadProfile(
            budget="low", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.TERSE,
            property_type="apartment", location="east side",
            surface_budget="high",
            verification_evasion_signals=frozenset({SignalKey.BUDGET}),
        ),
        # 6. Surface: medium/immediate -> True: high/immediate, DM -> qualified (budget is misleading)
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=True,
            motivation="investment", personality=Personality.FRIENDLY,
            property_type="penthouse", location="financial district",
            surface_budget="medium",
            verification_evasion_signals=frozenset({SignalKey.BUDGET}),
        ),
        # 7. Surface: medium/immediate → True: medium/6+, DM → unqualified (budget real, timeline fake)
        LeadProfile(
            budget="medium", timeline="6+ months", decision_maker=True,
            motivation="exploring", personality=Personality.EVASIVE,
            property_type="apartment", location="university area",
            surface_timeline="immediate",
            verification_evasion_signals=frozenset({SignalKey.TIMELINE}),
        ),
        # 8. Surface: high/immediate → True: medium/3-6, not DM → unqualified
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=False,
            motivation="investment", personality=Personality.VERBOSE,
            property_type="duplex", location="north end",
            surface_budget="high", surface_timeline="immediate",
            competitor_mention=True,
            verification_evasion_signals=frozenset({SignalKey.BUDGET, SignalKey.TIMELINE}),
        ),
        # 9. Surface: high → True: high, but timeline=3-6, DM, objection on budget → nurture
        LeadProfile(
            budget="high", timeline="3-6 months", decision_maker=True,
            motivation="investment", personality=Personality.EVASIVE,
            property_type="villa", location="hillside",
            surface_timeline="immediate",
            objection_on=SignalKey.BUDGET,
            verification_evasion_signals=frozenset({SignalKey.TIMELINE}),
        ),
        # 10. Surface: high/3-6 months -> True: medium/immediate, not DM -> unqualified
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=False,
            motivation="self_use", personality=Personality.TERSE,
            property_type="apartment", location="lakeside",
            surface_budget="high", surface_timeline="3-6 months",
            verification_evasion_signals=frozenset({SignalKey.BUDGET, SignalKey.TIMELINE}),
        ),
        # 11. Surface: high/immediate → True: low/6+, DM → unqualified
        LeadProfile(
            budget="low", timeline="6+ months", decision_maker=True,
            motivation="exploring", personality=Personality.FRIENDLY,
            property_type="studio", location="arts district",
            surface_budget="high", surface_timeline="immediate",
            competitor_mention=True,
            verification_evasion_signals=frozenset({SignalKey.BUDGET, SignalKey.TIMELINE}),
        ),
        # 12. Surface: high/immediate → True: medium/immediate, DM → qualified (partially misleading)
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="townhouse", location="garden district",
            surface_budget="high",
            verification_evasion_signals=frozenset({SignalKey.BUDGET}),
        ),
    ],
    TaskLevel.REQUALIFICATION: [
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=True,
            motivation="self_use", personality=Personality.DIRECT,
            property_type="townhouse", location="downtown",
            previous_qualification="nurture", motivation_shift=True,
            previous_crm={"status": "nurture", "last_contact": "3 months ago", "known_at_time": {"budget": "high", "timeline": "3-6 months", "decision_maker": True, "motivation": "investment"}},
        ),
        LeadProfile(
            budget="medium", timeline="6+ months", decision_maker=True,
            motivation="investment", personality=Personality.EVASIVE,
            property_type="apartment", location="suburbs",
            previous_qualification="nurture", motivation_shift=False,
            previous_crm={"status": "nurture", "last_contact": "3 months ago", "known_at_time": {"budget": "medium", "timeline": "3-6 months", "decision_maker": True, "motivation": "investment"}},
        ),
        LeadProfile(
            budget="high", timeline="immediate", decision_maker=False,
            motivation="self_use", personality=Personality.VERBOSE,
            property_type="villa", location="waterfront",
            previous_qualification="nurture", motivation_shift=False,
            previous_crm={"status": "nurture", "last_contact": "3 months ago", "known_at_time": {"budget": "high", "timeline": "3-6 months", "decision_maker": True, "motivation": "self_use"}},
        ),
        LeadProfile(
            budget="medium", timeline="immediate", decision_maker=True,
            motivation="investment", personality=Personality.FRIENDLY,
            property_type="condo", location="city center",
            previous_qualification="nurture", motivation_shift=True,
            previous_crm={"status": "nurture", "last_contact": "3 months ago", "known_at_time": {"budget": "medium", "timeline": "3-6 months", "decision_maker": True, "motivation": "exploring"}},
        ),
        LeadProfile(
            budget="medium", timeline="3-6 months", decision_maker=True,
            motivation="exploring", personality=Personality.TERSE,
            property_type="studio", location="arts district",
            previous_qualification="nurture", motivation_shift=False,
            previous_crm={"status": "nurture", "last_contact": "3 months ago", "known_at_time": {"budget": "medium", "timeline": "3-6 months", "decision_maker": True, "motivation": "exploring"}},
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


_PROPERTY_TYPES = tuple(_OPENER_TEMPLATES)
_LOCATIONS = (
    "downtown",
    "midtown",
    "suburbs",
    "waterfront",
    "financial district",
    "city center",
    "garden district",
    "arts district",
    "old town",
    "university area",
    "hillside",
    "lakeside",
    "tech park",
    "market district",
    "riverfront",
    "north end",
)


def _generate_profile(task: TaskLevel, index: int) -> LeadProfile:
    rng = random.Random(f"{task.value}:{index}")
    personality = rng.choice(list(Personality))
    property_type = rng.choice(_PROPERTY_TYPES)
    location = rng.choice(_LOCATIONS)

    if task == TaskLevel.EASY:
        decision_maker = rng.random() >= 0.15
        budget = "low" if rng.random() < 0.15 else rng.choice(["medium", "high"])
        timeline = "immediate"
        motivation = rng.choice(["self_use", "investment"])
        return LeadProfile(
            budget=budget,
            timeline=timeline,
            decision_maker=decision_maker,
            motivation=motivation,
            personality=personality,
            property_type=property_type,
            location=location,
        )

    if task == TaskLevel.MEDIUM:
        return LeadProfile(
            budget=rng.choice(["medium", "high"]),
            timeline="3-6 months",
            decision_maker=True,
            motivation=rng.choice(["self_use", "investment", "exploring"]),
            personality=personality,
            property_type=property_type,
            location=location,
        )

    if task == TaskLevel.REQUALIFICATION:
        motivation_shift = rng.random() > 0.5
        timeline_choice = rng.choice(["immediate", "3-6 months", "6+ months"])
        motivation_val = rng.choice(["self_use", "investment", "exploring"])
        prev_mot = "investment" if (motivation_shift and motivation_val == "self_use") else ("self_use" if motivation_shift else motivation_val)
        budget = rng.choice(["medium", "high"])
        return LeadProfile(
            budget=budget,
            timeline=timeline_choice,
            decision_maker=rng.random() > 0.2,
            motivation=motivation_val,
            personality=personality,
            property_type=property_type,
            location=location,
            previous_qualification="nurture",
            motivation_shift=motivation_shift,
            previous_crm={
                "status": "nurture",
                "last_contact": "3 months ago",
                "known_at_time": {
                    "budget": budget,
                    "timeline": "3-6 months",
                    "decision_maker": True,
                    "motivation": prev_mot,
                }
            }
        )

    decision_maker = rng.random() >= 0.3
    budget = rng.choice(["low", "medium", "high"])
    timeline = rng.choice(["immediate", "3-6 months", "6+ months"])
    motivation = rng.choice(["self_use", "investment", "exploring"])

    surface_budget = budget if rng.random() < 0.15 else None
    surface_timeline = timeline if rng.random() < 0.15 else None
    if surface_budget is None and rng.random() < 0.6:
        surface_budget = rng.choice([candidate for candidate in ("low", "medium", "high") if candidate != budget])
    if surface_timeline is None and rng.random() < 0.6:
        surface_timeline = rng.choice([candidate for candidate in ("immediate", "3-6 months", "6+ months") if candidate != timeline])

    return LeadProfile(
        budget=budget,
        timeline=timeline,
        decision_maker=decision_maker,
        motivation=motivation,
        personality=personality,
        property_type=property_type,
        location=location,
        surface_budget=surface_budget,
        surface_timeline=surface_timeline,
        competitor_mention=rng.random() < 0.35,
        objection_on=rng.choice([None, SignalKey.BUDGET, SignalKey.TIMELINE, SignalKey.DECISION_MAKER, SignalKey.MOTIVATION]),
    )


def build_profile_pool(task: TaskLevel, generated_count: int = 0) -> list[LeadProfile]:
    pool = list(TASK_PROFILES[task])
    for index in range(max(0, generated_count)):
        pool.append(_generate_profile(task, index))
    return pool


def sample_profile(
    task: TaskLevel,
    seed: int | None = None,
    *,
    generated_count: int = 0,
) -> LeadProfile:
    rng = random.Random(seed)
    return rng.choice(build_profile_pool(task, generated_count=generated_count))


def sample_opener(profile: LeadProfile, seed: int | None = None) -> str:
    rng = random.Random(seed)
    if profile.previous_qualification is not None:
        return f"Hi, we spoke a few months ago about the {profile.property_type}."
    templates = _OPENER_TEMPLATES.get(profile.property_type, [DEFAULT_OPENER])
    return rng.choice(templates)
