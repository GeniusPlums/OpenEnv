from __future__ import annotations

import random

from .models import LeadProfile, TaskLevel


TASK_PROFILES: dict[TaskLevel, list[LeadProfile]] = {
    TaskLevel.EASY: [
        LeadProfile(budget="medium", timeline="immediate", decision_maker=True, motivation="self_use"),
        LeadProfile(budget="high", timeline="immediate", decision_maker=True, motivation="investment"),
        LeadProfile(budget="medium", timeline="immediate", decision_maker=True, motivation="investment"),
    ],
    TaskLevel.MEDIUM: [
        LeadProfile(budget="high", timeline="3-6 months", decision_maker=True, motivation="investment"),
        LeadProfile(budget="medium", timeline="3-6 months", decision_maker=True, motivation="self_use"),
        LeadProfile(budget="high", timeline="3-6 months", decision_maker=True, motivation="self_use"),
    ],
    TaskLevel.HARD: [
        LeadProfile(
            budget="low",
            timeline="6+ months",
            decision_maker=False,
            motivation="exploring",
            surface_budget="high",
            surface_timeline="immediate",
        ),
        LeadProfile(
            budget="medium",
            timeline="6+ months",
            decision_maker=True,
            motivation="investment",
            surface_budget="high",
            surface_timeline="immediate",
        ),
        LeadProfile(
            budget="low",
            timeline="3-6 months",
            decision_maker=False,
            motivation="self_use",
            surface_budget="medium",
            surface_timeline="immediate",
        ),
    ],
}


def sample_profile(task: TaskLevel, seed: int | None = None) -> LeadProfile:
    rng = random.Random(seed)
    return rng.choice(TASK_PROFILES[task])
