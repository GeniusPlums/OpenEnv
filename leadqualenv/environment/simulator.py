from __future__ import annotations

import json
import os
from urllib import error, request

from .models import LeadProfile, ProbeQuality, SignalKey, TaskLevel


VAGUE_RESPONSES: dict[SignalKey, str] = {
    SignalKey.BUDGET: "somewhere in my range",
    SignalKey.TIMELINE: "still figuring out timing",
    SignalKey.DECISION_MAKER: "I need to discuss it a bit",
    SignalKey.MOTIVATION: "just exploring options",
}

PARAPHRASE_TEMPLATES: dict[SignalKey, dict[str | bool, str]] = {
    SignalKey.BUDGET: {
        "low": "I am looking at something on the lower end of the market.",
        "medium": "I have a mid-range budget in mind.",
        "high": "I can stretch for the right property.",
    },
    SignalKey.TIMELINE: {
        "immediate": "I want to move pretty quickly if something fits.",
        "3-6 months": "I am likely a few months away from making a move.",
        "6+ months": "This is more of a longer-term plan for me.",
    },
    SignalKey.DECISION_MAKER: {
        True: "I can make the purchase decision myself.",
        False: "I would need someone else involved before deciding.",
    },
    SignalKey.MOTIVATION: {
        "self_use": "This would be for my own use.",
        "investment": "I am mainly evaluating it as an investment.",
        "exploring": "I am still exploring what makes sense.",
    },
}

DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def resolve_signal(
    profile: LeadProfile,
    signal: SignalKey,
    probe_quality: ProbeQuality,
    task: TaskLevel,
) -> str | bool:
    if probe_quality == ProbeQuality.VAGUE:
        return VAGUE_RESPONSES[signal]

    if probe_quality == ProbeQuality.DIRECT:
        if task == TaskLevel.HARD and signal in profile.surface_signals:
            return profile.surface_signals[signal]
        return profile.true_signals[signal]

    if probe_quality == ProbeQuality.VERIFIED:
        return profile.true_signals[signal]

    raise ValueError(f"Unsupported probe quality: {probe_quality}")


def paraphrase_signal(signal: SignalKey, value: str | bool, probe_quality: ProbeQuality) -> str:
    if probe_quality == ProbeQuality.VAGUE:
        return str(value)
    template = PARAPHRASE_TEMPLATES[signal][value]
    return maybe_llm_paraphrase(signal, value, probe_quality, template)


def maybe_llm_paraphrase(
    signal: SignalKey,
    value: str | bool,
    probe_quality: ProbeQuality,
    fallback: str,
) -> str:
    if os.getenv("LEADQUALENV_USE_LLM", "").lower() not in {"1", "true", "yes"}:
        return fallback

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return fallback

    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    prompt = (
        "You are roleplaying a real-estate buyer in a lead qualification benchmark. "
        f"The canonical signal is {signal.value}={value!r}. "
        f"The probe quality was {probe_quality.value}. "
        "Reply with one short natural customer utterance that preserves exactly that fact and adds no new facts."
    )
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "Return a single plain-text customer utterance. Do not add extra facts.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"].strip()
        return content or fallback
    except (error.URLError, error.HTTPError, KeyError, IndexError, json.JSONDecodeError):
        return fallback


def generate_response(
    profile: LeadProfile,
    signal: SignalKey | None,
    probe_quality: ProbeQuality,
    task: TaskLevel,
) -> tuple[str, str | bool | None]:
    if probe_quality == ProbeQuality.IRRELEVANT or signal is None:
        return "Could you help me understand what you need from me?", None

    value = resolve_signal(profile, signal, probe_quality, task)
    return paraphrase_signal(signal, value, probe_quality), value
