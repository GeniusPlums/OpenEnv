from __future__ import annotations

import json
import os
from urllib import error, request

from .models import LeadProfile, Personality, ProbeQuality, SignalKey, TaskLevel

VAGUE_RESPONSES: dict[SignalKey, dict[Personality, str]] = {
    SignalKey.BUDGET: {
        Personality.DIRECT: "Somewhere in my range, I'd say.",
        Personality.EVASIVE: "I'd rather not get into exact numbers right now.",
        Personality.VERBOSE: "Well, you know, I've been looking at various options and my budget is flexible depending on what I find.",
        Personality.TERSE: "Flexible.",
        Personality.FRIENDLY: "Oh, I have some room to work with! Nothing locked in yet though.",
    },
    SignalKey.TIMELINE: {
        Personality.DIRECT: "Still figuring out the timing.",
        Personality.EVASIVE: "I'm not in a huge rush, but who knows.",
        Personality.VERBOSE: "I've been thinking about this for a while and honestly the timing depends on several factors I'm still working through.",
        Personality.TERSE: "Not sure yet.",
        Personality.FRIENDLY: "I'm pretty open on timing! Just seeing what's out there.",
    },
    SignalKey.DECISION_MAKER: {
        Personality.DIRECT: "I need to discuss it a bit.",
        Personality.EVASIVE: "There are a few people involved.",
        Personality.VERBOSE: "Well, it's not entirely up to me — I have family members who would want a say in something this big.",
        Personality.TERSE: "Not just me.",
        Personality.FRIENDLY: "Oh, I'd definitely want to loop in my partner before anything moves forward!",
    },
    SignalKey.MOTIVATION: {
        Personality.DIRECT: "Just exploring options for now.",
        Personality.EVASIVE: "Let's just say I'm keeping my options open.",
        Personality.VERBOSE: "I've been thinking about this from multiple angles — whether it's for living or as a financial move, I'm weighing everything.",
        Personality.TERSE: "Looking around.",
        Personality.FRIENDLY: "I'm just exploring what makes sense for me right now!",
    },
}

PARAPHRASE_TEMPLATES: dict[SignalKey, dict[str | bool, dict[Personality, str]]] = {
    SignalKey.BUDGET: {
        "low": {
            Personality.DIRECT: "I'm looking at something on the lower end of the market.",
            Personality.EVASIVE: "I'm being cautious with what I can put toward this.",
            Personality.VERBOSE: "Honestly, I've run the numbers and I need to be realistic — my budget is on the lower side right now.",
            Personality.TERSE: "Lower end.",
            Personality.FRIENDLY: "I'm keeping it modest! Looking for good value.",
        },
        "medium": {
            Personality.DIRECT: "I have a mid-range budget in mind.",
            Personality.EVASIVE: "I have a reasonable amount set aside.",
            Personality.VERBOSE: "I've been saving up and I think I can comfortably work within a mid-range budget for the right place.",
            Personality.TERSE: "Mid-range.",
            Personality.FRIENDLY: "I've got a decent budget — nothing crazy but solid!",
        },
        "high": {
            Personality.DIRECT: "I can stretch for the right property.",
            Personality.EVASIVE: "Let's say money isn't the main constraint here.",
            Personality.VERBOSE: "I'm in a fortunate position where budget isn't the primary limiting factor — the right property matters more than the price tag.",
            Personality.TERSE: "High end. Price isn't the issue.",
            Personality.FRIENDLY: "I'm ready to invest properly! The right place is worth paying for.",
        },
    },
    SignalKey.TIMELINE: {
        "immediate": {
            Personality.DIRECT: "I want to move pretty quickly if something fits.",
            Personality.EVASIVE: "I'd like to get things sorted soon if possible.",
            Personality.VERBOSE: "I've been looking for a while and I'm at the point where I want to move on this quickly — ideally within the next few weeks.",
            Personality.TERSE: "Soon. Ready to go.",
            Personality.FRIENDLY: "I'm excited to move fast! Really hoping to find something right away.",
        },
        "3-6 months": {
            Personality.DIRECT: "I'm likely a few months away from making a move.",
            Personality.EVASIVE: "There's no immediate rush on my end.",
            Personality.VERBOSE: "I'm thinking somewhere in the three to six month window — I want to take my time and make sure everything lines up properly.",
            Personality.TERSE: "Few months out.",
            Personality.FRIENDLY: "Probably in the next few months! No huge rush but definitely planning ahead.",
        },
        "6+ months": {
            Personality.DIRECT: "This is more of a longer-term plan for me.",
            Personality.EVASIVE: "I'm thinking further out, honestly.",
            Personality.VERBOSE: "This is really more of a longer-term consideration — I'd say at least six months out, maybe more, depending on how things develop.",
            Personality.TERSE: "Long term.",
            Personality.FRIENDLY: "Oh, this is more of a future thing! I'm just getting started with research.",
        },
    },
    SignalKey.DECISION_MAKER: {
        True: {
            Personality.DIRECT: "I can make the purchase decision myself.",
            Personality.EVASIVE: "I have the authority to move forward.",
            Personality.VERBOSE: "Yes, I'm the primary decision maker here — if I find what I'm looking for, I can move forward without needing anyone else's sign-off.",
            Personality.TERSE: "My call.",
            Personality.FRIENDLY: "Yep, it's all me! I make the final call on this.",
        },
        False: {
            Personality.DIRECT: "I would need someone else involved before deciding.",
            Personality.EVASIVE: "It's not entirely my decision alone.",
            Personality.VERBOSE: "I should be transparent — while I'm the one doing the initial research, the final decision involves my spouse and potentially my parents as well.",
            Personality.TERSE: "Need family approval.",
            Personality.FRIENDLY: "Oh, I'd definitely need to bring my partner into this! We make big decisions together.",
        },
    },
    SignalKey.MOTIVATION: {
        "self_use": {
            Personality.DIRECT: "This would be for my own use.",
            Personality.EVASIVE: "It's a personal matter.",
            Personality.VERBOSE: "I'm looking for a place where I can actually live — this would be my primary residence, so livability matters more than anything.",
            Personality.TERSE: "Personal use.",
            Personality.FRIENDLY: "It's for me! Looking for a place to call home.",
        },
        "investment": {
            Personality.DIRECT: "I'm mainly evaluating it as an investment.",
            Personality.EVASIVE: "I'm looking at the financial side of things.",
            Personality.VERBOSE: "I'm approaching this from an investment perspective — rental yields, appreciation potential, that sort of analysis.",
            Personality.TERSE: "Investment.",
            Personality.FRIENDLY: "I'm thinking of it as an investment! Love the idea of building a portfolio.",
        },
        "exploring": {
            Personality.DIRECT: "I'm still exploring what makes sense.",
            Personality.EVASIVE: "I'm just seeing what's available.",
            Personality.VERBOSE: "I'm at an early stage — I haven't fully committed to buying, I'm just exploring the market to see what's realistic given my situation.",
            Personality.TERSE: "Just looking.",
            Personality.FRIENDLY: "I'm just exploring for now! Keeping my options open and seeing what catches my eye.",
        },
    },
}

COMPETITOR_RESPONSES = [
    "By the way, I'm also talking to another agent about a similar property nearby.",
    "I should mention I'm comparing this with a few other options from different agents.",
    "Just so you know, I've got another viewing scheduled with a competitor this week.",
]

OBJECTION_RESPONSES: dict[SignalKey, str] = {
    SignalKey.BUDGET: "I'd rather not discuss exact numbers until I know more about the property.",
    SignalKey.TIMELINE: "Why does the timing matter so much at this stage?",
    SignalKey.DECISION_MAKER: "Does it matter who makes the final call? I'm the one interested.",
    SignalKey.MOTIVATION: "I'd rather keep my reasons private for now, if that's okay.",
}

DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def resolve_signal(
    profile: LeadProfile,
    signal: SignalKey,
    probe_quality: ProbeQuality,
    task: TaskLevel,
) -> str | bool:
    if probe_quality == ProbeQuality.VAGUE:
        return _get_vague_response(profile, signal)

    if probe_quality == ProbeQuality.DIRECT:
        if task == TaskLevel.HARD and signal in profile.surface_signals:
            return profile.surface_signals[signal]
        return profile.true_signals[signal]

    if probe_quality == ProbeQuality.VERIFIED:
        return profile.true_signals[signal]

    raise ValueError(f"Unsupported probe quality: {probe_quality}")


def _get_vague_response(profile: LeadProfile, signal: SignalKey) -> str:
    personality_responses = VAGUE_RESPONSES.get(signal, {})
    return personality_responses.get(profile.personality, "I'm not sure about that yet.")


def paraphrase_signal(
    profile: LeadProfile,
    signal: SignalKey,
    value: str | bool,
    probe_quality: ProbeQuality,
) -> str:
    if probe_quality == ProbeQuality.VAGUE:
        return _get_vague_response(profile, signal)

    templates = PARAPHRASE_TEMPLATES.get(signal, {})
    value_templates = templates.get(value, {})
    fallback = value_templates.get(Personality.DIRECT, str(value))
    text = value_templates.get(profile.personality, fallback)

    return maybe_llm_paraphrase(signal, value, probe_quality, text)


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


LOW_TEMPERATURE_RESPONSES = [
    "Look, I'm running short on time. Can we wrap this up?",
    "I've already answered a lot of questions. What else do you need?",
    "I'm starting to lose interest here. Do you have something for me or not?",
    "Honestly, I'm not sure this is going anywhere.",
    "I have other agents I could be talking to, you know.",
]


def generate_response(
    profile: LeadProfile,
    signal: SignalKey | None,
    probe_quality: ProbeQuality,
    task: TaskLevel,
    *,
    lead_temperature: float = 1.0,
) -> tuple[str, str | bool | None]:
    if probe_quality == ProbeQuality.IRRELEVANT or signal is None:
        return "Could you help me understand what you need from me?", None

    # Lead temperature affects buyer willingness to engage
    # At very low temperature, buyer may refuse to answer (returns vague/no value)
    if lead_temperature < 0.3 and probe_quality != ProbeQuality.VERIFIED:
        import random as _rng
        refusal = _rng.Random(hash((signal, lead_temperature))).choice(LOW_TEMPERATURE_RESPONSES)
        return refusal, None

    # At medium-low temperature, downgrade direct probes to vague responses
    if lead_temperature < 0.55 and probe_quality == ProbeQuality.DIRECT:
        vague_value = resolve_signal(profile, signal, ProbeQuality.VAGUE, task)
        vague_text = paraphrase_signal(profile, signal, vague_value, ProbeQuality.VAGUE)
        return vague_text, None  # Return no value — buyer is too disengaged for real info

    # Handle objection mechanic: lead pushes back on first probe of objection signal
    if profile.objection_on == signal and probe_quality == ProbeQuality.DIRECT:
        return OBJECTION_RESPONSES.get(signal, "I'd rather not answer that right now."), None

    value = resolve_signal(profile, signal, probe_quality, task)
    text = paraphrase_signal(profile, signal, value, probe_quality)

    # Competitor pressure mechanic
    if profile.competitor_mention and signal == SignalKey.TIMELINE and probe_quality == ProbeQuality.DIRECT:
        import random as _rng
        text += " " + _rng.Random(hash(text)).choice(COMPETITOR_RESPONSES)

    return text, value

