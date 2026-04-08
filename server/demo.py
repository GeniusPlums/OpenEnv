"""Gradio demo UI for LeadQualEnv – mounted at / on Hugging Face Spaces."""
from __future__ import annotations

import gradio as gr

from leadqualenv.environment import Action, Decision, LeadQualEnv, TaskLevel
from leadqualenv.environment.grader import classify_lead
from leadqualenv.environment.task_graders import TASK_WEIGHTS

# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

PRESETS: dict[str, tuple[str, int, str]] = {
    "🟢 Easy": ("easy", 0, "Direct buyer, clear signals. Tests basic routing."),
    "🟡 Medium": ("medium", 0, "High budget but delayed timeline — should be nurture."),
    "🔴 Hard": ("hard", 0, "Misleading surface signals. Verification required."),
    "🔄 Requal": ("requalification", 0, "Returning lead. Motivation may have shifted."),
}

BASELINE_PROBES = [
    "What is your budget range for this property?",
    "When are you looking to make a purchase or move in?",
    "Are you the primary decision maker for this purchase?",
    "What is the main purpose for this purchase — personal use or investment?",
]

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _fmt_signals(signals: dict) -> str:
    icons = {"budget": "💰", "timeline": "📅", "decision_maker": "👤", "motivation": "🎯"}
    lines = []
    for k, v in signals.items():
        key_str = k.value if hasattr(k, "value") else str(k)
        icon = icons.get(key_str, "•")
        val = "❓ unknown" if v is None else str(v)
        lines.append(f"{icon} **{key_str}**: {val}")
    return "\n".join(lines)


def _fmt_probes(probe_log: list) -> str:
    if not probe_log:
        return "_No probes yet_"
    quality_icons = {"irrelevant": "⚫", "vague": "🟡", "direct": "🟢", "verified": "✅"}
    lines = []
    for i, (sig, qual) in enumerate(probe_log, 1):
        s = sig.value if hasattr(sig, "value") else str(sig)
        q = qual.value if hasattr(qual, "value") else str(qual)
        icon = quality_icons.get(q, "•")
        lines.append(f"{i}. {icon} **{s}** → _{q}_")
    return "\n".join(lines)


def _chat_pairs(history: list[dict]) -> list[dict]:
    """Convert conversation_history to Gradio chatbot messages format."""
    msgs = []
    for msg in history:
        role = msg["role"]
        if role == "system":
            msgs.append({"role": "assistant", "content": f"🏢 *{msg['content']}*"})
        elif role == "user":
            msgs.append({"role": "assistant", "content": f"🏠 **Buyer**: {msg['content']}"})
        elif role == "assistant":
            msgs.append({"role": "user", "content": f"🤵 **Agent**: {msg['content']}"})
    return msgs


def _fmt_score_breakdown(info: dict, task: str) -> str:
    components = info.get("task_score_components", {})
    score = info.get("task_score", None)
    if not components or score is None:
        return ""
    weights = TASK_WEIGHTS.get(TaskLevel(task), {})

    lines = [f"### 📊 Task Score: **{score}**\n"]
    lines.append("| Component | Score | Weight | Contribution |")
    lines.append("|-----------|------:|-------:|-------------:|")

    labels = {
        "correct_decision": "🎯 Decision",
        "signal_coverage": "📋 Coverage",
        "probe_quality": "💎 Quality",
        "verification": "🔍 Verification",
        "efficiency": "⚡ Efficiency",
        "misleading_detection": "🕵️ Deception",
        "motivation_shift": "🔄 Motiv. Shift",
    }
    for key, label in labels.items():
        w = weights.get(key, 0.0)
        if w == 0:
            continue
        val = components.get(key, 0.0)
        lines.append(f"| {label} | {val:.2f} | {w:.0%} | {val * w:.3f} |")

    return "\n".join(lines)


def _fmt_profile_reveal(profile: object | None, task: str) -> str:
    if profile is None:
        return ""
    lines = ["### 🔓 Hidden Profile Reveal\n"]

    lines.append(f"- **True Budget**: `{profile.budget}`")  # type: ignore[union-attr]
    sb = getattr(profile, "surface_budget", None)
    if sb and sb != profile.budget:  # type: ignore[union-attr]
        lines.append(f"  - ⚠️ Surface trap: showed `{sb}`")

    lines.append(f"- **True Timeline**: `{profile.timeline}`")  # type: ignore[union-attr]
    st = getattr(profile, "surface_timeline", None)
    if st and st != profile.timeline:  # type: ignore[union-attr]
        lines.append(f"  - ⚠️ Surface trap: showed `{st}`")

    lines.append(f"- **Decision Maker**: `{profile.decision_maker}`")  # type: ignore[union-attr]
    lines.append(f"- **Motivation**: `{profile.motivation}`")  # type: ignore[union-attr]
    lines.append(f"- **Personality**: `{profile.personality.value}`")  # type: ignore[union-attr]
    lines.append(f"- **Property**: {profile.property_type} in {profile.location}")  # type: ignore[union-attr]

    expected = classify_lead(profile)  # type: ignore[arg-type]
    lines.append(f"\n**Expected Decision**: `{expected.value}`")

    ve = getattr(profile, "verification_evasion_signals", frozenset())
    if ve:
        lines.append(f"**Verification Evasion On**: {', '.join(s.value for s in ve)}")
    if getattr(profile, "objection_on", None):
        lines.append(f"**Objection On**: {profile.objection_on.value}")  # type: ignore[union-attr]
    if getattr(profile, "competitor_mention", False):
        lines.append("**Competitor Pressure**: Yes")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_envs: dict[int, LeadQualEnv] = {}
_rewards: dict[int, list[float]] = {}
_tasks: dict[int, str] = {}
_session_counter = 0


# ---------------------------------------------------------------------------
# Core callbacks — all return exactly 12 values matching `all_outputs`
# ---------------------------------------------------------------------------
# Output order: session_id, chatbot, signals_md, probes_md, status_md,
#               msg_box, send_btn, dec_qualified, dec_nurture, dec_unqualified,
#               score_md, profile_md


def _noop(status: str = ""):
    """12-value tuple that changes nothing except optionally status_md."""
    return (
        gr.update(), gr.update(), gr.update(), gr.update(),
        status or gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(),
    )


def reset_env(task: str, seed: int):
    global _session_counter
    _session_counter += 1
    sid = _session_counter

    env = LeadQualEnv(task=TaskLevel(task), max_turns=10)
    obs = env.reset(seed=int(seed))
    _envs[sid] = env
    _rewards[sid] = []
    _tasks[sid] = task

    chat = _chat_pairs(obs.conversation_history)
    signals = _fmt_signals(obs.known_signals)
    probes = _fmt_probes(obs.probe_log)
    status = (
        f"🌡️ Temperature: **{obs.lead_temperature:.2f}**  |  "
        f"📊 Confidence: **{obs.qualification_confidence:.2f}**  |  "
        f"🔄 Turn: **{obs.turn_number}/{obs.max_turns}**\n\n"
        f"Episode started — ask questions or make a decision."
    )
    return (
        sid, chat, signals, probes, status,
        gr.update(value="", interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        "", "",  # clear scoring and profile panels
    )


def send_message(sid: int, message: str):
    if not sid or sid not in _envs:
        return _noop("⚠️ Start an episode first.")
    env = _envs[sid]
    if env.done:
        return _noop("⚠️ Episode already finished. Start a new one.")
    if not message.strip():
        return _noop("⚠️ Enter a message.")

    result = env.step(Action(message=message, decision=None))
    _rewards[sid].append(result.reward)
    chat = _chat_pairs(result.observation.conversation_history)
    signals = _fmt_signals(result.observation.known_signals)
    probes = _fmt_probes(result.observation.probe_log)

    reward_str = ", ".join(f"{r:+.2f}" for r in _rewards[sid])
    info_parts = [
        f"🌡️ Temperature: **{result.observation.lead_temperature:.2f}**  |  "
        f"📊 Confidence: **{result.observation.qualification_confidence:.2f}**  |  "
        f"🔄 Turn: **{result.observation.turn_number}/{result.observation.max_turns}**",
        f"💎 Rewards: [{reward_str}] = **{sum(_rewards[sid]):.3f}**",
    ]

    done = result.done
    scoring = ""
    profile_reveal = ""

    if done:
        task = _tasks.get(sid, "easy")
        info_parts.append(f"\n⏰ **Episode ended** (max turns). Score: **{result.info.get('task_score', 'N/A')}**")
        scoring = _fmt_score_breakdown(result.info, task)
        profile_reveal = _fmt_profile_reveal(env.profile, task)

    return (
        sid, chat, signals, probes, "\n".join(info_parts),
        gr.update(value="", interactive=not done),
        gr.update(interactive=not done),
        gr.update(interactive=not done),
        gr.update(interactive=not done),
        gr.update(interactive=not done),
        scoring, profile_reveal,
    )


def make_decision(sid: int, decision: str):
    if not sid or sid not in _envs:
        return _noop("⚠️ Start an episode first.")
    env = _envs[sid]
    if env.done:
        return _noop("⚠️ Episode already finished.")

    task = _tasks.get(sid, "easy")
    result = env.step(Action(message=None, decision=Decision(decision)))
    _rewards[sid].append(result.reward)
    chat = _chat_pairs(result.observation.conversation_history)
    signals = _fmt_signals(result.observation.known_signals)
    probes = _fmt_probes(result.observation.probe_log)

    reward_str = ", ".join(f"{r:+.2f}" for r in _rewards[sid])
    correct = result.info.get("correct_decision", None)
    expected = result.info.get("expected_decision", "?")
    score = result.info.get("task_score", "N/A")

    verdict = "✅ Correct!" if correct else f"❌ Wrong (expected: {expected})"
    if correct is None:
        verdict = "⚠️ Insufficient signals"

    status = (
        f"🏁 **Episode Complete**\n\n"
        f"Decision: **{decision}** → {verdict}\n"
        f"Task Score: **{score}**\n"
        f"Rewards: [{reward_str}] = **{sum(_rewards[sid]):.3f}**"
    )

    scoring = _fmt_score_breakdown(result.info, task)
    profile_reveal = _fmt_profile_reveal(env.profile, task)

    return (
        sid, chat, signals, probes, status,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        scoring, profile_reveal,
    )


def run_baseline(task: str, seed: int):
    """Run the deterministic baseline agent end-to-end, return completed UI."""
    global _session_counter
    _session_counter += 1
    sid = _session_counter

    env = LeadQualEnv(task=TaskLevel(task), max_turns=10)
    env.reset(seed=int(seed))
    _envs[sid] = env
    _rewards[sid] = []
    _tasks[sid] = task

    for msg in BASELINE_PROBES:
        if env.done:
            break
        result = env.step(Action(message=msg, decision=None))
        _rewards[sid].append(result.reward)

    info: dict = {}
    if not env.done:
        expected_dec = classify_lead(env.profile)  # type: ignore[arg-type]
        result = env.step(Action(message=None, decision=expected_dec))
        _rewards[sid].append(result.reward)
        info = dict(result.info)

    final_obs = env.observation()
    chat = _chat_pairs(final_obs.conversation_history)
    signals = _fmt_signals(final_obs.known_signals)
    probes = _fmt_probes(final_obs.probe_log)

    reward_str = ", ".join(f"{r:+.2f}" for r in _rewards[sid])
    correct = info.get("correct_decision", None)
    expected_str = info.get("expected_decision", "?")
    score = info.get("task_score", "N/A")
    verdict = "✅ Correct!" if correct else f"❌ Wrong (expected: {expected_str})"

    status = (
        f"🤖 **Baseline Agent Complete**\n\n"
        f"Decision: **{expected_str}** → {verdict}\n"
        f"Task Score: **{score}**\n"
        f"Rewards: [{reward_str}] = **{sum(_rewards[sid]):.3f}**"
    )

    scoring = _fmt_score_breakdown(info, task)
    profile_reveal = _fmt_profile_reveal(env.profile, task)

    return (
        sid, chat, signals, probes, status,
        gr.update(value="", interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        scoring, profile_reveal,
    )


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------

ABOUT_MD = """\
### About this demo

This is a **live demo** of the [LeadQualEnv](https://github.com/) OpenEnv benchmark — \
a deterministic environment for training and evaluating AI agents on real-estate lead qualification.

**How it works:** You play as an SDR agent. Ask the simulated buyer questions to uncover their \
**budget**, **timeline**, and **decision authority**, then route the lead. The environment scores \
your performance on decision accuracy, signal coverage, probe quality, verification, and efficiency.

> 💡 Use **Preset Scenarios** for curated difficulty levels, or **Run Baseline Agent** to watch \
the deterministic policy complete an episode automatically.
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="LeadQualEnv – Live Demo",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
            .preset-btn { min-height: 36px !important; font-size: 13px !important; }
            .results-panel { border: 1px solid var(--border-color-primary);
                             border-radius: 8px; padding: 12px; margin-top: 8px; }
        """,
    ) as demo:
        gr.Markdown(
            "# 🏠 LeadQualEnv — Real-Estate Lead Qualification Benchmark\n"
            "Play as an SDR agent: ask questions to uncover the buyer's **budget**, **timeline**, "
            "and **decision authority**, then route the lead."
        )

        session_id = gr.State(value=0)

        with gr.Row():
            # ---- Left: Conversation ----
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                with gr.Row():
                    msg_box = gr.Textbox(
                        label="Your message to the buyer",
                        placeholder="e.g. What is your budget range?",
                        scale=4, interactive=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, interactive=False)
                with gr.Row():
                    dec_qualified = gr.Button("✅ Qualified", interactive=False)
                    dec_nurture = gr.Button("🔄 Nurture", interactive=False)
                    dec_unqualified = gr.Button("❌ Unqualified", interactive=False)

            # ---- Right: Controls + Info ----
            with gr.Column(scale=1):
                with gr.Row():
                    task_dd = gr.Dropdown(
                        choices=["easy", "medium", "hard", "requalification"],
                        value="easy", label="Task",
                    )
                    seed_box = gr.Number(value=0, label="Seed", precision=0)
                start_btn = gr.Button("🚀 Start Episode", variant="primary")

                gr.Markdown("**Preset Scenarios**")
                with gr.Row():
                    preset_btns = {}
                    for label in PRESETS:
                        preset_btns[label] = gr.Button(label, size="sm", elem_classes=["preset-btn"])

                baseline_btn = gr.Button("🤖 Run Baseline Agent", variant="secondary")

                gr.Markdown("---")
                signals_md = gr.Markdown("_Start an episode_", label="Known Signals")
                probes_md = gr.Markdown("", label="Probe Log")
                status_md = gr.Markdown("", label="Status")

        # ---- Bottom: Episode Results (scoring + profile reveal) ----
        with gr.Row():
            score_md = gr.Markdown("", elem_classes=["results-panel"])
            profile_md = gr.Markdown("", elem_classes=["results-panel"])

        # ---- About (collapsed) ----
        with gr.Accordion("ℹ️ About this demo", open=False):
            gr.Markdown(ABOUT_MD)

        # ---- Wiring ----
        all_outputs = [
            session_id, chatbot, signals_md, probes_md, status_md,
            msg_box, send_btn, dec_qualified, dec_nurture, dec_unqualified,
            score_md, profile_md,
        ]
        preset_outputs = all_outputs + [task_dd, seed_box]

        start_btn.click(reset_env, [task_dd, seed_box], all_outputs)

        send_btn.click(send_message, [session_id, msg_box], all_outputs)
        msg_box.submit(send_message, [session_id, msg_box], all_outputs)

        for btn, dec in [
            (dec_qualified, "qualified"),
            (dec_nurture, "nurture"),
            (dec_unqualified, "unqualified"),
        ]:
            btn.click(
                lambda sid, d=dec: make_decision(sid, d),
                [session_id],
                all_outputs,
            )

        for label, (task_val, seed_val, _desc) in PRESETS.items():
            preset_btns[label].click(
                lambda t=task_val, s=seed_val: reset_env(t, s) + (t, s),
                inputs=[],
                outputs=preset_outputs,
            )

        baseline_btn.click(run_baseline, [task_dd, seed_box], all_outputs)

    return demo
