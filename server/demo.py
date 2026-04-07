"""Gradio demo UI for LeadQualEnv – mounted at / on Hugging Face Spaces."""
from __future__ import annotations

import gradio as gr

from leadqualenv.environment import Action, Decision, LeadQualEnv, TaskLevel
from leadqualenv.environment.grader import classify_lead

# ---------------------------------------------------------------------------
# Helpers
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
    lines = []
    for i, (sig, qual) in enumerate(probe_log, 1):
        s = sig.value if hasattr(sig, "value") else str(sig)
        q = qual.value if hasattr(qual, "value") else str(qual)
        lines.append(f"{i}. **{s}** → _{q}_")
    return "\n".join(lines)


def _chat_pairs(history: list[dict]) -> list[dict]:
    """Convert conversation_history to Gradio chatbot messages format."""
    msgs = []
    for msg in history:
        role = msg["role"]
        if role == "system":
            msgs.append({"role": "assistant", "content": f"🏢 *{msg['content']}*"})
        elif role == "user":
            msgs.append({"role": "assistant", "content": f"🏠 Buyer: {msg['content']}"})
        elif role == "assistant":
            msgs.append({"role": "user", "content": f"🤵 Agent: {msg['content']}"})
    return msgs

# ---------------------------------------------------------------------------
# Core callbacks
# ---------------------------------------------------------------------------

_envs: dict[int, LeadQualEnv] = {}
_rewards: dict[int, list[float]] = {}
_session_counter = 0


def reset_env(task: str, seed: int):
    global _session_counter
    _session_counter += 1
    sid = _session_counter

    env = LeadQualEnv(task=TaskLevel(task), max_turns=10)
    obs = env.reset(seed=int(seed))
    _envs[sid] = env
    _rewards[sid] = []

    chat = _chat_pairs(obs.conversation_history)
    signals = _fmt_signals(obs.known_signals)
    probes = _fmt_probes(obs.probe_log)
    status = (
        f"🌡️ Temperature: **{obs.lead_temperature:.2f}**  |  "
        f"📊 Confidence: **{obs.qualification_confidence:.2f}**  |  "
        f"🔄 Turn: **{obs.turn_number}/{obs.max_turns}**"
    )
    return sid, chat, signals, probes, status, "", "Episode started – ask questions or make a decision.", gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)


def send_message(sid: int, message: str):
    if not sid or sid not in _envs:
        return None, "", "", "", "⚠️ Start an episode first.", message, gr.update(), gr.update(), gr.update()
    env = _envs[sid]
    if env.done:
        return None, "", "", "", "⚠️ Episode already finished. Start a new one.", "", gr.update(), gr.update(), gr.update()
    if not message.strip():
        return None, "", "", "", "⚠️ Enter a message.", message, gr.update(), gr.update(), gr.update()

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

    if result.done:
        info_parts.append(f"\n⏰ **Episode ended** (max turns). Score: **{result.info.get('task_score', 'N/A')}**")

    done = result.done
    return sid, chat, signals, probes, "\n".join(info_parts), "", gr.update(interactive=not done), gr.update(interactive=not done), gr.update(interactive=not done)


def make_decision(sid: int, decision: str):
    if not sid or sid not in _envs:
        return None, "", "", "", "⚠️ Start an episode first.", gr.update(), gr.update(), gr.update()
    env = _envs[sid]
    if env.done:
        return None, "", "", "", "⚠️ Episode already finished.", gr.update(), gr.update(), gr.update()

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

    info = (
        f"🏁 **Episode Complete**\n\n"
        f"Decision: **{decision}** → {verdict}\n"
        f"Task Score: **{score}**\n"
        f"Rewards: [{reward_str}] = **{sum(_rewards[sid]):.3f}**"
    )
    return sid, chat, signals, probes, info, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="LeadQualEnv – Live Demo") as demo:
        gr.Markdown(
            "# 🏠 LeadQualEnv – Real-Estate Lead Qualification Benchmark\n"
            "Play as an SDR agent: ask questions to uncover the buyer's **budget**, **timeline**, "
            "and **decision authority**, then route the lead.\n\n"
            "Choose a task difficulty, click **Start Episode**, type questions, and make a decision when ready."
        )

        session_id = gr.State(value=0)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                with gr.Row():
                    msg_box = gr.Textbox(
                        label="Your message to the buyer",
                        placeholder="e.g. What is your budget range?",
                        scale=4,
                        interactive=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, interactive=False)
                with gr.Row():
                    dec_qualified = gr.Button("✅ Qualified", interactive=False)
                    dec_nurture = gr.Button("🔄 Nurture", interactive=False)
                    dec_unqualified = gr.Button("❌ Unqualified", interactive=False)

            with gr.Column(scale=1):
                with gr.Row():
                    task_dd = gr.Dropdown(
                        choices=["easy", "medium", "hard", "requalification"],
                        value="easy", label="Task"
                    )
                    seed_box = gr.Number(value=0, label="Seed", precision=0)
                start_btn = gr.Button("🚀 Start Episode", variant="primary")

                signals_md = gr.Markdown("_Start an episode_", label="Known Signals")
                probes_md = gr.Markdown("", label="Probe Log")
                status_md = gr.Markdown("", label="Status")

        # -- Wiring --
        common_outputs = [session_id, chatbot, signals_md, probes_md, status_md]

        start_btn.click(
            reset_env, [task_dd, seed_box],
            common_outputs + [msg_box, status_md, msg_box, send_btn, dec_qualified],
        )

        send_btn.click(
            send_message, [session_id, msg_box],
            common_outputs + [msg_box, msg_box, send_btn, dec_qualified],
        )
        msg_box.submit(
            send_message, [session_id, msg_box],
            common_outputs + [msg_box, msg_box, send_btn, dec_qualified],
        )

        for btn, dec in [(dec_qualified, "qualified"), (dec_nurture, "nurture"), (dec_unqualified, "unqualified")]:
            btn.click(
                lambda sid, d=dec: make_decision(sid, d),
                [session_id],
                common_outputs + [msg_box, send_btn, dec_qualified],
            )

    return demo
