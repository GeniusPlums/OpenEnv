from server.leadqualenv_environment import LeadQualOpenEnv
from server.models import LeadQualActionModel


def test_server_state_exposes_internal_progress():
    env = LeadQualOpenEnv()
    obs = env.reset(seed=0, task="easy")

    assert obs.metadata["task"] == "easy"
    assert env.state.task == "easy"
    assert env.state.step_count == 0
    assert env.state.done is False
    assert env.state.lead_temperature == 1.0


def test_server_state_updates_after_step():
    env = LeadQualOpenEnv()
    env.reset(seed=0, task="easy")
    env.step(LeadQualActionModel(message="Are you the person who can make the purchase decision yourself?"))

    assert env.state.step_count == 1
    assert env.state.known_signals.get("decision_maker") is not None
    assert env.state.lead_temperature < 1.0


def test_server_observation_includes_property_context():
    env = LeadQualOpenEnv()
    obs = env.reset(seed=0, task="easy")

    assert obs.property_context is not None
    assert len(obs.property_context) > 0


def test_server_reset_changes_task():
    env = LeadQualOpenEnv()
    env.reset(seed=0, task="easy")
    assert env.state.task == "easy"

    env.reset(seed=0, task="hard")
    assert env.state.task == "hard"


def test_server_step_returns_reward():
    env = LeadQualOpenEnv()
    env.reset(seed=0, task="easy")
    obs = env.step(LeadQualActionModel(message="What budget range are you looking at exactly?"))

    assert obs.reward is not None


def test_server_full_episode():
    env = LeadQualOpenEnv()
    env.reset(seed=0, task="easy")

    env.step(LeadQualActionModel(message="Are you the person who can make the purchase decision yourself?"))
    env.step(LeadQualActionModel(message="When are you planning to move, specifically?"))
    env.step(LeadQualActionModel(message="What budget range are you looking at exactly?"))
    obs = env.step(LeadQualActionModel(decision="qualified"))

    assert obs.done is True
    assert "task_score" in obs.info
