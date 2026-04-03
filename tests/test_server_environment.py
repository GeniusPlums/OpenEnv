from server.leadqualenv_environment import LeadQualOpenEnv
from server.models import LeadQualActionModel


def test_server_state_exposes_internal_progress():
    env = LeadQualOpenEnv()
    obs = env.reset(seed=0, task="easy")

    assert obs.metadata["task"] == "easy"
    assert env.state.task == "easy"
    assert env.state.step_count == 0
    assert env.state.done is False


def test_server_state_updates_after_step():
    env = LeadQualOpenEnv()
    env.reset(seed=0, task="easy")
    env.step(LeadQualActionModel(message="Are you the person who can make the purchase decision yourself?"))

    assert env.state.step_count == 1
    assert env.state.known_signals["decision_maker"] is True
