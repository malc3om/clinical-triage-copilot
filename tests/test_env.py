"""
test_env.py — Unit tests for ClinicalTriageEnvironment logic.
"""

import pytest
from environment.env import ClinicalTriageEnvironment
from environment.models import TriageAction
from environment.task_registry import TASK_IDS

def test_env_initialization():
    env = ClinicalTriageEnvironment()
    assert env.current_state is None
    
def test_env_reset():
    env = ClinicalTriageEnvironment()
    obs = env.reset(task_id="task_stemi_code")
    assert obs.task_id == "task_stemi_code"
    assert len(obs.patients) > 0
    assert env.current_state is not None

def test_env_invalid_reset():
    env = ClinicalTriageEnvironment()
    # Should fallback to default stemi code
    obs = env.reset(task_id="bogus_task")
    assert obs.task_id == "task_stemi_code"

def test_env_step_logic():
    env = ClinicalTriageEnvironment()
    obs = env.reset(task_id="task_stemi_code")
    
    pid = obs.patients[0].patient_id
    action = TriageAction(
        action_type="order_diagnostic",
        parameter="ekg",
        patient_id=pid,
        rationale="Chest pain workup"
    )
    
    new_obs, reward, done, info = env.step(action)
    
    assert reward.score is not None
    assert env.current_state.elapsed_minutes > 0
    assert "ekg" in env.current_state.diagnostics_ordered

def test_env_state_access():
    env = ClinicalTriageEnvironment()
    with pytest.raises(RuntimeError):
        env.state()
    
    env.reset()
    state = env.state()
    assert state.episode_id is not None
