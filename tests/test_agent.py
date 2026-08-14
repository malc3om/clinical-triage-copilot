"""
test_agent.py — Tests for the heuristic triage policy.
"""

import pytest

from environment.agent import (
    _compact_observation,
    _extract_json_object,
    _llm_config,
    heuristic_next_action,
    normalize_parameter,
)
from environment.env import ClinicalTriageEnvironment
from environment.graders import GRADERS
from environment.models import TriageAction
from environment.task_registry import TASK_IDS
from environment.tasks import TASKS

VALID_ACTION_TYPES = {
    "order_diagnostic",
    "assign_esi_level",
    "activate_pathway",
    "disposition",
    "administer_medication",
    "wait",
}


def test_llm_config_prefers_featherless(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "rc-test")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("HF_TOKEN", "test-hf")
    cfg = _llm_config()
    assert "api.featherless.ai" in cfg["base_url"]
    assert cfg["model"] == "deepseek-ai/DeepSeek-V3.2"


def test_llm_config_prefers_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = _llm_config()
    assert cfg is not None
    assert "generativelanguage.googleapis.com" in cfg["base_url"]
    assert cfg["model"] == "gemini-flash-latest"


def test_llm_config_prefers_gemini_over_hf(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("HF_TOKEN", "test-hf")
    cfg = _llm_config()
    assert "generativelanguage.googleapis.com" in cfg["base_url"]


def test_llm_config_falls_back_to_hf(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "test-hf")
    cfg = _llm_config()
    assert cfg is not None
    assert "router.huggingface.co" in cfg["base_url"]


def test_llm_config_none_without_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert _llm_config() is None


def test_extract_json_handles_markdown_fence():
    raw = '```json\n{"action_type": "order_diagnostic", "parameter": "ekg"}\n```'
    assert _extract_json_object(raw)["action_type"] == "order_diagnostic"


def test_extract_json_handles_plain_and_junk():
    assert _extract_json_object('{"a": 1}')["a"] == 1
    assert _extract_json_object("no json here") == {}
    assert _extract_json_object("") == {}


def test_compact_observation_is_small_and_has_vitals():
    env = ClinicalTriageEnvironment()
    env.reset(task_id="task_stemi_code")
    obs = env._get_observation()
    compact = _compact_observation(obs)
    assert compact["patients"][0]["complaint"]
    assert "hr" in compact["patients"][0]
    assert compact["elapsed_minutes"] == 0


def test_normalize_parameter_diagnostics():
    assert normalize_parameter("order_diagnostic", "EKG") == "ekg"
    assert normalize_parameter("order_diagnostic", "CT pulmonary angiogram") == "ct_pa"
    assert normalize_parameter("order_diagnostic", "D-dimer") == "d_dimer"
    assert normalize_parameter("order_diagnostic", "head CT") == "ct_head"


def test_normalize_parameter_pathways_and_meds():
    assert normalize_parameter("activate_pathway", "STEMI/cath_lab") == "cath_lab"
    assert normalize_parameter("activate_pathway", "stroke code") == "stroke"
    assert normalize_parameter("administer_medication", "ASA 81mg") == "aspirin"
    assert normalize_parameter("administer_medication", "Rocephin") == "ceftriaxone"
    assert normalize_parameter("administer_medication", "nebulized albuterol") == "albuterol"


def test_normalize_parameter_disposition_and_esi():
    assert normalize_parameter("disposition", "admit to ICU") == "admit_icu"
    assert normalize_parameter("disposition", "Cardiac ICU") == "admit_cardiac_icu"
    assert normalize_parameter("disposition", "PICU") == "admit_picu"
    assert normalize_parameter("assign_esi_level", "ESI level 2") == "2"
    assert normalize_parameter("assign_esi_level", "1") == "1"


def test_heuristic_returns_valid_action_for_every_task():
    env = ClinicalTriageEnvironment()
    for task in TASK_IDS:
        env.reset(task_id=task)
        action = heuristic_next_action(env._get_observation(), task, env.state())
        assert isinstance(action, TriageAction)
        assert action.action_type in VALID_ACTION_TYPES
        assert action.patient_id
        assert action.parameter
        assert action.patient_id in {p.patient_id for p in env._get_observation().patients}


def test_heuristic_completes_every_task_above_baseline():
    for task in TASK_IDS:
        env = ClinicalTriageEnvironment()
        env.reset(task_id=task)
        done = False
        for _ in range(TASKS[task].max_steps):
            action = heuristic_next_action(env._get_observation(), task, env.state())
            _, _, done, _ = env.step(action)
            if done:
                break
        assert env.state().done, f"agent did not finish {task}"
        score = GRADERS[task](env.state().episode_history)
        assert score >= TASKS[task].baseline_score, (
            f"{task}: agent scored {score:.2f} below baseline {TASKS[task].baseline_score:.2f}"
        )


def test_heuristic_stemi_short_episode():
    env = ClinicalTriageEnvironment()
    env.reset(task_id="task_stemi_code")
    for _ in range(10):
        action = heuristic_next_action(env._get_observation(), "task_stemi_code", env.state())
        _, _, done, _ = env.step(action)
        if done:
            break
    assert len(env.state().episode_history) <= 5
