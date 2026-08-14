from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def _run(task_id: str, use_llm: bool = False) -> dict:
    r = client.post("/run-agent", json={"task_id": task_id, "use_llm": use_llm})
    assert r.status_code == 200
    return r.json()


def test_transcript_structure():
    """Every episode must return a complete step-by-step transcript."""
    d = _run("task_stemi_code")
    assert d["task_id"] == "task_stemi_code"
    assert d["mode"] in ("heuristic", "mixed")
    assert d["final_grade"] is not None
    assert d["step_count"] > 0
    assert len(d["steps"]) == d["step_count"]


def test_each_step_has_agent_action_and_rationale():
    """Every step must carry the agent origin, a structured action, and rationale."""
    d = _run("task_stemi_code")
    for s in d["steps"]:
        assert s["agent"] in ("heuristic", "llm")
        assert s["action"]["action_type"]
        assert s["action"]["parameter"] is not None
        assert s["action"]["patient_id"]
        assert s["action"]["rationale"]


def test_grade_is_scored_against_baseline():
    """Final grade is a float in [0,1] and the agent beats the baseline."""
    from environment.tasks import TASKS
    d = _run("task_sepsis_alert")
    assert 0.0 <= d["final_grade"] <= 1.0
    assert d["final_grade"] >= TASKS["task_sepsis_alert"].baseline_score