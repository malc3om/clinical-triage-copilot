"""
test_run_agent.py — API tests for the POST /run-agent endpoint.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from server import app


@pytest.mark.asyncio
async def test_run_agent_heuristic_stemi():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/run-agent", json={"task_id": "task_stemi_code", "use_llm": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["step_count"] > 0
    assert data["final_grade"] is not None
    assert data["final_grade"] >= 0.72
    assert data["steps"][0]["action"]["action_type"] == "order_diagnostic"


@pytest.mark.asyncio
async def test_run_agent_defaults_to_stemi():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/run-agent", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "task_stemi_code"
    assert data["done"] is True


@pytest.mark.asyncio
async def test_run_agent_invalid_task():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/run-agent", json={"task_id": "bogus_task"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_scenarios_metadata():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/scenarios")
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert len(tasks) == 6
    assert {"id", "name", "difficulty", "description", "max_steps", "baseline_score"} <= set(tasks[0].keys())
    assert all(t["name"] for t in tasks)
