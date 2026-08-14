import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from server import app

@pytest.mark.asyncio
async def test_ping():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/ping")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_reset_creates_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"task_id": "task_stemi_code"}
        response = await ac.post("/reset", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "observation" in data

@pytest.mark.asyncio
async def test_reset_empty_body():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/reset", json={})
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "observation" in data

@pytest.mark.asyncio
async def test_step_requires_session():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        action = {
            "action_type": "wait",
            "parameter": "1",
            "patient_id": "P1",
            "rationale": "testing"
        }
        # Missing Query param ?session_id=
        response = await ac.post("/step", json=action)
        assert response.status_code == 404
        
        # Invalid / missing session_id from DB
        response_invalid = await ac.post("/step?session_id=fake_sid", json=action)
        assert response_invalid.status_code == 404
        
@pytest.mark.asyncio
async def test_concurrent_sessions():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp_a = await ac.post("/reset", json={"task_id": "task_stemi_code"})
        sid_a = resp_a.json()["session_id"]
        
        resp_b = await ac.post("/reset", json={"task_id": "task_chest_pain_workup"})
        sid_b = resp_b.json()["session_id"]
        
        assert sid_a != sid_b
        
        resp_get_a = await ac.get(f"/state?session_id={sid_a}")
        resp_get_b = await ac.get(f"/state?session_id={sid_b}")
        
        assert resp_get_a.json()["task_id"] == "task_stemi_code"
        assert resp_get_b.json()["task_id"] == "task_chest_pain_workup"

@pytest.mark.asyncio
async def test_concurrent_sessions_are_isolated():
    """Proves two simultaneous sessions don't share state."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create two sessions with different tasks
        r1 = await client.post("/reset", json={"task_id": "task_stemi_code"})
        r2 = await client.post("/reset", json={"task_id": "task_sepsis_alert"})
        assert r1.status_code == 200
        assert r2.status_code == 200

        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]

        # Sessions must have different IDs
        assert sid1 != sid2

        # States must be independent — different task_ids
        state1 = await client.get(f"/state?session_id={sid1}")
        state2 = await client.get(f"/state?session_id={sid2}")
        assert state1.json()["task_id"] == "task_stemi_code"
        assert state2.json()["task_id"] == "task_sepsis_alert"

        # Step into session 1 only — session 2 must NOT change
        action = {
            "action_type": "assign_esi_level",
            "parameter": "1",
            "patient_id": "P1",
            "rationale": "Critical STEMI patient"
        }
        step_resp = await client.post(f"/step?session_id={sid1}", json=action)
        assert step_resp.status_code == 200

        # Session 1 should now have an action in history
        state1_after = await client.get(f"/state?session_id={sid1}")
        assert len(state1_after.json()["episode_history"]) == 1

        # Session 2 must remain pristine — 0 steps
        state2_after = await client.get(f"/state?session_id={sid2}")
        assert len(state2_after.json()["episode_history"]) == 0
        assert state2_after.json()["task_id"] == "task_sepsis_alert"
