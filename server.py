import os
import uuid
import logging
import asyncio
from datetime import datetime, timezone, timedelta
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from threading import Lock

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from environment.env import ClinicalTriageEnvironment
from environment.models import TriageAction
from environment.task_registry import TASK_IDS
from environment.tasks import TASKS
from environment.agent import choose_next_action

# â”€â”€â”€ LOGGING CONFIGURATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logger = logging.getLogger("SURGICAL")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s â€” %(message)s")

# â”€â”€â”€ SESSION MANAGEMENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_sessions: dict[str, ClinicalTriageEnvironment] = {}
_session_times: dict[str, datetime] = {}
_lock = Lock()
SESSION_TTL_MINUTES = 30

def get_session(session_id: str) -> ClinicalTriageEnvironment:
    with _lock:
        if session_id not in _sessions:
            raise KeyError(f"Session {session_id!r} not found")
        return _sessions[session_id]

def create_session() -> tuple[str, ClinicalTriageEnvironment]:
    session_id = str(uuid.uuid4())
    with _lock:
        _sessions[session_id] = ClinicalTriageEnvironment()
        _session_times[session_id] = datetime.now(timezone.utc)
    return session_id, _sessions[session_id]

async def _session_cleanup_loop() -> None:
    """Evict sessions older than SESSION_TTL_MINUTES every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
        with _lock:
            expired = [sid for sid, t in _session_times.items() if t < cutoff]
            for sid in expired:
                _sessions.pop(sid, None)
                _session_times.pop(sid, None)
        if expired:
            logger.info(f"TTL cleanup: evicted {len(expired)} expired session(s)")

# â”€â”€â”€ AUTHENTICATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_security = HTTPBearer(auto_error=False)

def verify_api_key(creds: HTTPAuthorizationCredentials = Depends(_security)):
    key = os.getenv("SURGICAL_API_KEY")
    if key and (not creds or creds.credentials != key):
        raise HTTPException(status_code=401, detail="Unauthorized")

# â”€â”€â”€ STATE RECORD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
START_TIME = time.time()

# â”€â”€â”€ WEBSOCKET MANAGEMENT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New dashboard connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Dashboard disconnected")

    async def broadcast(self, message: dict):
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                logger.warning("Client disconnected from WebSocket broadcast")
                dead.append(connection)
            except Exception as e:
                logger.warning(f"WebSocket broadcast failed: {type(e).__name__} - {e}")
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)

manager = ConnectionManager()

# â”€â”€â”€ LIFESPAN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("SURGICAL server starting...")
    logger.info(f"Available Tasks: {', '.join(TASK_IDS)}")
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("SURGICAL server shutting down â€” sessions cleared")
    with _lock:
        _sessions.clear()
        _session_times.clear()

# â”€â”€â”€ APP INITIALIZATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title="SURGICAL - Clinical Triage Environment",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€â”€ CORE API ROUTES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/", response_class=HTMLResponse)
async def root_dashboard():
    """Serves the dashboard as the Space homepage."""
    dashboard_file = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")
    try:
        with open(dashboard_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>SURGICAL</h1><p>Dashboard not found. API is running â€” try /ping</p>", status_code=200)

@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "SURGICAL",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/healthz")
async def healthz():
    """Diagnostic endpoint with active session counts."""
    return {
        "status": "healthy",
        "active_sessions": len(_sessions),
        "session_ttl_minutes": SESSION_TTL_MINUTES,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "available_tasks": TASK_IDS
    }

@app.get("/tasks")
async def list_tasks():
    """Returns the list of valid task IDs."""
    return {"tasks": TASK_IDS}

@app.get("/scenarios")
async def list_scenarios():
    """Returns rich metadata for every scenario (for the dashboard)."""
    return {
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": t.max_steps,
                "baseline_score": t.baseline_score,
                "total_beds": getattr(t, "total_beds", None),
            }
            for t in TASKS.values()
        ]
    }

@app.post("/reset", dependencies=[Depends(verify_api_key)])
async def reset(payload: Dict[str, Any] = None):
    """
    Initialize a new episode. Creates a session and returns session_id.
    """
    data = payload or {}
    task_id = data.get("task_id", "task_stemi_code")
    
    if task_id not in TASK_IDS:
        logger.warning(f"Invalid task_id requested: {task_id}")
        raise HTTPException(status_code=422, detail=f"Invalid task_id. Must be one of: {TASK_IDS}")
    
    # Create session
    sid, env = create_session()
    
    try:
        obs = env.reset(task_id=task_id)
    except Exception as e:
        logger.warning(f"Reset failed for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Environment reset failed")
    
    # Broadcast to dashboard
    await manager.broadcast({
        "type": "RESET",
        "session_id": sid,
        "task_id": task_id,
        "state": env.state().model_dump()
    })
    
    return {
        "observation": obs,
        "session_id": sid
    }

@app.post("/step", dependencies=[Depends(verify_api_key)])
async def step(action: TriageAction, session_id: str = Query(None)):
    """
    Execute one triage action for a specific session.
    """
    if not session_id:
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        env = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        obs, reward, done, info = env.step(action)
        
        # Broadcast to dashboard
        await manager.broadcast({
            "type": "STEP",
            "session_id": session_id,
            "state": env.state().model_dump(),
            "last_reward": reward.model_dump(),
            "last_action": action.model_dump()
        })
        
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        logger.warning(f"Error in session {session_id} during step: {type(e).__name__} - {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def state(session_id: str = Query(None)):
    """Retrieve full internal state for a session. Unauthenticated."""
    if not session_id:
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        env = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return env.state()

@app.post("/run-agent")
async def run_agent(payload: Dict[str, Any] = None):
    """Run an autonomous triage agent through a full episode and return the transcript."""
    data = payload or {}
    task_id = data.get("task_id", "task_stemi_code")

    if task_id not in TASK_IDS:
        raise HTTPException(status_code=422, detail=f"Invalid task_id. Must be one of: {TASK_IDS}")

    use_llm = bool(data.get("use_llm"))
    # LLM proposes, a deterministic clinical policy guarantees completion.
    # Capping LLM calls keeps episodes fast and scores reliable.
    max_llm_steps = max(0, int(data.get("max_llm_steps", 3 if use_llm else 0)))
    sid, env = create_session()
    env.reset(task_id=task_id)

    steps: List[Dict[str, Any]] = []
    total_reward = 0.0
    final_grade: Optional[float] = None
    llm_steps = 0

    max_steps = TASKS[task_id].max_steps
    for step_idx in range(max_steps):
        obs = env._get_observation()
        want_llm = use_llm and llm_steps < max_llm_steps
        action, used_llm = await asyncio.to_thread(
            choose_next_action, obs, task_id, env.state(), use_llm=want_llm
        )
        if used_llm:
            llm_steps += 1
        observation, reward, done, info = env.step(action)

        step_score = reward.score if reward.score is not None else 0.0
        total_reward += step_score

        step_record = {
            "step": step_idx + 1,
            "agent": "llm" if used_llm else "heuristic",
            "action": action.model_dump(exclude_none=True),
            "reward": reward.model_dump(),
            "done": done,
            "elapsed_minutes": observation.elapsed_minutes,
        }
        steps.append(step_record)

        await manager.broadcast({
            "type": "AGENT_STEP",
            "session_id": sid,
            "task_id": task_id,
            "state": env.state().model_dump(),
            "step": step_record,
        })

        if done:
            grading = info.get("grading")
            if grading:
                final_grade = grading.get("score")
            break

    actual_modes = {s["agent"] for s in steps}
    return {
        "session_id": sid,
        "task_id": task_id,
        "mode": next(iter(actual_modes)) if len(actual_modes) == 1 else "mixed",
        "steps": steps,
        "step_count": len(steps),
        "total_reward": round(total_reward, 4),
        "final_grade": final_grade,
        "done": final_grade is not None,
    }

# â”€â”€â”€ DASHBOARD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Redirect to root which serves the dashboard."""
    dashboard_file = os.path.join(os.path.dirname(__file__), "dashboard", "index.html")
    try:
        with open(dashboard_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("Dashboard file not found.", status_code=404)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time dashboard updates. Unauthenticated."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket unhandled exception: {type(e).__name__} - {e}")
        manager.disconnect(websocket)

# â”€â”€â”€ ENTRY POINT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    port = int(os.getenv("PORT", 7860))
    logger.info(f"Starting SURGICAL server on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
