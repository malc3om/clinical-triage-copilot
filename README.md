---
title: Surgical Clinical Triage Environment
emoji: 🏥
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - ai-agents
  - reinforcement-learning
  - medical-ai
  - fastapi
  - clinical-triage
---

<div align="center">

```
                       ███████╗██╗   ██╗██████╗  ██████╗ ██╗ ██████╗ █████╗ ██╗
                       ██╔════╝██║   ██║██╔══██╗██╔════╝ ██║██╔════╝██╔══██╗██║
                       ███████╗██║   ██║██████╔╝██║  ███╗██║██║     ███████║██║
                       ╚════██║██║   ██║██╔══██╗██║   ██║██║██║     ██╔══██║██║
                       ███████║╚██████╔╝██║  ██║╚██████╔╝██║╚██████╗██║  ██║███████╗
                       ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝
```

**Clinical Triage Environment for AI Agent Training — Healthcare AI that learns to save time-critical decisions**

[![CI](https://github.com/malc3om/clinical-triage-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/malc3om/clinical-triage-copilot/actions/workflows/ci.yml)
[![Agent](https://img.shields.io/badge/Agent%20Mode-Heuristic%20%2B%20LLM-yellow?style=flat-square)](#)
[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-Running-green?style=flat-square)](https://huggingface.co/spaces/sanskar1o7/clinical-triage-env)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-51%2F51%20passing-brightgreen?style=flat-square)](#)

*Healthcare AI · Emergency Medicine Triage*

</div>

---

## Why it matters

Every minute in an emergency department is a race against biology. A STEMI needs a cath lab within **90 minutes**. Sepsis needs antibiotics inside the **3-hour bundle**. Stroke needs a CT within **25 minutes**. When clinicians are overwhelmed, protocols fail and patients are lost.

**SURGICAL** is a clinical-triage copilot that watches a patient walk in and decides the *next critical action* — EKG? Cath lab? Antibiotics? — and explains itself in real time. It is a safe training ground for emergency triage decision-making: the agent learns from a dense clinical reward signal, then demonstrates its reasoning step-by-step so humans can learn too.

---

## What is Surgical?

**Surgical** is a production-grade reinforcement learning environment that simulates the highest-pressure moments in emergency medicine. It trains AI agents to think like senior emergency physicians — triaging patients, ordering diagnostics, activating protocols, and making irreversible disposition decisions under harsh time constraints.

Unlike toy environments, Surgical is grounded in **real clinical workflows**: ESI triage levels, STEMI protocols, sepsis care bundles, stroke door-to-CT windows, and mass casualty incident management. Every reward signal is shaped to provide dense, medically-meaningful feedback across the full decision trajectory — not just a binary pass/fail at episode end.

The environment exposes a fully **RESTful agent API** with session isolation, real-time WebSocket telemetry, and a brutalist live dashboard that makes the agent's reasoning visible at a glance.

---

## Live demo

**[▶ Run the AI Copilot on HF Spaces](https://huggingface.co/spaces/sanskar1o7/clinical-triage-env)** — pick a scenario, hit **RUN AI AGENT**, and watch the copilot triage a patient live: decision-flow nodes light up, the reward signal streams in, and the final grade is scored against the clinical baseline. No API key required.

## Screenshots

<img width="1400" alt="SURGICAL hero" src="https://raw.githubusercontent.com/malc3om/clinical-triage-copilot/main/screenshots/hero.png" />
<img width="1400" alt="AI copilot running a STEMI" src="https://raw.githubusercontent.com/malc3om/clinical-triage-copilot/main/screenshots/copilot_full.png" />

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SURGICAL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    HTTP/WS     ┌──────────────────────────────┐   │
│   │             │ ◄────────────► │        FastAPI Server        │   │
│   │  AI Agent   │                │  (server.py  —  port 7860)   │   │
│   │ agent.py    │                │                              │   │
│   │             │  OpenAI API    │  POST /reset  ─► new episode │   │
│   │ LLM policy  │  (optional)    │  POST /step   ─► take action │   │
│   │ heuristic   │                │  GET  /state  ─► get state   │   │
│   │ fallback    │                │  GET  /ping   ─► healthcheck │   │
│   └─────────────┘                │  WS   /ws     ─► live events │   │
│                                  └──────────┬───────────────────┘   │
│                                             │                       │ 
│   ┌─────────────────────────────────────────▼─────────────────┐     │
│   │                  Session Registry                         │     │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │     │
│   │  │ session1 │  │ session2 │  │ sessionN │  ← concurrency  │     │
│   │  │ (30 min) │  │ (30 min) │  │ (30 min) │    safe, TTL    │     │
│   │  └─────┬────┘  └──────────┘  └──────────┘                 │     │
│   └────────┼──────────────────────────────────────────────────┘     │
│            │                                                        │
│   ┌────────▼──────────────────────────────────────────────────┐     │
│   │              ClinicalTriageEnvironment (env.py)           │     │
│   │                                                           │     │
│   │  reset(task_id) ──► TaskRegistry ──► PatientGenerator     │     │
│   │                                                           │     │
│   │  step(action)   ──► ActionValidator                       │     │
│   │                  ──► RewardEngine (dense, shaped)         │     │
│   │                  ──► VitalsDeteriorationEngine            │     │
│   │                  ──► Grader (task-specific)               │     │
│   │                                                           │     │
│   │  state()        ──► TriageState (Pydantic snapshot)       │     │
│   └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │               Pydantic Data Layer (models.py)            │      │
│   │                                                          │      │
│   │  TriageAction  Observation  Reward  PatientState         │      │
│   │  VitalsModel   TriageState  TaskConfig  GradingResult    │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │            Live Dashboard (dashboard/index.html)         │      │
│   │                                                          │      │
│   │  ┌──────────────┐   WebSocket   ┌────────────────────┐   │      │
│   │  │  Browser UI  │ ◄───────────► │  Broadcast Manager │   │      │
│   │  │  (brutalist  │               │  (real-time events)│   │      │
│   │  │   yellow UI) │               └────────────────────┘   │      │
│   │  └──────────────┘                                        │      │
│   └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Clinical Scenarios

| Task ID | Difficulty | Baseline | Description |
|:--------|:----------:|:--------:|:------------|
| `task_stemi_code` | 🟢 Easy | 0.72 | STEMI recognition → immediate cath lab activation |
| `task_chest_pain_workup` | 🟡 Medium | 0.48 | PE vs STEMI differential diagnosis under uncertainty |
| `task_mci_surge` | 🔴 Hard | 0.31 | 5-victim mass casualty triage under resource scarcity |
| `task_sepsis_alert` | 🟡 Medium | 0.60 | Sepsis bundle: lactate, fluids, antibiotics within window |
| `task_stroke_code` | 🔴 Hard | 0.55 | Door-to-CT stroke pathway with NIHSS scoring |
| `task_pediatric_resp` | 🟡 Medium | 0.65 | Pediatric severe asthma, PICU escalation logic |

---

## Action & Observation Spaces

### Actions (`TriageAction`)
```python
class TriageAction(BaseModel):
    action_type: str   # "order_diagnostic" | "assign_esi_level" |
                       # "activate_pathway" | "administer_medication" |
                       # "disposition" | "wait"
    parameter:   str   # e.g. "troponin", "stemi", "aspirin", "admit_icu"
    patient_id:  str   # target patient
    rationale:   str   # agent's reasoning string
```

### Observations (`TriageObservation`)
Each step returns live `PatientState` objects with:
- Dynamic biometric vitals (HR, BP, SpO2, GCS, RR, Temp)
- Real-time trend vectors (↑ / ↓ / →)
- Chief complaint, history, and lab result accumulation
- Elapsed minutes and step count

### Reward Signal (Dense & Shaped)
```python
reward = 0.0
reward += 0.2   # correct diagnostic ordered
reward += 0.3   # partial progress toward protocol completion
reward += 0.5   # task objective fully met
reward -= 0.1   # destructive or incorrect action
reward  = max(0.0, min(1.0, reward))   # always clipped
```

---

## Quick Start

### Run locally
```bash
pip install -r requirements.txt
python server.py
# Dashboard → http://localhost:7860
```

### Run the agent from the CLI
Run all six scenarios headless — the agent (LLM or heuristic) drives every episode
and prints the final grade vs. the clinical baseline:
```bash
python -m environment.agent
```

### Run tests
```bash
pytest tests/ -v
# Expected: 51/51 passing
```

### Run the AI Copilot
Watch an autonomous agent triage a full episode live — no API key needed
(it falls back to a deterministic clinical policy; set `FEATHERLESS_API_KEY` to
drive it with **DeepSeek-V3.2**, or `GEMINI_API_KEY` / `HF_TOKEN` as fallback
providers). The LLM proposes the first actions; clinical guardrails guarantee
the protocol completes:

```bash
curl -X POST http://localhost:7860/run-agent \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_sepsis_alert", "use_llm": false}'
```

Then open the dashboard at `http://localhost:7860` and hit **RUN AI AGENT**.

### Docker
```bash
docker build -t surgical .
docker run -p 7860:7860 surgical
```

---

## REST API

| Method | Endpoint | Auth | Description |
|:-------|:---------|:----:|:------------|
| `GET` | `/ping` | ✗ | Health check → `{"status": "healthy"}` |
| `GET` | `/healthz` | ✗ | Full diagnostics + active sessions |
| `GET` | `/tasks` | ✗ | List all task IDs |
| `POST` | `/reset` | ✓ | Start episode → returns `session_id + observation` |
| `POST` | `/step?session_id=` | ✓ | Take action → returns `observation, reward, done, info` |
| `GET` | `/state?session_id=` | ✗ | Full internal state snapshot |
| `POST` | `/run-agent` | ✗ | Autonomous agent episode → full step-by-step transcript |
| `WS` | `/ws` | ✗ | Real-time broadcast for live dashboard |

Auth: `Bearer <SURGICAL_API_KEY>` (optional, activate via env var)

---

## Project Structure

```
surgical/
├── server.py                 ← FastAPI app, session management, WebSocket
├── env.yaml                  ← Environment metadata + task registry
├── Dockerfile                ← Production container (python:3.11-slim, port 7860)
├── requirements.txt          ← Python dependencies
├── .env.example              ← Environment variable template
│
├── environment/
│   ├── env.py                ← ClinicalTriageEnvironment (reset/step/state)
│   ├── agent.py              ← AI Copilot policies (heuristic + optional LLM)
│   ├── models.py             ← All Pydantic schemas
│   ├── tasks.py              ← Task definitions and patient generators
│   ├── graders.py            ← Deterministic per-task graders (0.0–1.0)
│   ├── logic.py              ← Reward engine + vitals deterioration
│   └── task_registry.py      ← TASK_IDS list
│
├── dashboard/
│   └── index.html            ← Self-contained brutalist live dashboard
│
└── tests/
    ├── test_reward.py
    ├── test_success_logic.py
    └── ...
```

---

## Deployment

The environment is live on Hugging Face Spaces:

**[🤗 Open in HF Space →](https://huggingface.co/spaces/sanskar1o7/clinical-triage-env)**

The Docker container:
1. Starts FastAPI on port `7860`
2. Serves the live dashboard at `/`
3. Exposes all REST agent endpoints
4. Returns `HTTP 200` on `/ping` (required by the judge pipeline)

---

<div align="center">

*Clinical Triage · AI for Health*

</div>
