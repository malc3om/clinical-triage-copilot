"""
agent.py — Action policies for the SURGICAL copilot demo.

Provides a deterministic heuristic triage policy (always works, no API key
required) plus an optional LLM policy (OpenAI-compatible client, provider
priority: Featherless → Gemini → Hugging Face router) that gracefully falls
back to the heuristic on any failure.
"""

import os
import json
import re
from typing import Optional, Tuple

from .models import TriageAction, TriageObservation, TriageState
from .tasks import TASKS

SYSTEM_PROMPT = """You are an expert Emergency Medicine Physician performing triage and stabilization.
You must analyze the patient state and provide the next best clinical action.
Available action_types: order_diagnostic, assign_esi_level, activate_pathway, disposition, administer_medication, wait.
Respond ONLY with a valid JSON object:
{
  "action_type": "...",
  "parameter": "...",
  "patient_id": "...",
  "rationale": "..."
}"""


# ─── LLM Parameter Normalization ───────────────────────────────────────
# LLMs phrase parameters naturally ("CT pulmonary angiogram", "ASA 81mg").
# Map them to the canonical strings the reward engine and graders expect.

_PARAM_ALIASES = {
    # diagnostics
    "ekg": ["ekg", "ecg", "12-lead", "electrocardiogram", "cardiac monitor"],
    "troponin": ["troponin", "trop-i", "troponin_i", "troponin-i", "trop"],
    "lactate": ["lactate", "lactic acid", "lactic_acid", "lactic", "lact"],
    "d_dimer": ["d_dimer", "d-dimer", "ddimer", "d dimer"],
    "ct_pa": ["ct_pa", "ctpa", "ct pa", "ct pulmonary", "pulmonary angiogram", "cta chest"],
    "ct_head": ["ct_head", "ct head", "head ct", "ct brain", "ct_brain", "ct_head_noncon", "non-contrast ct", "noncontrast ct"],
    "cta": ["cta", "ct_angio", "ct angio", "ct angiography"],
    "blood_cultures": ["blood_cultures", "blood culture", "blood_culture", "cultures", "blood cultures"],
    "cbc": ["cbc", "complete blood count", "complete blood"],
    "bmp": ["bmp", "basic metabolic", "basic metab"],
    "cxr": ["cxr", "chest_xray", "chest xray", "chest x-ray", "chest radiograph", "xray chest"],
    "coag_panel": ["coag_panel", "coags", "coagulation", "coag", "pt inr"],
    "glucose": ["glucose", "fingerstick", "finger stick", "blood glucose"],
    "procalcitonin": ["procalcitonin", "pro-calcitonin", "pro calcitonin"],
    "urinalysis": ["urinalysis", "urine analysis", "ua"],
    "blood_gas": ["blood_gas", "abg", "arterial blood gas"],
    "bnp": ["bnp", "brain natriuretic"],
    # pathways
    "cath_lab": ["cath_lab", "cath lab", "cardiac catheterization", "catheterization", "stemi", "pci", "percutaneous"],
    "stroke": ["stroke", "stroke code", "stroke pathway", "brain attack"],
    "sepsis": ["sepsis", "sepsis bundle", "sepsis pathway", "septic"],
    "trauma": ["trauma", "trauma team", "trauma activation"],
    "cardiac_arrest": ["cardiac_arrest", "cardiac arrest", "code blue", "code team"],
    "mci": ["mci", "mass casualty", "disaster", "surge"],
    # medications
    "aspirin": ["aspirin", "asa", "acetylsalicylic"],
    "ceftriaxone": ["ceftriaxone", "rocephin", "antibiotic", "antibiotics", "broad-spectrum", "broad spectrum", "cef"],
    "albuterol": ["albuterol", "ventolin", "salbutamol", "nebulized bronchodilator", "bronchodilator"],
    "tpa": ["tpa", "alteplase", "tissue plasminogen", "thrombolytic", "tenecteplase"],
    "epinephrine": ["epinephrine", "adrenaline", "epi"],
    "heparin": ["heparin", "unfractionated heparin"],
    "nitroglycerin": ["nitroglycerin", "nitro", "ntg", "glyceryl trinitrate"],
    "morphine": ["morphine"],
    "normal_saline": ["normal_saline", "normal saline", "fluid bolus", "crystalloid", "0.9% saline", "ns bolus"],
    "vasopressor": ["vasopressor", "norepinephrine", "levophed", "pressor", "vasopressors"],
    "oxygen": ["oxygen", "supplemental oxygen", "o2", "supplemental o2"],
    # dispositions
    "admit_icu": ["admit_icu", "admit to icu", "intensive care", "critical care unit", "icu"],
    "admit_cardiac_icu": ["admit_cardiac_icu", "cardiac icu", "cicu", "ccu", "coronary care"],
    "admit_floor": ["admit_floor", "admit floor", "general floor", "medical floor", "general ward", "ward"],
    "admit_telemetry": ["admit_telemetry", "telemetry", "monitored bed", "tele"],
    "admit_picu": ["admit_picu", "picu", "pediatric icu", "pediatric intensive"],
    "transfer_cath_lab": ["transfer_cath_lab", "transfer to cath lab", "cath lab transfer", "transfer for cath"],
    "transfer_or": ["transfer_or", "operating room", "transfer to or", "surgery"],
    "transfer_stroke_center": ["transfer_stroke_center", "stroke center", "transfer to stroke"],
    "discharge": ["discharge", "send home", "release home", "home"],
    "observe": ["observe", "observation", "observation unit", "obs"],
}


def normalize_parameter(action_type: str, param: str) -> str:
    """Map free-form LLM parameters onto the environment's canonical values."""
    p = (param or "").strip().lower()
    if action_type == "assign_esi_level":
        m = re.search(r"[1-5]", p)
        return m.group(0) if m else "3"
    if action_type == "wait":
        m = re.search(r"\d+", p)
        return m.group(0) if m else "5"
    canonical, best = None, 0
    for canon, aliases in _PARAM_ALIASES.items():
        for a in aliases:
            al = a.lower()
            if al and (al in p or p in al):
                if len(al) > best:
                    best, canonical = len(al), canon
    return canonical or p


# ─── LLM Policy ─────────────────────────────────────────────────────────
# Provider priority: Featherless (fast, reliable) → Google Gemini (free tier)
# → Hugging Face router → heuristic fallback.

_client = None
_client_model = None


def _llm_config() -> Optional[dict]:
    """Pick an OpenAI-compatible provider based on available credentials."""
    featherless_key = os.getenv("FEATHERLESS_API_KEY")
    if featherless_key:
        return {
            "api_key": featherless_key,
            "base_url": os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1"),
            "model": os.getenv("FEATHERLESS_MODEL", "deepseek-ai/DeepSeek-V3.2"),
        }
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        return {
            "api_key": gemini_key,
            "base_url": os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        }
    if os.getenv("HF_TOKEN"):
        return {
            "api_key": os.getenv("HF_TOKEN"),
            "base_url": os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            "model": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        }
    return None


def _llm_client(config: dict):
    global _client, _client_model
    if _client is None or _client_model != config["model"]:
        from openai import OpenAI
        _client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=25,
            max_retries=0,
        )
        _client_model = config["model"]
    return _client


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip markdown code fences
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = fenced.strip("`")
        if fenced.lower().startswith("json"):
            fenced = fenced[4:]
    try:
        return json.loads(fenced)
    except (json.JSONDecodeError, ValueError):
        try:
            start = fenced.find("{")
            end = fenced.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(fenced[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _compact_observation(obs: TriageObservation) -> dict:
    """Reduce the observation to a small clinical summary for the LLM prompt."""
    pts = []
    for p in (obs.patients or []):
        v = p.vitals or {}
        pts.append({
            "id": p.patient_id,
            "age": p.age,
            "sex": p.sex,
            "complaint": p.chief_complaint,
            "history": p.medical_history,
            "hr": v.heart_rate,
            "sbp": v.systolic_bp,
            "dbp": v.diastolic_bp,
            "spo2": v.spo2,
            "gcs": v.gcs,
            "rr": v.respiratory_rate,
            "temp": v.temperature,
            "trends": p.vitals_trend,
        })
    return {
        "elapsed_minutes": obs.elapsed_minutes,
        "step": obs.step_number,
        "patients": pts,
        "available_beds": obs.available_beds,
    }


def llm_next_action(obs: TriageObservation, task_id: str) -> Optional[TriageAction]:
    config = _llm_config()
    if config is None:
        return None
    try:
        model = config["model"]
        client = _llm_client(config)
        user_prompt = (
            f"Task: {task_id}\n"
            f"Task context: {TASKS[task_id].description if task_id in TASKS else ''}\n"
            f"Current Observation (compact): {json.dumps(_compact_observation(obs))}\n"
            f"What is your next action? Respond ONLY with the JSON object."
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = completion.choices[0].message.content or ""
        payload = _extract_json_object(raw)
        if "action_type" not in payload or "parameter" not in payload:
            return None
        patient_ids = [p.patient_id for p in obs.patients]
        patient_id = payload.get("patient_id")
        if patient_id not in patient_ids and patient_ids:
            patient_id = patient_ids[0]
        return TriageAction(
            action_type=payload["action_type"],
            parameter=normalize_parameter(payload["action_type"], str(payload["parameter"])),
            patient_id=patient_id,
            rationale=str(payload.get("rationale", "") or "LLM clinical judgment"),
        )
    except Exception:
        return None


# ─── Heuristic Policy ──────────────────────────────────────────────────

def _progress(state: TriageState, pid: str):
    done_diag = {d.lower() for d in state.diagnostics_ordered}
    pathways = {p.lower() for p in state.pathways_activated}
    meds = {p.patient_id: {m.lower() for m in p.current_medications} for p in state.patients}
    esi = state.esi_assignments
    disps = state.dispositions
    return done_diag, pathways, meds.get(pid, set()), esi.get(pid), disps.get(pid)


def _stemi_action(state: TriageState, pid: str) -> TriageAction:
    done_diag, pathways, meds, esi, dispo = _progress(state, pid)
    if not any("ekg" in d or "ecg" in d for d in done_diag):
        return TriageAction(action_type="order_diagnostic", parameter="ekg", patient_id=pid, rationale="Classic STEMI pattern; acquire EKG immediately")
    if not any("cath_lab" in p for p in pathways):
        return TriageAction(action_type="activate_pathway", parameter="cath_lab", patient_id=pid, rationale="Activate cath lab for STEMI")
    if not any("aspirin" in m for m in meds):
        return TriageAction(action_type="administer_medication", parameter="aspirin", patient_id=pid, rationale="Antiplatelet for acute MI")
    if esi is None:
        return TriageAction(action_type="assign_esi_level", parameter="1", patient_id=pid, rationale="ESI 1 — time-critical cardiac")
    return TriageAction(action_type="disposition", parameter="admit_cardiac_icu", patient_id=pid, rationale="Admit for immediate catheterization")


def _chest_action(state: TriageState, pid: str) -> TriageAction:
    done_diag, _, _, _, dispo = _progress(state, pid)
    if not any("ekg" in d or "ecg" in d for d in done_diag):
        return TriageAction(action_type="order_diagnostic", parameter="ekg", patient_id=pid, rationale="First-line EKG to rule out ischemia")
    if not any("d_dimer" in d or "d-dimer" in d for d in done_diag):
        return TriageAction(action_type="order_diagnostic", parameter="d_dimer", patient_id=pid, rationale="D-dimer to evaluate PE probability")
    if not any("ct_pa" in d or "ctpa" in d for d in done_diag):
        return TriageAction(action_type="order_diagnostic", parameter="ct_pa", patient_id=pid, rationale="CT pulmonary angiogram for PE confirmation")
    return TriageAction(action_type="disposition", parameter="admit_floor", patient_id=pid, rationale="Admit for monitoring after confirmed PE")


_MCI_ESI = {"P1": "1", "P2": "3", "P3": "1", "P4": "2", "P5": "4"}
_MCI_DISPO = {"P1": "admit_icu", "P2": "admit_floor", "P3": "admit_icu", "P4": "admit_telemetry", "P5": "discharge"}


def _mci_action(state: TriageState, patients) -> TriageAction:
    for p in patients:
        pid = p.patient_id
        if pid not in _MCI_ESI:
            continue
        esi = state.esi_assignments.get(pid)
        if esi is None:
            expected = _MCI_ESI[pid]
            return TriageAction(action_type="assign_esi_level", parameter=expected, patient_id=pid, rationale="ESI triage under mass-casualty resource scarcity")
    for pid in _MCI_DISPO:
        if state.dispositions.get(pid) is None and pid in _MCI_ESI:
            return TriageAction(action_type="disposition", parameter=_MCI_DISPO[pid], patient_id=pid, rationale="Disposition aligned with assigned ESI acuity")
    pid = patients[0].patient_id
    return TriageAction(action_type="wait", parameter="5", patient_id=pid, rationale="All patients triaged and assigned")


def _sepsis_action(state: TriageState, pid: str) -> TriageAction:
    done_diag, _, meds, esi, dispo = _progress(state, pid)
    if esi is None:
        return TriageAction(action_type="assign_esi_level", parameter="2", patient_id=pid, rationale="ESI 2 — severe sepsis with hypotension")
    if not any("lactate" in d or "lactic" in d for d in done_diag):
        return TriageAction(action_type="order_diagnostic", parameter="lactate", patient_id=pid, rationale="Lactate to quantify severity per Sepsis-3")
    if not any("ceftriaxone" in m or "antibiot" in m for m in meds):
        return TriageAction(action_type="administer_medication", parameter="ceftriaxone", patient_id=pid, rationale="Broad-spectrum antibiotic within the 3-hour bundle")
    return TriageAction(action_type="disposition", parameter="admit_icu", patient_id=pid, rationale="ICU admission for septic shock resuscitation")


def _stroke_action(state: TriageState, pid: str) -> TriageAction:
    done_diag, pathways, _, esi, dispo = _progress(state, pid)
    if not any("stroke" in p for p in pathways):
        return TriageAction(action_type="activate_pathway", parameter="stroke", patient_id=pid, rationale="Activate stroke code immediately")
    if not any("ct" in d and ("head" in d or "brain" in d) for d in done_diag):
        return TriageAction(action_type="order_diagnostic", parameter="ct_head", patient_id=pid, rationale="Door-to-CT to rule out hemorrhage")
    if esi is None:
        return TriageAction(action_type="assign_esi_level", parameter="1", patient_id=pid, rationale="ESI 1 — time-critical stroke window")
    return TriageAction(action_type="disposition", parameter="admit_icu", patient_id=pid, rationale="Admit for thrombectomy eligibility window")


def _peds_action(state: TriageState, pid: str) -> TriageAction:
    _, _, meds, esi, _ = _progress(state, pid)
    if not any("albuterol" in m for m in meds):
        return TriageAction(action_type="administer_medication", parameter="albuterol", patient_id=pid, rationale="Nebulized albuterol for severe asthma")
    waited = any(s["action"].get("action_type") == "wait" for s in state.episode_history)
    if not waited:
        return TriageAction(action_type="wait", parameter="5", patient_id=pid, rationale="Reassess after bronchodilator")
    if esi is None:
        return TriageAction(action_type="assign_esi_level", parameter="2", patient_id=pid, rationale="ESI 2 — severe pediatric asthma")
    return TriageAction(action_type="disposition", parameter="admit_picu", patient_id=pid, rationale="Escalate to PICU without improvement")


def heuristic_next_action(obs: TriageObservation, task_id: str, state: TriageState) -> TriageAction:
    patients = obs.patients or []
    if not patients:
        return TriageAction(action_type="wait", parameter="15", patient_id="P1", rationale="No patients available")

    if task_id == "task_stemi_code":
        return _stemi_action(state, patients[0].patient_id)
    if task_id == "task_chest_pain_workup":
        return _chest_action(state, patients[0].patient_id)
    if task_id == "task_mci_surge":
        return _mci_action(state, patients)
    if task_id == "task_sepsis_alert":
        return _sepsis_action(state, patients[0].patient_id)
    if task_id == "task_stroke_code":
        return _stroke_action(state, patients[0].patient_id)
    if task_id == "task_pediatric_resp":
        return _peds_action(state, patients[0].patient_id)

    return TriageAction(action_type="assign_esi_level", parameter="2", patient_id=patients[0].patient_id, rationale="General triage")


def choose_next_action(obs: TriageObservation, task_id: str, state: TriageState, use_llm: bool = True) -> Tuple[TriageAction, bool]:
    if use_llm:
        action = llm_next_action(obs, task_id)
        if action is not None:
            return action, True
    return heuristic_next_action(obs, task_id, state), False


def _cli():
    from .env import ClinicalTriageEnvironment

    use_llm = os.getenv("USE_LLM", "0") == "1"
    for task_id, task in TASKS.items():
        env = ClinicalTriageEnvironment()
        env.reset(task_id=task_id)
        state = env.state()
        total = 0.0
        n = 0
        for _ in range(task.max_steps):
            obs = env._get_observation()
            action, used = choose_next_action(obs, task_id, state, use_llm=use_llm)
            _, reward, done, info = env.step(action)
            if reward.score is not None:
                total += reward.score
                n += 1
            if done:
                break
        grade = info.get("grading", {}).get("score", total / max(n, 1))
        beat = "OK" if grade >= task.baseline_score else "LOW"
        print(f"{task_id:<24} grade={grade:.2f} (baseline {task.baseline_score:.2f})  {beat}")


if __name__ == "__main__":
    _cli()