"""
logic.py — Helper logic for ClinicalTriageEnv: Vitals, Rewards, and Patients.
"""

import random
from typing import List, Dict, Tuple, Any, Set
from .models import (
    VitalSigns, LabResult, PatientState, TriageAction, TriageState, GradeResult
)

# ─── Constants ─────────────────────────────────────────────────────────

INDICATED_TESTS = {
    "task_stemi_code": {"troponin_i", "troponin", "cbc", "bmp", "bnp", "aspirin", "iv_access", "ekg", "ecg"},
    "task_chest_pain_workup": {"ekg", "ecg", "d_dimer", "ct_pa", "ctpa", "troponin_i", "cbc", "bmp", "cxr"},
    "task_mci_surge": {"ekg", "cbc", "bmp", "epinephrine", "iv_access"},
    "task_sepsis_alert": {"lactate", "lactic_acid", "cbc", "bmp", "procalcitonin", "blood_cultures", "cxr"},
    "task_stroke_code": {"ct_head_noncon", "ct_head", "cta", "cbc", "bmp", "coags", "glucose"},
    "task_pediatric_resp": {"vbg", "abg", "cxr", "cbc", "bmp"},
}

ESI_1_PATIENTS = {
    "task_stemi_code": {"P1"},
    "task_mci_surge": {"P1", "P3"},
}

TIME_CRITICAL_PATIENTS = {
    "task_stemi_code": {"P1"},
    "task_mci_surge": {"P1", "P3"},
    "task_sepsis_alert": {"P1"},
    "task_stroke_code": {"P1"},
    "task_pediatric_resp": {"P1"},
}

LAB_RESULTS = {
    "task_stemi_code": {
        "troponin_i": LabResult(name="troponin_i", value=2.8, unit="ng/mL", reference_range="<0.04", critical=True),
        "bnp": LabResult(name="bnp", value=450, unit="pg/mL", reference_range="<100", critical=True),
    },
    "task_chest_pain_workup": {
        "d_dimer": LabResult(name="d_dimer", value=1.8, unit="mg/L", reference_range="<0.5", critical=True),
    },
    "task_sepsis_alert": {
        "lactate": LabResult(name="lactate", value=4.5, unit="mmol/L", reference_range="0.5-2.2", critical=True),
    },
    "task_pediatric_resp": {
        "vbg": LabResult(name="vbg", value="pH 7.28", unit="", reference_range="7.35-7.45", critical=True),
    }
}

IMAGING_RESULTS = {
    "task_stemi_code": {"EKG": "Acute inferior STEMI.", "CXR": "Normal size heart."},
    "task_chest_pain_workup": {"EKG": "Normal sinus rhythm.", "CT_PA": "Bilateral pulmonary emboli."},
    "task_stroke_code": {"CT_HEAD_NONCON": "No hemorrhage.", "CTA_HEAD_NECK": "Left MCA occlusion."},
}

# ─── Time Costs ────────────────────────────────────────────────────────

def get_action_time_cost(action: TriageAction) -> int:
    at, param = action.action_type, action.parameter.lower()
    if at == "order_diagnostic":
        if "ct" in param or "pa" in param: return 45
        if "ekg" in param or "ecg" in param: return 5
        if "cxr" in param or "x-ray" in param: return 15
        return 30
    if at == "administer_medication": return 1 if "epi" in param else 5
    if at == "activate_pathway": return 2
    if at == "assign_esi_level": return 1
    if at == "disposition": return 5
    if at == "wait":
        try: return int(param)
        except (ValueError, TypeError): return 15
    return 1

# ─── Vitals Engine ─────────────────────────────────────────────────────

def update_vitals(patients: List[PatientState], dt: int) -> None:
    for p in patients:
        p.vitals_trend = {k: "→" for k in ["HR", "BP", "SpO2", "Temp", "RR", "GCS"]}
        cc = p.chief_complaint.lower()
        
        # Simple deterioration logic
        if "anaphylaxis" in cc or "struggle" in cc:
            if not any("epi" in m.lower() for m in p.current_medications):
                p.vitals.systolic_bp -= int(2 * dt)
                p.vitals.spo2 -= 0.5 * dt
                p.vitals_trend["BP"], p.vitals_trend["SpO2"] = "↓", "↓"
        elif "crushing chest pain" in cc:
            if not any("cath" in m.lower() for m in p.current_medications):
                p.vitals.heart_rate += int(0.5 * dt)
                p.vitals.systolic_bp -= int(0.5 * dt)
                p.vitals_trend["HR"], p.vitals_trend["BP"] = "↑", "↓"
        elif "fever" in cc:
            if not any("antibiotic" in m.lower() for m in p.current_medications):
                p.vitals.temperature += 0.05 * dt
                p.vitals_trend["Temp"] = "↑"

        # Clamp
        v = p.vitals
        v.heart_rate = max(0, min(220, v.heart_rate))
        v.systolic_bp = max(0, min(250, v.systolic_bp))
        v.spo2 = max(0.0, min(100.0, v.spo2))
        v.temperature = round(max(32.0, min(42.0, v.temperature)), 1)
        if v.systolic_bp < 60 or v.spo2 < 85:
            v.gcs = max(3, v.gcs - 1); p.vitals_trend["GCS"] = "↓"

# ─── Patient Generator ─────────────────────────────────────────────────

def generate_patients(task_id: str) -> List[PatientState]:
    rng = random.Random()
    
    def bp(h, s, d, r, sp, t, g): return VitalSigns(heart_rate=h, systolic_bp=s, diastolic_bp=d, respiratory_rate=r, spo2=sp, temperature=t, gcs=g)
    
    if task_id == "task_stemi_code":
        return [PatientState(patient_id="P1", age=58, sex="M", chief_complaint="Crushing chest pain, diaphoretic", onset_minutes=30, vitals=bp(102, 88, 60, 22, 94.0, 37.1, 15), medical_history=["smoker"])]
    elif task_id == "task_chest_pain_workup":
        return [PatientState(patient_id="P1", age=44, sex="F", chief_complaint="Sharp pleuritic chest pain after flight", onset_minutes=240, vitals=bp(98, 122, 78, 20, 96.0, 37.0, 15))]
    elif task_id == "task_mci_surge":
        return [
            PatientState(patient_id="P1", age=72, sex="M", chief_complaint="Unresponsive, bradycardic", onset_minutes=15, vitals=bp(40, 70, 40, 8, 85.0, 36.2, 6)),
            PatientState(patient_id="P2", age=28, sex="F", chief_complaint="Deformed left forearm", onset_minutes=45, vitals=bp(82, 118, 72, 16, 99.0, 36.8, 15)),
            PatientState(patient_id="P3", age=15, sex="M", chief_complaint="Anaphylaxis, peanuts, stridor", onset_minutes=10, vitals=bp(130, 70, 40, 28, 88.0, 37.4, 14)),
            PatientState(patient_id="P4", age=60, sex="M", chief_complaint="Palpitations, rate 148", onset_minutes=90, vitals=bp(148, 110, 68, 18, 97.0, 36.9, 15)),
            PatientState(patient_id="P5", age=35, sex="F", chief_complaint="Anxiety, tingling hands", onset_minutes=60, vitals=bp(90, 128, 80, 24, 100.0, 36.7, 15)),
        ]
    # Fallback to sepsis
    return [PatientState(patient_id="P1", age=68, sex="M", chief_complaint="Fever, confusion, low BP", onset_minutes=1440, vitals=bp(118, 82, 48, 26, 92.0, 39.2, 13))]

# ─── Reward Computation ───────────────────────────────────────────────

def compute_step_reward(action: TriageAction, state: TriageState, task_id: str) -> Tuple[float, Dict[str, float], str]:
    comp: Dict[str, float] = {}
    at, param = action.action_type, action.parameter.lower()
    
    # Correctness
    indicated = INDICATED_TESTS.get(task_id, set())
    if at == "order_diagnostic":
        comp["correctness"] = 0.15 if param in indicated else -0.05
    elif at in ("activate_pathway", "assign_esi_level", "disposition"):
        comp["correctness"] = 0.15
    
    # Time Pressure
    critical = TIME_CRITICAL_PATIENTS.get(task_id, set())
    time_pen = 0.0
    for pid in critical:
        if pid not in state.dispositions: time_pen -= 0.02
    comp["time_pressure"] = time_pen
    
    # Safety
    if at == "disposition" and "discharge" in param:
        if action.patient_id in ESI_1_PATIENTS.get(task_id, set()):
            comp["safety"] = -5.0
            
    total = max(-10.0, min(1.0, sum(comp.values())))
    return total, comp, f"Step reward: {total:+.2f}"
