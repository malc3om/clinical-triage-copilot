"""
graders.py — Consolidated deterministic graders for ClinicalTriageEnv.
"""

from __future__ import annotations
from typing import List, Dict
from .models import GradeResult

# ─── Ground Truth Data ──────────────────────────────────────────────────

CORRECT_ESI_MCI = {
    "P1": 1,  # 72yo, unresponsive, GCS 6 → immediate
    "P2": 3,  # 28yo, broken arm → delayed
    "P3": 1,  # 15yo, anaphylaxis → immediate
    "P4": 2,  # 60yo, rapid afib → urgent
    "P5": 4,  # 35yo, anxiety → non-urgent
}

# ─── Grader Implementations ─────────────────────────────────────────────

def grade_stemi(episode_history: List[dict]) -> float:
    """Grade a completed STEMI Code episode."""
    score = 0.0
    actions = [step["action"] for step in episode_history]

    # ESI assigned correctly → +0.25
    esi_actions = [a for a in actions if a.get("action_type") == "assign_esi_level"]
    if any(str(a.get("parameter", "")).strip() == "1" for a in esi_actions):
        score += 0.25

    # Cath lab activated → +0.30
    pathway_actions = [a for a in actions if a.get("action_type") == "activate_pathway"]
    if any("cath_lab" in str(a.get("parameter", "")).lower() for a in pathway_actions):
        score += 0.30

    # Correct disposition: admit → +0.25
    disp_actions = [a for a in actions if a.get("action_type") == "disposition"]
    if any("admit" in str(a.get("parameter", "")).lower() for a in disp_actions):
        score += 0.25

    # Time penalty
    steps_taken = len(episode_history)
    if steps_taken > 4:
        penalty = -0.05 * ((steps_taken - 4) // 2)
        score += penalty

    # Bonus: aspirin ordered → +0.10
    diag_actions = [a for a in actions if a.get("action_type") == "order_diagnostic"]
    if any("aspirin" in str(a.get("parameter", "")).lower() for a in diag_actions):
        score += 0.10

    return max(0.01, min(0.99, score))


def grade_chest_workup(episode_history: List[dict]) -> float:
    """Grade a completed Chest Pain Workup episode."""
    score = 0.0
    actions = [s["action"] for s in episode_history]
    params = [str(a.get("parameter", "")).lower() for a in actions]

    ekg_idx = next((i for i, p in enumerate(params) if "ekg" in p or "ecg" in p), None)
    ctpa_idx = next((i for i, p in enumerate(params) if "ctpa" in p or ("ct" in p and "pa" in p)), None)

    if ekg_idx is not None:
        score += 0.20
        if ctpa_idx is not None and ekg_idx < ctpa_idx:
            score += 0.05

    if any("d_dimer" in p or "d-dimer" in p for p in params):
        score += 0.15

    if ctpa_idx is not None:
        score += 0.20
        if ekg_idx is None or ctpa_idx < ekg_idx:
            score -= 0.15

    disp_actions = [a for a in actions if a.get("action_type") == "disposition"]
    if any("admit" in str(a.get("parameter", "")).lower() for a in disp_actions):
        score += 0.25
    if any("discharge" in str(a.get("parameter", "")).lower() for a in disp_actions):
        score -= 0.30

    return max(0.01, min(0.99, score))


def grade_mci(episode_history: List[dict]) -> float:
    """Grade a completed MCI Surge episode."""
    score = 0.0
    esi_assignments: Dict[str, int] = {}
    disposition_map: Dict[str, str] = {}

    for step in episode_history:
        a = step["action"]
        pid, atype = str(a.get("patient_id", "")), a.get("action_type", "")
        if atype == "assign_esi_level":
            try: esi_assignments[pid] = int(a.get("parameter", "0"))
            except (ValueError, TypeError): pass
        if atype == "disposition":
            disposition_map[pid] = str(a.get("parameter", "")).lower()

    for pid, expected in CORRECT_ESI_MCI.items():
        assigned = esi_assignments.get(pid)
        if assigned == expected:
            score += 0.10
        elif assigned and abs(assigned - expected) == 1:
            score += 0.05

    immediate_admits = [pid for pid, disp in disposition_map.items() if "admit" in disp and pid in ("P1", "P3")]
    score += 0.10 * len(immediate_admits)

    p5_disp = disposition_map.get("P5", "")
    if p5_disp and "admit" not in p5_disp:
        score += 0.10
    elif "admit" in p5_disp:
        if "P1" not in disposition_map or "admit" not in disposition_map["P1"]: score -= 0.20
        if "P3" not in disposition_map or "admit" not in disposition_map["P3"]: score -= 0.20

    return max(0.01, min(0.99, score))


def grade_sepsis(history: list[dict]) -> float:
    """Grader for task_sepsis_alert."""
    score = 0.0
    ordered_lactate, meds_given, esi_correct, disposition_admit = False, False, False, False

    for step in history:
        a = step.get("action", {})
        atype, param = a.get("action_type"), a.get("parameter", "").lower()
        if atype == "order_diagnostic" and ("lactate" in param or "lactic" in param or "cultures" in param):
            if not ordered_lactate: ordered_lactate = True; score += 0.2
        if atype == "administer_medication" and ("fluid" in param or "antibiotic" in param or "ceftriaxone" in param):
            if not meds_given: meds_given = True; score += 0.3
        if atype == "assign_esi_level" and not esi_correct:
            try:
                if int(param) == 2: esi_correct = True; score += 0.2
                elif int(param) > 2: score -= 0.2
            except (ValueError, TypeError): pass
        if atype == "disposition" and ("admit" in param or "icu" in param) and not disposition_admit:
            disposition_admit = True; score += 0.3
        if atype == "disposition" and "discharge" in param: score -= 0.5

    return max(0.01, min(0.99, score))


def grade_stroke(history: list[dict]) -> float:
    """Grader for task_stroke_code."""
    score = 0.0
    code, ct, esi, dispo = False, False, False, False

    for step in history:
        a = step.get("action", {})
        atype, param = a.get("action_type"), a.get("parameter", "").lower()
        if atype == "activate_pathway" and "stroke" in param and not code:
            code = True; score += 0.3
        if atype == "order_diagnostic" and "ct" in param and ("head" in param or "brain" in param) and not ct:
            ct = True; score += 0.3
        if atype == "assign_esi_level" and not esi:
            try:
                if int(param) in [1, 2]: esi = True; score += 0.2
                else: score -= 0.2
            except (ValueError, TypeError): pass
        if atype == "disposition" and ("admit" in param or "transfer" in param) and not dispo:
            dispo = True; score += 0.2
        if atype == "disposition" and "discharge" in param: score -= 1.0

    return max(0.01, min(0.99, score))


def grade_pediatric(history: list[dict]) -> float:
    """Grader for task_pediatric_resp."""
    score = 0.0
    meds, esi, dispo, wait = False, False, False, False

    for step in history:
        a = step.get("action", {})
        atype, param = a.get("action_type"), a.get("parameter", "").lower()
        if atype == "administer_medication" and ("albuterol" in param or "steroid" in param) and not meds:
            meds = True; score += 0.3
        if atype == "wait" and meds and not wait:
            wait = True; score += 0.2
        if atype == "assign_esi_level" and not esi:
            try:
                if int(param) == 2: esi = True; score += 0.2
                elif int(param) > 2: score -= 0.2
            except (ValueError, TypeError): pass
        if atype == "disposition" and ("admit" in param or "picu" in param) and not dispo:
            dispo = True; score += 0.3
        if atype == "disposition" and "discharge" in param: score -= 0.5

    return max(0.01, min(0.99, score))


# ─── Grader Registry ────────────────────────────────────────────────────

GRADERS = {
    "task_stemi_code": grade_stemi,
    "task_chest_pain_workup": grade_chest_workup,
    "task_mci_surge": grade_mci,
    "task_sepsis_alert": grade_sepsis,
    "task_stroke_code": grade_stroke,
    "task_pediatric_resp": grade_pediatric,
}
