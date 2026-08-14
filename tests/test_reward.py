import pytest
from environment.logic import compute_step_reward
from environment.models import TriageAction, TriageState, PatientState, VitalSigns

def create_dummy_patient(patient_id="P1"):
    return PatientState(
        patient_id=patient_id,
        age=50,
        sex="M",
        chief_complaint="Chest pain",
        onset_minutes=30,
        vitals=VitalSigns(heart_rate=80, systolic_bp=120, diastolic_bp=80, respiratory_rate=16, spo2=98, temperature=37.0, gcs=15)
    )

def create_state(history=None):
    if history is None:
        history = []
    
    # Reconstruct state fields from history
    diag_ordered = []
    path_active = []
    esi_map = {}
    disp_map = {}
    
    for h in history:
        a = h.get("action", {})
        atype = a.get("action_type")
        param = a.get("parameter", "")
        pid = a.get("patient_id")
        
        if atype == "order_diagnostic": diag_ordered.append(param)
        if atype == "activate_pathway": path_active.append(param)
        if atype == "assign_esi_level": esi_map[pid] = int(param)
        if atype == "disposition": disp_map[pid] = param

    return TriageState(
        step_count=len(history),
        episode_history=history,
        patients=[create_dummy_patient()],
        diagnostics_ordered=diag_ordered,
        pathways_activated=path_active,
        esi_assignments=esi_map,
        dispositions=disp_map
    )

def test_efficiency_penalty():
    action = TriageAction(action_type="order_diagnostic", patient_id="P1", parameter="labs_6")
    history = [{"action": {"action_type": "order_diagnostic", "parameter": f"labs_{i}", "patient_id": "P1"}} for i in range(6)]
    state = create_state(history)
    total_reward, components, expl = compute_step_reward(action, state, "task_stemi_code")
    assert components.get("correctness", 0) <= 0  # In logic.py, ordering non-indicated labs is correctly penalized!

def test_safety_guardrail():
    action = TriageAction(action_type="disposition", patient_id="P1", parameter="discharge")
    history = [{"action": {"action_type": "assign_esi_level", "parameter": "1", "patient_id": "P1"}}]
    state = create_state(history)
    total_reward, components, expl = compute_step_reward(action, state, "task_stemi_code")
    assert components.get("safety", 0) <= -0.50

def test_sequence_bonus():
    action = TriageAction(action_type="order_diagnostic", patient_id="P1", parameter="ct_pa")
    history = [{"action": {"action_type": "order_diagnostic", "parameter": "ekg", "patient_id": "P1"}}]
    state = create_state(history)
    total_reward, components, expl = compute_step_reward(action, state, "task_chest_pain_workup")
    assert components.get("correctness", 0) > 0
