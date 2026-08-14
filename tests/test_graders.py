import pytest
from environment.graders import grade_stemi, grade_chest_workup, grade_mci, grade_sepsis, grade_pediatric, grade_stroke

def make_history(*actions):
    return [{"action": {"action_type": a[0], "parameter": a[1], "patient_id": a[2]}} for a in actions]

# --- STEMI Grader Tests ---
def test_stemi_perfect_score():
    hist = make_history(
        ("assign_esi_level", "1", "P1"),
        ("activate_pathway", "cath_lab", "P1"),
        ("order_diagnostic", "ekg", "P1"),
        ("order_diagnostic", "aspirin", "P1"),
        ("disposition", "admit", "P1")
    )
    res = grade_stemi(hist)
    assert 0.90 <= res <= 1.0

def test_stemi_fatal_violation():
    hist = make_history(
        ("order_diagnostic", "cbc", "P1"),
        ("wait", "5", "P1") # skipped cath lab activation
    )
    res = grade_stemi(hist)
    assert res <= 0.20

# --- Chest Pain Grader Tests ---
def test_chest_perfect_score():
    hist = make_history(
        ("assign_esi_level", "2", "P1"),
        ("order_diagnostic", "ekg", "P1"),
        ("order_diagnostic", "d_dimer", "P1"),
        ("order_diagnostic", "ct_pa", "P1"),
        ("disposition", "admit", "P1")
    )
    res = grade_chest_workup(hist)
    assert 0.80 <= res <= 1.0

def test_chest_fatal_violation():
    hist = make_history(
        ("disposition", "discharge", "P1")
    )
    res = grade_chest_workup(hist)
    assert res <= 0.20

# --- MCI Surge Grader Tests ---
def test_mci_perfect_score():
    hist = make_history(
        ("assign_esi_level", "1", "P1"),
        ("assign_esi_level", "3", "P2"),
        ("assign_esi_level", "1", "P3"),
        ("assign_esi_level", "2", "P4"),
        ("assign_esi_level", "4", "P5"),
        ("disposition", "admit", "P1"),
        ("disposition", "admit", "P3")
    )
    res = grade_mci(hist)
    assert 0.60 <= res <= 1.0 # MCI is complex, 0.8 is good baseline

# --- Sepsis Grader Tests ---
def test_sepsis_perfect_score():
    hist = make_history(
        ("assign_esi_level", "2", "P1"),
        ("order_diagnostic", "lactate", "P1"),
        ("order_diagnostic", "cultures", "P1"),
        ("administer_medication", "antibiotics", "P1"),
        ("disposition", "admit", "P1")
    )
    res = grade_sepsis(hist)
    assert 0.90 <= res <= 1.0

def test_sepsis_fatal_violation():
    hist = make_history(
        ("wait", "60", "P1"),
        ("disposition", "discharge", "P1")
    )
    res = grade_sepsis(hist)
    assert res <= 0.20

# --- Pediatric Grader Tests ---
def test_pediatric_perfect_score():
    hist = make_history(
        ("assign_esi_level", "2", "P1"),
        ("administer_medication", "albuterol", "P1"),
        ("wait", "10", "P1"),
        ("disposition", "admit", "P1")
    )
    res = grade_pediatric(hist)
    assert 0.90 <= res <= 1.0

# --- Stroke Grader Tests ---
def test_stroke_perfect_score():
    hist = make_history(
        ("assign_esi_level", "1", "P1"),
        ("activate_pathway", "stroke", "P1"),
        ("order_diagnostic", "ct_head", "P1"),
        ("disposition", "admit", "P1")
    )
    res = grade_stroke(hist)
    assert 0.90 <= res <= 1.0
