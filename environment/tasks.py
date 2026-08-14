"""
tasks.py — Clinical Triage Task definitions.
"""
from typing import Dict
from .models import TaskInfo

TASKS: Dict[str, TaskInfo] = {
    "task_stemi_code": TaskInfo(
        id="task_stemi_code",
        name="STEMI Code",
        difficulty="easy",
        description="Clear STEMI presentation. Agent must activate cath lab pathway within time window.",
        max_steps=15,
        baseline_score=0.72,
    ),
    "task_chest_pain_workup": TaskInfo(
        id="task_chest_pain_workup",
        name="Chest Pain Workup",
        difficulty="medium",
        description="Ambiguous chest pain. Agent must navigate differential diagnosis with ordered test sequencing.",
        max_steps=20,
        baseline_score=0.48,
    ),
    "task_mci_surge": TaskInfo(
        id="task_mci_surge",
        name="Mass Casualty Surge",
        difficulty="hard",
        description="Mass casualty: 5 simultaneous patients, 3 beds. Agent must correctly triage under scarcity.",
        max_steps=25,
        baseline_score=0.31,
        total_beds=3,
    ),
    "task_sepsis_alert": TaskInfo(
        id="task_sepsis_alert",
        name="Sepsis Alert",
        difficulty="medium",
        description="68yo with fever, hypotension, and altered mental status. Agent must recognize and respond to severe sepsis.",
        max_steps=20,
        baseline_score=0.60,
    ),
    "task_stroke_code": TaskInfo(
        id="task_stroke_code",
        name="Stroke Code",
        difficulty="hard",
        description="72yo with sudden onset facial droop and right-sided weakness. Time critical.",
        max_steps=18,
        baseline_score=0.55,
    ),
    "task_pediatric_resp": TaskInfo(
        id="task_pediatric_resp",
        name="Pediatric Respiratory",
        difficulty="medium",
        description="4yo with severe asthma exacerbation and retractions.",
        max_steps=18,
        baseline_score=0.65,
    ),
}
