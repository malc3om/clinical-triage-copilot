"""
env.py — Core environment implementation for ClinicalTriageEnv.
"""

import time
import uuid
from typing import Tuple, Dict, Any, Optional

from .models import (
    TriageAction, TriageObservation, TriageReward, TriageState,
    PatientState, TaskInfo, GradeResult
)
from .tasks import TASKS
from .graders import GRADERS
from .logic import (
    generate_patients, update_vitals, compute_step_reward,
    get_action_time_cost, LAB_RESULTS, IMAGING_RESULTS
)

class ClinicalTriageEnvironment:
    """
    Implements the agent environment contract: reset(), step(action), state().
    """
    
    def __init__(self):
        self.current_state: Optional[TriageState] = None
        self.task_id: Optional[str] = None
        
    def reset(self, task_id: str = "task_stemi_code") -> TriageObservation:
        """Initialize a new episode."""
        if task_id not in TASKS:
            task_id = "task_stemi_code"
            
        self.task_id = task_id
        patients = generate_patients(task_id)
        
        self.current_state = TriageState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            patients=patients,
            elapsed_minutes=0,
            diagnostics_ordered=[],
            pathways_activated=[],
            dispositions={},
            esi_assignments={},
            bed_assignments={},
            episode_history=[],
            done=False,
            total_reward=0.0
        )
        
        return self._get_observation()

    def step(self, action: TriageAction) -> Tuple[TriageObservation, TriageReward, bool, Dict[str, Any]]:
        """Execute one triage action."""
        if not self.current_state or self.current_state.done:
            raise RuntimeError("Episode is finished or not initialized. Call reset() first.")

        # 1. Update time and vitals
        dt = get_action_time_cost(action)
        self.current_state.elapsed_minutes += dt
        update_vitals(self.current_state.patients, dt)
        
        # 2. Record action
        step_idx = len(self.current_state.episode_history)
        
        # 3. Handle action specific updates (labs/imaging/dispo)
        at, param, pid = action.action_type, action.parameter.lower(), action.patient_id
        
        patient = next((p for p in self.current_state.patients if p.patient_id == pid), None)
        
        if patient:
            if at == "order_diagnostic":
                self.current_state.diagnostics_ordered.append(param)
                # Check for lab results in logic
                res = LAB_RESULTS.get(self.task_id, {}).get(param)
                if res and res not in patient.available_labs:
                    patient.available_labs.append(res)
                # Check imaging
                img = IMAGING_RESULTS.get(self.task_id, {}).get(param.upper())
                if img:
                    patient.chief_complaint += f" [IMAGING: {img}]"
            elif at == "activate_pathway":
                self.current_state.pathways_activated.append(param)
                patient.current_medications.append(f"PATHWAY_{param}")
            elif at == "assign_esi_level":
                try: self.current_state.esi_assignments[pid] = int(param)
                except (ValueError, TypeError): pass
            elif at == "disposition":
                self.current_state.dispositions[pid] = param
            elif at == "administer_medication":
                patient.current_medications.append(param)
        
        # 4. Compute Reward
        step_reward, comp, expl = compute_step_reward(action, self.current_state, self.task_id)
        self.current_state.total_reward += step_reward
        
        # 5. Check if done
        task_info = TASKS[self.task_id]
        all_disposed = len(self.current_state.dispositions) >= len(self.current_state.patients)
        max_steps_reached = step_idx + 1 >= task_info.max_steps
        
        self.current_state.done = all_disposed or max_steps_reached
        
        # 6. Record step in history
        self.current_state.episode_history.append({
            "step": step_idx,
            "action": action.model_dump(),
            "reward": step_reward,
            "explanation": expl
        })

        # 7. Final grading if done
        info = {"explanation": expl}
        if self.current_state.done:
            grader = GRADERS.get(self.task_id)
            if grader:
                final_grade = grader(self.current_state.episode_history)
                info["grading"] = {"score": final_grade}
                reward_obj = TriageReward(score=final_grade, reason=f"Final Grade: {final_grade:.2f}")
            else:
                reward_obj = TriageReward(score=0.0, reason="No grader found")
        else:
            reward_obj = TriageReward(score=step_reward, reason=expl)

        return self._get_observation(), reward_obj, self.current_state.done, info

    def state(self) -> TriageState:
        """Return full state snapshot."""
        if not self.current_state:
            raise RuntimeError("Environment not initialized.")
        return self.current_state

    def _get_observation(self) -> TriageObservation:
        """Filter state into agent-visible observation."""
        return TriageObservation(
            episode_id=self.current_state.episode_id,
            task_id=self.current_state.task_id,
            patients=self.current_state.patients,
            elapsed_minutes=self.current_state.elapsed_minutes,
            available_beds=TASKS[self.task_id].total_beds - len(self.current_state.bed_assignments),
            history_summary=[f"Step {s['step']}: {s['action']['action_type']} {s['action']['parameter']}" 
                            for s in self.current_state.episode_history[-5:]]
        )
