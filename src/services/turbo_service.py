from typing import Any, Dict, Union
from domain.domain_models import TurboState
from utils.json_utils import JsonUtils

class TurboService:

    def update_turbo_state(self, turbo_state: TurboState, y_new: float, function_number: int) -> TurboState:
        
        if turbo_state:
            improved = y_new > turbo_state.best_value

            if improved:
                turbo_state.best_value = y_new
                turbo_state.success_counter += turbo_state.batch_size
                turbo_state.failure_counter = 0
            else:
                turbo_state.failure_counter += turbo_state.batch_size
                turbo_state.success_counter = 0

            if turbo_state.success_counter >= turbo_state.success_tolerance:
                turbo_state.length = min(2.0 * turbo_state.length, turbo_state.length_max)
                turbo_state.success_counter = 0

            # Shrink on consecutive failures
            if turbo_state.failure_counter >= turbo_state.failure_tolerance:
                turbo_state.length = max(0.5 * turbo_state.length, turbo_state.length_min)
                turbo_state.failure_counter = 0

            # Trigger restart if TR has become too small
            turbo_state.restart_triggered = turbo_state.length <= turbo_state.length_min + 1e-12
            
            if turbo_state.restart_triggered:
                turbo_state.length = 0.8
                turbo_state.success_counter = 0
                turbo_state.failure_counter = 0
                turbo_state.restart_triggered = False
            JsonUtils.save_turbo_state(turbo_state.__dict__, function_number)
            return turbo_state


    def load_turbo_state(self, function_number: int) -> Union[TurboState, None]:
        turbo_state_dict = JsonUtils.get_last_turbo_state(function_number)
        return self._map_to_turbo_state(turbo_state_dict)

    def _map_to_turbo_state(self, turbo_state_dict: Dict[str, Any]) -> Union[TurboState, None]:
        try:
            if turbo_state_dict:
                turbo_state = TurboState(
                    dim=turbo_state_dict['dim'],
                    batch_size=turbo_state_dict['batch_size'],
                    length=turbo_state_dict['length'],
                    length_min=turbo_state_dict['length_min'],
                    length_max=turbo_state_dict['length_max'],
                    success_counter=turbo_state_dict['success_counter'],
                    failure_counter=turbo_state_dict['failure_counter'],
                    success_tolerance=turbo_state_dict['success_tolerance'],
                    failure_tolerance=turbo_state_dict['failure_tolerance'],
                    best_value=turbo_state_dict['best_value'],
                    restart_triggered=turbo_state_dict['restart_triggered']
                )
                return turbo_state
        except Exception as e:
            print(f"Error: {e}")
            return None