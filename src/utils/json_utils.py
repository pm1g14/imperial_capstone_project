
import json
from pathlib import Path
import numpy as np
from typing import Any, Dict, Tuple


class JsonUtils:

    @staticmethod
    def get_last_turbo_state(function_number: int) -> Tuple[np.ndarray, np.ndarray]:
        project_root = Path(__file__).resolve().parent.parent.parent
        input_path = project_root / "resources" / f"function_{function_number}" / "turbo_state.json"
        try:
            with open(input_path, 'r') as f:
                # Load the entire content, which is expected to be a list of dictionaries
                data: Dict[str, Any] = json.load(f)
            
                if not data:
                    # Handle empty file/list
                    print(f"JSON file at {input_path} is empty.")
                    return None
                
                return data
        except json.JSONDecodeError:
                print(f"Error: JSON file at {input_path} is not valid JSON.")
                return None

    @staticmethod
    def save_turbo_state(turbo_state: Dict[str, Any], function_number: int) -> bool:
        if turbo_state:
            project_root = Path(__file__).resolve().parent.parent.parent
            try:
                input_path = project_root / "resources" / f"function_{function_number}" / "turbo_state.json"
                with open(input_path, 'w') as f:
                    json.dump(turbo_state, f)
                return True
            except Exception as e:
                print(f"Error: {e}")
                return False
        
        return False