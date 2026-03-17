from __future__ import annotations

import numpy as np


def map_structured_action_to_fins(
    action: np.ndarray,
    trim_refs: np.ndarray,
    action_map: dict,
    ref_min: np.ndarray,
    ref_max: np.ndarray,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    refs = np.array(
        [
            trim_refs[0] + action_map["forward_flap"] * action[0],
            trim_refs[1] - action_map["turn_rot"] * action[1],
            trim_refs[2] + action_map["forward_flap"] * action[0],
            trim_refs[3] + action_map["turn_rot"] * action[1],
            trim_refs[4] + action_map["tail"] * action[2],
        ],
        dtype=np.float64,
    )
    return np.clip(refs, ref_min, ref_max)
