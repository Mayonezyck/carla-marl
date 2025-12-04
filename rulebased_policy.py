# rule_based_policy.py

import numpy as np


class RuleBasedPolicy:
    """
    Placeholder rule-based policy.

    IMPORTANT:
    - It will get obs['gnss'] and obs['waypoints_gnss'], where:
        obs['gnss'] = (lat, lon, alt)
        obs['waypoints_gnss'] = list of (lat, lon, alt) tuples
    - For now we ignore them and return a fixed action.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def reset(self):
        """Call this at episode start if you later keep internal state."""
        pass

    def act(self, obs=None, deterministic=True) -> np.ndarray:
        """
        obs: dict with GNSS info, e.g.
            {
                "gnss": (lat, lon, alt),
                "waypoints_gnss": [(lat1, lon1, alt1), ...]
            }

        Returns:
            np.ndarray of shape (1, 2) in [-1, 1]:
                [:, 0] = steer
                [:, 1] = accel (positive=gas, negative=brake)
        """
        # --- later you’ll use obs['gnss'] and obs['waypoints_gnss'] here ---
        steer = 0.0          # straight
        accel = 0.3          # mild throttle

        action = np.array([[steer, accel]], dtype=np.float32)
        return action
