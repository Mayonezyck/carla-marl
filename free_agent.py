from agents import Agent
from typing import Dict, Any, List, Optional
import carla


class Free_Agents(Agent):
    """
    Free agents:
    - Just spawn a vehicle (via Agent).
    - Enable autopilot.
    - No sensors (unless you decide to add some later).
    """

    def __init__(self, world: carla.World, index: int, config: Dict[str, Any]):
        super().__init__(world, index, config, role_prefix="free_agent")
        self._enable_autopilot()

    def _enable_autopilot(self) -> None:
        if self.vehicle is not None:
            # Simple autopilot; if you want to use a Traffic Manager,
            # you can hook it up here with vehicle.set_autopilot(True, tm_port)
            self.vehicle.set_autopilot(True)
            print(f"[Free_Agents] Enabled autopilot for free_agent_{self.index}")