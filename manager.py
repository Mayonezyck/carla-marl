from config import ConfigLoader
from controlled_agent import Controlled_Agents
from free_agent import Free_Agents  # adjust import path as needed

import random
import carla
from typing import List, Dict, Any, Optional
import queue


class Manager:
    def __init__(self, config: Dict[str, Any], world: carla.World):
        """
        The manager handles both the spawning of the controlled and free agents.
        According to the config, it will generate n controlled agents and m free agents.
        It keeps:
            - A list of all agents (controlled + free)
            - A parallel list of "active" flags
            - Separate lists of controlled and free agents
        """
        self.config = config
        self.world = world

        # Agent containers
        self.agents: List[Any] = []               # all agents (controlled + free)
        self.active_flags: List[bool] = []        # active mask, same length as agents
        self.controlled_agents: List[Controlled_Agents] = []
        self.free_agents: List[Free_Agents] = []
        

        # Map vehicle.id -> index in self.agents for quick lookup
        self.vehicle_to_index: Dict[int, int] = {}
        

        # Read counts from config
        nc, nf = config.get_headcounts()

        # Spawn everything
        self.spawn_both_controlled_and_free(nc, nf)

    # --------------------------------------------------
    # Spawning
    # --------------------------------------------------
    def spawn_both_controlled_and_free(self, nc: int, nf: int) -> None:
        """
        Spawn nc controlled agents and nf free agents.
        Ensure the actual successfully generated ones match the required count
        (or raise if impossible).
        """
        # Spawn controlled agents
        for i in range(nc):
            try:
                agent = Controlled_Agents(self.world, i, self.config)
            except Exception as e:
                print(f"[Manager] Failed to spawn controlled agent {i}: {e}")
                raise

            self._register_agent(agent, is_controlled=True)

        # Spawn free agents
        for i in range(nf):
            try:
                agent = Free_Agents(self.world, i, self.config)
            except Exception as e:
                print(f"[Manager] Failed to spawn free agent {i}: {e}")
                raise

            self._register_agent(agent, is_controlled=False)

        print(
            f"[Manager] Spawned {len(self.controlled_agents)} controlled agents "
            f"and {len(self.free_agents)} free agents."
        )

    def _register_agent(self, agent: Any, is_controlled: bool) -> None:
        """
        Internal helper to add an agent into all tracking structures.
        """
        self.agents.append(agent)
        self.active_flags.append(True)

        if agent.vehicle is not None:
            self.vehicle_to_index[agent.vehicle.id] = len(self.agents) - 1

        if is_controlled:
            self.controlled_agents.append(agent)
        else:
            self.free_agents.append(agent)

    # --------------------------------------------------
    # Cleanup & removal
    # --------------------------------------------------
    def cleanup(self) -> None:
        """
        Destroy all vehicles and sensors (by calling destroy_agent() of each agent),
        and mark them inactive.
        """
        print("[Manager] Cleaning up all agents...")
        for idx, agent in enumerate(self.agents):
            if agent is None:
                continue
            if self.active_flags[idx]:
                try:
                    agent.destroy_agent()
                except Exception as e:
                    print(f"[Manager] Error destroying agent {idx}: {e}")
                self.active_flags[idx] = False

        self.vehicle_to_index.clear()
        print("[Manager] Cleanup complete.")

    def remove_vehicle(self, vehicle: carla.Actor) -> None:
        """
        When called, remove the specific vehicle and its sensors from the world,
        also mark the vehicle as inactive.
        """
        if vehicle is None:
            print("[Manager] remove_vehicle called with None.")
            return

        idx = self.vehicle_to_index.get(vehicle.id, None)
        if idx is None:
            print(f"[Manager] Vehicle id {vehicle.id} not managed by this Manager.")
            return

        agent = self.agents[idx]
        if agent is None:
            print(f"[Manager] No agent found for index {idx}.")
            return

        print(f"[Manager] Removing vehicle id {vehicle.id} (agent index {idx})...")
        try:
            agent.destroy_agent()
        except Exception as e:
            print(f"[Manager] Error destroying agent {idx}: {e}")

        self.active_flags[idx] = False
        # Remove from vehicle lookup map
        self.vehicle_to_index.pop(vehicle.id, None)

        # Note: we don't physically remove the agent object from the lists,
        # just mark it inactive. That way indices stay stable.

    # Optional: convenience to remove by agent rather than by vehicle
    def remove_agent(self, agent: Any) -> None:
        if agent is None:
            return
        vehicle = getattr(agent, "vehicle", None)
        if vehicle is not None:
            self.remove_vehicle(vehicle)



