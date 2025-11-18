from config import ConfigLoader
from controlled_agent import Controlled_Agents
from free_agent import Free_Agents  # adjust import path as needed

import random
import numpy as np
import carla
from typing import List, Dict, Any, Optional, Tuple, Sequence
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

    def _process_lidar_measurement(
        self,
        lidar_measurement: Optional[carla.LidarMeasurement],
        target_n: int = 16000,
    ) -> np.ndarray:
        """
        Convert a raw CARLA LidarMeasurement into a fixed-size (target_n, 3) array.

        Typical RL preprocessing:
          - Convert raw buffer → (N, 3) XYZ points.
          - If N > target_n: random downsample.
          - If N < target_n: zero-pad at the end.

        Returns
        -------
        obs : np.ndarray
            Shape (target_n, 3), dtype float32.
        """
        # If no measurement yet, just return zeros
        if lidar_measurement is None:
            return np.zeros((target_n, 3), dtype=np.float32)

        # raw_data is float32 [x, y, z, intensity] for each point
        pts = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)
        pts = pts.reshape(-1, 4)[:, :3]  # (N, 3) → drop intensity

        n_points = pts.shape[0]

        if n_points >= target_n:
            # Randomly sample target_n points (no replacement)
            idx = np.random.choice(n_points, target_n, replace=False)
            obs = pts[idx]
        else:
            # Pad with zeros to reach target_n
            pad = np.zeros((target_n - n_points, 3), dtype=np.float32)
            obs = np.vstack([pts, pad])

        return obs.astype(np.float32)
    
    def apply_actions_to_controlled(self, actions: Sequence[Sequence[float]]) -> None:
        """
        Apply actions to all controlled agents.

        actions: iterable of (throttle, steer, brake) for each controlled agent,
                 in the same order as self.controlled_agents.
        """
        if len(actions) != len(self.controlled_agents):
            print(
                "[Manager] Warning: got {} actions for {} controlled agents.".format(
                    len(actions), len(self.controlled_agents)
                )
            )

        for agent, act in zip(self.controlled_agents, actions):
            if agent is None or agent.vehicle is None:
                continue
            try:
                #print(f'now applying {act} to {agent}')
                agent.apply_action(act)
            except Exception as e:
                print("[Manager] Error applying action to agent {}: {}".format(agent.index, e))


    def get_controlled_lidar_observations(
        self,
        target_n: int = 16000,
    ) -> Tuple[List[Optional[int]], np.ndarray]:
        """
        Collect latest lidar observations from all controlled agents and
        convert them into a fixed-size batch.
        """
        frames: List[Optional[int]] = []
        obs_list: List[np.ndarray] = []

        for agent in self.controlled_agents:
            latest = agent.get_lidar_latest()
            if latest is None:
                frame_id = None
                lidar_meas = None
            else:
                frame_id, lidar_meas = latest

            frames.append(frame_id)
            obs = self._process_lidar_measurement(lidar_meas, target_n=target_n)
            obs_list.append(obs)

        if len(obs_list) == 0:
            return [], np.zeros((0, target_n, 3), dtype=np.float32)

        batch_obs = np.stack(obs_list, axis=0)
        return frames, batch_obs
    
    # Compute reward and done for each controlled vehicle



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




