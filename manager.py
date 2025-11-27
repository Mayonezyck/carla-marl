from config import ConfigLoader
from controlled_agent import Controlled_Agents
from free_agent import Free_Agents  # adjust import path as needed
from road_points import RoadPointExtractor

import math
import numpy as np
import carla
import torch
from typing import List, Dict, Any, Optional, Tuple, Sequence
import queue


# These should match your constants module
MAX_SPEED = 100.0
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
MIN_REL_GOAL_COORD = -1000.0
MAX_REL_GOAL_COORD = 1000.0


def normalize_min_max_np(x: float, min_val: float, max_val: float) -> float:
    """GPUDrive-style [-1, 1] normalization."""
    rng = max_val - min_val
    if rng <= 0:
        return 0.0
    return float(2.0 * (x - min_val) / rng - 1.0)


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
        carla_map = world.get_map()
        step_size = self.config.get_step_size()
        self.search_radius = self.config.get_search_radius()
        self.search_n_points = config.get_search_n_points()
        self.road_extractor = RoadPointExtractor(carla_map, step=step_size, observation_radius=self.search_radius)


        # Agent containers
        self.agents: List[Any] = []               # all agents (controlled + free)
        self.active_flags: List[bool] = []        # active mask, same length as agents
        self.controlled_agents: List[Controlled_Agents] = []
        self.sensorlessFreeAgent = self.config.get_if_free_agent_sensorless()
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
                if self.sensorlessFreeAgent:
                    agent = Free_Agents(self.world, i, self.config) #clean agents
                else:
                    agent_output_dir = f"output/free_agent_{i}"
                    agent = Free_Agents(
                        self.world,
                        index=i,
                        config=self.config,
                        with_sensors=True,
                        output_dir=agent_output_dir,
                    )

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
    
    def get_agent_roadgraph_obs(self, agent) -> torch.Tensor:
        """
        Build a (200, 13) GPUDrive-style roadgraph obs for this agent,
        in the agent's local frame.
        """
        vehicle = agent.vehicle  # or however you store the carla.Actor
        transform = vehicle.get_transform()
        ego_loc = transform.location
        ego_yaw_deg = transform.rotation.yaw

        points9 = self.road_extractor.get_local_points_from_precomputed_knearest(
            self.road_extractor.driving_wps,
            ego_loc,
            radius=self.search_radius,
            n_points=self.search_n_points,
            step=self.road_extractor.step,
            all_landmarks=self.road_extractor.all_landmarks,
            all_crosswalks=self.road_extractor.all_crosswalks,
        )

        rg_np = self.road_extractor.carla_points_to_gpudrive_roadgraph(
            points9,
            ego_location=ego_loc,
            ego_yaw_deg=ego_yaw_deg,
        )

        rg_tensor = torch.from_numpy(rg_np).float()
        return rg_tensor  # shape (N, 13)
    
    def get_agent_ego_obs(
        self,
        agent,
        goal_location: carla.Location,
    ) -> torch.Tensor:
        """
        Build a normalized ego feature vector for one agent:

            [speed_norm,
             veh_len_norm,
             veh_width_norm,
             rel_goal_x_norm,
             rel_goal_y_norm,
             collision_flag]

        - speed is |v| in m/s, divided by MAX_SPEED
        - length / width are from CARLA bounding box extents (full size), normalized
        - rel_goal_{x,y} are in the *ego frame* then mapped to [-1, 1]
          using MIN_REL_GOAL_COORD / MAX_REL_GOAL_COORD
        - collision_flag is 0.0 or 1.0 (no normalization)
        """
        vehicle = agent.vehicle
        if vehicle is None:
            # dead agent → return zeros
            return torch.zeros(6, dtype=torch.float32)

        # --- speed (m/s) ---
        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed_norm = speed / MAX_SPEED

        # --- vehicle size (full length / width) ---
        bbox = vehicle.bounding_box
        # CARLA extents are half-dimensions
        veh_len = 2.0 * float(bbox.extent.x)
        veh_width = 2.0 * float(bbox.extent.y)

        veh_len_norm = veh_len / MAX_VEH_LEN
        veh_width_norm = veh_width / MAX_VEH_WIDTH

        # --- relative goal in ego frame ---
        tf = vehicle.get_transform()
        ego_loc = tf.location
        ego_yaw_rad = math.radians(tf.rotation.yaw)

        # world diff
        dx = float(goal_location.x) - float(ego_loc.x)
        dy = float(goal_location.y) - float(ego_loc.y)

        cos_e = math.cos(ego_yaw_rad)
        sin_e = math.sin(ego_yaw_rad)

        # rotate into ego frame (x forward, y left)
        rel_x = cos_e * dx + sin_e * dy
        rel_y = -sin_e * dx + cos_e * dy

        rel_goal_x_norm = normalize_min_max_np(
            rel_x, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD
        )
        rel_goal_y_norm = normalize_min_max_np(
            rel_y, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD
        )

        # --- collision flag ---
        # reuse your existing logic; swap to whatever flag you like
        collided = 1.0 if agent.get_fatal_flag() else 0.0

        ego_vec = torch.tensor(
            [
                speed_norm,
                veh_len_norm,
                veh_width_norm,
                rel_goal_x_norm,
                rel_goal_y_norm,
                collided,
            ],
            dtype=torch.float32,
        )
        return ego_vec
    
    def get_neighbor_obs(self, agent) -> torch.Tensor:
        """
        Placeholder partner/neighbor observation.

        For now we mimic 'no neighbors' by returning a fixed-size
        zero vector of length 378, matching GPUDrive's partner feature size.
        """
        return torch.zeros(378, dtype=torch.float32)
    

    def get_gpudrive_style_obs(
        self,
        agent,
        goal_location: carla.Location,
    ) -> torch.Tensor:
        """
        Build a single flattened observation vector for one agent in
        'GPUDrive style':

            ego_obs:        (6,)
            neighbor_obs:   (378,)   # currently all zeros
            roadgraph_obs:  (N, 13)  # N ~ 200 → (2600,)

        Concatenated into a 1D tensor of length 2984:

            6 + 378 + (N * 13) == 2984  (assuming N == 200)
        """
        # 1) Ego (normalized)
        ego_vec = self.get_agent_ego_obs(agent, goal_location)       # (6,)

        # 2) Neighbors (placeholder)
        neighbor_vec = self.get_neighbor_obs(agent)                  # (378,)

        # 3) Roadgraph (normalized, local frame)
        rg_tensor = self.get_agent_roadgraph_obs(agent)              # (N, 13)
        rg_flat = rg_tensor.reshape(-1)                              # (N*13,)

        obs = torch.cat([ego_vec, neighbor_vec, rg_flat], dim=0)

        # Safety check: for N=200 this should be 2984 = 6 + 378 + 200*13
        expected_len = 6 + 378 + self.search_n_points * 13
        assert obs.shape[0] == expected_len, (
            f"Got obs len {obs.shape[0]}, expected {expected_len}. "
            f"(search_n_points={self.search_n_points})"
        )

        return obs

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
    def get_rewards_and_dones(self):
        """
        Compute rewards and done flags for each controlled agent.

        Current logic:
          - If agent collided this step:
                reward = -10.0, done = True, agent is removed from world.
          - If agent is already inactive (destroyed / no vehicle):
                done = True, reward = 0.0
          - Otherwise:
                reward = 0.0, done = False

        Returns
        -------
        rewards : np.ndarray, shape (num_controlled,)
        dones   : np.ndarray, shape (num_controlled,)
        """
        num_ctrl = len(self.controlled_agents)
        rewards = np.zeros(num_ctrl, dtype=np.float32)
        dones = np.zeros(num_ctrl, dtype=bool)

        for i, agent in enumerate(self.controlled_agents):
            # Default: alive, zero reward
            if agent is None or agent.vehicle is None:
                # Agent has no vehicle → treat as already done
                dones[i] = True
                continue

            # Map to global index to check active_flags
            idx_all = self.vehicle_to_index.get(agent.vehicle.id, None)
            if idx_all is None or not self.active_flags[idx_all]:
                # Not active in global list → done
                dones[i] = True
                continue

            # Collision-based termination
            if agent.get_fatal_flag():
                # Big negative reward on collision
                rewards[i] = -10.0
                dones[i] = True

                # Clear its collision state and remove from world
                self.remove_agent(agent)
            else:
                # Stay alive, zero reward for now
                rewards[i] = 0.0
                dones[i] = False
        print(rewards)
        print(dones)
        return rewards, dones

    # Visualize Path for each controlled
    def visualize_path(self):
        for agent in self.controlled_agents:
            print('drawing drawing')
            agent.draw_current_route_debug()

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




