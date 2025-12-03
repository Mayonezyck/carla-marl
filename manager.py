from config import ConfigLoader
from controlled_agent import Controlled_Agents
from free_agent import Free_Agents  # adjust import path as needed
from road_points import RoadPointExtractor
from RL_handler import SEG_DEPTH_H, SEG_DEPTH_W, CMPE_OBS_DIM


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
MIN_REL_GOAL_COORD = -100.0
MAX_REL_GOAL_COORD = 100.0


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
        
        # Reward Stage 
        self.reward_stage = config.get_reward_stage()

        # Map vehicle.id -> index in self.agents for quick lookup
        self.vehicle_to_index: Dict[int, int] = {}
        

        # Read counts from config
        nc, nf = config.get_headcounts()
        # Spawn everything
        self.spawn_both_controlled_and_free(nc, nf)

        # ---------- Reward debugging stats ----------
        # Sums over all agents and steps between resets.
        self.reward_stats = {
            "steps": 0.0,
            "progress": 0.0,
            "backward": 0.0,
            "dist_pen": 0.0,
            "along": 0.0,
            "back_from_wp": 0.0,
            "lat_pen": 0.0,
            "idle_pen": 0.0,
            "speed_pen": 0.0,
            "steer_pen": 0.0,
            "steer_sustain": 0.0,   
            "intermediate": 0.0,
            "final": 0.0,
            "collision": 0.0,
            "lane_soft": 0.0,
            "lane_fatal": 0.0,
            "time_pen": 0.0,
        }



    # ---------- Reward debug helpers ----------

    def _accum_reward(self, key: str, value: float) -> None:
        """Accumulate weighted reward contribution by term."""
        if key in self.reward_stats:
            self.reward_stats[key] += float(value)
        else:
            # Just in case of typos, don't crash
            pass

    def print_and_reset_reward_stats(self) -> None:
        """
        Print per-step average contribution for each reward term,
        then reset stats. Call this once per rollout.
        """
        stats = self.reward_stats
        steps = max(int(stats.get("steps", 0.0)), 1)

        print("\n[REWARD STATS] Per-step averages over last window:")
        for k, v in stats.items():
            if k == "steps":
                continue
            avg = v / steps
            print(f"  {k:14s}: {avg:8.4f}")

        # Reset
        for k in stats.keys():
            self.reward_stats[k] = 0.0
        self.reward_stats["steps"] = 0.0

    def _segment_offset(self,
                        point: carla.Location,
                        start: carla.Location,
                        end: carla.Location) -> float:
        """
        Distance (in meters) from `point` to the line *segment* joining
        `start` and `end` (all carla.Location). Uses 2D (x, y).
        """
        ax, ay = float(start.x), float(start.y)
        bx, by = float(end.x), float(end.y)
        px, py = float(point.x), float(point.y)

        abx = bx - ax
        aby = by - ay
        ab_len2 = abx * abx + aby * aby
        if ab_len2 < 1e-6:
            # Degenerate segment → treat offset as 0
            return 0.0

        apx = px - ax
        apy = py - ay

        # Projection of AP onto AB, normalized by |AB|^2
        t = (apx * abx + apy * aby) / ab_len2
        # Clamp to segment [0, 1]
        t = max(0.0, min(1.0, t))

        cx = ax + t * abx
        cy = ay + t * aby

        dx = px - cx
        dy = py - cy
        return float(math.sqrt(dx * dx + dy * dy))


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
    
    def get_current_goal_location(self, agent) -> carla.Location:
        """
        For a given agent, return the current goal location:

          - If the agent has a route (current_waypoint_plan) and a valid
            current_wp_index, use that waypoint as the current intermediate goal.
          - Otherwise, fall back to agent.destination.

        Additionally:
          - If the current waypoint is *behind* the vehicle in the ego frame,
            we advance current_wp_index until we find one that is in front
            (rel_x >= 0), or fall back to the final destination.
        """
        vehicle = getattr(agent, "vehicle", None)
        if vehicle is None:
            # Fallback: arbitrary location
            return self.world.get_map().get_spawn_points()[0].location

        tf = vehicle.get_transform()
        loc = tf.location
        default_goal = getattr(agent, "destination", loc)

        route = getattr(agent, "current_waypoint_plan", None)
        wp_idx = int(getattr(agent, "current_wp_index", 0))

        if route is None or len(route) == 0:
            return default_goal

        if wp_idx < 0 or wp_idx >= len(route):
            # Index out of range → use final dest
            return default_goal

        # --- new logic: skip waypoints that are behind ego in ego frame ---
        yaw_rad = math.radians(tf.rotation.yaw)
        cos_e = math.cos(yaw_rad)
        sin_e = math.sin(yaw_rad)

        # how far "in front" we require (meters); 0.0 = strictly not behind
        FORWARD_THRESH = 0.0

        k = wp_idx
        last_valid_idx = len(route) - 1
        chosen_loc = None

        while k <= last_valid_idx:
            wp, _ = route[k]
            goal_loc = wp.transform.location

            dx = float(goal_loc.x) - float(loc.x)
            dy = float(goal_loc.y) - float(loc.y)

            # ego frame: x forward, y left
            rel_x = cos_e * dx + sin_e * dy
            # rel_y = -sin_e * dx + cos_e * dy  # not needed here

            if rel_x >= FORWARD_THRESH:
                # found a waypoint that is at or ahead of the vehicle
                chosen_loc = goal_loc
                break
            else:
                # this waypoint is behind → skip it
                k += 1

        if chosen_loc is None:
            # All route waypoints are behind ego: just use final destination
            return default_goal

        # Update current_wp_index so reward logic uses the same waypoint
        agent.current_wp_index = k
        return chosen_loc



    def get_agent_cmpe_style_obs(self, agent) -> np.ndarray:
        """
        CMPE observation for one agent.

        Core 10D vector:
            [speed_norm,
            heading_err_norm,
            rel_goal_x_norm,        # goal in ego frame: forward
            rel_goal_y_norm,        # goal in ego frame: left/right
            collision_flag,
            lane_inv_flag,
            traffic_light_red_flag,
            at_junction_flag,
            throttle_prev,
            steer_prev]

        Then we append:
            - flattened semantic segmentation (128x128), normalized to [0, 1]
            using 13 semantic classes
            - flattened depth (128x128), clipped to [0, 100] m and normalized to [0, 1]

        Final shape: (CMPE_OBS_DIM,) == 10 + 2 * 128 * 128
        """
        # If agent/vehicle is gone → return all zeros
        if agent is None or getattr(agent, "vehicle", None) is None:
            return np.zeros(CMPE_OBS_DIM, dtype=np.float32)

        vehicle: carla.Vehicle = agent.vehicle
        world = self.world
        amap = world.get_map()

        # ------------------------------------------------------------------
        # 1) Speed (normalize by 30 m/s)
        # ------------------------------------------------------------------
        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed_norm = float(np.clip(speed / 30.0, 0.0, 1.0))

        # ------------------------------------------------------------------
        # 2) Heading error vs lane direction (in [-1, 1])
        # ------------------------------------------------------------------
        tf = vehicle.get_transform()
        loc = tf.location
        yaw_rad = math.radians(tf.rotation.yaw)

        wp = amap.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        lane_yaw_rad = math.radians(wp.transform.rotation.yaw)
        heading_err = yaw_rad - lane_yaw_rad
        # wrap to [-pi, pi]
        while heading_err > math.pi:
            heading_err -= 2.0 * math.pi
        while heading_err < -math.pi:
            heading_err += 2.0 * math.pi
        heading_err_norm = float(heading_err / math.pi)  # [-1, 1]

        # ------------------------------------------------------------------
        # 3) Relative goal in ego frame (rel_x, rel_y), both in [-1, 1]
        #    IMPORTANT: use the same current goal as the reward function
        # ------------------------------------------------------------------
        goal_location: carla.Location = self.get_current_goal_location(agent)

        # world-frame difference
        dx = float(goal_location.x) - float(loc.x)
        dy = float(goal_location.y) - float(loc.y)

        cos_e = math.cos(yaw_rad)
        sin_e = math.sin(yaw_rad)

        # rotate into ego frame: x forward, y left
        rel_x = cos_e * dx + sin_e * dy
        rel_y = -sin_e * dx + cos_e * dy

        # Normalize to [-1, 1] using global bounds
        rel_goal_x_norm = normalize_min_max_np(
            rel_x, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD
        )
        rel_goal_y_norm = normalize_min_max_np(
            rel_y, MIN_REL_GOAL_COORD, MAX_REL_GOAL_COORD
        )

        # ------------------------------------------------------------------
        # 5) Flags: collision / lane invasion / traffic light / junction
        # ------------------------------------------------------------------
        collision_flag = 1.0 if agent.get_fatal_flag() else 0.0
        lane_inv_flag = 1.0 if getattr(agent, "had_lane_invasion", False) else 0.0

        tl = vehicle.get_traffic_light()
        traffic_light_red_flag = 0.0
        if tl is not None and tl.get_state() == carla.TrafficLightState.Red:
            traffic_light_red_flag = 1.0

        at_junction_flag = 1.0 if wp.is_junction else 0.0

        # ------------------------------------------------------------------
        # 6) Previous control (throttle, steer) – works with your dict storage
        # ------------------------------------------------------------------
        prev_ctrl = getattr(agent, "last_control", None)
        if prev_ctrl is None:
            throttle_prev = 0.0
            steer_prev = 0.0
        elif isinstance(prev_ctrl, dict):
            throttle_prev = float(prev_ctrl.get("throttle", 0.0))
            steer_prev = float(prev_ctrl.get("steer", 0.0))
        else:
            throttle_prev = float(getattr(prev_ctrl, "throttle", 0.0))
            steer_prev = float(getattr(prev_ctrl, "steer", 0.0))

        core_obs = np.array(
            [
                speed_norm,
                heading_err_norm,
                rel_goal_x_norm,
                rel_goal_y_norm,
                collision_flag,
                lane_inv_flag,
                traffic_light_red_flag,
                at_junction_flag,
                throttle_prev,
                steer_prev,
            ],
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # 7) Append seg + depth images (128x128 each)
        # ------------------------------------------------------------------
        seg = agent.get_seg_latest() if hasattr(agent, "get_seg_latest") else None
        depth = agent.get_depth_latest() if hasattr(agent, "get_depth_latest") else None

        if seg is None or depth is None:
            seg_flat = np.zeros(SEG_DEPTH_H * SEG_DEPTH_W, dtype=np.float32)
            depth_flat = np.zeros(SEG_DEPTH_H * SEG_DEPTH_W, dtype=np.float32)
        else:
            # seg: int labels, CARLA doc says 13 classes → IDs in [0, 12]
            seg = np.asarray(seg, dtype=np.int32)
            depth = np.asarray(depth, dtype=np.float32)

            assert seg.shape == (SEG_DEPTH_H, SEG_DEPTH_W), \
                f"seg shape {seg.shape} != {(SEG_DEPTH_H, SEG_DEPTH_W)}"
            assert depth.shape == (SEG_DEPTH_H, SEG_DEPTH_W), \
                f"depth shape {depth.shape} != {(SEG_DEPTH_H, SEG_DEPTH_W)}"

            # --- Segmentation normalization ---
            # 13 classes → labels in [0, 12], map to [0, 1]
            max_label = 12.0
            seg_norm = np.clip(seg.astype(np.float32), 0.0, max_label) / max_label

            # --- Depth normalization ---
            depth_cap = 100.0
            depth_clipped = np.clip(depth, 0.0, depth_cap)
            depth_norm = depth_clipped / depth_cap  # [0, 1]

            seg_flat = seg_norm.reshape(-1).astype(np.float32)
            depth_flat = depth_norm.reshape(-1).astype(np.float32)

        full_obs = np.concatenate([core_obs, seg_flat, depth_flat], axis=0)

        if full_obs.shape[0] != CMPE_OBS_DIM:
            raise ValueError(
                f"CMPE obs dim mismatch: got {full_obs.shape[0]}, expected {CMPE_OBS_DIM}"
            )

        return full_obs
        # return core_obs


    
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
    
    def _compute_throttle_brake(
        self,
        agent,
        target_speed_mps: float = 8.0,
    ) -> Tuple[float, float]:
        """
        Very simple hand-coded longitudinal controller:
        - Tries to keep the vehicle around target_speed_mps.
        - Returns (throttle, brake) in [0, 1].

        You can later tune target_speed_mps or make it depend on context.
        """
        vehicle = getattr(agent, "vehicle", None)
        if vehicle is None:
            return 0.0, 0.0

        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Speed error
        err = target_speed_mps - speed  # >0: we are too slow; <0: too fast

        # Simple piecewise “P controller”
        if err > 0.0:
            # Need to accelerate
            # Base throttle + proportional term
            throttle = 0.3 + 0.1 * err
            brake = 0.0
        else:
            # Too fast → no throttle, some brake
            throttle = 0.0
            brake = -0.2 * err  # err is negative here

        # Clamp to valid range
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        return throttle, brake

    def apply_actions_to_controlled(self, actions: Sequence) -> None:
        """
        Apply actions to all controlled agents.

        Phase 1: RL only controls STEERING.
        - PPO might still output 3D actions [throttle, steer, brake].
        - We extract steer from the action and ignore the rest.
        - Throttle / brake are computed by a hand-coded speed controller.
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
                # Convert to numpy array for uniform handling
                act_np = np.asarray(act, dtype=np.float32)

                # ---- Extract steering from action ----
                if act_np.ndim == 0:
                    # Single scalar (already "just steer")
                    raw_steer = float(act_np)
                elif act_np.ndim == 1:
                    if act_np.shape[0] == 1:
                        # shape (1,) -> one value, treat as steer
                        raw_steer = float(act_np[0])
                    else:
                        # shape (3,) like [throttle, steer, brake]
                        # we take index 1 as STEERING
                        raw_steer = float(act_np[1])
                else:
                    print(f"[Manager] Unexpected action shape {act_np.shape}, skipping agent {agent.index}")
                    continue

                # Clamp steering to [-1, 1]
                steer = float(np.clip(raw_steer, -1.0, 1.0))

                # ---- Hand-coded longitudinal controller (throttle / brake) ----
                throttle, brake = self._compute_throttle_brake(
                    agent,
                    target_speed_mps=2.0,
                )

                # Optional debug:
                # print(f"[Manager] agent {agent.index}: steer={steer:.3f}, throttle={throttle:.3f}, brake={brake:.3f}")

                # Store last_control so obs / reward code can use it
                agent.last_control = {
                    "throttle": throttle,
                    "steer": steer,
                    "brake": brake,
                }

                # Apply to CARLA
                agent.apply_action((throttle, steer, brake))

            except Exception as e:
                print(f"[Manager] Error applying action to agent {getattr(agent, 'index', '?')}: {e}")



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
        Simplified reward with normalized progress:

          progress_norm = (dist_prev - dist_current) / initial_final_dist

        where initial_final_dist is the straight-line distance from the
        agent's spawn position to its final destination, stored per-agent
        as `initial_final_goal_dist`.

        Kept terms:
          + normalized progress towards current goal
          + intermediate waypoint reward
          + final goal reward
          - collision penalty (fatal)
          - optional lane-fatal penalty (stage >= 3)
          - small per-step time penalty
        """
        num_ctrl = len(self.controlled_agents)
        rewards = np.zeros(num_ctrl, dtype=np.float32)
        dones = np.zeros(num_ctrl, dtype=bool)

        # ---- main hyper-parameters ----
        W_PROGRESS = 10.0          # scale for normalized progress
        GOAL_RADIUS = 3.0       # meters
        INTERMEDIATE_REWARD = 100
        STEER_PENALTY_SCALE = 0.0005
        FINAL_REWARD = 5000
        COLLISION_PENALTY = -500
        LANE_FATAL_PENALTY = -500
        SEG_SAFE_RADIUS = 1.5          # meters of “free” corridor
        SEG_PENALTY_SCALE = 0       # overall strength
        SEG_PENALTY_ALPHA = 0.2        # exponential growth rate


        # Stage-based time penalty (same as before)
        if self.reward_stage == 1:
            time_pen = 0
        elif self.reward_stage == 2:
            time_pen = 0
        else:  # stage 3+
            time_pen = 0

        for i, agent in enumerate(self.controlled_agents):
            # Count step for stats
            self.reward_stats["steps"] += 1.0

            r = 0.0
            done = False
            events = []

            # ---------------- Early exits for dead/finished agents ----------------
            if agent is None:
                dones[i] = True
                rewards[i] = 0.0
                continue

            vehicle = getattr(agent, "vehicle", None)
            if vehicle is None:
                dones[i] = True
                rewards[i] = 0.0
                continue

            if getattr(agent, "reached_final_goal", False):
                dones[i] = True
                rewards[i] = 0.0
                continue

            try:
                # If globally inactive (already removed)
                idx_all = self.vehicle_to_index.get(vehicle.id, None)
                if idx_all is None or not self.active_flags[idx_all]:
                    dones[i] = True
                    rewards[i] = 0.0
                    continue

                # ---------------- Current goal (intermediate or final) ----------------
                tf = vehicle.get_transform()
                loc = tf.location

                # Use same logic as observations to get the current goal location.
                goal_loc = self.get_current_goal_location(agent)

                seg_prev = getattr(agent, "segment_prev_goal", None)
                seg_curr = getattr(agent, "segment_curr_goal", None)

                if seg_prev is None or seg_curr is None:
                    # First time: start from current location to current goal
                    seg_prev = loc
                    seg_curr = goal_loc
                    agent.segment_prev_goal = seg_prev
                    agent.segment_curr_goal = seg_curr

                # After calling get_current_goal_location, current_wp_index may
                # have changed. Recompute route / is_final_goal consistently.
                route = getattr(agent, "current_waypoint_plan", None)
                wp_idx = int(getattr(agent, "current_wp_index", 0))

                use_route = route is not None and len(route) > 0
                if use_route and 0 <= wp_idx < len(route):
                    is_final_goal = (wp_idx == len(route) - 1)
                else:
                    use_route = False
                    is_final_goal = True

                # Distance to *current* goal (intermediate or final)
                dx = float(goal_loc.x) - float(loc.x)
                dy = float(goal_loc.y) - float(loc.y)
                dist = float(np.sqrt(dx * dx + dy * dy))

                # ---------------- Initial distance to FINAL destination ----------------
                # We normalize progress by the *episode-long* distance from spawn
                # to the FINAL destination, not a fixed global range.
                final_dest = getattr(agent, "destination", goal_loc)
                dx_f = float(final_dest.x) - float(loc.x)
                dy_f = float(final_dest.y) - float(loc.y)
                dist_to_final = float(np.sqrt(dx_f * dx_f + dy_f * dy_f))

                initial_final_dist = getattr(agent, "initial_final_goal_dist", None)
                if initial_final_dist is None:
                    # Set once per episode; avoid division by zero
                    initial_final_dist = max(dist_to_final, 1e-3)
                    agent.initial_final_goal_dist = initial_final_dist

                # ---------------- Progress toward current goal ----------------
                # Store previous distance to the current goal (can be intermediate).
                dist_prev = getattr(agent, "prev_dist_to_goal", None)
                if dist_prev is None:
                    dist_prev = dist
                progress_raw = dist_prev - dist  # > 0 if moving closer
                agent.prev_dist_to_goal = dist

                # Normalize by initial_final_dist (spawn → final destination)
                progress_norm = 0.0
                if initial_final_dist > 1e-3:
                    progress_norm = progress_raw / initial_final_dist

                if progress_norm > 0.0:
                    contrib = W_PROGRESS * progress_norm
                    r += contrib
                    self._accum_reward("progress", contrib)
                    events.append(f"+{contrib:.4f} progress_norm")

                # ---------------- Steering penalty (small, quadratic) ----------------
                prev_ctrl = getattr(agent, "last_control", None)
                if prev_ctrl is not None:
                    if isinstance(prev_ctrl, dict):
                        steer_val = float(prev_ctrl.get("steer", 0.0))
                    else:
                        steer_val = float(getattr(prev_ctrl, "steer", 0.0))

                    steer_pen = STEER_PENALTY_SCALE * (steer_val ** 2)
                    if steer_pen > 0.0:
                        r -= steer_pen
                        self._accum_reward("steer_pen", -steer_pen)
                        events.append(f"-{steer_pen:.4f} steer^2")
                
                #------------------ Segment Offset
                seg_offset = self._segment_offset(loc, seg_prev, seg_curr)

                if seg_offset > SEG_SAFE_RADIUS:
                    excess = seg_offset - SEG_SAFE_RADIUS
                    # Exponential growth: ~0 near corridor edge, grows fast when far
                    seg_pen = SEG_PENALTY_SCALE * (math.exp(SEG_PENALTY_ALPHA * excess) - 1.0)
                    r -= seg_pen
                    # Reuse "lat_pen" bucket in reward_stats for logging
                    self._accum_reward("lat_pen", -seg_pen)
                    events.append(
                        f"-{seg_pen:.3f} segment dev (d={seg_offset:.2f})"
                    )   

                    

                # ---------------- Goal reached logic ----------------
                if dist < GOAL_RADIUS and not done:
                    if use_route and not is_final_goal:
                        # Reached an intermediate waypoint
                        r += INTERMEDIATE_REWARD
                        self._accum_reward("intermediate", INTERMEDIATE_REWARD)
                        events.append(f"+{INTERMEDIATE_REWARD:.2f} intermediate goal")

                        # Advance to next waypoint in the route
                        agent.current_wp_index = wp_idx + 1
                        agent.prev_dist_to_goal = None
                        agent.idle_norm_steps = 0

                        # Update segment: previous = old goal, current = new goal
                        old_goal_loc = goal_loc
                        new_goal_loc = self.get_current_goal_location(agent)
                        agent.segment_prev_goal = old_goal_loc
                        agent.segment_curr_goal = new_goal_loc

                    else:
                        # Reached the final goal
                        r += FINAL_REWARD
                        self._accum_reward("final", FINAL_REWARD)
                        done = True
                        events.append(f"+{FINAL_REWARD:.2f} final goal reached")
                        setattr(agent, "reached_final_goal", True)
                        agent.prev_dist_to_goal = None
                        self.remove_agent(agent)


                # ---------------- Collision penalty ----------------
                if agent.get_fatal_flag() and not done:
                    r += COLLISION_PENALTY
                    self._accum_reward("collision", COLLISION_PENALTY)
                    done = True
                    events.append(f"{COLLISION_PENALTY:.2f} collision")
                    print("Yo fatal flag triggered, collision.")
                    self.remove_agent(agent)

                # ---------------- Lane invasion (optional fatal) ----------------
                lane_inv_flag = bool(getattr(agent, "had_lane_invasion", False))
                if lane_inv_flag and not done:
                    if self.reward_stage >= 3:
                        # Stage 3+: treat lane invasion as fatal
                        r += LANE_FATAL_PENALTY
                        self._accum_reward("lane_fatal", LANE_FATAL_PENALTY)
                        done = True
                        events.append(f"{LANE_FATAL_PENALTY:.2f} lane invasion (fatal)")
                        print("Lane invasion, fatal in stage 3+")
                        self.remove_agent(agent)
                    else:
                        # always treat lane invasion bad
                        r += LANE_FATAL_PENALTY
                        self._accum_reward("lane_fatal", LANE_FATAL_PENALTY)
                        done = True
                        events.append(f"{LANE_FATAL_PENALTY:.2f} lane invasion (fatal)")
                        print("Lane invasion, fatal in stage 3+")
                        self.remove_agent(agent)
                    agent.had_lane_invasion = False

                # ---------------- Per-step time penalty ----------------
                r -= time_pen
                self._accum_reward("time_pen", -time_pen)
                events.append(f"-{time_pen:.3f} time")

                agent.last_reward_events = events

                rewards[i] = float(r)
                dones[i] = done

            except RuntimeError as e:
                # Underlying CARLA actor destroyed → mark done & clean up.
                print(f"[Manager] Controlled agent {i} vehicle actor destroyed: {e}")
                rewards[i] = 0.0
                dones[i] = True
                self.remove_agent(agent)

        print("[RLHandler] rewards from manager:", rewards)   # TEMP debug
        return rewards, dones







    # Visualize Path for each controlled
    def visualize_path(self):
        for agent in self.controlled_agents:
            #print('drawing drawing')
            agent.draw_current_route_debug()

    # --------------------------------------------------
    # DEBUG: ego + goal locations
    # --------------------------------------------------
    def get_first_agent_goal_debug(self):
        """
        Return (ego_loc, goal_loc, rel_xy) for the first controlled agent.

        ego_loc, goal_loc are carla.Location
        rel_xy is (dx, dy) in world coordinates: goal - ego
        """
        if not self.controlled_agents:
            return None

        agent = self.controlled_agents[0]
        vehicle = getattr(agent, "vehicle", None)
        if vehicle is None:
            return None

        ego_tf = vehicle.get_transform()
        ego_loc = ego_tf.location

        goal_loc = self.get_current_goal_location(agent)

        dx = float(goal_loc.x) - float(ego_loc.x)
        dy = float(goal_loc.y) - float(ego_loc.y)

        return ego_loc, goal_loc, (dx, dy)

    # --------------------------------------------------
    # Cleanup & removal
    # --------------------------------------------------

    def reset_episode(self) -> None:
        """
        Destroy all existing agents and respawn fresh controlled + free agents.
        """
        # 1) Destroy current agents
        self.cleanup()

        # 2) Clear per-episode containers
        self.agents = []
        self.active_flags = []
        self.controlled_agents = []
        self.free_agents = []
        self.vehicle_to_index = {}

        # 3) Respawn with original headcounts
        nc, nf = self.config.get_headcounts()
        self.spawn_both_controlled_and_free(nc, nf)

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

    # DEBUGGING
    def get_agent_ego_raw_for_logging(
        self,
        agent,
        goal_location: carla.Location,
    ):
        """
        Same as get_agent_ego_obs but returns RAW (unnormalized) values:
            [speed, veh_len, veh_width, rel_x, rel_y, collided_flag]
        for debugging & visualization.
        """
        vehicle = agent.vehicle
        if vehicle is None:
            return np.zeros(6, dtype=np.float32)

        # --- speed (m/s) ---
        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # --- vehicle size (full length / width) ---
        bbox = vehicle.bounding_box
        veh_len = 2.0 * float(bbox.extent.x)
        veh_width = 2.0 * float(bbox.extent.y)

        # --- relative goal in ego frame (raw meters) ---
        tf = vehicle.get_transform()
        ego_loc = tf.location
        ego_yaw_rad = math.radians(tf.rotation.yaw)

        dx = float(goal_location.x) - float(ego_loc.x)
        dy = float(goal_location.y) - float(ego_loc.y)

        cos_e = math.cos(ego_yaw_rad)
        sin_e = math.sin(ego_yaw_rad)

        rel_x = cos_e * dx + sin_e * dy
        rel_y = -sin_e * dx + cos_e * dy

        collided = 1.0 if agent.get_fatal_flag() else 0.0

        return np.array(
            [speed, veh_len, veh_width, rel_x, rel_y, collided],
            dtype=np.float32,
        )


