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
MIN_REL_GOAL_COORD = -100.0
MAX_REL_GOAL_COORD = 100.0
SEG_DEPTH_H = 128
SEG_DEPTH_W = 128
#CMPE_OBS_DIM = 10 + 2 * SEG_DEPTH_H * SEG_DEPTH_W
CMPE_OBS_DIM = 10 


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
        """
        vehicle = getattr(agent, "vehicle", None)
        if vehicle is None:
            # Fallback: arbitrary location
            return self.world.get_map().get_spawn_points()[0].location

        # Default: use final destination if present
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

        wp, _ = route[wp_idx]
        return wp.transform.location


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

        # # ------------------------------------------------------------------
        # # 7) Append seg + depth images (128x128 each)
        # # ------------------------------------------------------------------
        # seg = agent.get_seg_latest() if hasattr(agent, "get_seg_latest") else None
        # depth = agent.get_depth_latest() if hasattr(agent, "get_depth_latest") else None

        # if seg is None or depth is None:
        #     seg_flat = np.zeros(SEG_DEPTH_H * SEG_DEPTH_W, dtype=np.float32)
        #     depth_flat = np.zeros(SEG_DEPTH_H * SEG_DEPTH_W, dtype=np.float32)
        # else:
        #     # seg: int labels, CARLA doc says 13 classes → IDs in [0, 12]
        #     seg = np.asarray(seg, dtype=np.int32)
        #     depth = np.asarray(depth, dtype=np.float32)

        #     assert seg.shape == (SEG_DEPTH_H, SEG_DEPTH_W), \
        #         f"seg shape {seg.shape} != {(SEG_DEPTH_H, SEG_DEPTH_W)}"
        #     assert depth.shape == (SEG_DEPTH_H, SEG_DEPTH_W), \
        #         f"depth shape {depth.shape} != {(SEG_DEPTH_H, SEG_DEPTH_W)}"

        #     # --- Segmentation normalization ---
        #     # 13 classes → labels in [0, 12], map to [0, 1]
        #     max_label = 12.0
        #     seg_norm = np.clip(seg.astype(np.float32), 0.0, max_label) / max_label

        #     # --- Depth normalization ---
        #     depth_cap = 100.0
        #     depth_clipped = np.clip(depth, 0.0, depth_cap)
        #     depth_norm = depth_clipped / depth_cap  # [0, 1]

        #     seg_flat = seg_norm.reshape(-1).astype(np.float32)
        #     depth_flat = depth_norm.reshape(-1).astype(np.float32)

        # full_obs = np.concatenate([core_obs, seg_flat, depth_flat], axis=0)

        # if full_obs.shape[0] != CMPE_OBS_DIM:
        #     raise ValueError(
        #         f"CMPE obs dim mismatch: got {full_obs.shape[0]}, expected {CMPE_OBS_DIM}"
        #     )

        #return full_obs
        return core_obs


    
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
                # ensure it's a 3-tuple
                throttle = float(act[0])
                steer    = float(act[1])
                brake    = float(act[2])

                # store last_control so CMPE obs can read it
                agent.last_control = {
                    "throttle": throttle,
                    "steer": steer,
                    "brake": brake,
                }

                # actually apply control using your existing helper
                agent.apply_action((throttle, steer, brake))

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

        Heuristic:
          - Progress toward goal       → positive reward
          - Moving away from goal      → stronger negative
          - Being far from goal        → small distance penalty
          - Reaching goal              → big positive + done
          - Collision (fatal flag)     → big negative + done (agent removed)
          - Lane invasion              → stage-based penalty (can be fatal)
          - Steering magnitude         → penalty to discourage circling
          - No-progress for long       → penalty + terminate (anti-orbit)
          - Speed / direction shaping:
                * reward motion along line to next waypoint
                * penalize sideways (lateral) motion
        """
        num_ctrl = len(self.controlled_agents)
        rewards = np.zeros(num_ctrl, dtype=np.float32)
        dones = np.zeros(num_ctrl, dtype=bool)

        # ---- global tuning knobs (feel free to tweak) ----
        W_FORWARD = 1.0             # reward for making progress
        W_BACKWARD = 2.0            # penalty for going away from goal
        DIST_PENALTY_SCALE = 0.01   # per-meter distance penalty (clipped)
        MAX_DIST_FOR_PENALTY = 100.0

        MAX_IDLE_STEPS = 20         # start counting "idle too long"
        IDLE_PENALTY = 1.0          # per-step penalty once idle kicks in
        NO_PROGRESS_DONE_STEPS = 80 # hard terminate if no progress this long

        STEER_MAG_PENALTY = 0.05    # penalty * |steer|

        # 🔹 NEW: “follow the line to waypoint” shaping
        ALONG_REWARD_SCALE = 0.1    # reward per m/s along line to waypoint
        LAT_PENALTY_SCALE  = 0.2    # penalty per m/s lateral to that line

        for i, agent in enumerate(self.controlled_agents):
            # Agent missing or no vehicle → treat as done with zero reward
            if agent is None:
                dones[i] = True
                rewards[i] = 0.0
                continue

            vehicle = getattr(agent, "vehicle", None)
            if vehicle is None:
                dones[i] = True
                rewards[i] = 0.0
                continue

            # If this agent already finished (reached final goal earlier),
            # treat as done with zero reward from now on.
            if getattr(agent, "reached_final_goal", False):
                dones[i] = True
                rewards[i] = 0.0
                continue

            try:
                # Check if this vehicle is still marked active in global structures
                idx_all = self.vehicle_to_index.get(vehicle.id, None)
                if idx_all is None or not self.active_flags[idx_all]:
                    dones[i] = True
                    rewards[i] = 0.0
                    continue

                # ------------------------------------------------------------------
                # Distance-based progress toward current goal (subgoal or final)
                # ------------------------------------------------------------------
                tf = vehicle.get_transform()
                loc = tf.location

                # Determine current goal:
                #   - intermediate waypoint if available
                #   - otherwise final destination
                route = getattr(agent, "current_waypoint_plan", None)
                wp_idx = int(getattr(agent, "current_wp_index", 0))

                use_route = route is not None and len(route) > 0
                is_final_goal = True  # assume final until proven otherwise

                if use_route and 0 <= wp_idx < len(route):
                    wp, _ = route[wp_idx]
                    goal_loc = wp.transform.location
                    is_final_goal = (wp_idx == len(route) - 1)
                else:
                    goal_loc = getattr(agent, "destination", loc)
                    is_final_goal = True

                dx = float(goal_loc.x) - float(loc.x)
                dy = float(goal_loc.y) - float(loc.y)
                dist = float(np.sqrt(dx * dx + dy * dy))

                # 🔹 NEW: velocity decomposition w.r.t. line to waypoint
                vel = vehicle.get_velocity()
                vx = float(vel.x)
                vy = float(vel.y)
                speed = math.sqrt(vx * vx + vy * vy)  # planar speed

                if dist > 1e-3:
                    # unit vector from ego → waypoint
                    ux = dx / dist
                    uy = dy / dist

                    # component of velocity along the line (dot product)
                    v_along = vx * ux + vy * uy   # m/s toward waypoint (can be negative)

                    # lateral speed = |v × u| in 2D
                    # cross(v, u)_z = vx*uy - vy*ux, magnitude is lateral component
                    lat_speed = abs(vx * uy - vy * ux)
                else:
                    v_along = 0.0
                    lat_speed = 0.0

                # previous distance stored on agent (per-episode), for THIS goal
                dist_prev = getattr(agent, "prev_dist_to_goal", None)
                if dist_prev is None:
                    dist_prev = dist
                progress = dist_prev - dist  # > 0 if moving closer
                agent.prev_dist_to_goal = dist

                # ---- idle / stall tracking based on progress ----
                idle_steps = getattr(agent, "idle_steps", 0)
                PROGRESS_EPS = 0.1  # meters; treat smaller as "no real progress"

                if abs(progress) < PROGRESS_EPS:
                    idle_steps += 1
                else:
                    idle_steps = 0
                agent.idle_steps = idle_steps

                r = 0.0
                events = []

                # ---------------- Progress shaping ----------------
                if progress > 0.0:
                    r += W_FORWARD * progress
                else:
                    r += W_BACKWARD * progress  # progress is negative here

                # ---------------- Distance penalty ----------------
                # Always slightly penalize being far from goal.
                # This breaks "orbiting" attractors where progress ~ 0.
                clipped_dist = min(dist, MAX_DIST_FOR_PENALTY)
                if clipped_dist > 0.0:
                    dist_pen = DIST_PENALTY_SCALE * clipped_dist
                    r -= dist_pen
                    events.append(f"-{dist_pen:.2f} distance penalty")

                # ---------------- NEW: line-following shaping ----------------
                # Reward moving along the line to the waypoint,
                # penalize moving sideways relative to that line.
                if v_along > 0.0:
                    along_reward = ALONG_REWARD_SCALE * v_along
                    r += along_reward
                    events.append(f"+{along_reward:.2f} along-waypoint speed")
                elif v_along < 0.0:
                    # moving directly away from waypoint → penalize
                    back_pen = ALONG_REWARD_SCALE * (-v_along)
                    r -= back_pen
                    events.append(f"-{back_pen:.2f} moving away from waypoint")

                if lat_speed > 0.0:
                    lat_pen = LAT_PENALTY_SCALE * lat_speed
                    r -= lat_pen
                    events.append(f"-{lat_pen:.2f} lateral deviation")

                # ---------------- Idle / no-progress penalties ----------------
                if idle_steps > MAX_IDLE_STEPS and dist > 10.0:
                    # per-step penalty when basically not reducing distance
                    r -= IDLE_PENALTY
                    events.append(f"-{IDLE_PENALTY:.2f} idle too long")

                # Hard terminate if we've made no progress for a long time
                done = False
                # if idle_steps > NO_PROGRESS_DONE_STEPS and dist > 10.0:
                #     done = True
                #     extra_pen = 5.0
                #     r -= extra_pen
                #     events.append(f"-{extra_pen:.2f} terminate: no progress")

                # ---------------- Steering penalty ----------------
                # Discourages tight circles: large |steer| over time is costly.
                prev_ctrl = getattr(agent, "last_control", None)
                if prev_ctrl is not None:
                    if isinstance(prev_ctrl, dict):
                        steer_val = float(prev_ctrl.get("steer", 0.0))
                    else:
                        steer_val = float(getattr(prev_ctrl, "steer", 0.0))

                    steer_pen = STEER_MAG_PENALTY * abs(steer_val)
                    if steer_pen > 0.0:
                        r -= steer_pen
                        events.append(f"-{steer_pen:.2f} steer penalty")

                # ------------------------------------------------------------------
                # Goal reached logic:
                #   - intermediate waypoint → small reward + advance wp index
                #   - final waypoint (or no route) → big reward + done
                # ------------------------------------------------------------------
                GOAL_RADIUS = 3.0  # meters
                INTERMEDIATE_REWARD = 10.0
                FINAL_REWARD = 100.0

                if dist < GOAL_RADIUS and not done:
                    if use_route and not is_final_goal:
                        # Reached an intermediate waypoint
                        r += INTERMEDIATE_REWARD
                        events.append(f"+{INTERMEDIATE_REWARD:.2f} intermediate goal")
                        # Advance to the next waypoint in the route
                        agent.current_wp_index = wp_idx + 1
                        # Reset distance baseline for the new goal
                        agent.prev_dist_to_goal = None
                        # Also reset idle tracking, so new segment starts fresh
                        agent.idle_steps = 0
                    else:
                        # Reached the final goal
                        r += FINAL_REWARD
                        done = True
                        events.append(f"+{FINAL_REWARD:.2f} final goal reached")
                        # Mark this agent as finished so we don't keep rewarding
                        setattr(agent, "reached_final_goal", True)
                        # Optional: reset distance baseline
                        agent.prev_dist_to_goal = None
                        self.remove_agent(agent)

                # ------------------------------------------------------------------
                # Collision penalty (using your fatal flag)
                # ------------------------------------------------------------------
                if agent.get_fatal_flag():
                    r -= 50.0
                    done = True
                    events.append("-50.00 collision")
                    print("Yo fatal flag triggered, -50 for this.")
                    # Remove this agent (vehicle + sensors) from the world
                    self.remove_agent(agent)

                # ------------------------------------------------------------------
                # Lane invasion penalty (stage-based)
                # ------------------------------------------------------------------
                lane_inv_flag = bool(getattr(agent, "had_lane_invasion", False))
                if lane_inv_flag:
                    if self.reward_stage == 1:
                        # Stage 1: ignore lane, just clear the flag
                        events.append("lane invasion (ignored in stage 1)")
                        agent.had_lane_invasion = False

                    elif self.reward_stage == 2:
                        # Stage 2: soft penalty, non-fatal
                        r -= 10.0
                        events.append("-10.00 lane invasion (stage 2)")
                        agent.had_lane_invasion = False

                    elif self.reward_stage >= 3:
                        # Stage 3+: fatal penalty, remove agent
                        r -= 50.0
                        done = True
                        events.append("-50.00 lane invasion (stage 3)")
                        print("Lane invasion, fatal in stage 3")
                        agent.had_lane_invasion = False
                        self.remove_agent(agent)

                # ------------- Per-step time penalty by stage -------------
                if self.reward_stage == 1:
                    r -= 0.01
                elif self.reward_stage == 2:
                    r -= 0.015
                else:  # stage 3+
                    r -= 0.02

                agent.last_reward_events = events

                rewards[i] = float(r)
                dones[i] = done

            except RuntimeError as e:
                # This happens if the underlying CARLA actor is already destroyed.
                print(f"[Manager] Controlled agent {i} vehicle actor destroyed: {e}")
                # Mark as done, zero reward, and clean up.
                rewards[i] = 0.0
                dones[i] = True
                self.remove_agent(agent)

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


