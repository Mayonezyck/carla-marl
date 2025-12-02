# RL_handler.py

import numpy as np
from typing import Optional, Callable, Any, Sequence, List, Dict
import torch

# Temporary constants
STEER_BINS = np.linspace(-np.pi, np.pi, 13)   # 13 values
ACCEL_BINS = np.array([-4.0, -2.6, -1.3, 0.0, 1.3, 2.6, 4.0])  # 7 values
CMPE_OBS_DIM = 10  # or 12, whatever we define below


class SimpleReplayBuffer:
    """
    Very simple replay buffer:
    - Stores transitions in a ring buffer.
    - Each transition is a dict with keys:
        'obs', 'action', 'disc_action', 'reward', 'next_obs', 'done'
    - 'action' is the continuous (throttle, steer, brake).
    - 'disc_action' is the discrete action index (int) or None.
    """
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = int(capacity)
        self.storage = []  # type: List[Dict[str, Any]]
        self.pos: int = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add_batch(
        self,
        obs: np.ndarray,              # (N, obs_dim)
        actions: np.ndarray,          # (N, action_dim) continuous
        rewards: np.ndarray,          # (N,)
        next_obs: np.ndarray,         # (N, obs_dim)
        dones: np.ndarray,            # (N,)
        disc_actions: Optional[np.ndarray] = None,  # (N,) discrete or None
    ) -> None:
        batch_size = obs.shape[0]

        # Normalize disc_actions to per-sample scalar or None
        if disc_actions is not None:
            disc_actions = np.asarray(disc_actions)
            if disc_actions.ndim == 0:
                disc_actions = disc_actions.reshape(1)
            assert disc_actions.shape[0] == batch_size, (
                f"disc_actions batch size {disc_actions.shape[0]} "
                f"!= obs batch size {batch_size}"
            )

        for i in range(batch_size):
            transition = {
                "obs":        np.array(obs[i], copy=True),
                "action":     np.array(actions[i], copy=True),
                "reward":     float(rewards[i]),
                "next_obs":   np.array(next_obs[i], copy=True),
                "done":       bool(dones[i]),
                "disc_action": (
                    int(disc_actions[i]) if disc_actions is not None else None
                ),
            }

            if len(self.storage) < self.capacity:
                self.storage.append(transition)
            else:
                self.storage[self.pos] = transition
                self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        idxs = np.random.randint(0, len(self.storage), size=batch_size)
        return [self.storage[i] for i in idxs]



class RLHandler:
    """
    RLHandler:
    - Talks to your Manager to:
        * get lidar observations from controlled agents
        * apply actions to controlled agents
    - Keeps track of last_obs and last_actions so it can
      build transitions (s, a, r, s', done).
    - Stores transitions in a SimpleReplayBuffer.
    """

    def __init__(
        self,
        manager: Any,
        action_dim: int = 3,
        replay_capacity: int = 100_000,
        policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """
        manager : your Manager instance
        action_dim : # of action components per agent (throttle, steer, brake → 3)
        replay_capacity : max transitions in buffer
        policy : optional callable(obs_batch) -> actions_batch
        obs_batch: (N, CMPE_OBS_DIM)
        returns:   (N,) discrete actions or (N, action_dim)

        """
        self.manager = manager
        self.action_dim = int(action_dim)
        self.policy = policy
        self.buffer = SimpleReplayBuffer(replay_capacity)

        # Cached from previous step
        self.last_obs = None       # type: Optional[np.ndarray]   # (N, P, 3)
        self.last_actions = None   # type: Optional[np.ndarray]   # (N, action_dim)
        self.last_disc_actions = None  # type: Optional[np.ndarray]  # (N,)

        self.step_count: int = 0
        self.debug_history = [] 

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def step(self):
        
        """
        One RL step:

        1. Read current GPUDrive-style observations from Manager.
        2. If we have (last_obs, last_actions), build and store
           the transition (last_obs, last_actions, reward, obs_t, done).
        3. Compute new actions from obs_t via policy.
        4. Apply actions to controlled agents via Manager.
        5. Cache (obs_t, actions_t) for the next call.

        Returns:
            obs_t    : np.ndarray, shape (N, obs_dim)    # e.g. 2984
            actions_t: np.ndarray, shape (N, action_dim)
            rewards  : np.ndarray, shape (N,) or None on the very first call
            dones    : np.ndarray, shape (N,) or None on the very first call
        """
        # 1) Get current obs from all controlled agents (GPUDrive-style)
        #obs_t = self._get_gpudrive_obs_from_manager()  # (num_agents, obs_dim) # This is only used when GPUDrive is needed
        obs_t = self._get_cmpe_obs_from_manager()

        rewards = None
        dones = None

        # 2) If we have a previous state-action, build transition
        if (
            self.last_obs is not None
            and self.last_actions is not None
            and self.last_disc_actions is not None
        ):
            rewards, dones = self._compute_reward_and_done(
                self.last_obs,
                self.last_actions,
                obs_t,
            )

            # Store transition in replay buffer
            self.buffer.add_batch(
                obs=self.last_obs,
                actions=self.last_actions,
                rewards=rewards,
                next_obs=obs_t,
                dones=dones,
                disc_actions=self.last_disc_actions,
            )


        # 3) Compute new actions from current obs
        if self.policy is not None:
            # GPUDrive returns discrete values; decode here
            discrete_actions = self.policy(obs_t)       # shape (N,) or scalar
            actions_t = self.decode_discrete_action(discrete_actions)
            disc_np = np.asarray(discrete_actions).reshape(-1)
        else:
            actions_t = self._default_policy(obs_t)
            disc_np = np.full((obs_t.shape[0],), np.nan, dtype=np.float32)

        # Cache discrete actions for the NEXT transition storage
        if self.policy is not None:
            self.last_disc_actions = disc_np.copy()
        else:
            self.last_disc_actions = None



        # Ensure actions are (N, action_dim)
        actions_t = np.asarray(actions_t, dtype=np.float32)
        if actions_t.ndim == 1:  # single agent edge case
            actions_t = actions_t.reshape(1, -1)

        if actions_t.shape[1] != self.action_dim:
            raise ValueError(
                "Expected actions shape (N, {}), got {}".format(
                    self.action_dim, actions_t.shape
                )
            )

        # >>> DEBUG LOGGING: focus on first controlled agent (index 0) <<<
        if len(self.manager.controlled_agents) > 0:
            ego_agent = self.manager.controlled_agents[0]
            if ego_agent is not None and getattr(ego_agent, "vehicle", None) is not None:
                ego_raw = self.manager.get_agent_ego_raw_for_logging(
                    ego_agent,
                    ego_agent.destination,
                )  # shape (6,)

                # Discrete action for agent 0
                disc_a0 = float(disc_np[0]) if disc_np.size > 0 else np.nan

                self.debug_history.append({
                    "step": self.step_count,
                    "ego_raw": ego_raw,                      # (6,)
                    "disc_action": disc_a0,                  # scalar
                    "throttle": float(actions_t[0, 0]),
                    "steer": float(actions_t[0, 1]),
                    "brake": float(actions_t[0, 2]),
                })

        # 4) Apply actions via Manager
        self.manager.apply_actions_to_controlled(actions_t)



        # 5) Cache for next step
        self.last_obs = obs_t
        self.last_actions = actions_t
        self.step_count += 1

        return obs_t, actions_t, rewards, dones
    
    def _get_cmpe_obs_from_manager(self) -> np.ndarray:
        """
        Build a batch of CMPE-style obs for all controlled agents.

        Assumes your Manager has:
            get_cmpe_style_obs(agent, destination) -> torch.Tensor or np.ndarray
        that returns a (CMPE_OBS_DIM,) vector per agent.
        """
        obs_list = []
        for agent in self.manager.controlled_agents:
            # If agent is dead / missing vehicle → just zero obs
            if agent is None or getattr(agent, "vehicle", None) is None:
                obs_vec = np.zeros(CMPE_OBS_DIM, dtype=np.float32)
                obs_list.append(obs_vec)
                continue

            obs = self.manager.get_agent_cmpe_style_obs(agent)

            # Handle torch or numpy
            if isinstance(obs, torch.Tensor):
                obs_vec = obs.detach().cpu().numpy().astype(np.float32)
            else:
                obs_vec = np.asarray(obs, dtype=np.float32)

            # Safety: enforce correct dim
            if obs_vec.shape != (CMPE_OBS_DIM,):
                raise ValueError(
                    f"get_cmpe_style_obs must return shape ({CMPE_OBS_DIM},), got {obs_vec.shape}"
                )

            obs_list.append(obs_vec)

        if len(obs_list) == 0:
            return np.zeros((0, CMPE_OBS_DIM), dtype=np.float32)

        obs_batch = np.stack(obs_list, axis=0)  # (N, CMPE_OBS_DIM)
        return obs_batch


    def _get_gpudrive_obs_from_manager(self) -> np.ndarray:
        """
        Build a batch of GPUDrive-style obs for all controlled agents.

        Assumes your Manager has:
            get_gpudrive_style_obs(agent) -> torch.Tensor or np.ndarray
        that returns a (2984,) vector per agent (ego + neighbors + roadgraph).
        """
        obs_list = []

        for agent in self.manager.controlled_agents:
            # If agent is dead / missing vehicle → just zero obs
            if agent is None or getattr(agent, "vehicle", None) is None:
                obs_vec = np.zeros(2984, dtype=np.float32)  # adjust if obs_dim changes
                obs_list.append(obs_vec)
                continue

            obs = self.manager.get_gpudrive_style_obs(agent, agent.destination)
            # Handle torch or numpy
            try:
                import torch
                if isinstance(obs, torch.Tensor):
                    obs_vec = obs.detach().cpu().numpy()
                else:
                    obs_vec = np.asarray(obs, dtype=np.float32)
            except ImportError:
                obs_vec = np.asarray(obs, dtype=np.float32)

            obs_list.append(obs_vec)

        if len(obs_list) == 0:
            return np.zeros((0, 2984), dtype=np.float32)

        obs_batch = np.stack(obs_list, axis=0)  # (N, obs_dim)
        return obs_batch

    
    def decode_discrete_action(self, discrete_action: np.ndarray) -> np.ndarray:
        """
        Converts discrete GPUDrive-style action(s) into CARLA continuous control:
            throttle, steer, brake

        discrete_action:
            - scalar: single action
            - or array-like shape (N,) for N agents

        Returns:
            actions: np.ndarray, shape (N, 3)
        """
        # Convert to 1D int array
        act = np.asarray(discrete_action, dtype=np.int64)

        # Case 1: scalar → reshape to (1,)
        if act.ndim == 0:
            act = act.reshape(1)

        # Now act.shape == (N,)
        num_steer = STEER_BINS.shape[0]   # 13
        num_accel = ACCEL_BINS.shape[0]   # 7
        num_actions = num_steer * num_accel

        # Optional safety check
        if np.any(act < 0) or np.any(act >= num_actions):
            raise ValueError(
                f"Discrete action out of range [0, {num_actions}): {act}"
            )

        # Vectorized indices
        steer_idx = act % num_steer        # shape (N,)
        accel_idx = act // num_steer       # shape (N,)

        # Map to actual values (broadcasted)
        steer = STEER_BINS[steer_idx]      # (N,)
        accel = ACCEL_BINS[accel_idx]      # (N,)

        # Convert accel → throttle/brake
        throttle = np.clip(accel, 0, None) / 4.0   # 0..1, shape (N,)
        brake    = np.clip(-accel, 0, None) / 4.0  # 0..1, shape (N,)

        # Normalize steering from [-pi, pi] to [-1, 1]
        steer_norm = steer / np.pi                 # (N,)

        # Stack into (N, 3)
        actions = np.stack([throttle, steer_norm, brake], axis=-1).astype(np.float32)
        return actions

    

    def reset(self, clear_buffer: bool = False) -> None:
        """
        Reset per-episode state in the handler.
        If clear_buffer=True, also empty the replay buffer.
        """
        self.last_obs = None
        self.last_actions = None
        self.last_disc_actions = None
        self.step_count = 0
        self.debug_history = []

        if clear_buffer:
            self.buffer = SimpleReplayBuffer(self.buffer.capacity)

    # --------------------------------------------------
    # Internals
    # --------------------------------------------------

    def _default_policy(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Very dumb default policy:
        - constant throttle,
        - tiny random steering,
        - zero brake.
        Just so things move without crashing your code.
        """
        num_agents = obs_batch.shape[0]
        actions = np.zeros((num_agents, self.action_dim), dtype=np.float32)

        # throttle ~ 0.3
        actions[:, 0] = 0.3

        # small random steering in [-0.1, 0.1]
        actions[:, 1] = np.random.uniform(-0.1, 0.1, size=num_agents)

        # brake = 0
        actions[:, 2] = 0.0

        return actions

    def _compute_reward_and_done(
        self,
        obs_t: np.ndarray,
        actions_t: np.ndarray,
        obs_tp1: np.ndarray,
    ):
        """
        Placeholder reward + done function.

        Right now:
            - reward = 0.0 for all agents
            - done   = False for all agents

        You can later plug in:
            - distance-to-goal rewards
            - collision penalties
            - off-road penalties, etc.

        Shapes:
            obs_t    : (N, obs_dim)      # CMPE-style obs
            actions_t: (N, action_dim)
            obs_tp1  : (N, obs_dim)

        Returns:
            rewards : (N,)
            dones   : (N,)
        """
        num_agents = obs_t.shape[0]
        rewards = np.zeros(num_agents, dtype=np.float32)
        dones = np.zeros(num_agents, dtype=bool)
        rewards, dones = self.manager.get_rewards_and_dones()

        return rewards, dones
    
    def save_debug_history(self, path: str = "carla_debug.pkl") -> None:
        """
        Save debug_history to disk for offline visualization.
        """
        import pickle
        if not self.debug_history:
            print("[RLHandler] No debug history to save.")
            return

        steps = np.array([e["step"] for e in self.debug_history], dtype=np.int64)
        ego_raw = np.stack([e["ego_raw"] for e in self.debug_history], axis=0)  # (T, 6)
        disc_actions = np.array([e["disc_action"] for e in self.debug_history], dtype=np.float32)
        decoded = np.stack(
            [[e["throttle"], e["steer"], e["brake"]] for e in self.debug_history],
            axis=0,
        )  # (T, 3)

        payload = {
            "steps": steps,
            "ego_raw": ego_raw,
            "disc_actions": disc_actions,
            "decoded_actions": decoded,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"[RLHandler] Saved debug history to {path} "
              f"(T={steps.shape[0]} steps).")
