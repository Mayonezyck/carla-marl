# RL_handler.py

import numpy as np
from typing import Optional, Callable, Any, Sequence, List, Dict


class SimpleReplayBuffer:
    """
    Very simple replay buffer:
    - Stores transitions in a ring buffer.
    - Each transition is a dict with keys:
        'obs', 'action', 'reward', 'next_obs', 'done'
    - This is enough to plug into a learner later.
    """
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = int(capacity)
        self.storage = []  # type: List[Dict[str, Any]]
        self.pos: int = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add_batch(
        self,
        obs: np.ndarray,        # (N, ..., ...)
        actions: np.ndarray,    # (N, action_dim)
        rewards: np.ndarray,    # (N,)
        next_obs: np.ndarray,   # (N, ..., ...)
        dones: np.ndarray,      # (N,)
    ) -> None:
        batch_size = obs.shape[0]
        for i in range(batch_size):
            transition = {
                "obs":      np.array(obs[i], copy=True),
                "action":   np.array(actions[i], copy=True),
                "reward":   float(rewards[i]),
                "next_obs": np.array(next_obs[i], copy=True),
                "done":     bool(dones[i]),
            }

            if len(self.storage) < self.capacity:
                self.storage.append(transition)
            else:
                self.storage[self.pos] = transition
                self.pos = (self.pos + 1) % self.capacity

    # Optional helper if you want to sample later
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
                 obs_batch: (N, num_points, 3)
                 returns:   (N, action_dim)
                 If None, a simple default policy is used.
        """
        self.manager = manager
        self.action_dim = int(action_dim)
        self.policy = policy
        self.buffer = SimpleReplayBuffer(replay_capacity)

        # Cached from previous step
        self.last_obs = None       # type: Optional[np.ndarray]   # (N, P, 3)
        self.last_actions = None   # type: Optional[np.ndarray]   # (N, action_dim)

        self.step_count: int = 0

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def step(self):
        
        """
        One RL step:

        1. Read current observations from Manager.
        2. If we have (last_obs, last_actions), build and store
           the transition (last_obs, last_actions, reward, obs_t, done).
        3. Compute new actions from obs_t via policy.
        4. Apply actions to controlled agents via Manager.
        5. Cache (obs_t, actions_t) for the next call.

        Returns:
            frames_t : list of frame ids (or None) for each controlled agent
            obs_t    : np.ndarray, shape (N, num_points, 3)
            actions_t: np.ndarray, shape (N, action_dim)
            rewards  : np.ndarray, shape (N,) or None on the very first call
            dones    : np.ndarray, shape (N,) or None on the very first call
        """
        # 1) Get current obs from all controlled agents
        frames_t, obs_t = self.manager.get_controlled_lidar_observations()
        # obs_t: (num_agents, num_points, 3)

        rewards = None
        dones = None

        # 2) If we have a previous state-action, build transition
        if self.last_obs is not None and self.last_actions is not None:
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
            )

        # 3) Compute new actions from current obs
        if self.policy is not None:
            actions_t = self.policy(obs_t)
        else:
            actions_t = self._default_policy(obs_t)

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

        # 4) Apply actions via Manager
        self.manager.apply_actions_to_controlled(actions_t)

        

        # 5) Cache for next step
        self.last_obs = obs_t
        self.last_actions = actions_t
        self.step_count += 1

        return frames_t, obs_t, actions_t, rewards, dones

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
            obs_t   : (N, P, 3)
            actions_t: (N, action_dim)
            obs_tp1 : (N, P, 3)

        Returns:
            rewards : (N,)
            dones   : (N,)
        """
        num_agents = obs_t.shape[0]
        print(num_agents)
        rewards = np.zeros(num_agents, dtype=np.float32)
        dones = np.zeros(num_agents, dtype=bool)
        print('checking the rewards')
        print('checking done')
        rewards, dones = self.manager.get_rewards_and_dones()

        return rewards, dones
