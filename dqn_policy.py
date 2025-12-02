# dqn_policy.py

from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNPolicy:
    """
    Simple DQN policy with:
      - epsilon-greedy exploration
      - target network
      - train_step(buffer) to do one gradient update
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay_steps: int = 50_000,
        target_update_freq: int = 1000,
        device: Optional[torch.device] = None,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.q_net = DQNNetwork(obs_dim, num_actions).to(self.device)
        self.target_net = DQNNetwork(obs_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

    # Make it callable: policy(obs_batch) -> discrete actions
    def __call__(self, obs_batch) -> np.ndarray:
        """
        obs_batch can be:
          - np.ndarray of shape (N, obs_dim), or
          - dict with at least key "cmpe": np.ndarray of shape (N, obs_dim)
            (other keys like "seg", "depth" are currently ignored here)

        returns: (N,) int32 discrete actions
        """
        # Extract the CMPE vector if a dict is passed
        if isinstance(obs_batch, dict):
            cmpe = obs_batch.get("cmpe", None)
            if cmpe is None:
                raise ValueError("DQNPolicy: expected 'cmpe' key in obs dict.")
            obs_np = np.asarray(cmpe, dtype=np.float32)
        else:
            obs_np = np.asarray(obs_batch, dtype=np.float32)

        if obs_np.ndim != 2 or obs_np.shape[1] != self.obs_dim:
            raise ValueError(
                f"DQNPolicy: expected obs shape (N, {self.obs_dim}), "
                f"got {obs_np.shape}"
            )

        self.total_steps += obs_np.shape[0]

        eps = self._current_epsilon()

        # Random actions with prob epsilon
        if np.random.rand() < eps:
            return np.random.randint(
                0, self.num_actions, size=(obs_np.shape[0],), dtype=np.int64
            )

        # Greedy actions from Q-network
        obs_t = torch.from_numpy(obs_np).float().to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)  # (N, num_actions)
            actions = torch.argmax(q_values, dim=1)  # (N,)
        return actions.cpu().numpy().astype(np.int64)


    def _current_epsilon(self) -> float:
        frac = min(1.0, self.total_steps / float(self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def train_step(self, buffer) -> None:
        """
        Perform one DQN update from the replay buffer.
        Only uses transitions that have a non-None 'disc_action'.
        """
        if len(buffer) < self.batch_size:
            return

        # Sample a batch and filter out None disc_actions
        raw_batch = buffer.sample(self.batch_size * 2)  # over-sample then filter
        batch: List[Dict] = [tr for tr in raw_batch if tr["disc_action"] is not None]

        if len(batch) < self.batch_size:
            return

        batch = batch[: self.batch_size]

        obs = np.stack([tr["obs"] for tr in batch], axis=0).astype(np.float32)
        next_obs = np.stack([tr["next_obs"] for tr in batch], axis=0).astype(np.float32)
        actions = np.array([tr["disc_action"] for tr in batch], dtype=np.int64)
        rewards = np.array([tr["reward"] for tr in batch], dtype=np.float32)
        dones = np.array([tr["done"] for tr in batch], dtype=np.float32)  # 0 or 1

        obs_t = torch.from_numpy(obs).to(self.device)       # (B, obs_dim)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)  # (B,)
        rewards_t = torch.from_numpy(rewards).to(self.device)  # (B,)
        dones_t = torch.from_numpy(dones).to(self.device)      # (B,)

        # Current Q(s, a)
        q_values = self.q_net(obs_t)  # (B, num_actions)
        q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Target Q(s', a') = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_net(next_obs_t)  # (B, num_actions)
            q_next_max, _ = torch.max(q_next, dim=1)  # (B,)
            target = rewards_t + self.gamma * q_next_max * (1.0 - dones_t)

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


    def save(self, path: str) -> None:
        """Save Q-network, target network, optimizer, and meta info."""
        state = {
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "total_steps": self.total_steps,
            "target_update_freq": self.target_update_freq,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load Q-network, target network, optimizer, and meta info."""
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state["q_net_state"])
        self.target_net.load_state_dict(state["target_net_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])

        # Restore meta info if present
        self.gamma = state.get("gamma", self.gamma)
        self.batch_size = state.get("batch_size", self.batch_size)
        self.epsilon_start = state.get("epsilon_start", self.epsilon_start)
        self.epsilon_end = state.get("epsilon_end", self.epsilon_end)
        self.epsilon_decay_steps = state.get("epsilon_decay_steps", self.epsilon_decay_steps)
        self.total_steps = state.get("total_steps", self.total_steps)
        self.target_update_freq = state.get("target_update_freq", self.target_update_freq)
