# dqn_policy.py
"""
DQN policy with a CNN+MLP backbone.

Assumes observations are flat vectors with layout:
    [cmpe_features (cmpe_dim), image_features (img_channels * H * W)]

If obs_dim == cmpe_dim + img_channels * H * W, we:
    - split obs into cmpe and image
    - reshape image to (B, C, H, W)
    - run a small CNN on the image
    - concatenate CNN features with cmpe vector
    - output Q-values over discrete actions

If obs_dim <= cmpe_dim (no image part), we fall back to a simple
MLP on the whole obs vector.
"""

from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        cmpe_dim: int = 10,
        img_channels: int = 2,
        img_height: int = 128,
        img_width: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.cmpe_dim = cmpe_dim
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width

        self.img_feat_dim = img_channels * img_height * img_width
        self.expected_obs_with_img = cmpe_dim + self.img_feat_dim

        # Decide whether to use CNN or pure MLP
        self.use_cnn = (obs_dim == self.expected_obs_with_img)

        if self.use_cnn:
            # --- CNN for (seg, depth) images ---
            self.conv = nn.Sequential(
                nn.Conv2d(img_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

            # Infer conv output size with a dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, img_channels, img_height, img_width)
                conv_out = self.conv(dummy)
                conv_out_dim = conv_out.view(1, -1).shape[1]

            fused_input_dim = conv_out_dim + cmpe_dim

            self.mlp_head = nn.Sequential(
                nn.Linear(fused_input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        else:
            # Fallback: pure MLP on the entire obs vector
            self.mlp_only = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim) float32 tensor
        returns: (B, num_actions)
        """
        if not self.use_cnn:
            return self.mlp_only(obs)

        # Split into cmpe (first cmpe_dim dims) and image (rest)
        cmpe = obs[:, : self.cmpe_dim]  # (B, cmpe_dim)
        img_flat = obs[:, self.cmpe_dim :]  # (B, img_feat_dim)
        B = obs.shape[0]

        # Reshape flat image to (B, C, H, W)
        img = img_flat.view(B, self.img_channels, self.img_height, self.img_width)

        feat = self.conv(img)          # (B, C', H', W')
        feat = feat.view(B, -1)        # (B, conv_out_dim)

        fused = torch.cat([cmpe, feat], dim=1)  # (B, fused_dim)
        q_values = self.mlp_head(fused)         # (B, num_actions)
        return q_values


class DQNPolicy:
    """
    Wrapper around DQNNetwork that:
      - handles epsilon-greedy action selection
      - maintains a target network
      - performs DQN updates from a replay buffer
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
        target_update_freq: int = 1_000,
        cmpe_dim: int = 10,
        img_channels: int = 2,
        img_height: int = 128,
        img_width: int = 128,
        device: Optional[str] = None,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Q-network and target network
        self.q_net = DQNNetwork(
            obs_dim=obs_dim,
            num_actions=num_actions,
            cmpe_dim=cmpe_dim,
            img_channels=img_channels,
            img_height=img_height,
            img_width=img_width,
        ).to(self.device)

        self.target_net = DQNNetwork(
            obs_dim=obs_dim,
            num_actions=num_actions,
            cmpe_dim=cmpe_dim,
            img_channels=img_channels,
            img_height=img_height,
            img_width=img_width,
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Step counter for epsilon schedule & target updates
        self.total_steps: int = 0

    # ------------------------------------------------------------------
    # Epsilon-greedy action selection
    # ------------------------------------------------------------------
    def _current_epsilon(self) -> float:
        """Linearly decay epsilon from epsilon_start to epsilon_end."""
        frac = min(1.0, self.total_steps / float(self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def __call__(self, obs_batch: np.ndarray) -> np.ndarray:
        """
        Make the policy callable so RLHandler can do:

            discrete_actions = policy(obs_batch)

        obs_batch: np.ndarray, shape (N, obs_dim)
        returns:   np.ndarray of shape (N,) with integer actions.
        """
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)

        obs_t = torch.from_numpy(obs_batch).float().to(self.device)
        batch_size = obs_t.shape[0]

        eps = self._current_epsilon()
        self.total_steps += 1

        # Epsilon-greedy
        if np.random.rand() < eps:
            actions = np.random.randint(0, self.num_actions, size=batch_size)
        else:
            with torch.no_grad():
                q_values = self.q_net(obs_t)  # (B, num_actions)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # If batch_size == 1, we still return a 1D array of length 1
        return actions.astype(np.int64)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(self, buffer) -> None:
        """
        One DQN gradient step from the replay buffer.

        buffer is assumed to be RL_handler.SimpleReplayBuffer, whose
        .sample(batch_size) returns a list of dicts with keys:
            'obs', 'next_obs', 'disc_action', 'reward', 'done'
        """
        if len(buffer) < self.batch_size:
            return

        # Sample a bit more and filter transitions that have disc_action
        raw_batch: List[Dict] = buffer.sample(self.batch_size * 2)
        batch: List[Dict] = [tr for tr in raw_batch if tr.get("disc_action") is not None]

        if len(batch) < self.batch_size:
            return

        batch = batch[: self.batch_size]

        obs = np.stack([tr["obs"] for tr in batch], axis=0).astype(np.float32)        # (B, obs_dim)
        next_obs = np.stack([tr["next_obs"] for tr in batch], axis=0).astype(np.float32)
        actions = np.array([tr["disc_action"] for tr in batch], dtype=np.int64)       # (B,)
        rewards = np.array([tr["reward"] for tr in batch], dtype=np.float32)         # (B,)
        dones = np.array([tr["done"] for tr in batch], dtype=np.float32)             # (B,)

        obs_t = torch.from_numpy(obs).to(self.device)           # (B, obs_dim)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)   # (B,)
        rewards_t = torch.from_numpy(rewards).to(self.device)   # (B,)
        dones_t = torch.from_numpy(dones).to(self.device)       # (B,)

        # Q(s, a) for taken actions
        q_values = self.q_net(obs_t)                            # (B, num_actions)
        q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_net(next_obs_t)                # (B, num_actions)
            q_next_max, _ = torch.max(q_next, dim=1)            # (B,)
            target = rewards_t + self.gamma * q_next_max * (1.0 - dones_t)

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        state = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_steps = state.get("total_steps", 0)
