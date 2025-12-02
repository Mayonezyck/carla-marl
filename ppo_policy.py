# ppo_policy.py

from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PPOBackbone(nn.Module):
    """
    Shared backbone: CMPE scalars + optional image (seg+depth).
    Very similar spirit to your DQNNetwork but split into:
        - shared feature extractor
        - separate policy + value heads
    """

    def __init__(
        self,
        obs_dim: int,
        cmpe_dim: int,
        img_channels: int = 2,
        img_height: int = 128,
        img_width: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.cmpe_dim = cmpe_dim
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width

        img_feat_dim = 0
        self.use_image = (obs_dim > cmpe_dim)
        if self.use_image:
            self.cnn = nn.Sequential(
                nn.Conv2d(img_channels, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            # figure out CNN output size
            with torch.no_grad():
                dummy = torch.zeros(1, img_channels, img_height, img_width)
                img_feat_dim = self.cnn(dummy).shape[1]

        in_dim = cmpe_dim + img_feat_dim

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim) = [cmpe (cmpe_dim), flat_image (...)]
        """
        if not self.use_image:
            x = obs
        else:
            cmpe = obs[:, :self.cmpe_dim]           # (B, cmpe_dim)
            img_flat = obs[:, self.cmpe_dim:]       # (B, C*H*W)
            img = img_flat.view(
                obs.shape[0], self.img_channels, self.img_height, self.img_width
            )
            img_feat = self.cnn(img)                # (B, img_feat_dim)
            x = torch.cat([cmpe, img_feat], dim=-1)
        return self.fc(x)


class PPOPolicy(nn.Module):
    """
    Continuous-action PPO:
        - action space: 2D (steer, accel) in [-1, 1]
        - we will decode accel → (throttle, brake) outside this class

    Interface:
        - act(obs) -> actions, log_probs, values
        - update(batch) -> apply PPO updates
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        cmpe_dim: int = 10,
        img_channels: int = 2,
        img_height: int = 128,
        img_width: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.backbone = PPOBackbone(
            obs_dim=obs_dim,
            cmpe_dim=cmpe_dim,
            img_channels=img_channels,
            img_height=img_height,
            img_width=img_width,
        )

        feat_dim = 256  # from backbone.fc output
        self.policy_head = nn.Linear(feat_dim, action_dim * 2)  # mean + log_std
        self.value_head = nn.Linear(feat_dim, 1)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _forward_backbone(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def _get_dist_and_value(
        self, obs: torch.Tensor
    ):
        feats = self._forward_backbone(obs)
        logits = self.policy_head(feats)
        mean, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(feats).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        obs_np: (N, obs_dim)
        Returns:
            actions: (N, action_dim) in [-1, 1] after tanh
            log_probs: (N,)
            values: (N,)
        """
        obs = torch.from_numpy(obs_np).float().to(self.device)
        dist, value = self._get_dist_and_value(obs)

        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        # squash to [-1, 1]
        action = torch.tanh(raw_action)
        # log_prob with tanh squashing correction (simplified: ignore)
        log_prob = dist.log_prob(raw_action).sum(-1)

        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        rewards, values, dones: (T, N)
        last_value: (N,)
        Returns:
            returns: (T, N)
            advantages: (T, N)
        """
        T, N = rewards.shape
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_value
                next_nonterminal = 1.0 - dones[t].astype(np.float32)
            else:
                next_values = values[t + 1]
                next_nonterminal = 1.0 - dones[t + 1].astype(np.float32)

            delta = (
                rewards[t]
                + self.gamma * next_values * next_nonterminal
                - values[t]
            )
            last_gae = (
                delta + self.gamma * self.lam * next_nonterminal * last_gae
            )
            adv[t] = last_gae

        returns = adv + values
        return returns, adv

    def update(
        self,
        batch: Dict[str, np.ndarray],
        epochs: int = 10,
        batch_size: int = 64,
    ):
        """
        batch fields: obs, actions, log_probs, returns, advantages
            obs:      (T*N, obs_dim)
            actions:  (T*N, action_dim)
            log_probs:(T*N,)
            returns:  (T*N,)
            advantages:(T*N,)
        """
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)
        actions = torch.from_numpy(batch["actions"]).float().to(self.device)
        old_log_probs = torch.from_numpy(batch["log_probs"]).float().to(self.device)
        returns = torch.from_numpy(batch["returns"]).float().to(self.device)
        advantages = torch.from_numpy(batch["advantages"]).float().to(self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = obs.shape[0]
        idxs = np.arange(num_samples)

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                dist, value = self._get_dist_and_value(mb_obs)

                # we stored already-squashed actions; invert tanh approximately
                atanh_actions = torch.atanh(
                    torch.clamp(mb_actions, -0.999, 0.999)
                )
                log_probs = dist.log_prob(atanh_actions).sum(-1)

                ratio = torch.exp(log_probs - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, mb_returns)

                entropy = dist.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
