# remote_policy.py  (runs in CARLA env)

import numpy as np
import requests
from typing import Sequence


class RemoteSimPolicy:
    """
    Thin client that calls a remote GPUDrive policy server via HTTP.

    Usage:
        policy = RemoteSimPolicy("http://localhost:7999")
        actions = policy(obs_batch)   # obs_batch: (N, 2984) np.ndarray
    """

    def __init__(self, base_url: str = "http://localhost:7999"):
        self.base_url = base_url.rstrip("/")

    def __call__(self, obs_batch: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        if obs_batch.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        payload = {
            "obs": obs_batch.tolist()
        }

        try:
            resp = requests.post(f"{self.base_url}/infer", json=payload, timeout=1.0)
            resp.raise_for_status()
        except Exception as e:
            print(f"[RemoteSimPolicy] Error calling remote policy: {e}")
            # Fallback: no-op / safe action
            num_agents = obs_batch.shape[0]
            actions = np.zeros((num_agents, 3), dtype=np.float32)
            return actions

        data = resp.json()
        actions = np.asarray(data["actions"], dtype=np.float32)  # (N, 3)
        return actions
