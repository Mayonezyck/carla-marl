import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("debug_ep_3.pkl", "rb") as f:
    data = pickle.load(f)

steps = data["steps"]                       # (T,)
ego_raw = data["ego_raw"]                  # (T, 6)
disc = data["disc_actions"]                # (T,)
decoded = data["decoded_actions"]          # (T, 3)

speed      = ego_raw[:, 0]
veh_len    = ego_raw[:, 1]
veh_width  = ego_raw[:, 2]
rel_x      = ego_raw[:, 3]
rel_y      = ego_raw[:, 4]
collided   = ego_raw[:, 5]

throttle = decoded[:, 0]
steer    = decoded[:, 1]
brake    = decoded[:, 2]

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))

# 1) Ego kinematics / geometry
axes[0].plot(steps, speed, label="speed (m/s)")
axes[0].plot(steps, rel_x, label="rel_goal_x (m)")
axes[0].plot(steps, rel_y, label="rel_goal_y (m)")
axes[0].set_ylabel("Ego state")
axes[0].legend()
axes[0].grid(True)

# 2) Vehicle size (should be mostly constant)
axes[1].plot(steps, veh_len, label="veh_len")
axes[1].plot(steps, veh_width, label="veh_width")
axes[1].set_ylabel("Size (m)")
axes[1].legend()
axes[1].grid(True)

# 3) Discrete action index
axes[2].step(steps, disc, where="post")
axes[2].set_ylabel("Discrete action idx")
axes[2].grid(True)

# 4) Decoded continuous actions
axes[3].plot(steps, throttle, label="throttle")
axes[3].plot(steps, brake, label="brake")
axes[3].plot(steps, steer, label="steer_norm")
axes[3].set_ylabel("Action")
axes[3].set_xlabel("Step")
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.show()

