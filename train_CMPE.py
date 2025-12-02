# train_CMPE.py
#
# Training-oriented version of the CMPE pipeline.
# - Uses CarlaWorld + Manager + RLHandler (same wiring style as main.py)
# - Runs multiple episodes
# - Fills the replay buffer
# - Has a hook for learning from the buffer

import traceback
from datetime import datetime

import numpy as np
import carla

from client import carlaClient
from config import ConfigLoader
from world import CarlaWorld
from RL_handler import RLHandler
from remote_policy import RemoteSimPolicy


def maybe_train_from_buffer(rl: RLHandler, step: int) -> None:
    """
    Placeholder hook for actual learning.

    Right now:
      - does nothing

    Later:
      - sample from rl.buffer
      - run a gradient step on your policy / critic
    """
    # Example skeleton (commented):
    # if len(rl.buffer) > 1000 and step % 10 == 0:
    #     batch = rl.buffer.sample(batch_size=64)
    #     # unpack batch and run your learner here
    pass


if __name__ == "__main__":
    # -----------------------------
    # 1) Load config & create world
    # -----------------------------
    config = ConfigLoader("config_cmpe.yaml")

    # Start CARLA client (your wrapper)
    client = carlaClient()

    # CarlaWorld should internally grab client.get_world(),
    # set sync mode, hold Manager as world.manager, etc.
    world = CarlaWorld(config)

    # -----------------------------
    # 2) Build policy for RLHandler
    # -----------------------------
    # If you want to use a remote policy server for actions:
    if config.get_use_policy() == "remote":
        policy = RemoteSimPolicy(base_url="http://0.0.0.0:7999")
        # RLHandler will call policy(obs_batch) and then decode_discrete_action.
    else:
        # If policy is None, RLHandler._default_policy() will be used.
        # You can plug in a local neural net policy here later.
        policy = None

    # -----------------------------
    # 3) RLHandler
    # -----------------------------
    rl = RLHandler(world.manager, policy=policy)

    # -----------------------------
    # 4) Episode settings
    # -----------------------------
    # If your config has these getters, use them; otherwise defaults.
    max_episodes = getattr(config, "max_episodes", 50)
    max_steps_per_episode = getattr(config, "max_steps_per_episode", 500)

    print(
        f"[train_CMPE] Starting training: "
        f"{max_episodes} episodes, up to {max_steps_per_episode} steps each."
    )

    global_step = 0

    try:
        for ep in range(1, max_episodes + 1):
            print(f"\n[train_CMPE] === Episode {ep} ===")

            # Reset environment (destroy & respawn agents)
            world.manager.reset_episode()

            # Reset RLHandler per-episode state (keep replay buffer)
            rl.reset(clear_buffer=False)

            for step_in_ep in range(1, max_steps_per_episode + 1):
                global_step += 1

                # 1) RL step: read obs, log transition, choose & apply new actions
                obs, act, rew, done = rl.step()
                # obs:  (N, CMPE_OBS_DIM)
                # act:  (N, 3)
                # rew:  (N,) or None on the very first ever step
                # done: (N,) or None on the very first ever step

                # Debug print (you can comment this out if too noisy)
                print(
                    f"[train_CMPE] ep {ep} | step {step_in_ep} "
                    f"| obs shape: {obs.shape}, act shape: {act.shape}"
                )

                # Optional: visualize planned routes
                if config.get_if_route_planning():
                    world.manager.visualize_path()

                # 2) Optionally run a training step from replay buffer
                if rew is not None:
                    maybe_train_from_buffer(rl, global_step)

                    # If all controlled agents are done, end the episode
                    if np.all(done):
                        print(
                            f"[train_CMPE] Episode {ep} finished early at "
                            f"step {step_in_ep} (all agents done)."
                        )
                        break

                # 3) Advance the CARLA world one tick
                world.tick()

    except KeyboardInterrupt:
        print("[train_CMPE] Stopping via KeyboardInterrupt...")

    except Exception as e:
        print("[train_CMPE] Unhandled exception in training loop:", repr(e))
        traceback.print_exc()

    finally:
        # Save debug history from RLHandler for offline inspection
        try:
            rl.save_debug_history("carla_debug_cmpe.pkl")
        except Exception as e:
            print("[train_CMPE] Error saving debug history:", repr(e))

        # Clean up CARLA actors via CarlaWorld
        world.cleanup()
        print("[train_CMPE] Cleanup complete.")
