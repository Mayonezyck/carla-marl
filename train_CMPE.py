# train_CMPE.py
#
# Training-oriented version of the CMPE pipeline.
# - Uses CarlaWorld + Manager + RLHandler (same wiring style as main.py)
# - Fills the replay buffer
# - Provides a hook for learning steps from the buffer

import time
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
    policy = None

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

    # Training hyperparams (can be moved into config if you like)
    max_total_steps = getattr(config, "max_total_steps", 10_000)
    print(f"[train_CMPE] Starting training loop for up to {max_total_steps} steps.")

    step_idx = 0

    try:
        while step_idx < max_total_steps:
            step_idx += 1

            try:
                # 1) RL step: read obs, log transition, choose & apply new actions
                obs, act, rew, done = rl.step()
                # obs:  (N, CMPE_OBS_DIM)
                # act:  (N, 3)
                # rew:  (N,) or None on very first step
                # done: (N,) or None on very first step

                # Debug print (can be throttled or removed later)
                print(
                    f"[train_CMPE] step {step_idx} | "
                    f"obs shape: {obs.shape}, act shape: {act.shape}"
                )

                # Optional: visualize planned routes
                if config.get_if_route_planning():
                    world.manager.visualize_path()

                # 2) Optionally run a training step from replay buffer
                if rew is not None:
                    maybe_train_from_buffer(rl, step_idx)

                    # If you later implement episode termination in Manager.get_rewards_and_dones(),
                    # you can also check for all-done here:
                    #
                    # if np.all(done):
                    #     print(f"[train_CMPE] all agents done at step {step_idx}")
                    #     # TODO: when you have a reset mechanism, call:
                    #     # world.manager.reset_episode()
                    #     # rl.reset(clear_buffer=False)
                    #     # and continue training into next episode.

                # 3) Advance the CARLA world one tick
                world.tick()

            except Exception as e:
                print("[train_CMPE] Exception inside main loop:", repr(e))
                traceback.print_exc()
                break   # exit while so we still hit finally

    except KeyboardInterrupt:
        print("[train_CMPE] Stopping via KeyboardInterrupt...")

    finally:
        # Save debug history from RLHandler for offline inspection
        try:
            rl.save_debug_history("carla_debug_cmpe.pkl")
        except Exception as e:
            print("[train_CMPE] Error saving debug history:", repr(e))

        # Clean up CARLA actors via CarlaWorld
        world.cleanup()
        print("[train_CMPE] Cleanup complete.")
