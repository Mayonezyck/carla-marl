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
from RL_handler import RLHandler, CMPE_OBS_DIM, STEER_BINS, ACCEL_BINS
from dqn_policy import DQNPolicy
from visualizer_cmpe import PygameVisualizer


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
    elif config.get_use_policy() == "local-dqn":
        num_steer = len(STEER_BINS)
        num_accel = len(ACCEL_BINS)
        num_actions = num_steer * num_accel

        policy = DQNPolicy(
            obs_dim=CMPE_OBS_DIM,
            num_actions=num_actions,
            lr=1e-3,
            gamma=0.99,
            batch_size=64,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=50_000,
            target_update_freq=1000,
        )

    # -----------------------------
    # 3) RLHandler
    # -----------------------------
    rl = RLHandler(world.manager, policy=policy)

    # -----------------------------
    # 4) Visualizer
    # -----------------------------
    visualizer = PygameVisualizer()

    # -----------------------------
    # 5) Episode settings
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

                # 1) RL step
                obs, act, rew, done = rl.step()

                # Debug print (optional)
                print(
                    f"[train_CMPE] ep {ep} | step {step_in_ep} "
                    f"| obs shape: {obs.shape}, act shape: {act.shape}"
                )

                # Optional: visualize planned routes
                if config.get_if_route_planning():
                    world.manager.visualize_path()

                # 2) DQN training step
                if rew is not None:
                    policy.train_step(rl.buffer)

                    # Early episode termination if all controlled agents done
                    if np.all(done):
                        print(
                            f"[train_CMPE] Episode {ep} finished early at "
                            f"step {step_in_ep} (all agents done)."
                        )
                        # we still want one last visualization before break
                        # so don't break yet, just mark and handle after vis if you want
                        episode_done_here = True
                    else:
                        episode_done_here = False
                else:
                    episode_done_here = False

                # 3) Visualization (top-down camera + ego obs + info)
                ego_rgb = None
                ego_obs_vec = None
                last_reward_ego = 0.0

                # Controlled vehicle 0
                if len(world.manager.controlled_agents) > 0:
                    ego_agent = world.manager.controlled_agents[0]
                    if ego_agent is not None and getattr(ego_agent, "vehicle", None) is not None:
                        # Top-down image
                        ego_rgb = ego_agent.get_forward_latest()

                        # Ego CMPE obs (10D)
                        ego_obs_vec = world.manager.get_agent_cmpe_style_obs(ego_agent)

                if rew is not None and rew.shape[0] > 0:
                    last_reward_ego = float(rew[0])

                # Build text lines for the right panel
                text_lines = [
                    f"Episode: {ep}",
                    f"Step in episode: {step_in_ep}",
                    f"Global step: {global_step}",
                    f"Last reward [ego]: {last_reward_ego:.3f}",
                    "",
                    "Ego CMPE obs (10D):",
                ]

                if ego_obs_vec is not None:
                    # Your get_agent_cmpe_style_obs docstring says:
                    # [speed_norm,
                    #  heading_err_norm,
                    #  dist_to_goal_norm,
                    #  lateral_norm,
                    #  collision_flag,
                    #  lane_invasion_flag,
                    #  traffic_light_red_flag,
                    #  at_junction_flag,
                    #  throttle_prev,
                    #  steer_prev]
                    names = [
                        "speed_norm",
                        "heading_err_norm",
                        "dist_to_goal_norm",
                        "lateral_norm",
                        "collision_flag",
                        "lane_invasion_flag",
                        "traffic_light_red_flag",
                        "at_junction_flag",
                        "throttle_prev",
                        "steer_prev",
                    ]
                    for name, val in zip(names, ego_obs_vec.tolist()):
                        text_lines.append(f"  {name}: {val:.3f}")
                else:
                    text_lines.append("  <no ego obs>")

                # Render in pygame (handles events internally)
                try:
                    visualizer.render(ego_rgb, text_lines)
                except KeyboardInterrupt:
                    # Closing the window or pressing ESC will bubble up
                    raise

                # 4) Advance CARLA world
                world.tick()

                # If we flagged early episode termination, break now
                if episode_done_here:
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
