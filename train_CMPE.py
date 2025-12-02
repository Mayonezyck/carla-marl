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

import os
import shutil
from pathlib import Path

import imageio
import pygame



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

    # Create per-run output directory with timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"cmpe_run_{run_timestamp}")
    videos_dir = base_output_dir / "videos"
    policies_dir = base_output_dir / "policies"

    base_output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config file into the run folder
    config_path = Path("config_cmpe.yaml")
    if config_path.exists():
        shutil.copy(config_path, base_output_dir / config_path.name)


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
    # 3b) Pygame visualization setup
    # -----------------------------
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CMPE Training Debug Viewer")
    font = pygame.font.SysFont("monospace", 16)

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

            # Per-episode logging
            episode_reward = 0.0
            video_frames = []

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

                # 2) RL update from replay buffer (once we have rewards)
                if rew is not None:
                    # Accumulate reward for logging (first controlled agent)
                    if rew.shape[0] > 0:
                        episode_reward += float(rew[0])

                    # One DQN update step from replay buffer (if we use local DQN)
                    if isinstance(policy, DQNPolicy):
                        policy.train_step(rl.buffer)

                    # If all controlled agents are done, end the episode
                    if np.all(done):
                        print(
                            f"[train_CMPE] Episode {ep} finished early at "
                            f"step {step_in_ep} (all agents done)."
                        )
                        break

                # 3) Pygame visualization: camera + text
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                screen.fill((0, 0, 0))

                # --- 3a) Try to get forward-facing camera from controlled agent 0 ---
                ego_cam_rgb = None
                if len(world.manager.controlled_agents) > 0:
                    ego_agent = world.manager.controlled_agents[0]
                    if (
                        ego_agent is not None
                        and getattr(ego_agent, "vehicle", None) is not None
                        and hasattr(ego_agent, "get_forward_latest")
                    ):
                        ego_cam_rgb = ego_agent.get_forward_latest()

                text_x = 10  # default text x offset

                if ego_cam_rgb is not None:
                    # ego_cam_rgb expected shape: (H, W, 3) uint8
                    cam_array = np.asarray(ego_cam_rgb, dtype=np.uint8)
                    h, w, _ = cam_array.shape

                    # Pygame wants (W, H, 3)
                    cam_surface = pygame.surfarray.make_surface(cam_array.swapaxes(0, 1))

                    # Scale to fit left portion of the 800x600 window
                    # e.g. 512x384
                    cam_surface = pygame.transform.scale(cam_surface, (512, 384))

                    # Blit at top-left
                    screen.blit(cam_surface, (0, 0))

                    # Shift text to the right of the camera
                    text_x = 520

                # --- 3b) Build text lines ---
                lines = [
                    f"Episode: {ep}",
                    f"Step in episode: {step_in_ep}",
                    f"Global step: {global_step}",
                    f"Episode reward (agent 0): {episode_reward:.3f}",
                ]
                if rew is not None and rew.shape[0] > 0:
                    lines.append(f"Reward[0] this step: {float(rew[0]):.3f}")

                # Show first few ego obs entries for controlled agent 0
                if obs.shape[0] > 0:
                    ego_obs = obs[0]
                    lines.append("Ego obs[0..9]:")
                    for i in range(min(len(ego_obs), 10)):
                        lines.append(f"  o[{i}] = {ego_obs[i]: .3f}")

                y = 10
                for line in lines:
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (text_x, y))
                    y += 20

                pygame.display.flip()


                # 4) Capture the current Pygame window into the episode video buffer
                frame_surface = pygame.display.get_surface()
                if frame_surface is not None:
                    frame_array = pygame.surfarray.array3d(frame_surface)  # (W, H, 3)
                    frame_array = np.transpose(frame_array, (1, 0, 2))      # (H, W, 3)
                    video_frames.append(frame_array)

                # 5) Advance the CARLA world one tick
                world.tick()

            # End of one episode
            # Save episode video
            if len(video_frames) > 0:
                video_path = videos_dir / f"episode_{ep:04d}.mp4"
                try:
                    imageio.mimsave(video_path, video_frames, fps=20)
                    print(f"[train_CMPE] Saved video for episode {ep} to {video_path}")
                except Exception as e:
                    print(f"[train_CMPE] Failed to save video for episode {ep}: {e}")

            # Save policy every 10 episodes if we are using local DQN
            if isinstance(policy, DQNPolicy) and (ep % 10 == 0):
                policy_path = policies_dir / f"dqn_policy_ep{ep:04d}.pt"
                try:
                    policy.save(str(policy_path))
                    print(f"[train_CMPE] Saved DQN policy at episode {ep} to {policy_path}")
                except Exception as e:
                    print(f"[train_CMPE] Failed to save policy at episode {ep}: {e}")


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
