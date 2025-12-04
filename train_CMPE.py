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
from RL_handler import RLHandler, CMPE_OBS_DIM, STEER_BINS, ACCEL_BINS, SEG_DEPTH_H, SEG_DEPTH_W

from dqn_policy import DQNPolicy
from visualizer_cmpe import PygameVisualizer

import os
import shutil
from pathlib import Path

import imageio
import pygame

# --------- DQN training hyperparams ----------
LEARNING_START = 2_000      # replay size before we start learning
TRAIN_EVERY    = 64          # gradient step every N env steps
GRAD_UPDATES_PER_CALL = 64   # how many batches per train_step call



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
    use_dict_policy = False

    if config.get_use_policy() == "remote":
        # Remote policy server expects flat CMPE obs
        policy = RemoteSimPolicy(base_url="http://0.0.0.0:7999")
        use_dict_policy = False

    elif config.get_use_policy() == "local-dqn":
        num_steer = len(STEER_BINS)

        # 1D action: steering only. Each action index corresponds to one steering bin.
        num_actions = num_steer

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
            cmpe_dim=10,          # first 10 are the scalar features
            img_channels=3,       # seg + depth
            img_height=128,
            img_width=128,
        )
        use_dict_policy = True
    else:
        policy = None
        use_dict_policy = False

    # -----------------------------
    # 3) RLHandler
    # -----------------------------
    rl = RLHandler(world.manager, policy=policy, use_dict_policy=use_dict_policy)


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
    max_episodes = getattr(config, "max_episodes", 2000)
    max_steps_per_episode = getattr(config, "max_steps_per_episode", 20000)

    print(
        f"[train_CMPE] Starting training: "
        f"{max_episodes} episodes, up to {max_steps_per_episode} steps each."
    )

    global_step = 0
    # ---- Training metrics history ----
    return_history = []
    length_history = []
    success_history = []  # 1 if reached goal, 0 otherwise


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
            
                # Optional: visualize planned routes
                if config.get_if_route_planning():
                    world.manager.visualize_path()

                # 2) RL update from replay buffer (once we have rewards)
                if rew is not None:
                    # Accumulate reward for logging (first controlled agent)
                    if rew.shape[0] > 0:
                        episode_reward += float(rew[0])

                    # DQN update from replay buffer with warmup + periodic updates
                    if isinstance(policy, DQNPolicy):
                        if len(rl.buffer) >= LEARNING_START and (global_step % TRAIN_EVERY == 0):
                            print('Now updating the policy')
                            for _ in range(GRAD_UPDATES_PER_CALL):
                                policy.train_step(rl.buffer)

                    # If all controlled agents are done, end the episode
                    if np.all(done):
                        print(
                            f"[train_CMPE] Episode {ep} finished early at "
                            f"step {step_in_ep} (all agents done)."
                        )
                        break


                # 3) Pygame visualization: RGB + segmentation + depth + text
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                screen.fill((0, 0, 0))

                ego_cam_rgb = None
                seg_img = None
                depth_img = None
                hud_img = None

                # --------- grab latest camera data from controlled agent 0 ---------
                if len(world.manager.controlled_agents) > 0:
                    ego_agent = world.manager.controlled_agents[0]

                    if ego_agent is not None and getattr(ego_agent, "vehicle", None) is not None:
                        # Forward RGB camera
                        if hasattr(ego_agent, "get_forward_latest"):
                            ego_cam_rgb = ego_agent.get_forward_latest()

                        # Segmentation
                        if hasattr(ego_agent, "get_seg_latest"):
                            seg_img = ego_agent.get_seg_latest()

                        # Depth
                        if hasattr(ego_agent, "get_depth_latest"):
                            depth_img = ego_agent.get_depth_latest()

                                # --------- reconstruct HUD goal channel from obs[0] ---------
                if obs.shape[0] > 0:
                    ego_obs = obs[0]
                    CORE_DIM = 10  # first 10 are scalar ego features

                    flat_len = SEG_DEPTH_H * SEG_DEPTH_W
                    needed = CORE_DIM + 3 * flat_len

                    if ego_obs.shape[0] >= needed:
                        offset = CORE_DIM
                        # seg_flat   = ego_obs[offset               : offset + flat_len]
                        # depth_flat = ego_obs[offset + flat_len    : offset + 2 * flat_len]
                        hud_flat   = ego_obs[offset + 2 * flat_len : offset + 3 * flat_len]
                        hud_img = hud_flat.reshape(SEG_DEPTH_H, SEG_DEPTH_W)


                # Layout:
                #   RGB:  left, size 480x360 at (0, 0)
                #   SEG:  top-right, size 320x180 at (480, 0)
                #   DEPTH:bottom-right, size 320x180 at (480, 180)
                #   Text: along bottom starting at y = 380

                # --------- draw RGB camera ---------
                if ego_cam_rgb is not None:
                    rgb_arr = np.asarray(ego_cam_rgb, dtype=np.uint8)
                    # Expect (H, W, 3)
                    if rgb_arr.ndim == 3 and rgb_arr.shape[2] >= 3:
                        rgb_vis = rgb_arr[:, :, :3]
                        rgb_surface = pygame.surfarray.make_surface(
                            rgb_vis.swapaxes(0, 1)
                        )
                        rgb_surface = pygame.transform.scale(rgb_surface, (480, 360))
                        screen.blit(rgb_surface, (0, 0))

                # --------- draw segmentation ---------
                if seg_img is not None:
                    seg_arr = np.asarray(seg_img)
                    # Handle (H, W) or (H, W, 1) or (H, W, 3)
                    if seg_arr.ndim == 2:
                        # normalize to [0, 255] for display
                        max_val = float(seg_arr.max()) if seg_arr.max() > 0 else 1.0
                        seg_norm = (seg_arr.astype(np.float32) / max_val) * 255.0
                        seg_norm = seg_norm.astype(np.uint8)
                        seg_vis = np.stack([seg_norm] * 3, axis=-1)  # grayscale to RGB
                    elif seg_arr.ndim == 3 and seg_arr.shape[2] == 1:
                        seg_vis = np.repeat(seg_arr.astype(np.uint8), 3, axis=2)
                    elif seg_arr.ndim == 3 and seg_arr.shape[2] >= 3:
                        seg_vis = seg_arr[:, :, :3].astype(np.uint8)
                    else:
                        seg_vis = None

                    if seg_vis is not None:
                        seg_surface = pygame.surfarray.make_surface(
                            seg_vis.swapaxes(0, 1)
                        )
                        seg_surface = pygame.transform.scale(seg_surface, (320, 180))
                        screen.blit(seg_surface, (480, 0))

                # --------- draw depth ---------
                if depth_img is not None:
                    depth_arr = np.asarray(depth_img, dtype=np.float32)
                    # Handle (H, W) or (H, W, 1) or (H, W, 3)
                    if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
                        depth_arr = depth_arr[:, :, 0]
                    elif depth_arr.ndim == 3 and depth_arr.shape[2] >= 3:
                        # if somehow RGB encoded, take one channel
                        depth_arr = depth_arr[:, :, 0]

                    if depth_arr.ndim == 2:
                        # normalize to [0, 255] using a cap (e.g., 100m)
                        depth_cap = 100.0
                        depth_norm = np.clip(depth_arr / depth_cap, 0.0, 1.0)
                        depth_vis = (depth_norm * 255.0).astype(np.uint8)
                        depth_vis = np.stack([depth_vis] * 3, axis=-1)  # grayscale to RGB
                    else:
                        depth_vis = None

                    if depth_vis is not None:
                        depth_surface = pygame.surfarray.make_surface(
                            depth_vis.swapaxes(0, 1)
                        )
                        depth_surface = pygame.transform.scale(depth_surface, (320, 180))
                        screen.blit(depth_surface, (480, 180))

                                # --------- draw HUD goal mask (from obs) ---------
                if hud_img is not None:
                    hud_arr = np.asarray(hud_img, dtype=np.float32)
                    # hud is in [0,1], map to [0,255] grayscale
                    hud_norm = np.clip(hud_arr, 0.0, 1.0)
                    hud_vis = (hud_norm * 255.0).astype(np.uint8)
                    hud_vis = np.stack([hud_vis] * 3, axis=-1)  # grayscale → RGB

                    hud_surface = pygame.surfarray.make_surface(
                        hud_vis.swapaxes(0, 1)
                    )
                    # Make it a bit shorter to leave space for text
                    hud_surface = pygame.transform.scale(hud_surface, (320, 120))
                    # Place under depth panel
                    screen.blit(hud_surface, (480, 360))


                # --------- build and draw text (multi-column) ---------
                info_lines = [
                    f"Episode: {ep}",
                    f"Step: {step_in_ep}",
                    f"Global: {global_step}",
                    f"Ep reward[0]: {episode_reward:.3f}",
                ]
                if rew is not None and rew.shape[0] > 0:
                    info_lines.append(f"Reward[0] step: {float(rew[0]):.3f}")

                obs_lines = []
                if obs.shape[0] > 0:
                    ego_obs = obs[0]
                    # compress obs a bit so it fits
                    obs_lines.append(
                        "Ego obs[0..4]: " +
                        ", ".join(f"{ego_obs[i]:.3f}" for i in range(min(len(ego_obs), 5)))
                    )
                    if len(ego_obs) > 5:
                        obs_lines.append(
                            "Ego obs[5..9]: " +
                            ", ".join(f"{ego_obs[i]:.3f}" for i in range(5, min(len(ego_obs), 10)))
                        )

                # Gather event strings (if any)
                event_lines = []
                if len(world.manager.controlled_agents) > 0:
                    ego_agent = world.manager.controlled_agents[0]
                    events = getattr(ego_agent, "last_reward_events", [])
                    for ev in events:
                        ev_str = str(ev).strip()
                        event_lines.append(ev_str)

                # ---- layout text in columns along the bottom ----
                base_y = 500
                line_h = 20

                # Left block: info + obs, packed in columns
                left_block = info_lines + obs_lines
                num_cols_left = 2          # how many columns to span horizontally
                col_width_left = 380       # width per column in pixels
                x0_left = 10

                for idx, line in enumerate(left_block):
                    col = idx % num_cols_left
                    row = idx // num_cols_left
                    x = x0_left + col * col_width_left
                    y = base_y + row * line_h
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (x, y))

                # Right block: events (colored), also in columns
                if event_lines:
                    # small header
                    header_surface = font.render("Events:", True, (200, 200, 200))
                    screen.blit(header_surface, (10, base_y + 3 * line_h))

                    # Put events under the header, starting from some offset
                    events_y_offset = base_y + 4 * line_h
                    num_cols_ev = 2
                    col_width_ev = 380
                    x0_ev = 10

                    for idx, ev_str in enumerate(event_lines):
                        col = idx % num_cols_ev
                        row = idx // num_cols_ev
                        x = x0_ev + col * col_width_ev
                        y = events_y_offset + row * line_h

                        if ev_str.startswith("+"):
                            color = (0, 255, 0)
                        elif ev_str.startswith("-"):
                            color = (255, 0, 0)
                        else:
                            color = (200, 200, 200)

                        ev_surface = font.render(ev_str, True, color)
                        screen.blit(ev_surface, (x, y))


                pygame.display.flip()



                # 4) Capture the current Pygame window into the episode video buffer
                frame_surface = pygame.display.get_surface()
                if frame_surface is not None:
                    frame_array = pygame.surfarray.array3d(frame_surface)  # (W, H, 3)
                    frame_array = np.transpose(frame_array, (1, 0, 2))      # (H, W, 3)
                    video_frames.append(frame_array)

                # 5) Advance the CARLA world one tick
                world.tick()

            # ---- Episode summary metrics ----
            # Note: step_in_ep will be the last value from the loop
            ep_len = step_in_ep
            ep_return = episode_reward

            # For now, define success as "ego reached its final goal"
            ep_success = 0
            if len(world.manager.controlled_agents) > 0:
                ego_agent = world.manager.controlled_agents[0]
                if getattr(ego_agent, "reached_final_goal", False):
                    ep_success = 1

            return_history.append(ep_return)
            length_history.append(ep_len)
            success_history.append(ep_success)

            # Print moving averages over the last 50 episodes
            window = 50
            recent_returns = return_history[-window:]
            recent_lengths = length_history[-window:]
            recent_success = success_history[-window:]

            avg_ret = np.mean(recent_returns)
            avg_len = np.mean(recent_lengths)
            avg_succ = np.mean(recent_success)

            print(
                f"[train_CMPE] Ep {ep} done | "
                f"R_ep={ep_return:.2f}, T_ep={ep_len}, success={ep_success} | "
                f"MA{window}: R={avg_ret:.2f}, T={avg_len:.1f}, succ={avg_succ:.2f}"
            )

            # Optional: print reward term breakdown
            world.manager.print_and_reset_reward_stats()

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
                # Save training metrics for plotting
        try:
            import csv
            metrics_path = base_output_dir / "training_metrics.csv"
            with open(metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "return", "length", "success"])
                for i, (r, t, s) in enumerate(zip(return_history, length_history, success_history), start=1):
                    writer.writerow([i, r, t, s])
            print(f"[train_CMPE] Saved training metrics to {metrics_path}")
        except Exception as e:
            print("[train_CMPE] Error saving training metrics:", repr(e))
    
        world.cleanup()
        print("[train_CMPE] Cleanup complete.")
