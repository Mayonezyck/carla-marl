import numpy as np
import torch

from datetime import datetime
from pathlib import Path

from client import carlaClient
from config import ConfigLoader
from world import CarlaWorld
from RL_handler import CMPE_OBS_DIM
from ppo_policy import PPOPolicy
import imageio


def decode_continuous_actions(raw_actions: np.ndarray) -> np.ndarray:
    """
    raw_actions: (N, 2) in [-1, 1]
        raw[:, 0] = steer
        raw[:, 1] = accel

    Returns (N, 3): throttle, steer, brake
    """
    steer = np.clip(raw_actions[:, 0], -1.0, 1.0)
    accel = np.clip(raw_actions[:, 1], -1.0, 1.0)

    throttle = np.clip(accel, 0.0, 1.0)
    brake = np.clip(-accel, 0.0, 1.0)

    actions = np.stack([throttle, steer, brake], axis=-1)
    return actions.astype(np.float32)


def main():
    print("[PPO] Loading config...")
    config = ConfigLoader("config_cmpe.yaml")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(f"cmpe_ppo_run_{run_timestamp}")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = base_output_dir / "videos"
    policies_dir = base_output_dir / "policies"
    videos_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)


    print("[PPO] Starting CARLA client and world...")
    client = carlaClient()
    world = CarlaWorld(config)

    # ---------- Pygame visualization setup ----------
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CMPE PPO Training Debug Viewer")
    font = pygame.font.SysFont("monospace", 16)

    print("[PPO] Building PPO policy...")
    policy = PPOPolicy(
        obs_dim=CMPE_OBS_DIM,
        action_dim=2,          # (steer, accel)
        cmpe_dim=10,
        img_channels=2,
        img_height=128,
        img_width=128,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=None,
    )

    rollout_horizon = 256
    num_updates = 1000
    ppo_epochs = 4
    ppo_batch_size = 64

    global_step = 0

    try:
        print("[PPO] Starting training loop...")
        for update_idx in range(num_updates):
            # Storage for one rollout (T, N, ...)
            obs_buf = []
            act_buf = []
            logp_buf = []
            rew_buf = []
            done_buf = []
            val_buf = []
            video_frames = []   # store Pygame frames for this rollout


            # fresh episode at start of rollout
            world.manager.reset_episode()

            t = 0
            while t < rollout_horizon:
                agents = world.manager.controlled_agents
                if len(agents) == 0:
                    print("[PPO] No controlled agents, resetting episode...")
                    world.manager.reset_episode()
                    agents = world.manager.controlled_agents
                    if len(agents) == 0:
                        raise RuntimeError("No controlled agents available after reset().")

                # Build obs for all controlled agents
                obs_list = []
                for agent in agents:
                    obs_vec = world.manager.get_agent_cmpe_style_obs(agent)
                    obs_list.append(obs_vec)
                obs_np = np.stack(obs_list, axis=0).astype(np.float32)  # (N, obs_dim)

                # Query PPO policy
                actions_np, logp_np, values_np = policy.act(
                    obs_np, deterministic=False
                )

                # Convert to CARLA controls
                carla_actions = decode_continuous_actions(actions_np)
                world.manager.apply_actions_to_controlled(carla_actions)

                # Step CARLA
                world.tick()

                if config.get_if_route_planning():
                    world.manager.visualize_path()

                # Rewards and dones
                rewards, dones = world.manager.get_rewards_and_dones()  # (N,), (N,)

                # ---------- Pygame visualization ----------
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

                screen.fill((0, 0, 0))

                ego_cam_rgb = None
                seg_img = None
                depth_img = None

                if len(world.manager.controlled_agents) > 0:
                    ego_agent = world.manager.controlled_agents[0]
                    if ego_agent is not None and getattr(ego_agent, "vehicle", None) is not None:
                        if hasattr(ego_agent, "get_forward_latest"):
                            ego_cam_rgb = ego_agent.get_forward_latest()
                        if hasattr(ego_agent, "get_seg_latest"):
                            seg_img = ego_agent.get_seg_latest()
                        if hasattr(ego_agent, "get_depth_latest"):
                            depth_img = ego_agent.get_depth_latest()

                # ---- draw RGB ----
                if ego_cam_rgb is not None:
                    rgb_arr = np.asarray(ego_cam_rgb, dtype=np.uint8)
                    if rgb_arr.ndim == 3 and rgb_arr.shape[2] >= 3:
                        rgb_vis = rgb_arr[:, :, :3]
                        rgb_surface = pygame.surfarray.make_surface(
                            rgb_vis.swapaxes(0, 1)
                        )
                        rgb_surface = pygame.transform.scale(rgb_surface, (480, 360))
                        screen.blit(rgb_surface, (0, 0))

                # ---- draw seg ----
                if seg_img is not None:
                    seg_arr = np.asarray(seg_img)
                    if seg_arr.ndim == 2:
                        max_val = float(seg_arr.max()) if seg_arr.max() > 0 else 1.0
                        seg_norm = (seg_arr.astype(np.float32) / max_val) * 255.0
                        seg_norm = seg_norm.astype(np.uint8)
                        seg_vis = np.stack([seg_norm] * 3, axis=-1)
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

                # ---- draw depth ----
                if depth_img is not None:
                    depth_arr = np.asarray(depth_img, dtype=np.float32)
                    if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
                        depth_arr = depth_arr[:, :, 0]
                    elif depth_arr.ndim == 3 and depth_arr.shape[2] >= 3:
                        depth_arr = depth_arr[:, :, 0]

                    if depth_arr.ndim == 2:
                        depth_cap = 100.0
                        depth_norm = np.clip(depth_arr / depth_cap, 0.0, 1.0)
                        depth_vis = (depth_norm * 255.0).astype(np.uint8)
                        depth_vis = np.stack([depth_vis] * 3, axis=-1)
                    else:
                        depth_vis = None

                    if depth_vis is not None:
                        depth_surface = pygame.surfarray.make_surface(
                            depth_vis.swapaxes(0, 1)
                        )
                        depth_surface = pygame.transform.scale(depth_surface, (320, 180))
                        screen.blit(depth_surface, (480, 180))

                                # ---- draw text (status, horizontally) ----
                status_items = [
                    f"Upd {update_idx + 1}",
                    f"t {t}",
                    f"Step {global_step}",
                ]
                if rewards is not None and rewards.shape[0] > 0:
                    status_items.append(f"R0 {float(rewards[0]):.3f}")

                text_y = 380
                text_x = 10
                for item in status_items:
                    surf = font.render(item, True, (255, 255, 255))
                    screen.blit(surf, (text_x, text_y))
                    text_x += surf.get_width() + 20  # move right

                # ---- ego obs (still useful, but compact) ----
                ego_obs = obs_np[0] if obs_np.shape[0] > 0 else None
                if ego_obs is not None:
                    text_y += 22
                    text_x = 10
                    label_surf = font.render("Ego[0..9]:", True, (200, 200, 200))
                    screen.blit(label_surf, (text_x, text_y))
                    text_x += label_surf.get_width() + 10

                    for i in range(min(len(ego_obs), 10)):
                        item = f"{ego_obs[i]: .2f}"
                        surf = font.render(item, True, (200, 200, 200))
                        screen.blit(surf, (text_x, text_y))
                        text_x += surf.get_width() + 8

                # ---- reward / penalty events (also horizontal) ----
                if len(world.manager.controlled_agents) > 0:
                    ego_agent = world.manager.controlled_agents[0]
                    events = getattr(ego_agent, "last_reward_events", [])
                    if events:
                        text_y += 24
                        text_x = 10
                        header_surface = font.render("Events:", True, (200, 200, 200))
                        screen.blit(header_surface, (text_x, text_y))
                        text_x += header_surface.get_width() + 10

                        for ev in events:
                            ev_str = str(ev).strip()
                            if ev_str.startswith("+"):
                                color = (0, 255, 0)
                            elif ev_str.startswith("-"):
                                color = (255, 0, 0)
                            else:
                                color = (200, 200, 200)

                            ev_surface = font.render(ev_str, True, color)
                            screen.blit(ev_surface, (text_x, text_y))
                            text_x += ev_surface.get_width() + 15


                pygame.display.flip()
                # Capture the current Pygame window into the video buffer
                frame_surface = pygame.display.get_surface()
                if frame_surface is not None:
                    frame_array = pygame.surfarray.array3d(frame_surface)  # (W, H, 3)
                    frame_array = np.transpose(frame_array, (1, 0, 2))      # → (H, W, 3)
                    video_frames.append(frame_array)


                # Store rollout data
                obs_buf.append(obs_np)
                act_buf.append(actions_np)
                logp_buf.append(logp_np)
                rew_buf.append(rewards.astype(np.float32))
                done_buf.append(dones.astype(np.float32))
                val_buf.append(values_np.astype(np.float32))

                global_step += 1
                t += 1

            # Save rollout video
            if len(video_frames) > 0:
                video_path = videos_dir / f"update_{update_idx + 1:04d}.mp4"
                try:
                    imageio.mimsave(video_path, video_frames, fps=20)
                    print(f"[PPO] Saved video for update {update_idx + 1} to {video_path}")
                except Exception as e:
                    print(f"[PPO] Failed to save video for update {update_idx + 1}: {e}")


            # ---- end of rollout, build PPO batch ----
            obs_arr = np.stack(obs_buf, axis=0)      # (T, N, obs_dim)
            act_arr = np.stack(act_buf, axis=0)      # (T, N, 2)
            logp_arr = np.stack(logp_buf, axis=0)    # (T, N)
            rew_arr = np.stack(rew_buf, axis=0)      # (T, N)
            done_arr = np.stack(done_buf, axis=0)    # (T, N)
            val_arr = np.stack(val_buf, axis=0)      # (T, N)

            last_obs = obs_arr[-1]  # (N, obs_dim)
            _, _, last_val_np = policy.act(last_obs, deterministic=True)  # (N,)

            returns, advantages = policy.compute_gae(
                rewards=rew_arr,
                values=val_arr,
                dones=done_arr,
                last_value=last_val_np,
            )

            T, N, _ = obs_arr.shape
            obs_flat = obs_arr.reshape(T * N, -1)
            act_flat = act_arr.reshape(T * N, -1)
            logp_flat = logp_arr.reshape(T * N)
            ret_flat = returns.reshape(T * N)
            adv_flat = advantages.reshape(T * N)

            batch = {
                "obs": obs_flat,
                "actions": act_flat,
                "log_probs": logp_flat,
                "returns": ret_flat,
                "advantages": adv_flat,
            }

            policy.update(batch, epochs=ppo_epochs, batch_size=ppo_batch_size)
            avg_return = rew_arr.sum(axis=0).mean()
            print(
                f"[PPO] Update {update_idx + 1}/{num_updates}, "
                f"global_step={global_step}, avg_return={avg_return:.2f}"
            )

            # Save policy periodically
            if (update_idx + 1) % 20 == 0:  # was 50; 10 is nicer to see stuff
                save_path = policies_dir / f"ppo_policy_update_{update_idx + 1:04d}.pt"
                torch.save(policy.state_dict(), save_path)
                print(f"[PPO] Saved policy to {save_path}")


    except Exception as e:
        # Catch anything and print it clearly
        import traceback
        print("[PPO] ERROR during training:", repr(e))
        traceback.print_exc()
    finally:
        world.cleanup()


if __name__ == "__main__":
    main()
