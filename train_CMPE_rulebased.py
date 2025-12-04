#!/usr/bin/env python3
import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import carla
import matplotlib.cm as cm
import numpy as np
import time
import random
import math
import queue
import pygame
import torch
import torch.nn as nn
from controlled_agent import Controlled_Agents
from config import ConfigLoader


from rulebased_policy import RuleBasedPolicy
from unet_depth_vis import make_unet_model, H, W, MAX_DEPTH_METERS  # make sure unet_depth_vis exposes these
from unet_seg_vis import make_seg_unet_model, predict_segmentation, numpy_to_pygame_surface as seg_np2surf


#"magma", "inferno", "plasma", "viridis"
COLOR_MAP_FOR_DEPTH = "viridis"

def decode_continuous_actions(raw_actions: np.ndarray) -> np.ndarray:
    """
    raw_actions: (N, 2) in [-1, 1]
        raw[:, 0] = steer      ([-1, 1])
        raw[:, 1] = accel      ([-1, 1])  positive: throttle, negative: brake

    Returns controls: (N, 3)
        [:, 0] = steer in [-1, 1]
        [:, 1] = throttle in [0, 1]
        [:, 2] = brake in [0, 1]
    """
    assert raw_actions.ndim == 2 and raw_actions.shape[1] == 2, \
        f"Expected (N, 2) actions, got {raw_actions.shape}"

    steer = np.clip(raw_actions[:, 0], -1.0, 1.0)

    accel = np.clip(raw_actions[:, 1], -1.0, 1.0)
    throttle = np.clip(accel, 0.0, 1.0)
    brake = np.clip(-accel, 0.0, 1.0)

    controls = np.stack([steer, throttle, brake], axis=-1)
    return controls

def latlon_to_xy(lat, lon, lat0, lon0):
    """
    Approximate conversion from (lat, lon) to local (x, y) in meters
    relative to (lat0, lon0) using an equirectangular approximation.

    x: east, y: north
    """
    # Earth radius in meters
    R = 6378137.0
    # degrees -> radians
    lat_rad  = math.radians(lat)
    lon_rad  = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    x = R * dlon * math.cos((lat_rad + lat0_rad) * 0.5)
    y = R * dlat
    return x, y


def init_pygame(width=960, height=540):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Rule-Based Agent + UNet Depth")
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()
    return screen, font, clock


def carla_rgb_to_numpy(image) -> np.ndarray:
    """
    Convert either:
      - carla.Image (BGRA uint8)  -> RGB (H, W, 3) uint8
      - numpy.ndarray (H, W, C)   -> RGB (H, W, 3) uint8 (passes through / fixes channels)

    No rotation; matches training orientation.
    """
    # Case 1: CARLA Image object
    if hasattr(image, "raw_data"):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb = array[:, :, :3]          # drop alpha
        rgb = rgb[:, :, ::-1]          # BGRA -> RGB
        return rgb

    # Case 2: already a numpy array
    if isinstance(image, np.ndarray):
        arr = image
        # If it's BGRA, convert to RGB
        if arr.ndim == 3 and arr.shape[2] == 4:
            rgb = arr[:, :, :3][:, :, ::-1]   # drop alpha + BGR -> RGB
        elif arr.ndim == 3 and arr.shape[2] == 3:
            rgb = arr                         # assume already RGB
        else:
            raise ValueError(f"Unsupported numpy image shape {arr.shape}")
        # ensure uint8
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    raise TypeError(f"Unsupported image type: {type(image)}")


def numpy_to_pygame_surface(arr: np.ndarray) -> pygame.Surface:
    """
    Convert (H, W, 3) uint8 numpy array to a pygame Surface.
    pygame.surfarray.make_surface expects (W, H, 3).
    """
    return pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))


def predict_depth(model: nn.Module, rgb_np: np.ndarray) -> np.ndarray:
    """
    Run the UNet on a single RGB frame.

    rgb_np: (H, W, 3) uint8
    returns: (H, W, 3) uint8 grayscale depth image
    """
    device = next(model.parameters()).device

    img = rgb_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)      # (1, 1, H, W)
        pred = torch.clamp(pred, 0.0, MAX_DEPTH_METERS)
        depth = pred.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

    depth_norm = np.clip(depth / MAX_DEPTH_METERS, 0.0, 1.0)


    cmap = cm.get_cmap(COLOR_MAP_FOR_DEPTH)

    # cmap returns RGBA in [0,1], shape (H, W, 4)
    depth_color = cmap(depth_norm)

    # Drop alpha, convert to uint8 RGB
    depth_rgb = (depth_color[..., :3] * 255.0).astype(np.uint8)  # (H, W, 3)
    return depth_rgb



def draw_visualizer(
    screen,
    font,
    rgb_surface: pygame.Surface,
    depth_surface: pygame.Surface,
    seg_surface: pygame.Surface,
    path_points,          # list[(rel_x, rel_y)] from spawn
    route_points,         # list[(rel_x, rel_y)] from spawn
    rel_x,
    rel_y,
    steer,
    throttle,
    brake,
    speed,
    step,
    sensor_names,
    current_wp_idx: int,
    final_wp_idx: int,
):
    """
    Layout:
      Top row (3 columns): RGB | Depth | Seg
      Bottom row:
        - Left : minimap centered on ego, with route & past trajectory
        - Right: text HUD
    """
    screen.fill((0, 0, 0))
    W_screen, H_screen = screen.get_size()

    # Split vertical space
    top_h = int(H_screen * 0.6)
    bottom_h = H_screen - top_h

    # --- Top row: three image panes ---
    col_w = W_screen // 3
    rect_rgb   = pygame.Rect(0 * col_w, 0, col_w, top_h)
    rect_depth = pygame.Rect(1 * col_w, 0, col_w, top_h)
    rect_seg   = pygame.Rect(2 * col_w, 0, col_w, top_h)

    def blit_scaled(surface, rect):
        if surface is None:
            pygame.draw.rect(screen, (40, 40, 40), rect)
            return
        w, h = surface.get_size()
        scale = min(rect.width / w, rect.height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        surf_scaled = pygame.transform.smoothscale(surface, (new_w, new_h))
        pos = (
            rect.left + (rect.width - new_w) // 2,
            rect.top + (rect.height - new_h) // 2,
        )
        screen.blit(surf_scaled, pos)

    blit_scaled(rgb_surface, rect_rgb)
    blit_scaled(depth_surface, rect_depth)
    blit_scaled(seg_surface, rect_seg)

    # --- Bottom row: minimap + text ---
    bottom_top = top_h

    bottom_left_w = W_screen // 2
    minimap_size = min(bottom_left_w - 20, bottom_h - 20)
    minimap_rect = pygame.Rect(
        10,
        bottom_top + (bottom_h - minimap_size) // 2,
        minimap_size,
        minimap_size,
    )

    text_area_rect = pygame.Rect(
        bottom_left_w + 10,
        bottom_top + 10,
        W_screen - bottom_left_w - 20,
        bottom_h - 20,
    )

        # ---- Minimap ----
    pygame.draw.rect(screen, (20, 20, 20), minimap_rect, border_radius=5)
    cx = minimap_rect.centerx
    cy = minimap_rect.centery

    scale = 2.0  # pixels per meter

    def world_to_mini(px, py):
        # center ego
        dx = px - rel_x
        dy = py - rel_y
        sx = cx + int(dx * scale)
        sy = cy - int(dy * scale)
        return sx, sy

    # --- Route waypoints (default + special colors) ---
    for i, (px, py) in enumerate(route_points):
        sx, sy = world_to_mini(px, py)
        if not minimap_rect.collidepoint(sx, sy):
            continue

        # Default color: yellow
        color = (255, 255, 0)
        radius = 2

        # Final waypoint: magenta, larger
        if i == final_wp_idx and final_wp_idx >= 0:
            color = (255, 0, 255)
            radius = 6

        # Current waypoint: orange, medium
        if i == current_wp_idx and current_wp_idx >= 0:
            color = (255, 140, 0)
            radius = 4

        pygame.draw.circle(screen, color, (sx, sy), radius)

    # --- Past path (blue polyline) ---
    if len(path_points) > 1:
        prev = None
        for px, py in path_points:
            sx, sy = world_to_mini(px, py)
            if minimap_rect.collidepoint(sx, sy):
                pygame.draw.circle(screen, (0, 120, 255), (sx, sy), 2)
                if prev is not None:
                    pygame.draw.line(screen, (0, 120, 255), prev, (sx, sy), 2)
                prev = (sx, sy)
            else:
                prev = None

    # Ego marker (green)
    ego_sx, ego_sy = world_to_mini(rel_x, rel_y)
    if minimap_rect.collidepoint(ego_sx, ego_sy):
        pygame.draw.circle(screen, (0, 255, 0), (ego_sx, ego_sy), 6)

    pygame.draw.rect(screen, (100, 100, 100), minimap_rect, width=1, border_radius=5)


    # ---- Text HUD ----
    lines = [
        f"step: {step}",
        f"speed: {speed:.2f} m/s",
        f"steer: {steer:.2f}",
        f"throttle: {throttle:.2f}",
        f"brake: {brake:.2f}",
        "",
        "Sensors:",
    ] + [f" - {name}" for name in sensor_names]

    x = text_area_rect.left
    y = text_area_rect.top
    for line in lines:
        surf = font.render(line, True, (255, 255, 255))
        screen.blit(surf, (x, y))
        y += 22

    pygame.display.flip()


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Take control of world ticking here.
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS sim
    world.apply_settings(settings)

    actors = []
    screen, font, clock = init_pygame(width=3 * 640, height=720)

    depth_unet = make_unet_model()
    seg_unet = make_seg_unet_model()

    policy = RuleBasedPolicy()
    policy.reset()

    latest_cam_image = None

    sensor_names = ["RGB Camera (front)"]

    try:
        # ---- Spawn vehicle via our custom Controlled_Agents ----
        # Adjust `config` according to what your Controlled_Agents expects.
        # For a quick start, an empty dict often works if you handle defaults inside the class.
        config_path = Path("config_cmpe.yaml")
        config = ConfigLoader(config_path)

        controlled_agent = Controlled_Agents(world, index=0, config=config)
        # Underlying CARLA vehicle; adjust attribute name if needed:
        vehicle = controlled_agent.vehicle

        print(f"Spawned Controlled_Agents vehicle: {vehicle.id}")
        vehicle.set_autopilot(False)

        # For cleanup later
        actors.append(controlled_agent)

        # ---- GNSS-based frame: use first GNSS fix as origin ----
        gnss_origin = None  # (lat0, lon0)
        print("Waiting for first GNSS fix...")

        # Tick until we get a GNSS reading
        while gnss_origin is None:
            world.tick()
            gnss = controlled_agent.get_gnss_latest()
            if gnss is not None:
                lat0, lon0, alt0 = gnss
                gnss_origin = (lat0, lon0)
                print(f"GNSS origin set to lat={lat0:.7f}, lon={lon0:.7f}")
                break

        # ---- Precompute route waypoints in *GNSS-relative* coordinates ----
        route_points_rel = []
        route_geo = controlled_agent.get_route_geo()  # list[(lat, lon, alt)]
        if route_geo:
            lat0, lon0 = gnss_origin
            for (lat, lon, alt) in route_geo:
                x, y = latlon_to_xy(lat, lon, lat0, lon0)
                route_points_rel.append((x, y))

        # Ego path (GNSS-relative)
        path_points = []

        step = 0
        running = True
        
        # Prime the world once so sensors start producing data
        world.tick()

        latest_rgb_surface = None
        latest_depth_surface = None
        latest_seg_surface = None

        while running:
            # Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            latest_cam_image = controlled_agent.get_forward_latest()
            gnss = controlled_agent.get_gnss_latest()
            if gnss is not None:
                lat, lon, alt = gnss
                # for now just print/debug
                if step % 50 == 0:
                    print(f"GNSS: lat={lat:.7f}, lon={lon:.7f}, alt={alt:.2f}")

            if latest_cam_image is not None:
                # Convert to numpy for UNet
                rgb_np = carla_rgb_to_numpy(latest_cam_image)

                # Ensure correct size for UNet
                if rgb_np.shape[0] != H or rgb_np.shape[1] != W:
                    import cv2
                    rgb_np = cv2.resize(rgb_np, (W, H), interpolation=cv2.INTER_NEAREST)

                depth_rgb = predict_depth(depth_unet, rgb_np)
                seg_rgb = predict_segmentation(seg_unet, rgb_np)

                # Convert to pygame surfaces
                latest_rgb_surface = numpy_to_pygame_surface(rgb_np)
                latest_depth_surface = numpy_to_pygame_surface(depth_rgb)
                latest_seg_surface = seg_np2surf(seg_rgb)

            # Build obs later; for now ignore
            obs = None
            raw_actions = policy.act(obs, deterministic=True)  # (1, 2)
            steer, throttle, brake = decode_continuous_actions(raw_actions)[0]

            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            control.hand_brake = False
            control.reverse = False
            vehicle.apply_control(control)

            # ---- Ego position from GNSS only ----
            gnss = controlled_agent.get_gnss_latest()
            if gnss is not None:
                lat, lon, alt = gnss
                lat0, lon0 = gnss_origin
                rel_x, rel_y = latlon_to_xy(lat, lon, lat0, lon0)

                path_points.append((rel_x, rel_y))
                if len(path_points) > 1000:
                    path_points.pop(0)
            else:
                # If GNSS not ready (rare after origin set), keep last rel_x/rel_y
                rel_x = rel_x if "rel_x" in locals() else 0.0
                rel_y = rel_y if "rel_y" in locals() else 0.0

            vel = vehicle.get_velocity() #speedometer
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            if step % 50 == 0:
                print(
                    f"[step {step}] steer={steer:.2f}, thr={throttle:.2f}, "
                    f"brk={brake:.2f}, speed={speed:.2f} m/s, rel=({rel_x:.1f},{rel_y:.1f})"
                )

                        # Waypoint indices for coloring
            if route_points_rel:
                current_wp_idx = getattr(controlled_agent, "current_wp_index", 0)
                current_wp_idx = max(0, min(current_wp_idx, len(route_points_rel) - 1))
                final_wp_idx = len(route_points_rel) - 1
            else:
                current_wp_idx = -1
                final_wp_idx = -1

            draw_visualizer(
                screen,
                font,
                latest_rgb_surface,
                latest_depth_surface,
                latest_seg_surface,
                path_points,
                route_points_rel,
                rel_x,
                rel_y,
                steer,
                throttle,
                brake,
                speed,
                step,
                sensor_names,
                current_wp_idx,
                final_wp_idx,
            )


            step += 1

            # Limit visualization FPS
            clock.tick(60)

            # 🔁 Advance the world to generate the *next* frame
            world.tick()

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        print("Cleaning up actors and restoring world settings...")
        for a in actors:
            try:
                if isinstance(a, Controlled_Agents):
                    a.destroy_agent()  # or whatever your destructor is called
                else:
                    a.destroy()
            except Exception:
                pass

        world.apply_settings(original_settings)
        pygame.quit()
        print("Done.")


if __name__ == "__main__":
    main()
