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
from unet_seg_vis import make_seg_unet_model, predict_segmentation, predict_segmentation_ids, numpy_to_pygame_surface as seg_np2surf


#"magma", "inferno", "plasma", "viridis"
COLOR_MAP_FOR_DEPTH = "viridis"

# --- Ego state machine constants ---
STATE_NORMAL = "normal"
STATE_APPROACH_GOAL = "approach_goal"
STATE_STOP = "stop"

# --- Obstacle state machine constants ---
OBST_CLEAR = "clear"
OBST_SLOW  = "slow"
OBST_STOP  = "stop_for_obstacle"

# Obstacle distance thresholds (meters) for behavior
OBST_STOP_DIST = 6.0    # <= this -> full stop
OBST_SLOW_DIST = 12.0   # <= this -> slow
OBST_CLEAR_DIST = 25.0  # beyond this -> totally clear


TARGET_SPEED_NORMAL = 5.0   # m/s ~ 36 km/h
TARGET_SPEED_APPROACH = 3.0  # m/s ~ 18 km/h
STOP_DISTANCE_M = 5        # within 3m of goal -> stop
APPROACH_DISTANCE_M = 20.0   # within 20m of goal -> approach_goal

LOOKAHEAD_STEPS = 1    # how far ahead along the route to aim
K_STEER_PP      = 1.0  # gain for pure-pursuit steering (tune 0.8–1.5)
WHEELBASE_M     = 2.8  # approximate vehicle wheelbase for curvature calc

# Segmentation classes we care about
PEDESTRIAN_CLASS = 4
VEHICLE_CLASS    = 10

ROADLINE_CLASS = 6
TRAFFIC_SIGN_CLASS = 12
OBSTACLE_CLASSES = {PEDESTRIAN_CLASS, VEHICLE_CLASS}

# ROI for "in our way" region in image coordinates (for seg_ids with shape (H, W))
# Tune these based on your camera FOV / mounting:
ROI_Y_START = int(H * 0.4)   # start halfway down (ignore far away stuff)
ROI_Y_END   = H              # bottom of image (near to car)
ROI_X_START = int(W * 0.35)  # narrow-ish central band
ROI_X_END   = int(W * 0.65)

WP_REACHED_DIST = 3       # how close we must get to a route point to "reach" it


current_wp_idx = -1
final_wp_idx = -1



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

def compute_dist_to_goal(rel_x, rel_y, route_points_rel):
    """
    Distance from ego (rel_x, rel_y) to final route point (in the same GNSS-local frame).
    Returns None if no route.
    """
    if not route_points_rel:
        return None
    goal_x, goal_y = route_points_rel[-1]
    return math.hypot(goal_x - rel_x, goal_y - rel_y)


def select_current_waypoint(rel_x, rel_y, route_points_rel, search_radius=50.0) -> int:
    """
    Pick the closest route waypoint to the ego in GNSS-local frame.
    Returns index in [0, len-1], or -1 if route is empty.
    """
    if not route_points_rel:
        return -1

    best_idx = -1
    best_dist2 = float("inf")
    max_d2 = search_radius * search_radius

    # First pass: closest within search radius
    for i, (wx, wy) in enumerate(route_points_rel):
        dx = wx - rel_x
        dy = wy - rel_y
        d2 = dx * dx + dy * dy
        if d2 < best_dist2 and d2 <= max_d2:
            best_dist2 = d2
            best_idx = i

    # If none within radius, fall back to global closest
    if best_idx == -1:
        for i, (wx, wy) in enumerate(route_points_rel):
            dx = wx - rel_x
            dy = wy - rel_y
            d2 = dx * dx + dy * dy
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = i

    return best_idx


def update_ego_state(current_state, dist_to_goal, speed) -> str:
    """
    Simple state transition logic based on distance to goal (GNSS) and speed.
    For now:
      - if no route: default to normal
      - if very close to goal: STOP
      - else if moderately close: APPROACH_GOAL
      - else: NORMAL
    """
    if dist_to_goal is None:
        return STATE_NORMAL

    if dist_to_goal <= STOP_DISTANCE_M:
        return STATE_STOP
    elif dist_to_goal <= APPROACH_DISTANCE_M:
        return STATE_APPROACH_GOAL
    else:
        return STATE_NORMAL


def update_obstacle_state_from_depth(current_state: str, min_depth) -> str:
    """
    Decide obstacle state from the nearest obstacle depth (in meters).

    min_depth:
      None -> no obstacle
      small -> very close
    """
    if min_depth is None:
        return OBST_CLEAR

    if min_depth <= OBST_STOP_DIST:
        return OBST_STOP
    elif min_depth <= OBST_SLOW_DIST:
        return OBST_SLOW
    else:
        return OBST_CLEAR



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


def predict_depth(model: nn.Module, rgb_np: np.ndarray):
    """
    Run the UNet on a single RGB frame.

    rgb_np: (H, W, 3) uint8
    returns:
      depth_rgb : (H, W, 3) uint8 colored depth for visualization
      depth_map : (H, W) float32 in meters
    """
    device = next(model.parameters()).device

    img = rgb_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)      # (1, 1, H, W)
        pred = torch.clamp(pred, 0.0, MAX_DEPTH_METERS)
        depth = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)  # (H, W)

    depth_norm = np.clip(depth / MAX_DEPTH_METERS, 0.0, 1.0)

    cmap = cm.get_cmap(COLOR_MAP_FOR_DEPTH)
    depth_color = cmap(depth_norm)                    # (H, W, 4)
    depth_rgb = (depth_color[..., :3] * 255.0).astype(np.uint8)  # (H, W, 3)

    return depth_rgb, depth     # <--- new: also return metric depth


def spawn_vehicles_and_walkers(
    client: carla.Client,
    world: carla.World,
    num_vehicles: int = 15,
    num_walkers: int = 30,
) -> list:
    """
    Spawn some background traffic:
      - num_vehicles vehicles with autopilot
      - num_walkers pedestrians with AI controllers

    Returns a flat list of actors (vehicles + walkers + walker_controllers)
    so the caller can destroy them on shutdown.
    """
    spawned_actors = []

    blueprint_library = world.get_blueprint_library()
    tm = client.get_trafficmanager()  # use default TM port
    tm.set_synchronous_mode(True)

    # ------------- Vehicles -------------
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicle_blueprints = blueprint_library.filter("vehicle.*")

    vehicles_to_spawn = min(num_vehicles, len(spawn_points))
    for i in range(vehicles_to_spawn):
        bp = random.choice(vehicle_blueprints)

        # Avoid bikes if you want "proper" vehicles only:
        if bp.has_attribute("number_of_wheels"):
            wheels = int(bp.get_attribute("number_of_wheels"))
            if wheels < 4:
                continue

        # safer: make sure we have a spawn point
        transform = spawn_points[i]
        try:
            veh = world.try_spawn_actor(bp, transform)
            if veh is not None:
                veh.set_autopilot(True, tm.get_port())
                spawned_actors.append(veh)
        except RuntimeError:
            continue

    # ------------- Walkers -------------
    walker_blueprints = blueprint_library.filter("walker.pedestrian.*")
    walker_actors = []
    walker_controllers = []

    for i in range(num_walkers):
        # Pick a random navmesh position
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue

        walker_bp = random.choice(walker_blueprints)
        # optional: randomize speed if attribute exists
        if walker_bp.has_attribute("speed"):
            speed = random.uniform(
                float(walker_bp.get_attribute("speed").recommended_values[1]),
                float(walker_bp.get_attribute("speed").recommended_values[-1]),
            )
            walker_bp.set_attribute("speed", f"{speed:.2f}")

        transform = carla.Transform(loc, carla.Rotation(0, random.uniform(-180, 180), 0))
        try:
            walker = world.try_spawn_actor(walker_bp, transform)
        except RuntimeError:
            walker = None

        if walker is None:
            continue

        walker_actors.append(walker)
        spawned_actors.append(walker)

    # Now spawn controllers for each walker
    controller_bp = blueprint_library.find("controller.ai.walker")
    for walker in walker_actors:
        try:
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        except RuntimeError:
            controller = None

        if controller is None:
            continue

        walker_controllers.append(controller)
        spawned_actors.append(controller)

    # Start walker controllers and give them random destinations
    world.tick()
    for controller in walker_controllers:
        controller.start()
        controller.set_max_speed(1.5 + random.random())  # m/s
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            controller.go_to_location(dest)

    return spawned_actors



def draw_visualizer(
    screen,
    font,
    rgb_surface: pygame.Surface,
    depth_surface: pygame.Surface,
    seg_surface: pygame.Surface,
    path_points,
    route_points,
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
    ego_state: str,
    obstacle_state: str,
    min_obstacle_depth: float,
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

    def blit_scaled_with_roi(surface, rect, draw_roi=False):
        """
        Draws a surface scaled into rect.
        If draw_roi is True, overlay the (ROI_X/Y_*) box in green,
        assuming ROI_* are in original image coords.
        """
        if surface is None:
            pygame.draw.rect(screen, (40, 40, 40), rect)
            return

        # Original image size
        img_w, img_h = surface.get_size()

        # Scale to fit inside rect while keeping aspect ratio
        scale = min(rect.width / img_w, rect.height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        surf_scaled = pygame.transform.smoothscale(surface, (new_w, new_h))

        # Center it inside the rect
        pos_x = rect.left + (rect.width - new_w) // 2
        pos_y = rect.top + (rect.height - new_h) // 2

        screen.blit(surf_scaled, (pos_x, pos_y))

        if draw_roi:
            # ROI is defined in original image coordinates (W, H)
            # Map to screen coordinates using same scale + offset
            roi_x = int(ROI_X_START * scale)
            roi_y = int(ROI_Y_START * scale)
            roi_w = int((ROI_X_END - ROI_X_START) * scale)
            roi_h = int((ROI_Y_END - ROI_Y_START) * scale)

            roi_rect = pygame.Rect(
                pos_x + roi_x,
                pos_y + roi_y,
                roi_w,
                roi_h,
            )
            pygame.draw.rect(screen, (0, 255, 0), roi_rect, width=2)

    # Draw RGB with ROI box overlay
    blit_scaled_with_roi(rgb_surface, rect_rgb, draw_roi=True)

    # Depth & seg: no ROI box (or set True too if you want)
    blit_scaled_with_roi(depth_surface, rect_depth, draw_roi=False)
    blit_scaled_with_roi(seg_surface, rect_seg, draw_roi=False)


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
        f"route_state: {ego_state}",
        f"obstacle_state: {obstacle_state}",
        f"steer: {steer:.2f}",
        f"throttle: {throttle:.2f}",
        f"brake: {brake:.2f}",
        "",
        # GNSS-local debug
        f"ego_rel_xy: ({rel_x:.1f}, {rel_y:.1f}) m",
        f"wp_idx: {current_wp_idx}",
    ]
    if min_obstacle_depth is None:
        lines.append("min_obs_depth: None")
    else:
        lines.append(f"min_obs_depth: {min_obstacle_depth:.1f} m")

    # If we have a valid current waypoint, show its position and distance
    if 0 <= current_wp_idx < len(route_points):
        wx, wy = route_points[current_wp_idx]
        dx_wp = wx - rel_x
        dy_wp = wy - rel_y
        dist_wp = math.hypot(dx_wp, dy_wp)
        lines.append(
            f"wp_rel_xy: ({wx:.1f}, {wy:.1f}) m"
        )
        lines.append(
            f"wp_offset_from_ego: (dx={dx_wp:.1f}, dy={dy_wp:.1f}) m, dist={dist_wp:.1f} m"
        )

    lines += [
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

def compute_obstacle_proximity(seg_ids: np.ndarray) -> float:
    """
    Look in a central-bottom ROI of the segmentation map for pedestrians/vehicles.

    Returns a proximity score in [0, 1]:
      0.0 -> no obstacle in ROI
      1.0 -> obstacle at the very bottom (very close)
    """
    if seg_ids is None:
        return 0.0

    h, w = seg_ids.shape
    y0 = max(0, min(ROI_Y_START, h))
    y1 = max(0, min(ROI_Y_END,   h))
    x0 = max(0, min(ROI_X_START, w))
    x1 = max(0, min(ROI_X_END,   w))

    if y1 <= y0 or x1 <= x0:
        return 0.0

    roi = seg_ids[y0:y1, x0:x1]   # (roi_h, roi_w)

    # mask for "danger" pixels: pedestrians or vehicles
    mask = np.isin(roi, list(OBSTACLE_CLASSES))
    if not mask.any():
        return 0.0

    # We approximate "closer" by how low the obstacle pixels go in the ROI
    ys, xs = np.where(mask)
    max_y = ys.max()                      # 0 = top of ROI, roi_h-1 = bottom
    roi_h = roi.shape[0]
    proximity = max_y / max(1, roi_h-1)   # ~0 top, ~1 bottom

    return float(np.clip(proximity, 0.0, 1.0))

def compute_min_obstacle_depth(seg_ids: np.ndarray,
                               depth_map: np.ndarray,
                               use_roi: bool = True):
    """
    Use segmentation + depth to estimate the nearest obstacle distance.

    seg_ids   : (H, W) int class indices
    depth_map : (H, W) float32 in meters, aligned with seg_ids

    If use_roi:
      only consider a central-bottom ROI (same spirit as compute_obstacle_proximity),
      so we react mainly to obstacles actually in our lane & near in front.

    Returns:
      min_depth (float, meters) of any pedestrian/vehicle pixel in ROI,
      or None if no such pixels.
    """
    if seg_ids is None or depth_map is None:
        return None

    if seg_ids.shape != depth_map.shape:
        raise ValueError(f"seg_ids shape {seg_ids.shape} != depth_map shape {depth_map.shape}")

    if use_roi:
        h, w = seg_ids.shape
        y0 = max(0, min(ROI_Y_START, h))
        y1 = max(0, min(ROI_Y_END,   h))
        x0 = max(0, min(ROI_X_START, w))
        x1 = max(0, min(ROI_X_END,   w))

        if y1 <= y0 or x1 <= x0:
            return None  # bad ROI => no info

        seg_roi   = seg_ids[y0:y1, x0:x1]
        depth_roi = depth_map[y0:y1, x0:x1]
    else:
        seg_roi   = seg_ids
        depth_roi = depth_map

    # mask of all ped/vehicle pixels in ROI
    mask = np.isin(seg_roi, list(OBSTACLE_CLASSES))
    if not mask.any():
        return None

    obstacle_depths = depth_roi[mask]
    obstacle_depths = obstacle_depths[np.isfinite(obstacle_depths)]
    obstacle_depths = obstacle_depths[obstacle_depths > 0.1]  # ignore junk very close to 0

    if obstacle_depths.size == 0:
        return None

    return float(obstacle_depths.min())






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

        # For cleanup later
        actors.append(controlled_agent)
        # Spawn background NPC traffic (cars + pedestrians)
        npc_actors = spawn_vehicles_and_walkers(
            client,
            world,
            num_vehicles=80,
            num_walkers=80,
        )
        actors.extend(npc_actors)
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

        # Waypoint indices along the route
        current_wp_idx = 0 if route_points_rel else -1
        final_wp_idx = len(route_points_rel) - 1 if route_points_rel else -1


        # Ego state for state machine
        ego_state = STATE_NORMAL
        ego_yaw = 0.0
        step = 0
        running = True

        obstacle_state = OBST_CLEAR
        min_obstacle_depth = None

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

                depth_rgb, depth_map = predict_depth(depth_unet, rgb_np)
                seg_rgb = predict_segmentation(seg_unet, rgb_np)
                seg_ids = predict_segmentation_ids(seg_unet, rgb_np)  # (H, W) int32 or int64

                # Convert to pygame surfaces
                latest_rgb_surface = numpy_to_pygame_surface(rgb_np)
                latest_depth_surface = numpy_to_pygame_surface(depth_rgb)
                latest_seg_surface = seg_np2surf(seg_rgb)

                # 🔎 nearest obstacle distance (meters)
                min_obstacle_depth = compute_min_obstacle_depth(seg_ids, depth_map)

                # update obstacle state using distance
                obstacle_state = update_obstacle_state_from_depth(
                    obstacle_state,
                    min_obstacle_depth,
                )
            # ---- Ego position from GNSS only ----
            gnss = controlled_agent.get_gnss_latest()
            if gnss is not None:
                lat, lon, alt = gnss
                lat0, lon0 = gnss_origin
                rel_x, rel_y = latlon_to_xy(lat, lon, lat0, lon0)

                path_points.append((rel_x, rel_y))
                
                if len(path_points) > 1000:
                    path_points.pop(0)

                # Update ego heading from last GNSS step
                if len(path_points) >= 2:
                    x_prev, y_prev = path_points[-2]
                    x_curr, y_curr = path_points[-1]
                    if x_curr != x_prev or y_curr != y_prev:
                        ego_yaw = math.atan2(y_curr - y_prev, x_curr - x_prev)
                 # --- Initialize current_wp_idx from closest route point (first time only) ---
                if route_points_rel and current_wp_idx < 0:
                    current_wp_idx = select_current_waypoint(rel_x, rel_y, route_points_rel)
                    final_wp_idx = len(route_points_rel) - 1
                    print(f"[init] current_wp_idx set to {current_wp_idx}")


                        
            else:
                # If GNSS not ready (rare after origin set), keep last rel_x/rel_y
                rel_x = rel_x if "rel_x" in locals() else 0.0
                rel_y = rel_y if "rel_y" in locals() else 0.0

            
            # ---- Mark current waypoint as "reached" if we're close enough ----
            if route_points_rel and current_wp_idx >= 0:
                wx, wy = route_points_rel[current_wp_idx]
                dist_wp = math.hypot(wx - rel_x, wy - rel_y)

                if dist_wp < WP_REACHED_DIST and current_wp_idx < final_wp_idx:
                    current_wp_idx += 1


            vel = vehicle.get_velocity()  # speedometer
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # Distance to final GNSS waypoint (if any)
            dist_to_goal = compute_dist_to_goal(rel_x, rel_y, route_points_rel)

            # Update ego state based on GNSS distance + speed
            ego_state = update_ego_state(ego_state, dist_to_goal, speed)

            # # --- Select current waypoint index (GNSS-local), monotonic forward ---
            # if route_points_rel:
            #     raw_wp_idx = select_current_waypoint(rel_x, rel_y, route_points_rel)

            #     if current_wp_idx < 0:
            #         # first time
            #         current_wp_idx = raw_wp_idx
            #     else:
            #         # do not go backwards in the route
            #         current_wp_idx = max(current_wp_idx, raw_wp_idx)

            #     final_wp_idx = len(route_points_rel) - 1
            # else:
            #     current_wp_idx = -1
            #     final_wp_idx = -1

            # # --- Lateral control: follow route using heading error ---
            # if current_wp_idx >= 0:
            #     wx, wy = route_points_rel[current_wp_idx]
            #     dx_wp = wx - rel_x
            #     dy_wp = wy - rel_y

            #     # 1) Check if this waypoint is mostly behind the ego
            #     # ego heading vector from ego_yaw
            #     hx = math.cos(ego_yaw)
            #     hy = math.sin(ego_yaw)
            #     dot = dx_wp * hx + dy_wp * hy  # projection along heading

            #     # # if dot < 0, waypoint lies behind the car in heading frame
            #     # if dot < 0.0 and current_wp_idx < len(route_points_rel) - 1:
            #     #     # skip this waypoint once and aim for the next one instead
            #     #     current_wp_idx += 1
            #     #     wx, wy = route_points_rel[current_wp_idx]
            #     #     dx_wp = wx - rel_x
            #     #     dy_wp = wy - rel_y
            #     #     # (You could recompute dot here again if you want.)

            #     desired_yaw = math.atan2(dy_wp, dx_wp)
            #     # shortest signed angle difference
            #     yaw_error = (desired_yaw - ego_yaw + math.pi) % (2.0 * math.pi) - math.pi

            #     # if abs(yaw_error) > 0.7:  # ~40 degrees, tweak as you like
            #     #     print(
            #     #         f"[DEBUG yaw] step={step}, wp_idx={current_wp_idx}, "
            #     #         f"dx_wp={dx_wp:.1f}, dy_wp={dy_wp:.1f}, "
            #     #         f"ego_yaw={ego_yaw:.2f}, des_yaw={desired_yaw:.2f}, "
            #     #         f"yaw_err={yaw_error:.2f}"
            #     #     )

            #     # small dead-zone to avoid jitter
            #     if abs(yaw_error) < 0.02:  # ~1.1 degrees
            #         yaw_error = 0.0

            #     K_STEER = 0.3  # slightly softer
            #     steer = max(-1.0, min(1.0, K_STEER * yaw_error))
            # else:
            #     # fallback: just go straight for now
            #     steer = 0.0

            # --- Lateral control: pure-pursuit style on route ---
            if (ego_state == STATE_STOP) or (obstacle_state == OBST_STOP):
                # If we intend to be stopped (goal or hard obstacle),
                # don't keep spinning the steering wheel.
                steer = 0.0

            elif current_wp_idx >= 0 and route_points_rel:
                # 1) Choose a lookahead waypoint along the route
                lookahead_idx = min(current_wp_idx + LOOKAHEAD_STEPS, final_wp_idx)
                wx, wy = route_points_rel[lookahead_idx]

                # 2) Vector from ego to waypoint in world frame
                dx = wx - rel_x
                dy = wy - rel_y

                # 3) Transform into ego frame (x forward, y left)
                #    Rotate by -ego_yaw
                cos_y = math.cos(-ego_yaw)
                sin_y = math.sin(-ego_yaw)
                x_rel = dx * cos_y - dy * sin_y
                y_rel = dx * sin_y + dy * cos_y

                # Guard: if the point is slightly behind us due to noise,
                # clamp x_rel to small positive so we don't flip
                if x_rel < 0.5:
                    x_rel = 0.5

                # 4) Compute pure-pursuit curvature
                Ld = max(1.0, math.hypot(x_rel, y_rel))   # lookahead distance
                curvature = 2.0 * y_rel / (Ld * Ld)       # standard PP formula

                # 5) Map curvature to steering in [-1, 1]
                steer_cmd = curvature * WHEELBASE_M * K_STEER_PP
                steer = max(-1.0, min(1.0, -steer_cmd))

                # (Optional) tiny dead-zone to avoid steering noise on straight road
                if abs(steer) < 0.01:
                    steer = 0.0
            else:
                # fallback: just go straight
                steer = 0.0


            # --- Longitudinal control: route state -> desired route speed ---
            if ego_state == STATE_NORMAL:
                route_speed = TARGET_SPEED_NORMAL
            elif ego_state == STATE_APPROACH_GOAL:
                route_speed = TARGET_SPEED_APPROACH
            elif ego_state == STATE_STOP:
                route_speed = 0.0
            else:
                route_speed = TARGET_SPEED_NORMAL

            # --- Obstacle-driven speed cap ---
            if obstacle_state == OBST_CLEAR:
                obstacle_speed_cap = TARGET_SPEED_NORMAL
            elif obstacle_state == OBST_SLOW:
                obstacle_speed_cap = TARGET_SPEED_APPROACH
            elif obstacle_state == OBST_STOP:
                obstacle_speed_cap = 0.0
            else:
                obstacle_speed_cap = TARGET_SPEED_NORMAL

            target_speed = min(route_speed, obstacle_speed_cap)


            speed_error = target_speed - speed
            throttle = 0.0
            brake = 0.0
            if speed_error > 0.5:
                throttle = max(0.0, min(1.0, speed_error / 5.0))
            elif speed_error < -0.5:
                brake = max(0.0, min(1.0, -speed_error / 5.0))

            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            control.hand_brake = False
            control.reverse = False
            vehicle.apply_control(control)



            if step % 50 == 0:
                print(
                    f"[step {step}] state={ego_state}, dist={dist_to_goal or -1:.1f} m, "
                    f"steer={steer:.2f}, thr={throttle:.2f}, "
                    f"brk={brake:.2f}, speed={speed:.2f} m/s, rel=({rel_x:.1f},{rel_y:.1f}), "
                    f"wp={current_wp_idx}"
                )

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
                ego_state,
                obstacle_state,
                min_obstacle_depth,
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
