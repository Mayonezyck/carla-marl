#!/usr/bin/env python3
import sys
import numpy as np
import pygame
import carla

# ---------- USER SETTINGS ----------

# Each entry: (x, y, z, pitch, yaw, roll)
# Add/remove transforms as needed.
CAMERA_TRANSFORMS = [
    (0.0, 0.0, 30.0, -45.0,   0.0, 0.0),
    (-105.0, 15.0, 30.0, -45.0, -90.0, 0.0),
    (0.0, 50.0, 30.0, -45.0, 180.0, 0.0),
    (50.0, 50.0, 30.0, -45.0,  90.0, 0.0),
    # You can add up to 9; unused grid cells will be black.
    # (100.0, 0.0, 30.0, -45.0, 0.0, 0.0),
    # ...
]

# Per-camera resolution
CAM_W = 400
CAM_H = 300

# -----------------------------------

latest_surfaces = {}  # cam_index -> pygame surface


def make_image_callback(cam_index: int):
    def _callback(image):
        global latest_surfaces

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # BGRA -> RGB
        array = array[:, :, :3][:, :, ::-1]
        array = np.swapaxes(array, 0, 1)  # (width, height, 3)

        surface = pygame.surfarray.make_surface(array)
        latest_surfaces[cam_index] = surface

    return _callback


def get_grid_shape(num_cams: int):
    """
    Simple rule:
      - up to 4 cams -> 2x2
      - up to 9 cams -> 3x3
    """
    if num_cams <= 4:
        return 2, 2
    else:
        return 3, 3


def main():
    pygame.init()

    num_cams = len(CAMERA_TRANSFORMS)
    if num_cams == 0:
        print("No CAMERA_TRANSFORMS specified.")
        return

    grid_rows, grid_cols = get_grid_shape(num_cams)
    total_cells = grid_rows * grid_cols

    window_w = grid_cols * CAM_W
    window_h = grid_rows * CAM_H

    display = pygame.display.set_mode(
        (window_w, window_h),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("CARLA Multi-CCTV Viewer")
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # IMPORTANT:
    # We do NOT change world settings or tick the world here.

    actor_list = []

    try:
        bp_lib = world.get_blueprint_library()
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_W))
        cam_bp.set_attribute("image_size_y", str(CAM_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", "0.0")  # every server frame

        # Spawn cameras
        for i, (x, y, z, pitch, yaw, roll) in enumerate(CAMERA_TRANSFORMS):
            loc = carla.Location(x=x, y=y, z=z)
            rot = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
            transform = carla.Transform(loc, rot)

            cam = world.spawn_actor(cam_bp, transform)
            actor_list.append(cam)

            cam.listen(make_image_callback(i))

        running = True
        black_surface = pygame.Surface((CAM_W, CAM_H))
        black_surface.fill((0, 0, 0))

        while running:
            clock.tick(60)  # just limits pygame loop, not CARLA

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            display.fill((0, 0, 0))

            # Draw each cell in the grid
            for cell_idx in range(total_cells):
                row = cell_idx // grid_cols
                col = cell_idx % grid_cols

                x0 = col * CAM_W
                y0 = row * CAM_H

                if cell_idx < num_cams and cell_idx in latest_surfaces:
                    surf = latest_surfaces[cell_idx]
                    # Make sure it matches expected size
                    if surf.get_width() != CAM_W or surf.get_height() != CAM_H:
                        surf = pygame.transform.smoothscale(surf, (CAM_W, CAM_H))
                    display.blit(surf, (x0, y0))
                else:
                    # No camera for this cell or no frame yet -> black tile
                    display.blit(black_surface, (x0, y0))

            pygame.display.flip()

    finally:
        print("Cleaning up cameras...")
        for actor in actor_list:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass

        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser interrupted, exiting.")
        pygame.quit()
        sys.exit(0)
