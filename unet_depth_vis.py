import numpy as np
import pygame
import torch
import torch.nn as nn
import carla

from depth_train_UNet_multiVehicle import CARLA_UNet, H, W, MAX_DEPTH_METERS

CHECKPOINT_PATH = "unet_depth/unet_depth_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_unet_model():
    """Instantiate and load the trained CARLA_UNet."""
    model = CARLA_UNet(in_channels=3, n_filters=16, n_classes=1)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


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
    # To tensor, normalize to [0,1], shape (1, 3, H, W)
    img = rgb_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)      # (1, 1, H, W)
        pred = torch.clamp(pred, 0.0, MAX_DEPTH_METERS)
        depth = pred.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

    # Normalize depth to [0,1] and convert to grayscale uint8
    depth_norm = np.clip(depth / MAX_DEPTH_METERS, 0.0, 1.0)
    depth_gray = (depth_norm * 255.0).astype(np.uint8)  # (H, W)

    # Make it 3-channel grayscale for pygame
    depth_rgb = np.stack([depth_gray] * 3, axis=-1)  # (H, W, 3)
    return depth_rgb


def unet_visualization_step(world, screen, shared, depth_unet, clock=None, fps=20):
    """
    One visualization step:
      - tick the world
      - if we have an RGB frame in `shared["rgb"]`, run UNet
      - draw RGB + depth side by side in pygame

    Args:
        world: carla.World (in synchronous mode)
        screen: pygame display surface
        shared: dict with at least {"rgb": np.ndarray or None}
        depth_unet: loaded CARLA_UNet model
        clock: optional pygame.time.Clock
        fps: target FPS for `clock.tick`
    """
    # Tick world (sync mode)
    world.tick()

    if shared.get("rgb") is not None:
        rgb_np = shared["rgb"]

        # Make sure size matches H,W (safety)
        if rgb_np.shape[0] != H or rgb_np.shape[1] != W:
            import cv2
            rgb_np = cv2.resize(rgb_np, (W, H), interpolation=cv2.INTER_NEAREST)

        depth_rgb = predict_depth(depth_unet, rgb_np)

        rgb_surface = numpy_to_pygame_surface(rgb_np)
        depth_surface = numpy_to_pygame_surface(depth_rgb)

        # RGB left, depth right
        screen.blit(rgb_surface, (0, 0))
        screen.blit(depth_surface, (W, 0))

    pygame.display.flip()

    if clock is not None:
        clock.tick(fps)
