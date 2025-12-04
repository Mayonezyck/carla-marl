# unet_seg_vis.py

import numpy as np
import torch
import torch.nn as nn
import pygame

from seg_train_UNet_multiVehicle import CARLA_UNet as Seg_UNet, colorize_mask

SEG_CHECKPOINT_PATH = "unet_seg/unet_seg_29.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_seg_unet_model() -> nn.Module:
    """
    Instantiate and load the trained segmentation CARLA_UNet.
    """
    model = Seg_UNet(in_channels=3, n_filters=16, n_classes=28)
    state = torch.load(SEG_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_segmentation(model: nn.Module, rgb_np: np.ndarray) -> np.ndarray:
    """
    Run the segmentation UNet on a single RGB frame.

    Args:
        model: loaded segmentation UNet
        rgb_np: (H, W, 3) uint8 RGB image

    Returns:
        seg_rgb: (H, W, 3) uint8 segmentation colored with CARLA palette
    """
    # (H, W, 3) uint8 -> (1, 3, H, W) float32
    img = rgb_np.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)            # (1, C, H, W)
        preds = torch.argmax(logits, dim=1)   # (1, H, W)

    mask = preds.squeeze(0)                   # (H, W), int
    seg_rgb = colorize_mask(mask)             # (H, W, 3) uint8
    return seg_rgb


def numpy_to_pygame_surface(arr: np.ndarray) -> pygame.Surface:
    """
    Convert (H, W, 3) uint8 numpy array to a pygame Surface.
    pygame.surfarray.make_surface expects (W, H, 3).
    """
    return pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))


class SegUNetVis:
    """
    Tiny convenience wrapper:
      vis = SegUNetVis()
      seg_rgb = vis(rgb_np)
      seg_surface = vis.to_surface(rgb_np)
    """

    def __init__(self):
        self.model = make_seg_unet_model()

    def __call__(self, rgb_np: np.ndarray) -> np.ndarray:
        return predict_segmentation(self.model, rgb_np)

    def to_surface(self, rgb_np: np.ndarray) -> pygame.Surface:
        seg_rgb = self(rgb_np)
        return numpy_to_pygame_surface(seg_rgb)
