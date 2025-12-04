#!/usr/bin/env python3
"""
Generate a color legend image for the CARLA segmentation classes.

Output: carla_seg_color_legend.png
"""

from math import ceil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Option 1: import directly from your training file
try:
    from seg_train_UNet_multiVehicle import CARLA_PALETTE  # noqa: F401
except ImportError:
    # If import fails, fall back to hard-coded palette (copy from seg_train_UNet_multiVehicle.py)
    CARLA_PALETTE = [
        (0, 0, 0),          # 0  None
        (70, 70, 70),       # 1  Buildings
        (190, 153, 153),    # 2  Fences
        (250, 170, 160),    # 3  Other
        (220, 20, 60),      # 4  Pedestrians
        (153, 153, 153),    # 5  Poles
        (157, 234, 50),     # 6  RoadLines
        (128, 64, 128),     # 7  Roads
        (244, 35, 232),     # 8  Sidewalks
        (107, 142, 35),     # 9  Vegetation
        (0, 0, 142),        # 10 Vehicles
        (102, 102, 156),    # 11 Walls
        (220, 220, 0),      # 12 TrafficSigns
        (70, 130, 180),     # 13 Sky
        (81, 0, 81),        # 14 Ground
        (150, 100, 100),    # 15 Bridge
        (230, 150, 140),    # 16 RailTrack
        (180, 165, 180),    # 17 GuardRail
        (250, 170, 30),     # 18 TrafficLight
        (110, 190, 160),    # 19 Static
        (170, 120, 50),     # 20 Dynamic
        (45, 60, 150),      # 21 Water
        (145, 170, 100),    # 22 Terrain
        (255, 255, 255),    # 23 Unused_23
        (255, 0, 255),      # 24 Unused_24
        (0, 255, 255),      # 25 Unused_25
        (255, 255, 0),      # 26 Unused_26
        (0, 255, 0),        # 27 Unused_27
    ]

# Class labels (same order as the palette)
CLASS_LABELS = [
    "None", "Buildings", "Fences", "Other", "Pedestrians", "Poles", "RoadLines",
    "Roads", "Sidewalks", "Vegetation", "Vehicles", "Walls", "TrafficSigns", "Sky",
    "Ground", "Bridge", "RailTrack", "GuardRail", "TrafficLight", "Static", "Dynamic",
    "Water", "Terrain", "Unused_23", "Unused_24", "Unused_25", "Unused_26", "Unused_27"
]


def make_color_legend(
    palette,
    labels,
    cols: int = 4,
    block_w: int = 260,
    block_h: int = 70,
    margin: int = 20,
    text_margin: int = 10,
    bg_color=(0, 0, 0),
):
    """
    Build an image with colored blocks and text labels.

    Each block: one class color with "index name" written on top of it.
    Arranged in a cols x rows grid.
    """
    n = len(palette)
    assert n == len(labels), "palette and labels must have same length"

    rows = ceil(n / cols)

    # Total image size
    img_w = margin * 2 + cols * block_w
    img_h = margin * 2 + rows * block_h

    img = Image.new("RGB", (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font; fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for idx, (rgb, label) in enumerate(zip(palette, labels)):
        row = idx // cols
        col = idx % cols

        x0 = margin + col * block_w
        y0 = margin + row * block_h
        x1 = x0 + block_w - 1
        y1 = y0 + block_h - 1

        # Fill block with class color
        draw.rectangle([x0, y0, x1, y1], fill=tuple(rgb))

        # Label text: "idx label"
        text = f"{idx:02d}  {label}"
        # Text color: choose black or white for contrast
        # Compute simple luminance to decide
        r, g, b = rgb
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

        # Position text with small margin inside block
        tx = x0 + text_margin
        ty = y0 + (block_h - 20) // 2  # roughly centered vertically
        draw.text((tx, ty), text, font=font, fill=text_color)

    return img


if __name__ == "__main__":
    legend_img = make_color_legend(CARLA_PALETTE, CLASS_LABELS)
    out_path = "carla_seg_color_legend.png"
    legend_img.save(out_path)
    print(f"Saved legend to {out_path}")
