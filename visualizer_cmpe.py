# visualizer_cmpe.py

from typing import List, Optional
import numpy as np
import pygame


class PygameVisualizer:
    """
    Simple Pygame visualizer:
      - Left: top-down RGB camera
      - Right: text panel with ego obs + episode/step/reward
    """

    def __init__(self, width: int = 960, height: int = 540, font_size: int = 18):
        pygame.init()
        pygame.display.set_caption("CARLA CMPE Debug Viewer")

        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("monospace", font_size)

        # Split window: left for camera, right for text
        self.cam_width = 640
        self.cam_height = 480

    def _handle_events(self) -> None:
        """
        Process Pygame events. Raise KeyboardInterrupt on window close so that
        your main loop can cleanly stop.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

    def render(
        self,
        rgb: Optional[np.ndarray],
        text_lines: List[str],
    ) -> None:
        """
        rgb: (H, W, 3) uint8 or None
        text_lines: list of strings to print on the right panel
        """
        self._handle_events()

        # Fill background
        self.screen.fill((0, 0, 0))

        # Draw camera view on the left
        if rgb is not None:
            # Ensure uint8 (H, W, 3)
            rgb = np.asarray(rgb, dtype=np.uint8)
            h, w, _ = rgb.shape

            # Convert to a Pygame surface (expects (W, H, 3))
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

            # Scale to our cam panel size if needed
            surface = pygame.transform.scale(surface, (self.cam_width, self.cam_height))

            self.screen.blit(surface, (0, 0))

        # Draw text on the right side
        x0 = self.cam_width + 10
        y = 10
        for line in text_lines:
            surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(surf, (x0, y))
            y += surf.get_height() + 4

        pygame.display.flip()
