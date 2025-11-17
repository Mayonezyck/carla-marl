import carla
from agents import Agent
import random
from typing import Dict, Any, List, Optional
import queue


class Controlled_Agents(Agent):
    """
    Controlled agents:
    - Spawn a vehicle (via Agent).
    - Check config to see what sensors to add:
        - If lidar: attach full lidar.
        - If lidar_partial: attach partial lidar (e.g. different range/FOV).
        - If camera: attach camera.
    - Store both vehicle and sensors in this class.
    - destroy_agent() removes both vehicle and sensors.
    """

    def __init__(self, world: carla.World, index: int, config: Dict[str, Any]):
        super().__init__(world, index, config, role_prefix="controlled_agent")
        # self.starting_point: carla.Location = (
        #     self.vehicle.get_transform().location if self.vehicle is not None else None
        # )

        # Destination = random valid driving waypoint in the world
        self._lidar_queue: "queue.Queue[tuple[int, carla.LidarMeasurement]]" = queue.Queue()
        self._last_lidar: tuple[int, carla.LidarMeasurement] | None = None

        self._setup_sensors_from_config()

        

    def _setup_sensors_from_config(self) -> None:
        sensors_cfg = self.config.get_sensor_type()
        if sensors_cfg == "lidar":
            lidar_tf = self.config.get_lidar_tf()
            #lidar_cfg = self.config.get_lidar_cfg()
            def _lidar_callback(data: carla.LidarMeasurement, idx=self.index):
                # Just push into queue + keep last for non-blocking access
                print('debug: using actual callback')
                item = (data.frame, data)
                self._last_lidar = item
                # Non-blocking: drop if queue is "full" to avoid buildup (optional)
                try:
                    self._lidar_queue.put_nowait(item)
                except queue.Full:
                    pass
            self._spawn_lidar(lidar_tf,_lidar_callback,'full')
            

    def get_lidar_blocking(self, timeout: float = 1.0):
        """Get the next lidar measurement (blocking)."""
        try:
            return self._lidar_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_lidar_latest(self):
        """Get the latest lidar measurement (non-blocking, may be None)."""
        return self._last_lidar

        # Lidar (full)
        # lidar_cfg = sensors_cfg.get("lidar", {})
        # if lidar_cfg.get("enabled", False):
        #     self._spawn_lidar(lidar_cfg, name_suffix="full")

        # # Lidar (partial)
        # lidar_partial_cfg = sensors_cfg.get("lidar_partial", {})
        # if lidar_partial_cfg.get("enabled", False):
        #     self._spawn_lidar(lidar_partial_cfg, name_suffix="partial")

        # # Camera
        # camera_cfg = sensors_cfg.get("camera", {})
        # if camera_cfg.get("enabled", False):
        #     self._spawn_camera(camera_cfg, name_suffix="main")