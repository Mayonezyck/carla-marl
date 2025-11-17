import carla
from agents import Agent
import random
from typing import Dict, Any, List, Optional


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
        self.starting_point: carla.Location = (
            self.vehicle.get_transform().location if self.vehicle is not None else None
        )

        # Destination = random valid driving waypoint in the world
        self.destination: carla.Location = self._pick_random_destination()
        self._setup_sensors_from_config()
        

    def _setup_sensors_from_config(self) -> None:
        sensors_cfg = self.config.get_sensor_type()
        if sensors_cfg == "lidar":
            lidar_tf = self.config.get_lidar_tf()
            #lidar_cfg = self.config.get_lidar_cfg()
            self._spawn_lidar(lidar_tf,'full')

    def _pick_random_destination(self) -> carla.Location:
        """
        Pick a random valid driving waypoint as destination.
        Returns its location.
        """
        carla_map = self.world.get_map()

        # generate dense-enough waypoints, e.g. every 2 meters
        waypoints = carla_map.generate_waypoints(2.0)

        # only keep driving lanes
        driving_wps = [wp for wp in waypoints if wp.lane_type == carla.LaneType.Driving]
        if not driving_wps:
            raise RuntimeError("[Controlled_Agents] No driving waypoints found to choose a destination from.")

        # try to avoid picking a destination extremely close to the start
        if self.starting_point is not None:
            for _ in range(20):
                wp = random.choice(driving_wps)
                loc = wp.transform.location
                if loc.distance(self.starting_point) > 10.0:  # 10 meters away
                    return loc
            # fallback: if all are close (we’re on a tiny map), just return any
            return random.choice(driving_wps).transform.location
        else:
            return random.choice(driving_wps).transform.location

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