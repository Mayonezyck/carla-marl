import carla, os, sys,glob
from customAgents import Agent
import random
from typing import Dict, Any, List, Optional
import queue
CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/zb536/Carla-9.15")
try:
    egg_pattern = os.path.join(
        CARLA_ROOT, "PythonAPI", "carla", "dist", "carla-*py3*.egg"
    )
    sys.path.append(glob.glob(egg_pattern)[0])
    sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))
    sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI"))
except IndexError:
    print("Could not find CARLA egg, please check CARLA_ROOT.")
    sys.exit(1)

from agents.navigation.global_route_planner import GlobalRoutePlanner


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
        self._route_planner: GlobalRoutePlanner | None = None
        self.current_waypoint_plan: list[tuple[carla.Waypoint, Any]] = []
        self._route_debug_enabled: bool = False

        # Destination = random valid driving waypoint in the world
        self._lidar_queue: "queue.Queue[tuple[int, carla.LidarMeasurement]]" = queue.Queue()
        self._last_lidar: tuple[int, carla.LidarMeasurement] | None = None
        self._collision_event = [] #I use a list to record the events, should I just cache one?
        self._fatal_collision = False #Currently this will log any collision as fatal, later can be tweeked to check only significant events (e.g. collision with vehicle/ped)
        # Initialize start/end and plan a route
        if self.config.get_if_route_planning() and self.vehicle is not None:

            # Build planner once per agent
            self._route_planner = GlobalRoutePlanner(self.world.get_map(), 10.0)

            # Compute the route between start and end
            if self.starting_point is not None and self.destination is not None:
                self.current_waypoint_plan = self._route_planner.trace_route(
                    self.starting_point,
                    self.destination
                )
                self._route_debug_enabled = True
        
        self._add_other_sensors()
        self._setup_sensors_from_config()
        
        

    def _add_other_sensors(self):
        collision_sensor = self._add_collision_sensor()

        def collision_callback(event: carla.CollisionEvent):
            def _check_fatal(event: carla.CollisionEvent):
                #Label any collision as fatal
                return True
            actor_we_collided_with = event.other_actor
            impulse = event.normal_impulse  # carla.Vector3D
            intensity = (impulse.x**2 + impulse.y**2 + impulse.z**2)**0.5
            self._collision_event.append(event)
            if _check_fatal(event):
                self._fatal_collision = True
            print(f"[COLLISION] with {actor_we_collided_with.type_id}, intensity={intensity:.2f}")
        
        collision_sensor.listen(collision_callback)
        self.sensors.append(collision_sensor)
        
    def _add_collision_sensor(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')

        # You can set attributes on the sensor if needed
        # e.g. collision_bp.set_attribute('some_attribute', 'value')

        # Attach the sensor to the parent (e.g., vehicle)
        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.0),  # place it somewhere on the car
            carla.Rotation()
        )

        collision_sensor = self.world.spawn_actor(
            collision_bp,
            transform,
            attach_to=self.vehicle
        )
        return collision_sensor
            

    def _setup_sensors_from_config(self) -> None:
        sensors_cfg = self.config.get_sensor_type()
        if sensors_cfg == "lidar":
            lidar_tf = self.config.get_lidar_tf()
            #lidar_cfg = self.config.get_lidar_cfg()
            def _lidar_callback(data: carla.LidarMeasurement, idx=self.index):
                # import open3d as o3d
                # import numpy as np
                # Just push into queue + keep last for non-blocking access
                print('debug: using actual callback')
                print(len(data.raw_data))
                item = (data.frame, data)
                self._last_lidar = item
                # points = np.frombuffer(data.raw_data, dtype=np.float32)
                # points = points.reshape(-1, 4)[:, :3]

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(points)
                # o3d.io.write_point_cloud('test_points.pcd', pcd)
                # Non-blocking: drop if queue is "full" to avoid buildup (optional)
                try:
                    self._lidar_queue.put_nowait(item)
                except queue.Full:
                    pass
            self._spawn_lidar(lidar_tf,_lidar_callback,'full')
            
            
    def apply_action(self, action) -> None:
        """
        action: for example [throttle, steer, brake]
        """
        throttle, steer, brake = float(action[0]), float(action[1]), float(action[2])
        control = carla.VehicleControl(
            throttle=max(0.0, min(1.0, throttle)),
            steer=max(-1.0, min(1.0, steer)),
            brake=max(0.0, min(1.0, brake)),
        )
        self.vehicle.apply_control(control)

    def get_fatal_flag(self) -> bool:
        return self._fatal_collision

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

    def draw_current_route_debug(self, life_time: float = 0.2) -> None:
        """
        Draw the current waypoint plan as:
          - Blue points at each waypoint
          - Green lines between consecutive waypoints
        Call this every tick from the Manager.
        """
        if not self.current_waypoint_plan:
            return

        dbg = self.world.debug

        # Draw points
        for wp, _ in self.current_waypoint_plan:
            loc = wp.transform.location
            loc.z += 0.3   # lift slightly above road so it's not z-fighting
            dbg.draw_point(
                loc,
                size=0.1,
                color=carla.Color(0, 0, 255),  # blue points
                life_time=life_time,
            )

        # Draw lines
        for (wp1, _), (wp2, _) in zip(
            self.current_waypoint_plan[:-1], self.current_waypoint_plan[1:]
        ):
            loc1 = wp1.transform.location
            loc2 = wp2.transform.location
            loc1.z += 0.3
            loc2.z += 0.3
            dbg.draw_line(
                loc1,
                loc2,
                thickness=0.1,
                color=carla.Color(0, 255, 0),  # green line
                life_time=life_time,
            )


    def _clear_route_debug(self) -> None:
        """
        Stop drawing the route and drop the plan.
        Debug shapes in CARLA are time-limited, so once we stop redrawing
        and the lifetime expires, the visualization is effectively 'cleaned'.
        """
        self._route_debug_enabled = False
        self.current_waypoint_plan = []
