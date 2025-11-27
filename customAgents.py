import random
from typing import Dict, Any, List, Optional

import carla
import math
import yaml

class Agent:
    """
    Superclass for both Controlled_Agents and Free_Agents.

    Responsibilities:
    - Get a spawn point.
    - Pick a random vehicle blueprint.
    - Try to spawn the vehicle on the map.
    - Store the vehicle instance and any attached sensors.
    - Provide destroy_agent() to cleanly remove vehicle + sensors.
    """

    def __init__(
        self,
        world: carla.World,
        index: int,
        config: Dict[str, Any],
        role_prefix: str,
    ):
        self.world = world
        self.index = index
        self.config = config
        self.role_prefix = role_prefix
        self.starting_point = None
        self.destination = None


        self.vehicle: Optional[carla.Actor] = None
        self.sensors: List[carla.Actor] = []

        self._spawn_vehicle()

    # ---------- vehicle spawning ----------

    def _spawn_vehicle(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter("vehicle.*")

        if not vehicle_blueprints:
            raise RuntimeError("[Agent] No vehicle blueprints found.")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("[Agent] No spawn points available in this map.")

        random.shuffle(spawn_points)

        max_attempts = self.config.get_spawn_max_attempts()
        attempts = 0
        vehicle = None

        while attempts < max_attempts and vehicle is None:
            bp = random.choice(vehicle_blueprints)
            # Random color if available
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)

            # Role name for debugging & filtering
            
            spawn_point = self.find_closest_spawn(spawn_points) #FOR EASE of OBSERVATION I USE THIS SPAWN FINDING 
            #spawn_point = spawn_points[attempts % len(spawn_points)]
            self.starting_point = spawn_point.location
            if self.role_prefix == "controlled_agent":
                start = self.starting_point
                self.destination = self._pick_nearby_destination(
                    world=self.world,
                    start_location=self.starting_point,
                    min_distance=50.0,   # or whatever you want
                    step=2.0,
                )
                dest = self.destination
                print(dest)
                bp.set_attribute("role_name", f"{self.role_prefix}_{self.index}|{start.x:.2f},{start.y:.2f},{start.z:.2f}|{dest.x:.2f},{dest.y:.2f},{dest.z:.2f}")
            else:
                bp.set_attribute("role_name", f"{self.role_prefix}_{self.index}")
            vehicle = self.world.try_spawn_actor(bp, spawn_point)

            attempts += 1

        if vehicle is None:
            raise RuntimeError(
                f"[Agent] Could not spawn vehicle for {self.role_prefix}_{self.index} "
                f"after {max_attempts} attempts."
            )
        self.initial_pos = spawn_point
        self.vehicle = vehicle
        print(f"[Agent] Spawned {self.role_prefix}_{self.index} at {self.initial_pos}")

    # ---------- sensor helpers & callbacks ----------

    def _dict_to_transform(self, tf_dict: Dict[str, float]) -> carla.Transform:
        """Convert a dict {x,y,z,roll,pitch,yaw} to carla.Transform."""
        x = tf_dict.get("x", 0.0)
        y = tf_dict.get("y", 0.0)
        z = tf_dict.get("z", 0.0)
        roll = tf_dict.get("roll", 0.0)
        pitch = tf_dict.get("pitch", 0.0)
        yaw = tf_dict.get("yaw", 0.0)

        location = carla.Location(x=x, y=y, z=z)
        rotation = carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
        return carla.Transform(location, rotation)

    def _spawn_lidar(self, sensor_tf, cb, name_suffix: str) -> Optional[carla.Actor]:
    #def _spawn_lidar(self,sensor_tf) -> Optional[carla.Actor]:
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        if lidar_bp is None:
            print("[Agent] Lidar blueprint not found.")
            return None

        lidar_bp.set_attribute('rotation_frequency', '20.0')      # 20 rev/s
        lidar_bp.set_attribute('sensor_tick', '0.1')    # 0.1s per measurement
        lidar_bp.set_attribute('horizontal_fov', '360.0') 
        # attributes = sensor_cfg.get("attributes", {})
        # for key, value in attributes.items():
        #     if lidar_bp.has_attribute(key):
        #         lidar_bp.set_attribute(key, str(value))

        #lidar_bp.set_attribute("role_name", f"{self.role_prefix}_{self.index}_lidar_{name_suffix}")

        #tf = self._dict_to_transform(sensor_tf)
        tf = carla.Transform(carla.Location(sensor_tf[0],sensor_tf[1],sensor_tf[2]),carla.Rotation(sensor_tf[3],sensor_tf[4],sensor_tf[5]))

        lidar = self.world.try_spawn_actor(lidar_bp, tf, attach_to=self.vehicle)
        if lidar is None:
            print(f"[Agent] Failed to spawn lidar ({name_suffix}) for {self.role_prefix}_{self.index}")
            return None

        # Attach callback
        lidar.listen(cb)
        self.sensors.append(lidar)
        print("rotation_frequency:", float(lidar.attributes['rotation_frequency']))
        print("sensor_tick:", float(lidar.attributes['sensor_tick']))

        print(f"[Agent] Spawned lidar ({name_suffix}) for {self.role_prefix}_{self.index}")
        return lidar

    def _spawn_camera(self, sensor_cfg: Dict[str, Any], name_suffix: str) -> Optional[carla.Actor]:
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        if camera_bp is None:
            print("[Agent] Camera blueprint not found.")
            return None

        attributes = sensor_cfg.get("attributes", {})
        for key, value in attributes.items():
            if camera_bp.has_attribute(key):
                camera_bp.set_attribute(key, str(value))

        camera_bp.set_attribute("role_name", f"{self.role_prefix}_{self.index}_camera_{name_suffix}")

        tf = self._dict_to_transform(sensor_cfg.get("transform", {}))

        camera = self.world.try_spawn_actor(camera_bp, tf, attach_to=self.vehicle)
        if camera is None:
            print(f"[Agent] Failed to spawn camera ({name_suffix}) for {self.role_prefix}_{self.index}")
            return None

        camera.listen(lambda image, self=self, tag=name_suffix: self.camera_callback(image, tag))
        self.sensors.append(camera)
        print(f"[Agent] Spawned camera ({name_suffix}) for {self.role_prefix}_{self.index}")
        return camera

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

    def _pick_nearby_destination(
        self,
        world: carla.World,
        start_location: carla.Location,
        min_distance: float = 50.0,
        step: float = 2.0,
        max_steps: int = 200,
    ) -> carla.Location:
        """
        Starting from start_location, walk along the lane using waypoint.next(step)
        until we are at least `min_distance` meters away (or we run out of road).

        Returns a carla.Location for the destination.
        """
        carla_map = world.get_map()
        # Get closest driving waypoint to the start
        wp = carla_map.get_waypoint(
            start_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        start = wp.transform.location
        total_dist = 0.0
        current_wp = wp

        for _ in range(max_steps):
            next_wps = current_wp.next(step)
            if not next_wps:
                # Reached end of lane; stop here
                break

            # You can randomize branch choice if there are multiple:
            current_wp = random.choice(next_wps)
            loc = current_wp.transform.location
            seg = math.sqrt(
                (loc.x - start.x) ** 2 +
                (loc.y - start.y) ** 2 +
                (loc.z - start.z) ** 2
            )
            total_dist = seg

            if total_dist >= min_distance:
                # Good enough, stop here
                return loc

        # Fallback: if we never reached min_distance, just return
        # the last waypoint location we got
        return current_wp.transform.location

    def find_closest_spawn(self, spawn_points,
                        target_xyz=( -100.0, 56.0, 0 )):
        """
        Returns the spawn point whose location is closest to target_xyz.
        """
        target_x, target_y, target_z = target_xyz
        target_loc = carla.Location(target_x, target_y, target_z)

        if not spawn_points:
            raise RuntimeError("No spawn points available in this map.")

        closest_sp = None
        closest_dist = float("inf")

        for sp in spawn_points:
            loc = sp.location
            dx = loc.x - target_loc.x
            dy = loc.y - target_loc.y
            dz = loc.z - target_loc.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < closest_dist:
                closest_dist = dist
                closest_sp = sp

        print(f"[Closest Spawn] Distance = {closest_dist:.2f} m")
        return closest_sp


    # Default callbacks: you can override or extend in subclasses if needed
    def lidar_callback(self, data: carla.LidarMeasurement, tag: str) -> None:
        """
        Default lidar callback.
        Override in subclass or modify for logging/saving.
        """
        # For now, just print a tiny debug line occasionally
        print('Warning, this is a default method that does nothing')
        print(f"[{self.role_prefix}_{self.index}] Lidar ({tag}) received frame {data.frame}")

    def camera_callback(self, image: carla.Image, tag: str) -> None:
        """
        Default camera callback.
        Override in subclass or modify for logging/saving.
        """
        print(f"[{self.role_prefix}_{self.index}] Camera ({tag}) received frame {image.frame}")

    # ---------- cleanup ----------

    def destroy_agent(self) -> None:
        """
        Destroy vehicle and all attached sensors.
        """
        print(f"[Agent] Destroying {self.role_prefix}_{self.index} and its sensors...")

        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass
        self.sensors.clear()
        if hasattr(self, "_clear_route_debug"):
            self._clear_route_debug()
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None
