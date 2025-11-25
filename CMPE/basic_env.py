import random
#random.seed(618)
import carla
import os, sys, glob
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
#from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
# from agents.navigation.global_route_planner import (
#     GlobalRoutePlanner,
#     GlobalRoutePlannerDAO,
# )



class BasicEnv:
    """
    Very simple environment:
      - Takes an existing carla.World
      - Picks two random spawn points
      - Plans a route (waypoints) between them
      - Spawns a vehicle at the start
      - Visualizes the route with debug points/lines
    """

    def __init__(self, world, sampling_resolution: float = 2.0):
        self.world = world
        self.map = world.get_map()
        self.sampling_resolution = sampling_resolution

        self.route_planner = self._init_route_planner()
        self.vehicle = None           # carla.Actor
        self.route = []               # list of (Waypoint, RoadOption)

    def _init_route_planner(self) -> GlobalRoutePlanner:
        """
        Initialize CARLA's GlobalRoutePlanner.
        sampling_resolution is the distance between internal waypoints (in meters).
        """
        #dao = GlobalRoutePlannerDAO(self.map, self.sampling_resolution)
        grp = GlobalRoutePlanner(self.map, self.sampling_resolution)
        #grp.setup()
        return grp

    def reset(self, visualize: bool = True, life_time: float = 30.0):
        """
        - Destroy old vehicle (if any)
        - Choose two random spawn points as start and end
        - Plan a route between them
        - Spawn a vehicle at the start
        - Optionally visualize the route

        Returns:
            vehicle: the spawned carla.Actor
            route:   list[(Waypoint, RoadOption)]
        """
        # 1. Clean up old ego vehicle if it exists
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except RuntimeError:
                pass
            self.vehicle = None

        # 2. Choose random start & end spawn points
        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) < 2:
            raise RuntimeError("Not enough spawn points on this map.")

        start_transform, end_transform = random.sample(spawn_points, 2)
        start_loc = start_transform.location
        end_loc = end_transform.location

        # 3. Plan a route (sequence of waypoints) between them
        #    route is a list of (carla.Waypoint, RoadOption)
        self.route = self.route_planner.trace_route(start_loc, end_loc)

        # 4. Spawn a vehicle at the start
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        if not vehicle_bps:
            raise RuntimeError("No vehicle blueprints found!")

        vehicle_bp = random.choice(vehicle_bps)
        # optional: tag it as ego
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "ego")

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, start_transform)
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle at chosen spawn point.")

        # 5. Visualize the route
        if visualize:
            self._draw_route(self.route, life_time=life_time)

        return self.vehicle, self.route

    def _draw_route(self, route, life_time: float = 30.0):
        """
        Visualize route waypoints as green points and lines between them.
        """
        debug = self.world.debug

        # Draw points
        prev_wp = None
        for wp, _ in route:
            loc = wp.transform.location + carla.Location(z=0.3)
            debug.draw_point(
                loc,
                size=0.1,
                color=carla.Color(0, 255, 0),
                life_time=life_time
            )

            if prev_wp is not None:
                prev_loc = prev_wp.transform.location + carla.Location(z=0.3)
                debug.draw_line(
                    prev_loc,
                    loc,
                    thickness=0.05,
                    color=carla.Color(0, 255, 0),
                    life_time=life_time
                )

            prev_wp = wp
