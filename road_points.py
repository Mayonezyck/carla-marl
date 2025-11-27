import carla
import math
import numpy as np
import random
import torch


# TODO: MOVE those constants to somewhere else
MAX_ORIENTATION_RAD = 2 * np.pi

MIN_RG_COORD = -1000.0
MAX_RG_COORD = 1000.0
MAX_ROAD_LINE_SEGMENT_LEN = 100.0
MAX_ROAD_SCALE = 100.0


class RoadPointExtractor:
    def __init__(self, carla_map, step, observation_radius):
        self.step = step
        self.carla_map = carla_map
        self.observation_radius = observation_radius

        self.driving_wps = self.precompute_driving_waypoints(carla_map, step=step)
        self.all_landmarks = self.precompute_landmarks(carla_map)
        self.all_crosswalks = self.precompute_crosswalks(carla_map)

    def precompute_driving_waypoints(self, carla_map, step=2.0):
        """
        Precompute all driving waypoints for the map at given resolution.
        Call this once and reuse the result.
        """
        all_wps = carla_map.generate_waypoints(distance=step)
        driving_wps = [wp for wp in all_wps if wp.lane_type == carla.LaneType.Driving]
        print(f"Precomputed {len(driving_wps)} driving waypoints (step={step})")
        return driving_wps

    def precompute_landmarks(self, carla_map):
        """
        Precompute all landmarks from the map.
        """
        all_landmarks = carla_map.get_all_landmarks()
        print(f"Precomputed {len(all_landmarks)} landmarks")
        return all_landmarks

    def precompute_crosswalks(self, carla_map):
        """
        Precompute all crosswalks from the map.
        """
        all_crosswalks = carla_map.get_crosswalks()
        print(f"Precomputed {len(all_crosswalks)} crosswalks")
        return all_crosswalks

    def classify_landmark(self, lm: carla.Landmark) -> str:
        """
        Return a small semantic tag for a CARLA landmark.
        """
        t = (lm.type or "").strip()
        name = (lm.name or "").lower()

        if t == "206" or "stop" in name:
            return "stop_sign"
        if t == "205" or "yield" in name:
            return "yield_sign"
        if t == "1000001" or "signal" in name:
            return "traffic_light"

        return "other"

    def get_local_points_from_precomputed_knearest(
        self, 
        driving_wps,
        ego_loc,
        radius=60.0,
        n_points=200,
        step=2.0,
        all_landmarks=None,
        all_crosswalks=None,
    ):
        """
        Using precomputed driving waypoints:
        - Find waypoints whose *center* is within radius of ego_loc.
        - Build lane center points for them.
        - Build lane marking points (left/right) from them.
        - Keep only those boundary points whose own position is within radius.
        - Sort all candidates by distance and keep up to n_points.

        Returns:
            points: np.ndarray of shape (n_points, 9), rows:

                [x_world, y_world, yaw_rad,
                half_length, half_width,
                lane_type_code, road_id,
                geom_type_code, lane_mark_type_code]

            geom_type_code:
                0 = lane center
                1 = left lane marking
                2 = right lane marking

            lane_mark_type_code:
                0 for centers
                numeric code (float) for lane marking types for boundary points
        """
        candidates = []

        # First: select only waypoints whose center is within radius
        wps_near = []
        for wp in driving_wps:
            loc = wp.transform.location
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            dist = math.hypot(dx, dy)
            if dist <= radius:
                wps_near.append((wp, loc, dist))

        # 1) Lane center points
        for wp, loc, dist in wps_near:
            yaw_rad = math.radians(wp.transform.rotation.yaw)
            half_length = step * 0.5
            half_width = wp.lane_width * 0.5
            lane_type_code = float(wp.lane_type)
            road_id = float(wp.road_id)

            geom_type_code = 0.0
            lm_type = 0.0

            candidates.append(
                (
                    dist,
                    loc.x, loc.y,
                    yaw_rad,
                    half_length, half_width,
                    lane_type_code, road_id,
                    geom_type_code, lm_type,
                )
            )

        # 2) Lane marking points (left/right), built from the same nearby waypoints
        for wp, loc, _ in wps_near:
            yaw_rad = math.radians(wp.transform.rotation.yaw)
            nx = -math.sin(yaw_rad)
            ny =  math.cos(yaw_rad)
            half_width = wp.lane_width * 0.5
            half_length = step * 0.5
            lane_type_code = float(wp.lane_type)
            road_id = float(wp.road_id)

            # LEFT marking
            lm_left = wp.left_lane_marking
            if lm_left is not None and lm_left.type != carla.LaneMarkingType.NONE:
                bx = loc.x - nx * half_width
                by = loc.y - ny * half_width
                dx = bx - ego_loc.x
                dy = by - ego_loc.y
                dist = math.hypot(dx, dy)
                if dist <= radius:
                    geom_type_code = 1.0
                    lm_type = float(int(lm_left.type))
                    candidates.append(
                        (
                            dist,
                            bx, by,
                            yaw_rad,
                            half_length, 0.1,      # very thin strip
                            lane_type_code, road_id,
                            geom_type_code, lm_type,
                        )
                    )

            # RIGHT marking
            lm_right = wp.right_lane_marking
            if lm_right is not None and lm_right.type != carla.LaneMarkingType.NONE:
                bx = loc.x + nx * half_width
                by = loc.y + ny * half_width
                dx = bx - ego_loc.x
                dy = by - ego_loc.y
                dist = math.hypot(dx, dy)
                if dist <= radius:
                    geom_type_code = 2.0
                    lm_type = float(int(lm_right.type))
                    candidates.append(
                        (
                            dist,
                            bx, by,
                            yaw_rad,
                            half_length, 0.1,
                            lane_type_code, road_id,
                            geom_type_code, lm_type,
                        )
                    )
            # 3) Landmarks (optional)
        if all_landmarks is not None:
            for lm in all_landmarks:
                loc = lm.transform.location
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y
                dist = math.hypot(dx, dy)
                if dist > radius:
                    continue

                yaw_rad = math.radians(lm.transform.rotation.yaw)
                half_length = 1.0
                half_width = 1.0

                kind = self.classify_landmark(lm)

                # geom_type = 3: "landmark-like object"
                geom_type_code = 3.0

                if kind == "stop_sign":
                    subtype = 1.0
                elif kind == "yield_sign":
                    subtype = 2.0
                elif kind == "traffic_light":
                    subtype = 3.0
                else:
                    subtype = 0.0  # other / ignore later if you want

                lm_id = float(lm.id)

                candidates.append(
                    (
                        dist,
                        loc.x, loc.y,
                        yaw_rad,
                        half_length, half_width,
                        0.0, lm_id,        # lane_type=0 for landmarks
                        geom_type_code, subtype,
                    )
                )

        if all_crosswalks is not None:
            for lm in all_crosswalks:
                loc = lm
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y
                dist = math.hypot(dx, dy)
                if dist > radius:
                    continue

                #yaw_rad = math.radians(lm.transform.rotation.yaw)
                yaw_rad = 0 # set to rad as lm is just location
                # Rough size guesses; you can tune these per lm.type if you like
                half_length = 1.0
                half_width = 1.0

                geom_type_code = 4.0  # "crosswalks"
                subtype = 0.0
    #             # landmark.type is often a numeric string like "1000001"
    #             try:
    #                 subtype = float(int(lm.type))
    #             except (ValueError, TypeError):
    #                 subtype = 0.0
                # Store landmark id in the "road_id" slot
                lm_id = 0.0

                candidates.append(
                    (
                        dist,
                        loc.x, loc.y,
                        yaw_rad,
                        half_length, half_width,
                        0.0, lm_id,        # lane_type=0 for landmarks
                        geom_type_code, subtype,
                    )
                )

        D = 9
        points = np.zeros((n_points, D), dtype=np.float32)
        if not candidates:
            return points

        # Sort by distance and keep up to n_points
        candidates.sort(key=lambda t: t[0])
        selected = candidates[:n_points]

        for i, (_, x, y, yaw, hlen, hwid,
                lane_t, rid, geom_t, lm_type) in enumerate(selected):
            points[i, :] = [x, y, yaw, hlen, hwid, lane_t, rid, geom_t, lm_type]

        return points


    def draw_local_map_points_precomputed(
        self,
        world,
        driving_wps,
        all_landmarks,
        all_crosswalks,
        geo_location,
        radius=60.0,
        n_points=200,
        step=2.0,
        life_time=20.0,
        max_draw=None,
    ):
        """
        Same as before, but also draws landmarks (geom_type 3) in a distinct color.
        """
        if max_draw is None:
            max_draw = n_points

        carla_map = world.get_map()
        debug = world.debug

        ego_wp = carla_map.get_waypoint(
            geo_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        ego_loc = ego_wp.transform.location

        points = self.get_local_points_from_precomputed_knearest(
            driving_wps,
            ego_loc,
            radius=radius,
            n_points=n_points,
            step=step,
            all_landmarks=all_landmarks,
            all_crosswalks=all_crosswalks,
        )

        num_drawn = 0
        for row in points:
            x, y, yaw, hlen, hwid, lane_t, rid, geom_t, subtype = row
            if x == 0 and y == 0 and hlen == 0 and hwid == 0:
                continue

            loc = carla.Location(x=float(x), y=float(y), z=ego_loc.z + 0.5)

            if int(geom_t) == 0:
                color = carla.Color(0, 255, 0)       # lane center
            elif int(geom_t) == 1:
                color = carla.Color(0, 0, 255)       # left marking
            elif int(geom_t) == 2:
                color = carla.Color(255, 255, 0)     # right marking
            elif int(geom_t) == 3:
                color = carla.Color(0, 255, 255)     # landmarks (stop sign, etc.)
            else:
                color = carla.Color(255, 0, 0)       # crosswalks 

            debug.draw_point(
                loc,
                size=0.08,
                color=color,
                life_time=life_time,
            )

            num_drawn += 1
            if num_drawn >= max_draw:
                break

        debug.draw_point(
            ego_loc + carla.Location(z=0.7),
            size=0.15,
            color=carla.Color(255, 0, 255),
            life_time=life_time,
        )

        print(f"Drew {num_drawn} map points (centers + lane drawings + landmarks) and ego.")
        return points, ego_loc

    def wrap_to_pi(self, angle):
        """Wrap angle (radians) to [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def carla_points_to_gpudrive_roadgraph(
        self,
        points: np.ndarray,
        ego_location,
        ego_yaw_deg: float,
    ) -> np.ndarray:
        """
        Convert CARLA map points (world-frame) into GPUDrive-style local roadgraph features.

        Args:
            points: (N, 9) numpy array:
                [x_w, y_w, yaw_w,
                half_length, half_width,
                lane_type, road_id,
                geom_type, subtype]
            ego_location: carla.Location (or any object with .x and .y attributes)
            ego_yaw_deg: ego yaw in degrees (CARLA yaw, usually from waypoint rotation)

        Returns:
            The normalized array
            features: (N, 13) numpy array:
                [pos_x_rel, pos_y_rel,
                seg_len, scale_x, scale_y,
                yaw_rel,
                one_hot[0..6]]  # None, RoadLine, RoadEdge, RoadLane, CrossWalk, Speedbump, Stopsign
        """
        assert points.shape[1] == 9, f"Expected (N, 9) points, got {points.shape}"

        ego_x = float(ego_location.x)
        ego_y = float(ego_location.y)
        ego_yaw_rad = math.radians(ego_yaw_deg)

        cos_e = math.cos(ego_yaw_rad)
        sin_e = math.sin(ego_yaw_rad)

        N = points.shape[0]
        out = np.zeros((N, 13), dtype=np.float32)

        for i in range(N):
            x_w, y_w, yaw_w, hlen, hwid, lane_t, rid, geom_t, subtype = points[i]

            # Treat fully-zero geometry as padding; leave output row as zeros.
            if hlen == 0.0 and hwid == 0.0 and x_w == 0.0 and y_w == 0.0:
                continue

            # 1) World -> ego local coordinates for position
            dx = x_w - ego_x
            dy = y_w - ego_y

            # Rotate by -ego_yaw
            x_rel = cos_e * dx + sin_e * dy
            y_rel = -sin_e * dx + cos_e * dy

            # 2) Relative orientation
            yaw_rel = self.wrap_to_pi(yaw_w - ego_yaw_rad)

            # 3) Decide semantic class from geom_type
            geom = int(round(geom_t))

            # Default: None
            class_id = 0
            seg_len = 0.0
            scale_x = 0.0
            scale_y = 0.0

            if geom == 0:
                # Lane center -> RoadLane
                class_id = 3  # RoadLane
                seg_len = float(hlen)          # half-length, like GPUDrive
                scale_x = 0.001                # thin line convention
                scale_y = 0.001
            elif geom in (1, 2):
                # Left/right lane marking -> RoadLine
                class_id = 1  # RoadLine
                seg_len = float(hlen)
                scale_x = 0.001
                scale_y = 0.001
            elif geom == 3:
                # Landmark → decide based on subtype
                st = int(round(subtype))
                if st == 1:
                    # stop_sign
                    class_id = 6  # Stopsign
                    seg_len = 0.002
                    scale_x = 0.002
                    scale_y = 0.001
                else:
                    # yield, traffic lights, others → currently ignored in GPUDrive classes
                    class_id = 0  # None
                    seg_len = 0.0
                    scale_x = 0.0
                    scale_y = 0.0

            elif geom == 4:
                # Crosswalk: for now, "set aside" -> None class
                class_id = 0
                seg_len = 0.0
                scale_x = 0.0
                scale_y = 0.0

            # 4) One-hot encoding
            one_hot = np.zeros(7, dtype=np.float32)
            if 0 <= class_id < 7:
                one_hot[class_id] = 1.0

            # 5) Pack into output
            out[i, 0:6] = [
                x_rel,
                y_rel,
                seg_len,
                scale_x,
                scale_y,
                yaw_rel,
            ]
            out[i, 6:] = one_hot
            
        # -------------------------------
        # 6) NORMALIZATION (GPUDrive-style)
        # -------------------------------

        # x,y → [-1, 1] using MIN_RG_COORD, MAX_RG_COORD
        rg_range = (MAX_RG_COORD - MIN_RG_COORD)
        # avoid div by zero just in case
        if rg_range > 0:
            out[:, 0] = (2.0 * (out[:, 0] - MIN_RG_COORD) / rg_range) - 1.0
            out[:, 1] = (2.0 * (out[:, 1] - MIN_RG_COORD) / rg_range) - 1.0

        # segment_length / MAX_ROAD_LINE_SEGMENT_LEN
        if MAX_ROAD_LINE_SEGMENT_LEN > 0:
            out[:, 2] /= MAX_ROAD_LINE_SEGMENT_LEN

        # scale_x, scale_y / MAX_ROAD_SCALE
        if MAX_ROAD_SCALE > 0:
            out[:, 3] /= MAX_ROAD_SCALE
            out[:, 4] /= MAX_ROAD_SCALE

        # heading / MAX_ORIENTATION_RAD
        if MAX_ORIENTATION_RAD > 0:
            out[:, 5] /= MAX_ORIENTATION_RAD

        
        return out

# if __name__ == '__main__':
#     client = carla.Client("localhost", 2000)
#     client.set_timeout(10.0)
#     world = client.get_world()
#     carla_map = world.get_map()

#     STEP = 5.0
#     driving_wps = precompute_driving_waypoints(carla_map, step=STEP)
#     all_landmarks = precompute_landmarks    (carla_map)
#     all_crosswalks = precompute_crosswalks(carla_map)

#     # pick some geo location (e.g., from ego vehicle)
#     all_wps_for_test = carla_map.generate_waypoints(distance=10.0)
#     ego_wp = random.choice(all_wps_for_test)

#     #geo_location = ego_wp.transform.location
#     ego_location = carla.Location(x=-105.1, y=19.5)

#     RADIUS = 50.0
#     N_POINTS = 2000

#     points, ego_loc = draw_local_map_points_precomputed(
#         world,
#         driving_wps,
#         all_landmarks,
#         all_crosswalks,
#         geo_location=ego_location,
#         radius=RADIUS,
#         n_points=N_POINTS,
#         step=STEP,
#         life_time=20.0,
#         max_draw=200,  
#     )

#     print("Points shape:", points.shape)
#     print("First 5 points:\n", points[:5])
#     ego_wp = carla_map.get_waypoint(
#         ego_loc,
#         project_to_road=True,
#         lane_type=carla.LaneType.Driving,
#     )
#     ego_yaw_deg = ego_wp.transform.rotation.yaw

#     rg_features = carla_points_to_gpudrive_roadgraph(
#         points,               # (200, 9) from your sampler
#         ego_location=ego_loc,
#         ego_yaw_deg=ego_yaw_deg,
#     )
#     tensor_rg_feature = torch.tensor(rg_features)

#     print("Roadgraph features shape:", tensor_rg_feature.shape)  # (200, 13)
#     print("First row:", tensor_rg_feature[0])
