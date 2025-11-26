from customAgents import Agent
from typing import Dict, Any, Optional
import carla
import os


class Free_Agents(Agent):
    """
    Free agents:
    - Spawn a vehicle (via Agent).
    - Enable autopilot.
    - Optionally attach 4 sensors (RGB, seg, seg_raw, depth) that save images to disk.
    """

    TM_PORT = 8000

    def __init__(
        self,
        world: carla.World,
        index: int,
        config: Dict[str, Any],
        with_sensors: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        :param with_sensors: if True, attach sensors and save images
        :param output_dir: base folder where images are saved; must be provided if with_sensors=True
                          The structure will be:
                              output_dir/
                                rgb/
                                seg/
                                seg_raw/
                                depth/
        """
        super().__init__(world, index, config, role_prefix="free_agent")

        self.with_sensors = with_sensors
        self.output_dir = output_dir

        # Camera actors
        self.rgb_cam: Optional[carla.Actor] = None
        self.seg_cam: Optional[carla.Actor] = None
        self.seg_raw_cam: Optional[carla.Actor] = None
        self.depth_cam: Optional[carla.Actor] = None

        # Per-modality directories
        self.rgb_dir: Optional[str] = None
        self.seg_dir: Optional[str] = None
        self.seg_raw_dir: Optional[str] = None
        self.depth_dir: Optional[str] = None

        self._enable_autopilot()

        if self.with_sensors:
            if self.output_dir is None:
                raise ValueError("output_dir must be provided when with_sensors=True")
            self._prepare_output_dirs()
            self._setup_sensors()

    def _enable_autopilot(self) -> None:
        if self.vehicle is not None:
            self.vehicle.set_autopilot(True, self.TM_PORT)
            print(f"[Free_Agents] Enabled autopilot for free_agent_{self.index}")

    # ========= OUTPUT FOLDER SETUP =========

    def _prepare_output_dirs(self) -> None:
        """
        Create rgb / seg / seg_raw / depth subfolders under output_dir.
        """
        base = self.output_dir
        self.rgb_dir = os.path.join(base, "rgb")
        self.seg_dir = os.path.join(base, "seg")
        self.seg_raw_dir = os.path.join(base, "seg_raw")
        self.depth_dir = os.path.join(base, "depth")
        self.depth_raw_dir = os.path.join(base, "depth_raw")

        for d in [self.rgb_dir, self.seg_dir, self.seg_raw_dir, self.depth_dir]:
            os.makedirs(d, exist_ok=True)

        print(f"[Free_Agents] Image output dirs created under: {base}")

    # ========= SENSOR SETUP (same intrinsics & pose as your script) =========

    def _setup_sensors(self) -> None:
        if self.vehicle is None:
            print(f"[Free_Agents] Cannot attach sensors, vehicle is None for free_agent_{self.index}")
            return

        world = self.world
        blueprints = world.get_blueprint_library()

        CAMERA_WIDTH = 640
        CAMERA_HEIGHT = 480
        CAMERA_FOV = 110.0
        CAPTURE_INTERVAL = 0.1

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        def make_camera(bp_name: str) -> carla.Actor:
            bp = blueprints.find(bp_name)
            bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
            bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
            bp.set_attribute("fov", str(CAMERA_FOV))
            bp.set_attribute("sensor_tick", f"{CAPTURE_INTERVAL}")
            return world.spawn_actor(bp, camera_transform, attach_to=self.vehicle)

        # Spawn cameras
        self.rgb_cam = make_camera("sensor.camera.rgb")
        self.seg_cam = make_camera("sensor.camera.semantic_segmentation")
        self.seg_raw_cam = make_camera("sensor.camera.semantic_segmentation")
        self.depth_cam = make_camera("sensor.camera.depth")

        # Attach callbacks that save directly to disk
        self.rgb_cam.listen(self._rgb_callback)
        self.seg_cam.listen(self._seg_callback)
        self.seg_raw_cam.listen(self._seg_raw_callback)
        self.depth_cam.listen(self._depth_callback)

        print(f"[Free_Agents] Attached sensors for free_agent_{self.index}")

    # ========= CALLBACKS: SAVE IMAGES (NO QUEUES) =========

    def _make_filename(self, image: carla.Image) -> str:
        """
        Use CARLA frame number as unique, shared base name.
        All modalities from the same tick will share this name.
        """
        return f"{image.frame:06d}.png"

    def _rgb_callback(self, image: carla.Image) -> None:
        if not self.rgb_dir:
            return
        filename = self._make_filename(image)
        path = os.path.join(self.rgb_dir, filename)
        image.save_to_disk(path)

    def _seg_callback(self, image: carla.Image) -> None:
        if not self.seg_dir:
            return
        filename = self._make_filename(image)
        path = os.path.join(self.seg_dir, filename)
        image.save_to_disk(path, carla.ColorConverter.CityScapesPalette)

    def _seg_raw_callback(self, image: carla.Image) -> None:
        if not self.seg_raw_dir:
            return
        filename = self._make_filename(image)
        path = os.path.join(self.seg_raw_dir, filename)
        image.save_to_disk(path, carla.ColorConverter.Raw)

    def _depth_callback(self, image: carla.Image) -> None:
        if not self.depth_dir:
            return
        filename = self._make_filename(image)
        path = os.path.join(self.depth_dir, filename)
        path_raw = os.path.join(self.depth_raw_dir, filename)
        image.save_to_disk(path, carla.ColorConverter.LogarithmicDepth)
        image.save_to_disk(path_raw, carla.ColorConverter.Raw)

    # ========= CLEANUP =========

    def destroy_sensors(self) -> None:
        """
        Stop and destroy all attached sensors.
        """
        for cam_name in ["rgb_cam", "seg_cam", "seg_raw_cam", "depth_cam"]:
            cam = getattr(self, cam_name, None)
            if cam is not None:
                try:
                    cam.stop()
                except Exception:
                    pass
                try:
                    cam.destroy()
                except Exception:
                    pass
                setattr(self, cam_name, None)
        print(f"[Free_Agents] Destroyed sensors for free_agent_{self.index}")
