import time
import carla
from config import ConfigLoader
from manager import Manager


class CarlaWorld:
    def __init__(self, config):
        self.config = config

        # 1) Connect to server
        world_host = config.get_host()
        world_port = config.get_port()
        self.client = carla.Client(world_host, world_port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()

        # Save original settings so we can restore them in cleanup()
        self.original_settings = self.world.get_settings()

        # 2) Enable synchronous mode + fixed delta from config
        settings = self.world.get_settings()
        settings.synchronous_mode = True

        # Option A: you store a fixed delta directly
        # (e.g. 0.05 for 20 Hz sim)
        try:
            fixed_dt = self.config.get_fixed_delta_seconds()
        except AttributeError:
            # Option B: you store a sim FPS instead
            # (e.g. 20 → 0.05)
            sim_fps = getattr(self.config, "get_sim_fps", lambda: 20)()
            fixed_dt = 1.0 / sim_fps

        settings.fixed_delta_seconds = fixed_dt
        self.world.apply_settings(settings)

        # 3) Optional: if you use Traffic Manager, sync it too
        # self.tm = self.client.get_trafficmanager()
        # self.tm.set_synchronous_mode(True)

        # 4) Initialize your Manager with this world
        self.manager = Manager(config, self.world)

    def setup(self):
        """
        Let the manager create/spawn everything here.
        No ticking yet.
        """
        if hasattr(self.manager, "setup"):
            self.manager.setup()
        else:
            # Or whatever your manager uses for creation
            self.manager.create_agents()

    def run(self):
        """
        Main simulation loop:
        - we tick the world manually
        - we call manager.step/tick per frame
        - we pace wall-clock time using the configured rate
        """
        fixed_dt = self.world.get_settings().fixed_delta_seconds or 0.05

        # If you also have a "wall clock rate" in config, use it;
        # otherwise simulate in (roughly) real-time.
        try:
            wall_dt = self.config.get_wall_dt()
        except AttributeError:
            wall_dt = fixed_dt

        try:
            while True:
                # Advance simulation by fixed_delta_seconds
                snapshot = self.world.tick()

                # Let your Manager do per-timestep logic
                if hasattr(self.manager, "tick"):
                    self.manager.tick(snapshot)
                elif hasattr(self.manager, "step"):
                    self.manager.step(snapshot)

                # Optional: slow down to approximately real-time
                time.sleep(wall_dt)

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        # Have the agent manager remove all generated agents
        self.manager.cleanup()

        # Restore original world settings
        self.world.apply_settings(self.original_settings)


if __name__ == '__main__':
    cfg = ConfigLoader()
    cw = CarlaWorld(cfg)
    cw.setup()   # manager spawns everything here
    cw.run()     # now start manual ticking loop
