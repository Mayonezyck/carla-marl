import time
import carla
from config import ConfigLoader
from manager import Manager


class CarlaWorld:
    def __init__(self, config):
        self.config = config

        world_host = config.get_host()
        world_port = config.get_port()
        self.client = carla.Client(world_host, world_port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Save original settings
        self.original_settings = self.world.get_settings()

        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True

        # Fixed sim step from config (e.g. 0.05 sec)
        self.fixed_dt = getattr(config, "get_fixed_delta_seconds", lambda: 0.05)()
        settings.fixed_delta_seconds = self.fixed_dt
        self.world.apply_settings(settings)

        # Create manager
        self.manager = Manager(config, self.world)

    def setup(self):
        # Let manager spawn stuff, only once
        if hasattr(self.manager, "setup"):
            self.manager.setup()
        elif hasattr(self.manager, "create_agents"):
            self.manager.create_agents()
        # else: do nothing

    def run(self):
        # Wall-clock pacing (can be same as fixed_dt)
        wall_dt = getattr(self.config, "get_wall_dt", lambda: self.fixed_dt)()

        try:
            while True:
                # *** THIS is the only place we tick the world ***
                snapshot = self.world.tick()
                print("tick")

                # Optional: if manager has per-step logic, call it
                if hasattr(self.manager, "tick"):
                    self.manager.tick(snapshot)
                elif hasattr(self.manager, "step"):
                    self.manager.step(snapshot)

                time.sleep(wall_dt)

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        finally:
            self.cleanup()

    def cleanup(self):
        # Despawn agents
        if hasattr(self.manager, "cleanup"):
            self.manager.cleanup()

        # Restore original world settings
        self.world.apply_settings(self.original_settings)


if __name__ == "__main__":
    cfg = ConfigLoader()
    cw = CarlaWorld(cfg)
    cw.setup()
    cw.run()
