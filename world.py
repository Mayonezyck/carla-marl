import time
import carla
from config import ConfigLoader
from manager import Manager


class CarlaWorld:
    def __init__(self, config):
        world_host = config.get_host()
        world_port = config.get_port()
        self.client = carla.Client(world_host, world_port)
        self.client.set_timeout(10.0)
        self.config = config
        self.world = self.client.get_world()

        # Save original settings for cleanup
        self.original_settings = self.world.get_settings()

        # --- Make world synchronous ---
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        #settings.fixed_delta_seconds = config.get_fixed_delta_seconds()  # e.g. 0.05
        self.world.apply_settings(settings)

        # --- Traffic Manager in sync mode ---
        self.tm = self.client.get_trafficmanager(8000)  # or pass tm_port from config
        self.tm.set_synchronous_mode(True)

        # Manager spawns agents (free agents will use autopilot + TM)
        self.manager = Manager(config, self.world)
        self.tick()
        self.fixed_dt = config.get_fixed_delta_seconds()

    def tick(self):
        return self.world.tick()
    
    def run(self):
        # Wall-clock pacing (can be same as fixed_dt)
        wall_dt = getattr(self.config, "get_wall_dt", lambda: self.fixed_dt)()

        try:
            while True:
                # *** THIS is the only place we tick the world ***
                print('tick')

                _, batch_obs=self.manager.get_controlled_lidar_observations()
                print("batch_obs shape:", batch_obs.shape)
                snapshot = self.world.tick()
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
        self.tm.set_synchronous_mode(False)


if __name__ == "__main__":
    cfg = ConfigLoader()
    cw = CarlaWorld(cfg)
    cw.run()
