# This code is the one and only pipeline that handles different modules, wire them together, and 
# import 
import carla
import yaml
import time
from client import carlaClient
from config import ConfigLoader
from world import CarlaWorld
from RL_handler import RLHandler

if __name__ == "__main__": 
    #Starting Carla Client
    config = ConfigLoader()
    client = carlaClient()
    world = CarlaWorld(config)
    rl = RLHandler(world.manager)

    from datetime import datetime
    t = 0 
    import traceback

    try:
        while True:
            t += 1
            try:
                # 1) RL step: read obs, log transition, choose & apply new actions
                obs, act, rew, done = rl.step()
                print("step", t, "obs shape:", obs.shape, "act shape:", act.shape)

                if config.get_if_route_planning():
                    world.manager.visualize_path()

                # 2) Advance the CARLA world one tick
                world.tick()

            except Exception as e:
                print("[PIPELINE] Exception inside loop:", repr(e))
                traceback.print_exc()
                break   # exit while True so we still hit finally

    except KeyboardInterrupt:
        print("Stopping via KeyboardInterrupt...")
    finally:
        world.cleanup()