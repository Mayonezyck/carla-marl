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
    client = carlaClient() #assess if the class is needed
    world = CarlaWorld(config)
    rl = RLHandler(world.manager)
    from datetime import datetime

    try:
        while True:
            # 1) RL step: read obs, log transition, choose & apply new actions
            #print(f'Before rl step: {datetime.now().time()}')
            frames, obs, actions, rewards, dones = rl.step()

            # 2) Advance the CARLA world one tick
            #print(f'After RL tick: {datetime.now().time()}')
            world.tick()

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        world.cleanup()
    # world.run()
    # world.cleanup()
    #time.sleep(10) #Sleep for 10 seconds so that we can see the world

    pass