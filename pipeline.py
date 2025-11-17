# This code is the one and only pipeline that handles different modules, wire them together, and 
# import 
import carla
import yaml
import time
from client import carlaClient
from config import ConfigLoader
from world import CarlaWorld

if __name__ == "__main__": 
    #Starting Carla Client
    config = ConfigLoader()
    client = carlaClient() #assess if the class is needed
    world = CarlaWorld(config)
    world.run()
    #time.sleep(10) #Sleep for 10 seconds so that we can see the world

    pass