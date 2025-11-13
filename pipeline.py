# This code is the one and only pipeline that handles different modules, wire them together, and 
# import 
import carla
import yaml
from client import carlaClient
from config import ConfigLoader
from world import carlaWorld

if __name__ == "__main__": 
    #Starting Carla Client
    config = ConfigLoader()
    client = carlaClient() #assess if the class is needed
    world = carlaWorld(config)
    

    pass