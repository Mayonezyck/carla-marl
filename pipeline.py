# This code is the one and only pipeline that handles different modules, wire them together, and 
# import 
import carla
import yaml
from client import carlaClient
from config import ConfigLoader

if __name__ == "__main__": 
    #Starting Carla Client
    configLoader = ConfigLoader()
    client = carlaClient()
    

    pass