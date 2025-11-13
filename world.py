import carla
from config import ConfigLoader
class carlaWorld():
    def __init__(self, config):
        # Starting the world, connect to the carla world using a client
        world_host = config.get_host()
        world_port = config.get_port()
        self.client = carla.Client(world_host,world_port)
        self.world = self.client.get_world()
        print(self.world)
        

if __name__ == '__main__':
    carlaWorld(ConfigLoader())