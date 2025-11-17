import yaml
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

class ConfigLoader():
    def __init__(self, config_path='config.yaml'):
        print('Loading config from: ', config_path)
        try:
            cfg = load_config(config_path)
            self.host = cfg["carla"]["host"]
            self.port = cfg["carla"]["port"]
            self.controlled_count = cfg["experiment"]["controlled_count"]
            self.free_count = cfg["experiment"]["free_count"]
            self.spawn_max_attempts = cfg["experiment"]["spawn_max_attempts"]
            self.sensor_type = cfg["experiment"]["sensor_type"]
            self.lidar_tf = [cfg["experiment"]['lidar']['x'],cfg["experiment"]['lidar']['y'],cfg["experiment"]['lidar']['z'],cfg["experiment"]['lidar']['roll'],cfg["experiment"]['lidar']['pitch'],cfg["experiment"]['lidar']['yaw']]

        except FileNotFoundError:
            print(f'File {config_path} Not Found')
    
    # A bunch of getter methods
    def get_host(self):
        return self.host
    def get_port(self):
        return self.port
    def get_controlled_count(self):
        return self.controlled_count
    def get_free_count(self):
        return self.free_count
    def get_headcounts(self):
        return self.controlled_count, self.free_count
    def get_spawn_max_attempts(self):
        return self.spawn_max_attempts
    def get_sensor_type(self):
        return self.sensor_type
    def get_lidar_tf(self):
        return self.lidar_tf

if __name__ == "__main__":
    testLoader = ConfigLoader()
