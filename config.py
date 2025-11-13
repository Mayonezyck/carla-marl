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
            self.agent_count = cfg["experiment"]["agent_count"]
            self.actor_count = cfg["experiment"]["actor_count"]

        except FileNotFoundError:
            print(f'File {config_path} Not Found')
    
    # A bunch of getter methods
    def get_host(self):
        return self.host
    def get_port(self):
        return self.port
    def get_agent_count(self):
        return self.agent_count
    def get_actor_count(self):
        return self.actor_count

if __name__ == "__main__":
    testLoader = ConfigLoader()
