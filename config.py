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

        except FileNotFoundError:
            print(f'File {config_path} Not Found')
    
    def get_host(self):
        return self.host

if __name__ == "__main__":
    testLoader = ConfigLoader()
