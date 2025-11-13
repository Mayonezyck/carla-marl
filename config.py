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
            print(cfg)
        except FileNotFoundError:
            print(f'File {config_path} Not Found')

if __name__ == "__main__":
    testLoader = ConfigLoader()
    