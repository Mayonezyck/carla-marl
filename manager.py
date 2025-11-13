from config import ConfigLoader
class Manager():
    def __init__(self, config):
        self.controlled_count = config.get_controlled_count()
        self.free_count = config.get_free_count()
        print(f'We are going to add {self.controlled_count} agents, and {self.free_count} actors.')

        pass
    def cleanup(self):
        print("Now we remove all the generated agents")

if __name__ == "__main__":
    Manager()