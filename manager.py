from config import ConfigLoader
class Manager():
    def __init__(self, config):
        self.agent_count = config.get_agent_count()
        self.actor_count = config.get_actor_count()
        print(f'We are going to add {self.agent_count} agents, and {self.actor_count} actors.')
        pass

if __name__ == "__main__":
    Manager()