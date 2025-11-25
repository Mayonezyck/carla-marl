import carla
from basic_env import BasicEnv

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    env = BasicEnv(world, sampling_resolution=2.0)

    vehicle, route = env.reset(visualize=True, life_time=60.0)

    print(f"Spawned vehicle id: {vehicle.id}")
    print(f"Route length (waypoints): {len(route)}")

    # If the world is in synchronous mode, don't forget to tick in your main loop.
    # Here we just tick a few times so you see the debug route.
    for _ in range(200):
        world.tick()
    
if __name__ == "__main__":
    main()
