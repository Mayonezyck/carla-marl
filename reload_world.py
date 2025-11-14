#!/usr/bin/env python3
import carla

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    print("[*] Reloading CARLA world...")
    world = client.reload_world()
    print("[*] World reloaded:", world.get_map().name)

if __name__ == "__main__":
    main()
