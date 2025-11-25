#!/usr/bin/env python3
import carla
import random
import time


NUM_WALKERS = 100
HOST = "localhost"
PORT = 2000


def spawn_walkers(world, num_walkers):
    blueprint_library = world.get_blueprint_library()

    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    if not walker_bps:
        print("[PedSpawner] No walker blueprints found!")
        return [], []

    try:
        controller_bp = blueprint_library.find("controller.ai.walker")
    except RuntimeError as e:
        print(f"[PedSpawner] Could not find controller.ai.walker: {e}")
        return [], []

    walkers = []
    controllers = []

    spawned = 0
    max_trials = num_walkers * 10  # avoid infinite loops
    trials = 0

    while spawned < num_walkers and trials < max_trials:
        trials += 1

        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue

        spawn_transform = carla.Transform(loc)
        walker_bp = random.choice(walker_bps)

        # Optional: make them mortal
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        walker = world.try_spawn_actor(walker_bp, spawn_transform)
        if walker is None:
            continue

        controller = world.try_spawn_actor(
            controller_bp,
            carla.Transform(),  # transform usually ignored
            attach_to=walker
        )

        if controller is None:
            print("[PedSpawner] Controller spawn failed, destroying walker.")
            walker.destroy()
            continue

        walkers.append(walker)
        controllers.append(controller)

        controller.start()
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            controller.go_to_location(dest)
        else:
            controller.go_to_location(walker.get_location())
        controller.set_max_speed(1.5 + random.random())

        spawned += 1
        print(f"[PedSpawner] Spawned walker {walker.id} with controller {controller.id}")

    print(f"[PedSpawner] Total spawned walkers: {len(walkers)} / requested {num_walkers}")
    return walkers, controllers


def cleanup_walkers(walkers, controllers):
    print("[PedSpawner] Cleaning up pedestrians...")
    # Stop/destroy controllers first
    for controller in controllers:
        if controller is None:
            continue
        try:
            controller.stop()
        except Exception:
            pass
        try:
            controller.destroy()
        except Exception as e:
            print(f"[PedSpawner] Error destroying controller: {e}")

    # Destroy walkers
    for walker in walkers:
        if walker is None:
            continue
        try:
            walker.destroy()
        except Exception as e:
            print(f"[PedSpawner] Error destroying walker: {e}")

    print("[PedSpawner] Cleanup complete.")


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()
    print("[PedSpawner] Connected to world:", world.get_map().name)

    walkers, controllers = spawn_walkers(world, NUM_WALKERS)

    try:
        print("[PedSpawner] Walkers are active. Press Ctrl+C to stop and clean up.")
        while True:
            # We DON'T touch world settings here.
            # Just block a bit so the script doesn't exit.
            # If your world is sync, something else should be ticking it.
            world.wait_for_tick()
            # or: time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[PedSpawner] Ctrl+C received.")
    finally:
        cleanup_walkers(walkers, controllers)


if __name__ == "__main__":
    main()
