from config import ConfigLoader
import random
import carla
class Manager():
    def __init__(self, config, world):
        self.controlled_count = config.get_controlled_count() #TODO: Determine if they should be stored as fields? Or they should just be passed.
        self.free_count = config.get_free_count()
        print(f'We are going to add {self.controlled_count} agents, and {self.free_count} actors.')
        #Start a roster of controlled agents
        self.spawn_both_controlled_and_free(self.controlled_count, self.free_count, world)
        
    
    def spawn_both_controlled_and_free(self, nc, nf, world):
        # This method spawn both the controlled vehicles and free vehicles.
        self.roster_controlled = self.spawn_random_vehicles(nc, world)
        self.roster_free = self.spawn_random_vehicles(nf, world)
        for v in self.roster_free:
            v.set_autopilot(True)

    def cleanup(self):
        print("Now we remove all the generated agents")
        self.destroy_actors(self.roster_controlled)
        self.destroy_actors(self.roster_free)
    

    def spawn_random_vehicles(self, n: int, world: carla.World):
        """
        Spawn up to n random vehicles at random spawn points in the given CARLA world.

        Args:
            n (int): Number of vehicles to try to spawn.
            world (carla.World): The world in which to spawn vehicles.

        Returns:
            list[carla.Actor]: List of successfully spawned vehicle actors.
        """
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter("vehicle.*")

        # All available spawn points in the map
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("[spawn_random_vehicles] No spawn points available in this map.")
            return []

        # Shuffle spawn points for randomness
        random.shuffle(spawn_points)

        num_to_spawn = min(n, len(spawn_points))
        spawned_vehicles = []

        for i in range(num_to_spawn):
            bp = random.choice(vehicle_blueprints)

            # Randomize color if supported
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)

            # Optional role name
            bp.set_attribute("role_name", f"random_vehicle_{i}")

            transform = spawn_points[i]

            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                spawned_vehicles.append(vehicle)
                print(f'Spawned {vehicle}')
            else:
                # If spawn failed (e.g., collision at spawn), try a few more random points
                extra_trials = 5
                success = False
                for _ in range(extra_trials):
                    transform = random.choice(spawn_points)
                    vehicle = world.try_spawn_actor(bp, transform)
                    if vehicle is not None:
                        spawned_vehicles.append(vehicle)
                        success = True
                        break
                if not success:
                    print("[spawn_random_vehicles] Failed to spawn a vehicle after extra trials.")

        if len(spawned_vehicles) < n:
            print(f"[spawn_random_vehicles] Requested {n}, but only spawned {len(spawned_vehicles)} vehicles.")

        return spawned_vehicles
    

    def destroy_actors(self, actors):
        """
        Destroys all CARLA actors in the given iterable.

        Args:
            actors (Iterable[carla.Actor]): List (or any iterable) of CARLA actors to destroy.
        """
        if not actors:
            print("[destroy_actors] No actors to destroy.")
            return

        for actor in actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError as e:
                # Actor may already be destroyed or invalid
                print(f"[destroy_actors] Failed to destroy actor {actor.id if actor else 'unknown'}: {e}")

        print(f"[destroy_actors] Destroyed {len(actors)} actors.")



if __name__ == "__main__":
    Manager()