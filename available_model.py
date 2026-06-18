# env = gym.make('roundabout-v0', render_mode="rgb_array")
# roundabout_config = {
#     "observation": {
#         "type": "Kinematics",
#         "features": ["presence", "x", "y", "vx", "vy"],
#         "absolute": True,
#         "normalize": False,
#         "vehicles_count": VEHICLE_COUNT,
#         "see_behind": True,
#     },
#     "action": {"type": "DiscreteMetaAction"},
#     "duration": 11,
#     "vehicles_density": 1.0,
# }
# env.configure(roundabout_config)

# critical_config = {
#     "vehicles_density": 3.0,       # Très dense
#     "vehicles_count": 15,
#     "duration": 60,
#     "controlled_vehicles": 1,
#     "initial_spacing": 1.0,        # Véhicules très proches au départ
# }

# # Dans ton main, après env.reset() :
# from highway_env.vehicle.behavior import AggressiveVehicle

# def inject_critical_vehicle(env):
#     road = env.unwrapped.road
#     ego = env.unwrapped.vehicle
#     # Spawn un véhicule juste devant l'ego sur la même voie
#     v = AggressiveVehicle.create_random(
#         road,
#         speed=ego.speed - 5,   # plus lent = forçage de freinage
#         lane_id=ego.lane_index,
#         spacing=0.5             # très proche
#     )
#     road.vehicles.append(v)

# obs, _ = env.reset(seed=42)  # même scène à chaque run