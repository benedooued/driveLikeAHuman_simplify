import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain_groq import ChatGroq
import time
from math import sqrt
from datetime import datetime
from highway_env.vehicle.behavior import AggressiveVehicle, IDMVehicle

from scenario.scenario import Scenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.customTools import (
    getAvailableActions, getAvailableLanes, getLaneInvolvedCar,
    isChangeLaneConflictWithCar, isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar, isDecelerationSafe,
)





def build_llm(model_name: str, max_tokens: int = 512) -> ChatGroq:
    kwargs = {
        "model_name": model_name,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    # Qwen 3.x
    if model_name.startswith("qwen/"):
        kwargs["reasoning_effort"] = "none"

    # GPT-OSS
    elif model_name.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"
        kwargs["max_tokens"] = max(max_tokens, 1024)

    return ChatGroq(**kwargs)


# ── Config ────────────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open('config.yaml'))
os.environ["GROQ_API_KEY"] = cfg['GROQ_API_KEY']

ENV_TYPE = cfg.get('ENV_TYPE', 'highway')      # "highway" ou "roundabout"
env_cfg  = cfg[ENV_TYPE]                        # sous-section du config

llm = build_llm(
    cfg.get(
        "GROQ_MODEL",
        "qwen/qwen3-32b"
    )
)

# ── Configs par environnement ─────────────────────────────────────────────────
VEHICLE_COUNT = env_cfg['vehicle_count']

BASE_OBS = {
    "type": "Kinematics",
    "features": ["presence", "x", "y", "vx", "vy"],
    "absolute": True,
    "normalize": False,
    "vehicles_count": VEHICLE_COUNT,
    "see_behind": True,
}

ENV_CONFIGS = {
    "highway": {
        "observation": BASE_OBS,
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": np.linspace(0, 32, 9).tolist(),
        },
        "duration": env_cfg['duration'],
        "vehicles_density": env_cfg['vehicles_density'],
        "lanes_count": env_cfg['lanes_count'],
        "show_trajectories": True,
        "render_agent": True,
    },
    "roundabout": {
        "observation": BASE_OBS,
        "action": {"type": "DiscreteMetaAction"},
        "duration": env_cfg['duration'],
        "vehicles_density": env_cfg['vehicles_density'],
        "render_agent": True,
    },
}

# ── Environment ───────────────────────────────────────────────────────────────
gym_id = env_cfg['gym_id']
env = gym.make(gym_id, render_mode="rgb_array")
env.configure(ENV_CONFIGS[ENV_TYPE])

# Dossier du run (un par expérience, rien n'est écrasé)
ts       = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_id = cfg.get('GROQ_MODEL', 'llm').split('-')[0]
run_dir  = f'results/{ENV_TYPE}_{model_id}_{ts}'
os.makedirs(f'{run_dir}/video', exist_ok=True)

env = RecordVideo(env, f'{run_dir}/video', name_prefix=ENV_TYPE)

# ── Scenario & Agent ──────────────────────────────────────────────────────────
sce = Scenario(
    VEHICLE_COUNT,
    database=f'{run_dir}/sim.db',
    env_type=ENV_TYPE,
    experiment_name=f'{ENV_TYPE}_{model_id}',
)

tools = [
    getAvailableActions(env),
    getAvailableLanes(sce),
    getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce),
    isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
]

# ── Scènes critiques ──────────────────────────────────────────────────────────


def inject_critical_vehicle(env, scenario: str = "cut_in"):
    """
    Injecte un véhicule critique après env.reset().

    Scénarios disponibles :
      - "cut_in"      : véhicule rapide sur la voie adjacente, force un dépassement
      - "hard_brake"  : véhicule lent juste devant l'ego sur la même voie
      - "tailgater"   : véhicule agressif très proche derrière l'ego
      - "dense"       : 3 véhicules lents encerclant l'ego

    :param env:      l'env gymnasium (déjà resetté)
    :param scenario: nom du scénario critique
    """
    road = env.unwrapped.road
    ego  = env.unwrapped.vehicle

    # lane_index de l'ego : (origin, dest, local_idx)
    ego_lane  = ego.lane_index
    ego_pos   = ego.position        # [x, y]
    ego_speed = ego.speed

    if scenario == "cut_in":
        # Véhicule rapide sur la voie adjacente, très proche latéralement
        # va probablement changer de voie devant l'ego
        adj_lane_id = max(0, ego_lane[2] - 1)   # voie à gauche (ou même voie si bord)
        v = AggressiveVehicle.make_on_lane(
            road,
            lane_index=(ego_lane[0], ego_lane[1], adj_lane_id),
            longitudinal=ego.lane_distance_to(ego) + 15,   # 15m devant
            speed=ego_speed + 8,                            # plus rapide → va couper
        )
        road.vehicles.append(v)

    elif scenario == "hard_brake":
        # Véhicule lent juste devant sur la même voie → force freinage d'urgence
        v = AggressiveVehicle.make_on_lane(
            road,
            lane_index=ego_lane,
            longitudinal=ego.lane_distance_to(ego) + 12,   # seulement 12m devant
            speed=max(5, ego_speed - 10),                   # bien plus lent
        )
        road.vehicles.append(v)

    elif scenario == "tailgater":
        # Véhicule agressif collé derrière l'ego
        v = AggressiveVehicle.make_on_lane(
            road,
            lane_index=ego_lane,
            longitudinal=ego.lane_distance_to(ego) - 8,    # 8m derrière
            speed=ego_speed + 5,                            # plus rapide → rattrape
        )
        road.vehicles.append(v)

    elif scenario == "dense":
        # 3 véhicules lents encerclant l'ego (devant, derrière, côté)
        offsets = [
            (ego_lane,                                   +18, ego_speed - 8),   # devant
            (ego_lane,                                   -10, ego_speed + 3),   # derrière
            ((ego_lane[0], ego_lane[1], max(0, ego_lane[2] - 1)), +5, ego_speed - 5),  # côté
        ]
        for lane_idx, dist, spd in offsets:
            v = IDMVehicle.make_on_lane(
                road,
                lane_index=lane_idx,
                longitudinal=ego.lane_distance_to(ego) + dist,
                speed=max(5, spd),
            )
            road.vehicles.append(v)

agent = DriverAgent(llm, tools, sce, verbose=False)

obs, _ = env.reset()

CRITICAL_SCENARIO = cfg.get('critical_scenario', 'none')
SEED              = cfg.get('seed', None)          # None = aléatoire, 42 = reproductible

obs, _ = env.reset(seed=SEED)
# Construit le graphe de lanes depuis l'env réel (critique pour roundabout)
if ENV_TYPE == "roundabout":
    sce._build_roadgraph_from_env(env)

if CRITICAL_SCENARIO != "none":
    inject_critical_vehicle(env, scenario=CRITICAL_SCENARIO)
    print(f"[yellow]⚠ Critical scenario injected: {CRITICAL_SCENARIO}[/yellow]")

# ── Boucle principale ─────────────────────────────────────────────────────────
terminated = truncated = False
frame = 0
decision = None
total_reward = 0.0
previous_speed = None
lane_change_count = 0
previous_action = None
latencies = []
last_collision = 0

try:
    while not (terminated or truncated):
        # Passe env pour récupérer les lane_index réels en roundabout
        sce.update_vehicles(obs, frame, env=env)

        t0 = time.perf_counter()
        decision = agent.run(decision)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        action_id = decision.get("action_id", 1) if decision else 1
        obs, reward, terminated, truncated, info = env.step(action_id)
        time.sleep(cfg.get('frame_delay_s', 3.0))
        env.render()
        total_reward += reward

        ego = sce.vehicles['ego']
        current_speed = ego.speed
        dt = 1.0 / env.config["policy_frequency"]

        acceleration = (
            (current_speed - previous_speed) / dt
            if previous_speed is not None
            else 0.0
        )
        previous_speed = current_speed
        last_collision = int(info.get('crashed', False))

        other_distances = [
            sqrt((v.x - ego.x)**2 + (v.y - ego.y)**2)
            for v in sce.vehicles.values()
            if v.presence and v.id != 'ego'
        ]
        min_distance = min(other_distances) if other_distances else 999.0

        # lane_offset : valide en highway, approximatif en roundabout
        lane_offset = ego.lane_offset

        if previous_action in (0, 2) and action_id not in (0, 2):
            lane_change_count += 1
        previous_action = action_id

        tokens_in  = decision.get('_tokens_in',  0) if decision else 0
        tokens_out = decision.get('_tokens_out', 0) if decision else 0

        sce.commit_metrics(
            reward=reward, action_id=action_id,
            collision=last_collision, min_distance=min_distance,
            lane_offset=lane_offset, acceleration=acceleration,
            tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms,
        )

        frame += 1
        print(f"[{ENV_TYPE}] frame={frame:3d} | action={action_id} | "
              f"reward={reward:.2f} | speed={current_speed:.1f} | "
              f"dist={min_distance:.1f}m | collision={last_collision} | "
              f"latency={latency_ms:.0f}ms")

finally:
    env.close()
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    sce.commit_episode(
        total_frames=frame, collision=last_collision,
        lane_change_count=lane_change_count,
        average_latency=avg_latency, total_reward=total_reward,
    )
    print(f"\nDone [{ENV_TYPE}] — {frame} frames | "
          f"reward={total_reward:.2f} | run: {run_dir}")