"""
DriveLikeAHuman — simplified with Groq (free LLM).
"""

import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain_groq import ChatGroq

from scenario.scenario import Scenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.customTools import (
    getAvailableActions, getAvailableLanes, getLaneInvolvedCar,
    isChangeLaneConflictWithCar, isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar, isDecelerationSafe,
)

# ── Config ────────────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open('config.yaml'))
os.environ["GROQ_API_KEY"] = cfg['GROQ_API_KEY']

llm = ChatGroq(
    model_name=cfg.get('GROQ_MODEL', 'llama-3.3-70b-versatile'),
    temperature=0,
    max_tokens=1024,
)

# ── Environment ───────────────────────────────────────────────────────────────
VEHICLE_COUNT = 10  # keep low → smaller context, faster inference

env_config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        "vehicles_count": VEHICLE_COUNT,
        "see_behind": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),
    },
    "duration": 40,
    "vehicles_density": 1.5,
    "show_trajectories": True,
    "render_agent": True,
}

env = gym.make('highway-v0', render_mode="rgb_array")
env.configure(env_config)
env = RecordVideo(env, './results-video', name_prefix="highway")

obs, _ = env.reset()
env.render()

# ── Scenario & Agent ──────────────────────────────────────────────────────────
os.makedirs('results-db', exist_ok=True)
sce = Scenario(VEHICLE_COUNT, database='results-db/highway.db')

tools = [
    getAvailableActions(env),
    getAvailableLanes(sce),
    getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce),
    isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
]

agent = DriverAgent(llm, tools, sce, verbose=True)

# ── Main loop ─────────────────────────────────────────────────────────────────
# decision = None

# frame = 0
# terminated = truncated = False
# try:
#     while not (terminated or truncated):
#         sce.update_vehicles(obs, frame)
#         decision = agent.run(decision)
#         if decision is None or "action_id" not in decision:
#             action_id = 1  # IDLE safe default
#         else:
#             action_id = decision["action_id"]
#         env.render()
       
#         obs, reward, terminated, truncated, _ = env.step(decision["action_id"])
#         frame += 1
#         print("frame:", frame)
#         print("action:", decision)
#         print("terminated:", terminated, "truncated:", truncated)
# finally:
#     env.close()
#     print(f"Done after {frame} frames. DB: results-db/highway.db")


obs, _ = env.reset()

terminated = truncated = False
frame = 0

decision = None

try:
    while not (terminated or truncated):

        sce.update_vehicles(obs, frame)

        decision = agent.run(decision)

        if decision is None or "action_id" not in decision:
            action_id = 1
        else:
            action_id = decision["action_id"]

        obs, reward, terminated, truncated, _ = env.step(action_id)

        env.render()

        frame += 1

finally:
    env.close()
    print(f"Done after {frame} frames. DB: results-db/highway.db")