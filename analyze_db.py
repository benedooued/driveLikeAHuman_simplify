import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "/home/benedo/Téléchargements/driveLikeAHuman_simplify-main/results/highway_llama_2026-06-16_12-43-22/sim.db"

conn = sqlite3.connect(DB_PATH)

metrics = pd.read_sql_query("SELECT * FROM metricsINFO", conn)
episodes = pd.read_sql_query("SELECT * FROM episodeINFO", conn)
decisions = pd.read_sql_query("SELECT * FROM decisionINFO", conn)

conn.close()

print("=" * 60)
print("SAFETY METRICS")
print("=" * 60)

# Collision Rate
collision_rate = episodes["collision"].mean() * 100

# Near Miss Rate (seuil = 2 m)
near_miss_rate = (
    (metrics["min_vehicle_distance"] < 2.0).sum()
    / len(metrics)
    * 100
)

# Hard Braking Frequency
hard_braking_count = (
    metrics["ego_acceleration"] < -3.0
).sum()

hard_braking_rate = (
    hard_braking_count
    / len(metrics)
    * 100
)

print(f"Collision Rate (%): {collision_rate:.2f}")
print(f"Near Miss Rate (%): {near_miss_rate:.2f}")
print(f"Hard Braking Frequency (%): {hard_braking_rate:.2f}")


print("\n" + "=" * 60)
print("DRIVING CONTROL")
print("=" * 60)

# Lane Following Quality
mean_lane_offset = metrics["lane_center_offset"].abs().mean()

# Driving Smoothness
acc_std = metrics["ego_acceleration"].std()

# Acceleration Stability
acc_mean = metrics["ego_acceleration"].mean()

print(f"Mean Lane Offset (m): {mean_lane_offset:.3f}")
print(f"Acceleration Std (m/s²): {acc_std:.3f}")
print(f"Mean Acceleration (m/s²): {acc_mean:.3f}")


print("\n" + "=" * 60)
print("REASONING QUALITY")
print("=" * 60)

valid_actions = [
    "IDLE",
    "FASTER",
    "SLOWER",
    "LANE_LEFT",
    "LANE_RIGHT"
]

valid_count = decisions["parsedAction"].isin(valid_actions).sum()

action_validity_rate = (
    valid_count
    / len(decisions)
    * 100
)

print(f"Action Validity Rate (%): {action_validity_rate:.2f}")

print("Thought-Action Consistency: Manual Evaluation")
print("Hallucination Rate: Manual Evaluation")


print("\n" + "=" * 60)
print("PERFORMANCE")
print("=" * 60)

distance = episodes["total_distance"].mean()

survival_time = episodes["survival_time"].mean()

avg_speed = episodes["average_speed"].mean()

reward = episodes["total_reward"].mean()

print(f"Distance Travelled: {distance:.2f}")
print(f"Survival Time: {survival_time:.2f}")
print(f"Average Speed: {avg_speed:.2f}")
print(f"Total Reward: {reward:.2f}")


print("\n" + "=" * 60)
print("HUMAN-LIKE BEHAVIOUR")
print("=" * 60)

decision_latency = metrics["decision_latency_ms"].mean()

print(f"Decision Latency (ms): {decision_latency:.2f}")

# Human-Like Score simplifié

safety_score = 1 - (collision_rate / 100)

lane_score = max(
    0,
    1 - mean_lane_offset
)

smoothness_score = 1 / (
    1 + acc_std
)

human_like_score = (
    0.4 * safety_score
    + 0.3 * lane_score
    + 0.3 * smoothness_score
)

print(f"Human-Like Score: {human_like_score:.3f}")

print("\nRule Violation Rate: Not Available")