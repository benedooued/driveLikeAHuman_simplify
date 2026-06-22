import sqlite3
import pandas as pd
import json
from pathlib import Path

RESULTS_DIR = Path("results")


# --------------------------------------------------
# Find latest sim.db
# --------------------------------------------------

db_files = list(RESULTS_DIR.rglob("sim.db"))

if not db_files:
    raise FileNotFoundError("No sim.db found")

latest_db = max(
    db_files,
    key=lambda p: p.stat().st_mtime
)

print(f"Using DB: {latest_db}")


# --------------------------------------------------
# Scenario name
# --------------------------------------------------

scenario_name = latest_db.parent.name


# --------------------------------------------------
# Load database
# --------------------------------------------------

conn = sqlite3.connect(latest_db)

metrics = pd.read_sql_query(
    "SELECT * FROM metricsINFO",
    conn
)

episodes = pd.read_sql_query(
    "SELECT * FROM episodeINFO",
    conn
)

decisions = pd.read_sql_query(
    "SELECT * FROM decisionINFO",
    conn
)

conn.close()

# print("\n===== DECISIONS COLUMNS =====")
# print(decisions.columns)

# print("\n===== PARSEDACTION SAMPLE =====")
# print(decisions["parsedAction"].head(10).tolist())
# --------------------------------------------------
# Safety
# --------------------------------------------------

collision_rate = float(
    episodes["collision"].mean() * 100
)

near_miss_rate = float(
    (
        metrics["min_vehicle_distance"] < 2.0
    ).mean()
    * 100
)

hard_braking_rate = float(
    (
        metrics["ego_acceleration"] < -3.0
    ).mean()
    * 100
)


# --------------------------------------------------
# Driving Control
# --------------------------------------------------

mean_lane_offset = float(
    metrics["lane_center_offset"]
    .abs()
    .mean()
)

acc_std = float(
    metrics["ego_acceleration"]
    .std()
)

acc_mean = float(
    metrics["ego_acceleration"]
    .mean()
)


# --------------------------------------------------
# Reasoning
# --------------------------------------------------



valid_actions = {
    "IDLE",
    "FASTER",
    "SLOWER",
    "LANE_LEFT",
    "LANE_RIGHT"
}

actions = decisions["parsedAction"].apply(
    lambda x: json.loads(x).get("action_name", "")
)

valid_count = actions.isin(valid_actions).sum()

action_validity_rate = (
    valid_count / len(actions) * 100
)




hallucination_rate = None
consistency_failure_rate = None

if "hallucination" in decisions.columns:

    hallucination_rate = float(
        decisions["hallucination"]
        .mean()
        * 100
    )

if "consistency_failure" in decisions.columns:

    consistency_failure_rate = float(
        decisions["consistency_failure"]
        .mean()
        * 100
    )


# --------------------------------------------------
# Performance
# --------------------------------------------------

distance = float(
    episodes["total_distance"].mean()
)

survival_time = float(
    episodes["survival_time"].mean()
)

avg_speed = float(
    episodes["average_speed"].mean()
)

reward = float(
    episodes["total_reward"].mean()
)


# --------------------------------------------------
# Human Like
# --------------------------------------------------

decision_latency = float(
    metrics["decision_latency_ms"].mean()
)

safety_score = 1 - collision_rate / 100

lane_score = max(
    0,
    1 - mean_lane_offset
)

smoothness_score = 1 / (
    1 + acc_std
)

human_like_score = float(
    0.4 * safety_score
    + 0.3 * lane_score
    + 0.3 * smoothness_score
)


# --------------------------------------------------
# JSON export
# --------------------------------------------------

results = {
    "scenario": scenario_name,

    "safety": {
        "collision_rate": collision_rate,
        "near_miss_rate": near_miss_rate,
        "hard_braking_rate": hard_braking_rate,
    },

    "driving_control": {
        "mean_lane_offset": mean_lane_offset,
        "acceleration_std": acc_std,
        "mean_acceleration": acc_mean,
    },

    "reasoning": {
        "action_validity_rate": action_validity_rate,
        "hallucination_rate": hallucination_rate,
        "consistency_failure_rate": consistency_failure_rate,
    },

    "performance": {
        "distance": distance,
        "survival_time": survival_time,
        "average_speed": avg_speed,
        "reward": reward,
    },

    "human_like": {
        "decision_latency_ms": decision_latency,
        "human_like_score": human_like_score,
    }
}


output_file = (
    latest_db.parent /
    "metrics_summary.json"
)

with open(output_file, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )

print(f"\nSaved: {output_file}")