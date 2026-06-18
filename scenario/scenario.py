from scenario.baseClass import Lane, Vehicle
from typing import List, Dict
from datetime import datetime
import sqlite3
import json
import os

# Generate the env for simulation, either highway or roundabout, create table in DB, and store the scenario data in the DB.

class Scenario:
    def __init__(self,
                 vehicleCount: int,
                 database: str = None,
                 env_type: str = "highway",
                 experiment_name: str = None
                 ) -> None:
        self.env_type = env_type
        self.lanes: Dict[str, Lane] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.vehicleCount = vehicleCount
        self.frame = 0
        self.total_distance = 0.0
        self.previous_ego_x = None
        self.previous_ego_y = None

        # Roadgraph statique seulement en highway
        if env_type == "highway":
            self._build_roadgraph()

        self._init_vehicles()

        # --- Persistance : un dossier par run, rien n'est écrasé ---
        if database is None:
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')   # corrigé : datatime → datetime
            name = experiment_name or env_type
            run_dir = f'results/{name}_{ts}'
            os.makedirs(run_dir, exist_ok=True)
            database = f'{run_dir}/sim.db'

        self.database = database
        self.run_dir = os.path.dirname(database)

        self._init_db()

    # ------------------------------------------------------------------
    # Roadgraph
    # ------------------------------------------------------------------

    def _build_roadgraph(self):
        """Lanes rectilignes highway (statique, connu à l'avance)."""
        for i in range(4):
            lid = f'lane_{i}'
            self.lanes[lid] = Lane(
                id=lid, laneIdx=i,
                left_lanes=[f'lane_{k}' for k in range(0, i)],
                right_lanes=[f'lane_{j}' for j in range(i + 1, 4)],
            )

    def _build_roadgraph_from_env(self, env):
        """
    Roundabout : construit le graphe depuis le RoadNetwork réel de l'env.
        À appeler après env.reset(), avant la boucle principale.
        """
        if self.env_type != "roundabout":
            return

        self.lanes.clear()
        graph = env.unwrapped.road.network.graph  # dict[origin: dict[dest: list[AbstractLane]]]
        idx = 0
        for origin, destinations in graph.items():
            for dest, lanes in destinations.items():
                for local_idx, _ in enumerate(lanes):
                    lid = f'{origin}_{dest}_{local_idx}'
                    self.lanes[lid] = Lane(id=lid, laneIdx=idx)
                    idx += 1

    # ------------------------------------------------------------------
    # Vehicles
    # ------------------------------------------------------------------

    def _init_vehicles(self):
        for i in range(self.vehicleCount):
            vid = 'ego' if i == 0 else f'veh{i}'
            self.vehicles[vid] = Vehicle(id=vid)

    def update_vehicles(self, observation: List[List], frame: int, env=None):
        """
        env est optionnel mais nécessaire en roundabout pour récupérer
        les lane_index réels depuis le RoadNetwork.
        """
        self.frame = frame
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()

        # Récupère la liste ordonnée des véhicules depuis l'env si disponible
        env_vehicles = []
        if env is not None:
            env_vehicles = env.unwrapped.road.vehicles

        for i, obs in enumerate(observation):
            vid = 'ego' if i == 0 else f'veh{i}'
            presence, x, y, vx, vy = obs
            if presence:
                veh = self.vehicles[vid]
                veh.presence = True

                # Récupère lane_index depuis le vrai objet véhicule si possible
                lane_index = None
                if i < len(env_vehicles):
                    lane_index = getattr(env_vehicles[i], 'lane_index', None)

                veh.updateProperty(x, y, vx, vy,
                                   lane_index=lane_index,
                                   env_type=self.env_type)

                cur.execute(
                    'INSERT OR REPLACE INTO vehINFO VALUES (?,?,?,?,?,?,?);',
                    (frame, vid, float(x), float(y),
                     veh.lane_id, float(vx), float(vy))
                )
            else:
                self.vehicles[vid].clear()

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _init_db(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vehINFO(
                frame    INT,
                id       TEXT,
                x        REAL,
                y        REAL,
                lane_id  TEXT,
                speedx   REAL,
                speedy   REAL,
                PRIMARY KEY (frame, id)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS decisionINFO(
    frame           INT PRIMARY KEY,
    scenario        TEXT,
    thoughts        TEXT,
    finalAnswer     TEXT,
    parsedAction    TEXT,
    formatFailure   INT DEFAULT 0,
    hallucination INT DEFAULT 0,
                    consistencyFail  INT DEFAULT 0
        );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metricsINFO(
                frame                INT PRIMARY KEY,
                ego_x                REAL,
                ego_y                REAL,
                ego_speed            REAL,
                ego_acceleration     REAL,
                ego_lane             TEXT,
                reward               REAL,
                action_id            INT,
                collision            INT,
                min_vehicle_distance REAL,
                lane_center_offset   REAL,
                llm_tokens_in        INT,
                llm_tokens_out       INT,
                decision_latency_ms  REAL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodeINFO(
                episode_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                total_frames     INT,
                total_distance   REAL,
                collision        INT,
                survival_time    REAL,
                lane_change_count INT DEFAULT 0,
                average_speed    REAL,
                average_latency  REAL,
                total_reward     REAL
            );
        """)

        
        conn.commit()
        conn.close()

    def commit_metrics(
        self,
        reward: float,
        action_id: int,
        collision: int,
        min_distance: float,
        lane_offset: float,
        acceleration: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ):
        ego = self.vehicles['ego']
        if self.previous_ego_x is not None:

            dx = ego.x - self.previous_ego_x
            dy = ego.y - self.previous_ego_y

            self.total_distance += (dx**2 + dy**2) ** 0.5

        self.previous_ego_x = ego.x
        self.previous_ego_y = ego.y
        conn = sqlite3.connect(self.database)
        conn.execute("""
            INSERT OR REPLACE INTO metricsINFO(
                frame, ego_x, ego_y, ego_speed, ego_acceleration, ego_lane,
                reward, action_id, collision, min_vehicle_distance,
                lane_center_offset, llm_tokens_in, llm_tokens_out, decision_latency_ms
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?);
        """, (
            self.frame, ego.x, ego.y, ego.speed, acceleration, ego.lane_id,
            reward, action_id, collision, min_distance,
            lane_offset, tokens_in, tokens_out, latency_ms,
        ))
        conn.commit()
        conn.close()

    def commit_episode(
        self,
        total_frames: int,
        collision: int,
        lane_change_count: int,
        average_latency: float,
        total_reward: float,
    ):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute("""
            SELECT
                SUM(ego_speed) / COUNT(*) AS average_speed, 
                AVG(ego_speed)          AS average_speed,
                MAX(frame)              AS survival_time
            FROM metricsINFO
        """)
        row = cur.fetchone()
        total_distance = self.total_distance
        average_speed  = row[1] or 0.0
        survival_time  = row[2] or 0.0

        conn.execute("""
            INSERT INTO episodeINFO(
                total_frames, total_distance, collision, survival_time,
                lane_change_count, average_speed, average_latency, total_reward
            ) VALUES (?,?,?,?,?,?,?,?)
        """, (
            total_frames, total_distance, collision, survival_time,
            lane_change_count, average_speed, average_latency, total_reward,
        ))
        conn.commit()
        conn.close()

    def export2json(self) -> str:
        ego = self.vehicles['ego']
        nearby = [
            v.export2json() for v in self.vehicles.values()
            if v.presence and v.id != 'ego' and abs(v.x - ego.x) < 50
        ]
        scenario = {
            'env_type': self.env_type,
            'lanes': [l.export2json() for l in self.lanes.values()],
            'ego': ego.export2json(),
            'nearby_vehicles': nearby,
        }
        return json.dumps(scenario)

    def commit_decision(
        self,
        thoughts: str,
        final_answer: str,
        parsed_action: str,
        format_failure: int = 0,
        hallucination: int = 0,
        consistency_failure: int = 0
    ):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """
        INSERT INTO decisionINFO
        VALUES (?,?,?,?,?,?,?,?);
        """,
        (
            self.frame,
            self.export2json(),
            thoughts,
            final_answer,
            parsed_action,
            format_failure,
            hallucination,
            consistency_failure
        ),
        )
        conn.commit()
        conn.close()