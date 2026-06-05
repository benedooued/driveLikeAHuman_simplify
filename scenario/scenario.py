from scenario.baseClass import Lane, Vehicle
from typing import List, Dict
from datetime import datetime
import sqlite3
import json
import os


class Scenario:
    def __init__(self, vehicleCount: int, database: str = None) -> None:
        self.lanes: Dict[str, Lane] = {}
        self._build_roadgraph()
        self.vehicles: Dict[str, Vehicle] = {}
        self.vehicleCount = vehicleCount
        self._init_vehicles()
        self.frame = 0

        self.database = database or datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.db'
        if os.path.exists(self.database):
            os.remove(self.database)

        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS vehINFO(
            frame INT, id TEXT, x REAL, y REAL, lane_id TEXT, speedx REAL, speedy REAL,
            PRIMARY KEY (frame, id));""")
        cur.execute("""CREATE TABLE IF NOT EXISTS decisionINFO(
            frame INT PRIMARY KEY, scenario TEXT, thoughts TEXT, finalAnswer TEXT, parsedAction TEXT);""")
        conn.commit()
        conn.close()

    def _build_roadgraph(self):
        for i in range(4):
            lid = 'lane_' + str(i)
            self.lanes[lid] = Lane(
                id=lid, laneIdx=i,
                left_lanes=['lane_' + str(k) for k in range(0, i)],
                right_lanes=['lane_' + str(j) for j in range(i + 1, 4)]
            )

    def _init_vehicles(self):
        for i in range(self.vehicleCount):
            vid = 'ego' if i == 0 else f'veh{i}'
            self.vehicles[vid] = Vehicle(id=vid)

    def update_vehicles(self, observation: List[List], frame: int):
        self.frame = frame
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        for i, obs in enumerate(observation):
            vid = 'ego' if i == 0 else f'veh{i}'
            presence, x, y, vx, vy = obs
            if presence:
                veh = self.vehicles[vid]
                veh.presence = True
                veh.updateProperty(x, y, vx, vy)
                cur.execute('INSERT INTO vehINFO VALUES (?,?,?,?,?,?,?);',
                            (frame, vid, float(x), float(y), veh.lane_id, float(vx), float(vy)))
            else:
                self.vehicles[vid].clear()
        conn.commit()
        conn.close()

    def export2json(self) -> str:
        scenario = {
            'lanes': [l.export2json() for l in self.lanes.values()],
            'ego_info': self.vehicles['ego'].export2json(),
            'vehicles': [v.export2json() for v in self.vehicles.values() if v.presence],
        }
        return json.dumps(scenario)

    def commit_decision(self, thoughts: str, final_answer: str, parsed_action: str):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute("INSERT INTO decisionINFO VALUES (?,?,?,?,?);",
                    (self.frame, self.export2json(), thoughts, final_answer, parsed_action))
        conn.commit()
        conn.close()