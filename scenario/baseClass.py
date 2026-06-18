from typing import List, Dict, Literal
from dataclasses import dataclass, field
from math import sqrt

@dataclass
class Lane:
    id: str
    laneIdx: int
    left_lanes: List[str] = field(default_factory=list)
    right_lanes: List[str] = field(default_factory=list)

    def export2json(self):
        return {
            'id': self.id,
            'lane index': self.laneIdx,
            'left_lanes': self.left_lanes,
            'right_lanes': self.right_lanes,
        }


@dataclass
class Vehicle:
    id: str
    lane_id: str = ''
    x: float = 0.0
    y: float = 0.0
    speedx: float = 0.0
    speedy: float = 0.0
    presence: bool = False
    # Nouveau : stocker l'index brut reçu de l'env
    _raw_lane_index: tuple = field(default=None, repr=False)

    def clear(self) -> None:
        self.lane_id = ''
        self.x = 0.0
        self.y = 0.0
        self.speedx = 0.0
        self.speedy = 0.0
        self.presence = False
        self._raw_lane_index = None

    def updateProperty(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        lane_index: tuple = None,      # ← nouveau paramètre
        env_type: str = "highway",     # ← nouveau paramètre
    ) -> None:
        self.x = x
        self.y = y
        self.speedx = vx
        self.speedy = vy
        self._raw_lane_index = lane_index

        if env_type == "highway":
            # Logique originale : lane déduite de y
            idx = max(0, min(3, round(y / 4.0)))
            self.lane_id = f'lane_{idx}'

        elif env_type == "roundabout":
            if lane_index is not None:
                # lane_index = (origin, destination, local_idx)
                # On utilise une représentation symbolique lisible
                origin, dest, local = lane_index
                self.lane_id = f'{origin}_{dest}_{local}'
            else:
                # Fallback : on garde l'ancienne logique
                idx = max(0, round(y / 4.0))
                self.lane_id = f'lane_{idx}'

    @property
    def lane_center(self) -> float:
        """Valide uniquement en highway (lanes rectilignes)."""
        try:
            lane_idx = int(self.lane_id.split('_')[1])
            return lane_idx * 4.0
        except (IndexError, ValueError):
            return self.y  # fallback roundabout

    @property
    def lane_offset(self) -> float:
        return abs(self.y - self.lane_center)

    @property
    def speed(self) -> float:
        return sqrt(self.speedx ** 2 + self.speedy ** 2)

    @property
    def lanePosition(self) -> float:
        """Position longitudinale dans la lane (= x en highway)."""
        return self.x

    def export2json(self) -> Dict:
        return {
            'id': self.id,
            'current lane': self.lane_id,
            'lane position': round(float(self.x), 2),
            'speed': round(float(self.speed), 2),
        }