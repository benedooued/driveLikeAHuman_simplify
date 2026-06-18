import sqlite3
import base64
import json
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib import patches as mpatch
import numpy as np

#Allows to replay a scenario from the database, either highway or roundabout, and visualize it using matplotlib.
class ScenarioReplay:
    def __init__(self, database: str) -> None:
        self.database = database
        self.env_type = self._detect_env_type()

    # ------------------------------------------------------------------
    # Détection automatique highway vs roundabout
    # ------------------------------------------------------------------

    def _detect_env_type(self) -> str:
        """Lit env_type depuis le premier scénario enregistré en DB."""
        try:
            conn = sqlite3.connect(self.database)
            cur  = conn.cursor()
            cur.execute("SELECT scenario FROM decisionINFO LIMIT 1;")
            row = cur.fetchone()
            conn.close()
            if row:
                data = json.loads(row[0])
                return data.get('env_type', 'highway')
        except Exception:
            pass
        return 'highway'   # fallback safe

    # ------------------------------------------------------------------
    # Lecture véhicules
    # ------------------------------------------------------------------

    def get_vehicles(self, frame: int):
        conn = sqlite3.connect(self.database)
        cur  = conn.cursor()
        cur.execute("SELECT id, x, y FROM vehINFO WHERE frame=?;", (frame,))
        rows = cur.fetchall()
        conn.close()
        return rows

    # ------------------------------------------------------------------
    # Dispatcher : choisit le bon renderer selon env_type
    # ------------------------------------------------------------------

    def plot_scene(self, frame: int) -> str:
        if self.env_type == 'roundabout':
            return self._plot_roundabout(frame)
        return self._plot_highway(frame)

    # ------------------------------------------------------------------
    # Renderer highway (logique originale, inchangée)
    # ------------------------------------------------------------------

    def _plot_highway(self, frame: int) -> str:
        vehicles = self.get_vehicles(frame)
        if not vehicles:
            return self._empty_plot("No data for this frame")

        vehicles.sort(key=lambda v: v[1])
        minx = vehicles[0][1]  - 20
        maxx = vehicles[-1][1] + 20

        plt.figure(figsize=(22, 6))
        plt.axis('equal')

        # Lignes de voie
        for lane_y, style in [(-2, 'solid'), (2, 'dashed'), (6, 'dashed'),
                               (10, 'dashed'), (14, 'solid')]:
            plt.plot([minx, maxx], [lane_y, lane_y],
                     linestyle=style, color='#778ca3', linewidth=1.2)

        # Véhicules
        for vid, x, y in vehicles:
            color = '#fd9644' if vid == 'ego' else '#0fb9b1'
            rect  = mpatch.Rectangle((x - 2.5, y - 1), 5.0, 2.0, color=color, zorder=3)
            plt.gca().add_patch(rect)
            plt.annotate(vid, (x - 2.5, y + 1.5), fontsize=11, color='#fc5c65', zorder=4)

        plt.gca().invert_yaxis()
        plt.title(f"Highway — frame {frame}", fontsize=13)
        plt.axis('off')
        return self._fig_to_base64()

    # ------------------------------------------------------------------
    # Renderer roundabout
    # ------------------------------------------------------------------

    def _plot_roundabout(self, frame: int) -> str:
        vehicles = self.get_vehicles(frame)
        if not vehicles:
            return self._empty_plot("No data for this frame")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')

        # ── Roundabout geometry (highway-env roundabout-v0 hardcodé) ──
        # Centre du rond-point : (0, 0) dans le repère de l'env
        CENTER    = (0, 0)
        # Rayons intérieur / extérieur des deux anneaux circulaires
        RINGS = [
            (14, 18),   # anneau intérieur
            (18, 22),   # anneau extérieur
        ]
        ROAD_COLOR  = '#4a4a4a'
        LANE_COLOR  = '#778ca3'
        BG_COLOR    = '#2d2d2d'

        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Fond du rond-point (disque plein)
        outer_disk = plt.Circle(CENTER, RINGS[-1][1] + 2,
                                color=ROAD_COLOR, zorder=1)
        ax.add_patch(outer_disk)

        # Îlot central (blanc)
        inner_island = plt.Circle(CENTER, RINGS[0][0],
                                  color='#e8e8e8', zorder=2)
        ax.add_patch(inner_island)

        # Lignes de séparation des anneaux
        for r_in, r_out in RINGS:
            for r in [r_in, r_out]:
                circle = plt.Circle(CENTER, r, fill=False,
                                    edgecolor=LANE_COLOR,
                                    linewidth=1.2, linestyle='dashed', zorder=3)
                ax.add_patch(circle)

        # Branches d'entrée/sortie (4 branches à N/S/E/W)
        BRANCH_WIDTH = 8
        BRANCH_LEN   = 15
        for angle_deg in [0, 90, 180, 270]:
            angle = np.radians(angle_deg)
            x_start = CENTER[0] + np.cos(angle) * RINGS[-1][1]
            y_start = CENTER[1] + np.sin(angle) * RINGS[-1][1]
            x_end   = CENTER[0] + np.cos(angle) * (RINGS[-1][1] + BRANCH_LEN)
            y_end   = CENTER[1] + np.sin(angle) * (RINGS[-1][1] + BRANCH_LEN)
            # Dessine la branche comme un rectangle orienté
            perp = np.array([-np.sin(angle), np.cos(angle)]) * BRANCH_WIDTH / 2
            pts  = np.array([
                [x_start - perp[0], y_start - perp[1]],
                [x_start + perp[0], y_start + perp[1]],
                [x_end   + perp[0], y_end   + perp[1]],
                [x_end   - perp[0], y_end   - perp[1]],
            ])
            branch = mpatch.Polygon(pts, closed=True,
                                    color=ROAD_COLOR, zorder=1)
            ax.add_patch(branch)

        # ── Véhicules ──────────────────────────────────────────────────
        for vid, x, y in vehicles:
            color  = '#fd9644' if vid == 'ego' else '#0fb9b1'
            circle = plt.Circle((x, y), 2.0, color=color, zorder=5)
            ax.add_patch(circle)
            ax.annotate(vid, (x, y + 2.5),
                        fontsize=9, color='white',
                        ha='center', zorder=6)

        # Cadrage automatique sur les véhicules avec marge
        xs = [v[1] for v in vehicles]
        ys = [v[2] for v in vehicles]
        margin = 30
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        ax.set_title(f"Roundabout — frame {frame}",
                     fontsize=13, color='white')
        ax.axis('off')
        return self._fig_to_base64()

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _fig_to_base64(self) -> str:
        buf = BytesIO()
        plt.savefig(buf, bbox_inches='tight')
        plt.close()
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    def _empty_plot(self, msg: str) -> str:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14)
        plt.axis('off')
        return self._fig_to_base64()