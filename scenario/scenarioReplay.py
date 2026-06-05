import sqlite3
import base64
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib import patches as mpatch


class ScenarioReplay:
    def __init__(self, database: str) -> None:
        self.database = database

    def get_vehicles(self, frame: int):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(f"SELECT id, x, y FROM vehINFO WHERE frame={frame};")
        rows = cur.fetchall()
        conn.close()
        return rows

    def plot_scene(self, frame: int) -> str:
        vehicles = self.get_vehicles(frame)
        vehicles.sort(key=lambda v: v[1])
        minx = vehicles[0][1] - 20
        maxx = vehicles[-1][1] + 20

        plt.figure(figsize=(22, 6))
        plt.axis('equal')
        for vid, x, y in vehicles:
            color = '#fd9644' if vid == 'ego' else '#0fb9b1'
            rect = mpatch.Rectangle((x - 2.5, y - 1), 5.0, 2.0, color=color)
            plt.annotate(vid, (x - 2.5, y + 1.5), fontsize=14, color='#fc5c65')
            plt.gca().add_patch(rect)

        for y, style in [(-2, 'solid'), (2, 'dashed'), (6, 'dashed'), (10, 'dashed'), (14, 'solid')]:
            plt.plot([minx, maxx], [y, y], linestyle=style, color='#778ca3')

        plt.gca().invert_yaxis()
        buf = BytesIO()
        plt.savefig(buf, bbox_inches='tight')
        plt.close()
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()