"""
Replay viewer — browse decisions frame by frame.
Run: python app.py
"""

import sqlite3
from flask import Flask, jsonify, render_template, request
from scenario.scenarioReplay import ScenarioReplay

DATABASE = './results-db/highway.db'
sr = ScenarioReplay(DATABASE)
app = Flask(__name__)


@app.route("/")
def index():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT min(frame), max(frame) FROM decisionINFO;")
    min_f, max_f = cur.fetchone()
    conn.close()
    return render_template('index.html', minFrame=min_f, maxFrame=max_f)


@app.route("/frame", methods=['POST'])
def get_frame():
    frame = int(request.form['frame'])
    img = sr.plot_scene(frame)
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute(
        "SELECT scenario, thoughts, finalAnswer, parsedAction FROM decisionINFO WHERE frame=?;",
        (frame,)
    )
    row = cur.fetchone()
    conn.close()
    scenario, thoughts, final_answer, parsed = row
    return jsonify(img=img, scenario=scenario, thoughts=thoughts,
                   finalAnswer=final_answer, parsedAction=parsed)


if __name__ == '__main__':
    app.run(debug=True)