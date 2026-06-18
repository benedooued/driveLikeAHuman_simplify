DriverAgent

DriverAgent is an autonomous driving agent powered by a Large Language Model (LLM). The agent observes the driving environment, reasons about the current situation, and selects driving actions such as lane changes, acceleration, or braking.

Features
LLM-based decision making
Highway driving simulation
Decision logging in SQLite
Replay and visualization tools
Scenario analysis frame by frame
Generated simulation videos
Project Structure
.
├── main.py                 # Runs the driving simulation
├── app.py                  # Replay web interface
├── scenario/
│   └── scenarioReplay.py
├── results-db/
│   └── highway.db
├── templates/
│   └── index.html
└── outputs/
    └── simulation_video.mp4
Requirements
pip install -r requirements.txt

Main dependencies:

Python 3.10+
Flask
SQLite
Highway-env
Gymnasium
LangChain
Groq API (if used)
Running a Simulation

Launch the driving agent:

python main.py

The simulation will:

Run the driving environment.
Query the LLM for decisions.
Store decisions in the SQLite database.
Generate a simulation video.

Output database:

results-db/highway.db
Replay and Decision Analysis

After the simulation finishes, launch the replay interface:

python app.py

Open:

http://127.0.0.1:5000

The interface allows you to inspect each frame individually and visualize:

Environment state
Scenario description
LLM reasoning
Raw model response
Parsed driving action
Database Content

The decisionINFO table stores:

frame
scenario
thoughts
finalAnswer
parsedAction

This information is used by the replay interface to reconstruct the decision process.

Workflow
main.py
   ↓
Simulation
   ↓
highway.db
   ↓
app.py
   ↓
Interactive Replay Viewer
Use Cases
Autonomous driving research
LLM decision analysis
Explainable AI experiments
Reinforcement learning evaluation
Driving policy debugging
License

For academic and research purposes.