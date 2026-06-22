# DriveLikeAHuman-Simplify

### LLM-Based Autonomous Driving with Reasoning, Tool Use, and Explainable Decision Making

DriveLikeAHuman-Simplify is a research-oriented autonomous driving framework that integrates Large Language Models (LLMs) into the decision-making layer of an autonomous vehicle.

The agent observes the environment, reasons about the current traffic situation using a ReAct-style framework, interacts with dedicated driving tools, and generates structured driving actions such as lane changes, acceleration, or deceleration.

The project is designed to investigate the capabilities, interpretability, and limitations of Foundation Models in autonomous driving.

---

## Highlights

* Large Language Model based driving agent
* ReAct reasoning framework
* Tool-augmented decision making
* Explainable driving behavior
* Scenario replay and visualization
* SQLite-based experiment logging
* Automated metrics extraction
* Support for multiple LLM providers
* Foundation Model evaluation for autonomous driving research

---

## System Architecture

Environment Observation
│
▼
Scenario Representation
│
▼
Prompt Construction
│
▼
LLM Agent (ReAct)
│
┌──────┼──────┐
▼      ▼      ▼
Tools  Tools  Tools
│
▼
Driving Decision
│
▼
Simulation Execution
│
▼
Database Logging
│
▼
Replay & Analysis

---

## Available Driving Actions

| Action     | Description                     |
| ---------- | ------------------------------- |
| LANE_LEFT  | Change lane to the left         |
| IDLE       | Maintain current speed and lane |
| LANE_RIGHT | Change lane to the right        |
| FASTER     | Accelerate                      |
| SLOWER     | Decelerate                      |

---

## Features

### Reasoning

The agent uses a ReAct-style reasoning process:

* Observe traffic situation
* Query driving tools
* Evaluate safety constraints
* Generate structured action

### Tool Calling

Available tools include:

* Get Available Actions
* Get Available Lanes
* Get Lane Vehicles
* Check Safety With Vehicle
* Check Safety For Action

### Explainability

Each decision stores:

* Scenario description
* Full reasoning trace
* Raw LLM response
* Final parsed action

allowing complete post-hoc analysis.

---

## Project Structure

.
├── main.py
├── app.py
├── scenario/
├── database/
├── outputs/
├── prompts/
├── templates/
├── metrics/
├── reports/
├── requirements.txt
└── README.md

---

## Installation

Clone the repository:

```bash
git clone https://github.com/benedooued/driveLikeAHuman_simplify.git

cd driveLikeAHuman_simplify
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

This project requires valid API credentials for supported LLM providers.

Create a local configuration file:

```bash
cp config.example.yaml config.yaml
```

Then edit:

```yaml
llm:
  provider: groq

api:
  groq_api_key: YOUR_API_KEY
```

The configuration file is excluded from version control and should never be committed.

---

## Running a Simulation

Launch the driving simulation:

```bash
python main.py
```

The simulation will:

1. Initialize the environment
2. Build scenario descriptions
3. Query the LLM
4. Execute driving actions
5. Log all decisions
6. Store experiment data

---

## Replay and Visualization

After a simulation completes:

```bash
python app.py
```

Open:

http://127.0.0.1:5000

The replay interface allows frame-by-frame inspection of:

* Traffic state
* Scenario description
* LLM reasoning
* Tool interactions
* Raw model outputs
* Final driving actions

---

## Logged Information

The framework stores:

### Vehicle Information

* Position
* Speed
* Lane assignment

### Decision Information

* Scenario description
* Thoughts
* Final answer
* Parsed action

### Driving Metrics

* Reward
* Collision status
* Lane offset
* Minimum vehicle distance
* Decision latency
* Token usage

### Episode Statistics

* Total distance
* Survival time
* Average speed
* Total reward
* Collision outcome

---

## Evaluation Metrics

### Safety

* Collision Rate
* Near-Miss Rate
* Hard-Braking Rate

### Driving Control

* Lane Offset
* Acceleration Stability

### Reasoning Quality

* Action Validity Rate
* Hallucination Rate
* Thought-Action Consistency

### Performance

* Distance Travelled
* Survival Time
* Total Reward

### Human-Likeness

* Decision Latency
* Rule Violation Rate

---

## Research Applications

This project can be used for:

* Autonomous Driving Research
* Foundation Model Evaluation
* Explainable AI Studies
* Human-like Driving Analysis
* LLM Reasoning Evaluation
* Safety-Critical Decision Making
* Autonomous Vehicle Benchmarking

---

## Citation

If you use this project in your research, please consider citing:

```bibtex
@software{ouedraogo2026drivelikehuman,
  title={DriveLikeAHuman-Simplify},
  author={Ariel Ouedraogo},
  year={2026}
}
```

---

## License

This project is released for academic and research purposes.
