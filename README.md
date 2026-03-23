# 🤖 Energy-Aware Robot Navigation System

Welcome to the **Energy-Aware Robot Navigation System**! This project is an advanced simulation and machine-learning framework designed to teach an Artificial Neural Network (ANN) how to autonomously navigate an environment by learning from a heuristic teacher algorithm (A*). The focus is specifically structured around "Energy-Awareness"—costing operations differently based on distances, rotations, and steep 3D terrain slopes.

---

## 🌟 Key Features

1. **Dual-Algorithm Navigation**: Compare an optimal heuristic baseline (A* Cost-Search) against an unpredictable, fast student (Multilayer Perceptron Neural Network).
2. **Dynamic 2D/3D Environments**: 
   - **Phase 1**: Flat 2D grid environments with configurable obstacle density.
   - **Phase 2**: Complex 3D topographies (hills, steep cliffs) where elevation changes drastically affect the energy required to traverse.
3. **Automated ML Dataset Generation**: Scrape thousands of perfect algorithmic path trajectories in seconds to build massive behavior-cloning datasets.
4. **Centralized UI Dashboard**: Explore, simulate, compare, and modify simulations live through an interactive Streamlit multipage application.
5. **Timeline Scrubbing HUD**: Easily pause, step forward, backward, and analyze sensor features during runtime using the isolated Simulation Scrubber.

---

## 📂 Project Structure

- `ui/` - Contains the Streamlit frontend.
  - `pages/` - Isolated interactive modules (`map_setup.py`, `training.py`, `simulation.py`, `comparison.py`, `dashboard.py`).
  - `session_manager.py` - Core single source of truth for UI state.
  - `app.py` - Primary application entry and overview file.
- `navigation/` & `planning/` - Core A* heuristic logic and neural network agent runner blocks.
- `dataset/` - Generator functions capable of extracting and packing feature sets (distances, raycast angles, slopes).
- `model/` - The ANN (Multilayer Perceptron) initialization and validation code (`train_test_split`).
- `plots/` - Matplotlib and Plotly visual rendering tools.
- `main.py` - Easy launch wrapper.

---

## ⚡ Energy Metrics & Cost Function

The environment utilizes a highly specific cost calculation:
- **Forward translation**: `1.0` units of energy per grid cell.
- **Rotation (90° turn)**: `5.0` units of energy computationally (encourages straight paths).
- **Incline Climbing**: `10.0` units per height increment over the slope terrain.
- **Max Slope Rules**: Slopes exceeding `0.5` delta height act as absolute walls/obstacles.

---

## 🚀 Setup & Installation

### Requirements
Ensure you are using **Python 3.8+**. 
This project uses several standard visual and data science distributions (NumPy, Pandas, Scikit-Learn, Streamlit, Matplotlib, Plotly). See `requirements.txt` for exact constraints.

```bash
# 1. Clone the repository and CD into it
cd energy_aware_robot

# 2. Build and activate the virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Launching the Application

Start the centralized dashboard module right away through the Streamlit wrapper:

```bash
python main.py
```

The server will initialize on `http://localhost:8501`.

---

## 🛠️ Usage Workflow

1. **Map Setup:** Open the sidebar, go to Map Setup, and generate a new topographic grid and goal targets.
2. **Model Training:** Go to the Training Center. Pick Phase 1 (2D) or Phase 2 (3D). Generate a massive internal dataset representing A* solutions, then train building the new model logic!
3. **Simulation Engine:** Hop to Simulation, load your target map, and run the Teacher (A*) and Student (ANN) successively. Use the slider to step-by-step scrub their decision making processes.
4. **Comparison Metrics:** Check the Head-to-Head module to mathematically contrast total energy burned, path duration, and overall efficiency between the two!
