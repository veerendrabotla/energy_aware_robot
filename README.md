# 🤖 Energy-Aware Robot Navigation System

An intelligent robot navigation platform that uses **Artificial Neural Networks (ANN)** to learn energy-optimal pathfinding by imitating the **A\* search algorithm** across 2D obstacle grids and 3D terrain topographies.

## 🎯 Project Objective

> **Minimize total energy consumption** — not just path length — by training a neural network to suppress unnecessary turns and avoid steep terrain slopes.

The system explicitly considers three energy cost factors:
- **Forward Movement:** 1.0 energy unit per cell
- **Rotation (Turn):** 5.0 energy units per 90° turn
- **Slope Climbing:** 10.0 × height difference (slopes > 0.5 are impassable walls)

## 🏗️ Architecture

```
energy_aware_robot/
├── main.py                          # Entry point (launches Streamlit)
├── config.py                        # Global constants (costs, directions)
├── environment/
│   ├── map_generator.py             # 2D obstacle grid generation
│   └── terrain_generator.py         # 3D height map generation (flat/hills/steep/mixed)
├── planning/
│   └── astar.py                     # Energy-weighted A* search algorithm
├── dataset/
│   └── dataset_generator.py         # Feature extraction & labeled dataset builder
├── model/
│   └── train_ann.py                 # MLPClassifier training & persistence
├── navigation/
│   └── run_navigation.py            # ANN inference loop with heading hysteresis
├── evaluation/
│   └── metrics.py                   # Energy, path length & turn calculations
├── plots/
│   ├── plot_2d.py                   # Matplotlib 2D grid with terrain underlay
│   └── plot_3d.py                   # Plotly 3D interactive terrain surface
└── ui/
    ├── app.py                       # Main Streamlit dashboard (all-in-one)
    ├── session_manager.py           # Centralized state management
    └── pages/
        ├── dashboard.py             # Landing page with project overview
        ├── map_setup.py             # Environment configuration & preview
        ├── training.py              # Dataset generation & ANN training
        ├── simulation.py            # Live path execution & timeline scrubber
        ├── analytics.py             # Neural network diagnostics & heatmap
        └── comparison.py            # A* vs ANN metrics & academic conclusion
```

## 🧠 Neural Network Details

| Property | Value |
|---|---|
| **Model** | Scikit-Learn `MLPClassifier` |
| **Topology** | `(64, 64, 32)` — 3 hidden layers |
| **Activation** | ReLU |
| **Solver** | Adam with adaptive learning rate |
| **Max Epochs** | 2,000 |
| **Validation** | 15% train-test split |
| **Heading Hysteresis** | `+0.25` probability bias toward Forward to suppress zig-zagging |

### Feature Sets
- **Phase 1 (5 features):** Distance to Goal, Relative Angle, Obstacle Forward/Left/Right
- **Phase 2 (8 features):** Distance, Angle, Slope Forward/Left/Right, Obstacle Forward/Left/Right

## 🏔️ Terrain Types

| Type | Description | Elevation Range |
|---|---|---|
| **Flat** | Zero elevation everywhere | 0 m |
| **Hills** | Gentle, smooth rolling slopes (σ=3.5 Gaussian blur) | 0–6 m |
| **Steep** | Sharp, jagged spikes (σ=0.8 Gaussian blur) | 0–15 m |
| **Mixed** | Hills base layer + steep spike overlay | 0–15 m |

## 🚀 Setup & Run

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
cd energy_aware_robot
pip install -r requirements.txt
```

### Launch
```bash
python main.py
```
The application opens at `http://localhost:8501`.

### Workflow
1. **Map Setup** → Configure grid size, density, terrain → Generate Map
2. **Training** → Build dataset → Train MLP Neural Network
3. **Simulation** → Run A* → Run ANN → Use Timeline Scrubber
4. **Analytics** → Inspect model architecture & loss curve
5. **Comparison** → Review energy/length/turns charts & academic conclusion

## 📊 Key Features

- **3D Interactive Terrain** — Plotly surface with white background, fixed Z-axis (0–15m), Earth colorscale, diamond Start/Goal markers
- **2D Terrain Underlay** — Matplotlib grid overlays terrain elevation heatmap behind obstacles
- **Toggle 3D Preview** — Checkbox to show/hide 3D terrain even for flat maps
- **Phase Lock** — Flat terrain auto-disables Phase 2 (3D slopes are meaningless on flat)
- **Heading Hysteresis** — ANN biases toward going straight, drastically reducing turn energy costs
- **Academic Conclusion Panel** — Auto-generates mathematical proof explaining energy vs. distance tradeoffs
- **Individual Comparison Charts** — Separate bar charts for Path Length, Energy, and Turns
- **Neural Network Topology Display** — Shows Input Nodes, Hidden Layers, Output Nodes explicitly
- **Dataset Export** — Download training data as CSV
- **Timeline Scrubber** — Step through paths frame-by-frame with live sensor HUD

## 📦 Dependencies

```
streamlit
numpy
pandas
matplotlib
plotly
scikit-learn
scipy
```

## 👥 Team

Energy-Aware Robot Navigation — Academic Research Project, March 2026
