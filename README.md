# 🛰️ Satellite-Inertia-ID

A lightweight framework for estimating spacecraft inertia properties through **controlled torque excitation** and **state-based estimation** using Least Squares (LS) and Extended Kalman Filtering (EKF).

This repository accompanies the paper:

> **Towards Active Excitation-Based Inertia Identification in Satellites**  
> *M. El Hariry et al., 2025*  

---

## 🌌 Overview

This package provides the code and configurations used to study how **excitation design** affects spacecraft inertia identification accuracy.  
It models three satellites (CubeSat, MicroSat, SmallSat) under **different excitation profiles** and **dynamic inertia conditions**, allowing reproduction of all simulation results from the paper.

### Main Features
- 🚀 Nonlinear 3‑DoF rotational dynamics with reaction‑wheel coupling  
- ⚙️ Configurable **torque excitation profiles** (chirp, multi‑sine, PRBS, step, etc.)  
- 📈 Inertia estimation via **Least Squares** and **Extended Kalman Filter**  
- 🔊 Realistic sensor noise and actuation limits  
- 🧩 Easy configuration via YAML files  
- 🧪 Reproducible experiments for static and dynamic inertia scenarios  

---

## 📁 Project Structure


```
satellite-inertia-id/
├── sim/ # Core simulation
│ ├── dynamics.py # Attitude propagation & Euler equations
│ ├── actuators.py # Reaction-wheel model
│ └── sensors.py # Gyro & noise models
│
├── control/ # Excitation design
│ └── torque_generators.py # Chirp, PRBS, multi-sine, etc.
│
├── estimation/ # Estimation algorithms
│ ├── ls_estimator.py # Batch Least Squares
│ └── ekf.py # Extended Kalman Filter
│
├── scripts/ # Experiment runners
│ ├── run_ls_simulation.py # Static LS estimation
│ └── visualize.py # Plotting and post-processing
│
├── utils/ # Support functions
│ ├── metrics_analysis.py # RMSE & error statistics
│ └── signal_processing.py # Filtering utilities
│
├── config_sat1.yaml # CubeSat configuration
├── config_sat2.yaml # MicroSat configuration
├── config_sat3.yaml # SmallSat configuration
├── requirements.txt
└── README.md
```


---

## ⚙️ Installation

```bash
git clone https://github.com/<user>/satellite-inertia-id.git
cd satellite-inertia-id
python3 -m venv venv
source venv/bin/activate        # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

```

## 🔬 Running Experiments

### 1. Iterative LS-EKF Pipeline (Recommended)

The main estimation pipeline combines Least Squares and Extended Kalman Filtering iteratively:

```bash
python scripts/run_ls_simulation.py
```

**What it does:**
- **Iteration 1:** LS estimates inertia from noisy measurements → EKF filters angular velocity
- **Iteration 2:** LS re-estimates inertia using filtered ω → EKF improves ω estimates
- **Iteration 3:** Final refinement until convergence

**Features:**
- 🔄 Automatic convergence detection
- 📈 Performance comparison with single-pass methods
- 📊 Detailed iteration tracking and visualization
- ⚙️ Configurable torque profiles and disturbances



## ⚙️ Configuration

### Key Parameters in `config.yaml`:

```yaml
satellite:
  inertia_tensor: [0.3, 0.3, 0.5]  # True inertia [kg⋅m²]
  
sim:
  dt: 0.1                          # Time step [s]
  t_max: 100.0                     # Simulation duration [s]
  
estimation:
  max_iterations: 3                # LS-EKF iterations
  convergence_threshold: 1e-3      # Convergence criteria
  
sensors:
  gyro_noise_std: 0.01            # Gyroscope noise [rad/s]
  bias_stability: 1e-6            # Bias drift [rad/s]
```


## 🔧 Extending the Framework

### Adding New Estimation Methods:
1. Implement in `estimation/` following the base interface
2. Add to comparison pipeline in `run_estimation_loop.py`
3. Update visualization in `scripts/visualize.py`

### Custom Torque Profiles:
1. Add profile function to `control/torque_generators.py`
2. Register in profile selection logic
3. Configure parameters in YAML



## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Implement your changes
4. Update documentation (if needed)
5. Submit pull request
