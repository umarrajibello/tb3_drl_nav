# tb3_drl_nav

# 🧭 TurtleBot3 Hybrid Navigation (RRT‑Guided Deep Reinforcement Learning)

This repository implements a hybrid navigation system for TurtleBot3 in PyBullet, combining classical **RRT waypoint planning** with modern **deep reinforcement learning**. The system supports both **hybrid navigation** (RRT + RL) and **end‑to‑end RL**, using LiDAR‑based perception and temporal feature extractors such as **CNN‑GRU**, **LSTM**, and **Transformer encoders**.

The training pipeline is built on **Stable‑Baselines3**, with support for **SAC**, **PPO**, and custom temporal extractors.

---

## 🚀 Key Features

- **Hybrid navigation** using RRT for global planning and RL for local obstacle avoidance.
- **End‑to‑end RL** using LiDAR only.
- **Temporal observation stacking** via **TemporalObservationWrapper**.
- **Three temporal feature extractors**:
  - CNN‑GRU (recommended)
  - LSTM
  - Transformer
- **SB3‑compatible custom policies** for seamless training.
- **VecNormalize** for stable learning.
- **Evaluation metrics** including success rate, collisions, path efficiency, jerk, clearance, and stuck events.
- **Model registry** for experiment tracking.

---

## 📦 Installation

Clone the repository inside your ROS workspace:

```
cd ~/tb3_drl_ws/src
git clone https://github.com/<your-username>/tb3_drl_nav.git
```

Install dependencies:

```
pip install -r requirements.txt
```

Dependencies include:

- PyBullet
- Stable‑Baselines3
- Torch
- NumPy
- Gym / Gymnasium

---

## 🗂 Project Structure

```
tb3_drl_ws/src/tb3_drl_nav/
│
├── train.py
├── eval.py
├── run_eval.sh
├── model_registry.json
├── README.md
├── config.py
│
├── env/
│   ├── tb3_env.py
│   ├── tb3_env_hybrid.py
│   ├── observation_wrapper.py
│   ├── lidar.py
│   └── __init__.py
│
├── policies/
│   ├── cnn_gru_extractor.py
│   ├── lstm_extractor.py
│   ├── transformer_extractor.py
│   └── __init__.py
│
├── models/
└── logs/
```

---

## 🏋️ Training Experiments

Training is controlled by:

```
python3 train.py \
    --algorithm <algo> \
    --env_type <static|mixed|dense> \
    [--hybrid] \
    --total_timesteps <N> \
    --gpu
```

### Supported Algorithms

| Algorithm | Hybrid | End‑to‑End | Temporal Model |
|----------|--------|------------|----------------|
| `cnn-gru-sac` | ✔️ | ✔️ | CNN‑GRU |
| `lstm-sac` | ✔️ | ✔️ | LSTM |
| `tx-sac` | ✔️ | ✔️ | Transformer |
| `sac` | ✔️ | ✔️ | MLP |
| `ppo` | ✔️ | ✔️ | MLP |

---

## 🧪 Training Examples

### Hybrid CNN‑GRU SAC (recommended)

```
python3 train.py \
    --algorithm cnn-gru-sac \
    --hybrid \
    --env_type mixed \
    --total_timesteps 500000 \
    --gpu
```

### End‑to‑end CNN‑GRU SAC

```
python3 train.py \
    --algorithm cnn-gru-sac \
    --env_type mixed \
    --total_timesteps 500000 \
    --gpu
```

### Hybrid LSTM SAC

```
python3 train.py \
    --algorithm lstm-sac \
    --hybrid \
    --env_type dense
```

### Hybrid Transformer SAC

```
python3 train.py \
    --algorithm tx-sac \
    --hybrid \
    --env_type static
```

### Vanilla SAC (MLP)

```
python3 train.py --algorithm sac --env_type mixed
```

### PPO Baseline

```
python3 train.py --algorithm ppo --env_type mixed
```

---

## 📊 Evaluation

Use the helper script:

```
./run_eval.sh <model.zip> <algorithm> <hybrid|end2end> <env_type>
```

Example:

```
./run_eval.sh models/mixed_cnn-gru-sac_hybrid/model.zip cnn-gru-sac hybrid mixed
```

Or run manually:

```
python3 eval.py \
    --model_path models/.../model.zip \
    --algorithm cnn-gru-sac \
    --env_type mixed \
    --hybrid \
    --num_episodes 50 \
    --gui
```

Evaluation outputs:

- `eval_results.csv`
- printed summary statistics

---

## 📁 Model Registry

`model_registry.json` tracks experiments:

```json
{
  "experiments": [
    {
      "name": "mixed_cnn-gru-sac_hybrid_seed42",
      "algorithm": "cnn-gru-sac",
      "env_type": "mixed",
      "hybrid": true,
      "timestamp": "2026-04-12T23:15:00",
      "model_path": "models/mixed_cnn-gru-sac_hybrid_seed42/model.zip",
      "vecnormalize_path": "models/mixed_cnn-gru-sac_hybrid_seed42/vecnormalize.pkl",
      "notes": "CNN-GRU hybrid navigation with RRT + RL"
    }
  ]
}
```

---

## 📈 TensorBoard

```
tensorboard --logdir logs
```

---

## 📚 Citations

### Stable‑Baselines3
```
@misc{raffin2021stable,
  title={Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  author={Raffin, Antonin and Hill, Ashley and Ernestus, Adam and Gleave, Adam and Kanervisto, Anssi and Dormann, Noah},
  year={2021},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={268},
  pages={1--8}
}
```

### PyBullet
```
@misc{coumans2016pybullet,
  title={PyBullet, a Python module for physics simulation},
  author={Coumans, Erwin and Bai, Yunfei},
  year={2016},
  howpublished={http://pybullet.org}
}
```

### TurtleBot3
```
@inproceedings{turtlebot3,
  title={TurtleBot3: An Open-Source Modular Robot Platform},
  author={Kim, Sungeun and Kim, Hyun Myung},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2017}
}
```

### Transformer Encoder
```
@article{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and others},
  journal={Advances in Neural Information Processing Systems},
  year={2017}
}
```

---

