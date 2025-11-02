# Autonomous Vehicle Reinforcement Learning Agent

🚗 A state-based autonomous driving agent trained with PPO reinforcement learning in CARLA 0.9.14 simulator. The agent learns lane-following, speed control, and obstacle avoidance through reward signals.

## Overview

This project demonstrates autonomous driving using state-based reinforcement learning. The agent receives 8 state features (speed, steering, throttle, brake, lane offset, heading error, waypoint X/Y) and outputs 3 continuous actions (steering, throttle, brake). Trained for 100k steps using PPO with VecNormalize reward scaling.

### Results

✅ **Autonomous Lane-Following** - Drives 1000+ steps continuously
✅ **Consistent Speed Control** - Maintains 25-35 km/h safely
✅ **Smooth Steering** - Stays centered with curve navigation
✅ **Obstacle Avoidance** - No collisions during driving

## Quick Start

### 1. Download CARLA 0.9.14

Official release: https://github.com/carla-simulator/carla/releases/tag/0.9.14

Extract to `C:\CARLA\CARLA_0914` (Windows) or `/opt/carla` (Linux)

```bash
cd C:\CARLA\CARLA_0914
CarlaUE4.exe  # Verify launch
```


### 2. Install Python 3.7.9

Official download: https://www.python.org/downloads/release/python-379/

### 3. Setup Python Environment

```bash
conda create -n av_env python=3.7.9
conda activate av_env
pip install -r requirements_py37.txt
```


### 4. Configure CARLA Path

Edit `carla_env_state.py` line 10:

```python
CARLA_PATH = r'C:\CARLA\CARLA_0914' # or your actual path
```


## Running the Agent

### View Autonomous Driving Demo

```bash
# Terminal 1: Start CARLA
C:\CARLA\CARLA_0914\CarlaUE4.exe

# Terminal 2: Run demo
python demo_showcase.py
```


### Evaluate Performance

```bash
python check_behavior.py
```


### Train New Agent (100k steps, ~6-7 hours)

**Step 1: Create batch file**

Save as `train_headless.bat`:

```batch
@echo off
echo ========================================
echo CARLA 0.9.14 - HEADLESS TRAINING MODE
echo No Graphics = No VRAM Issues
echo ========================================

CarlaUE4.exe -RenderOffScreen

pause
```

**Step 2: Run training**

```bash
# Terminal 1: Run batch file
train_headless.bat

# Terminal 2: Start training (pause/resume with Ctrl+C)
python train_state_fast.py
```


## Architecture

**State Space:** 8 features (Speed, Steering, Throttle, Brake, Lane Offset, Heading Error, Waypoint X, Waypoint Y)

**Action Space:** 3 continuous values (Steering [-1,1], Throttle, Brake )[^11]

**Algorithm:** PPO with VecNormalize
**Training Steps:** 100,000
**GPU Required:** NVIDIA RTX 3050+ (6GB VRAM minimum)

## Project Structure

```
AV_Project/
├── carla_env_state.py              # Environment
├── train_state_fast.py             # Training script
├── check_behavior.py               # Evaluation
├── demo_showcase.py                # Demo
├── ppo_carla_state_fixed.zip       # Trained model
├── carla_vec_normalize.pkl         # Normalization stats
├── train_headless.bat              # Headless training batch
└── requirements_py37.txt           # Dependencies
```


## Requirements

- **GPU:** NVIDIA RTX 3050+ (6GB VRAM)
- **RAM:** 16GB
- **Python:** 3.7.9
- **Packages:** carla==0.9.14, stable-baselines3==1.8.0, torch==1.13.1, numpy, gym, opencv-python


## Real-World Application

This represents the **Planning Layer** of autonomous systems. In production:

- **Perception Layer:** Sensors → YOLOv8/LaneNet → State vector
- **Planning Layer:** State vector → PPO agent → Actions (this project)
- **Control Layer:** Actions → Actuators


## Official Resources

- CARLA Documentation: https://carla.readthedocs.io/en/0.9.14/
- CARLA GitHub: https://github.com/carla-simulator/carla
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- PPO Paper: https://arxiv.org/abs/1707.06347



