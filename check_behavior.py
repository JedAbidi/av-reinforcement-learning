# check_behavior.py - See what the agent learned

from stable_baselines3 import PPO
from carla_env_state import CarlaEnvState
import numpy as np
import time

model = PPO.load('ppo_carla_state_fixed.zip')
env = CarlaEnvState()

print("\n" + "="*70)
print("CHECKING AGENT BEHAVIOR")
print("="*70)

for ep in range(3):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    speeds = []
    distances = []
    actions_taken = []
    
    last_loc = env.vehicle.get_location()
    
    while not done and step < 2000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        # Track behavior
        speeds.append(info['speed_kmh'])
        
        curr_loc = env.vehicle.get_location()
        dist_moved = curr_loc.distance(last_loc)
        distances.append(dist_moved)
        last_loc = curr_loc
        
        actions_taken.append(action)
        
        if step % 100 == 0:
            print(f"  Step {step:>4} | Speed: {info['speed_kmh']:>5.1f} km/h | "
                  f"Moved: {dist_moved:>5.2f}m | Reward: {reward:>7.1f}")
    
    print(f"\n[Episode {ep+1}]")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Steps: {step}")
    print(f"  Avg speed: {np.mean(speeds):.1f} km/h")
    print(f"  Max speed: {np.max(speeds):.1f} km/h")
    print(f"  Avg distance per step: {np.mean(distances):.3f}m")
    print(f"  Total distance: {np.sum(distances):.1f}m")
    
    # Check for suspicious patterns
    if np.mean(speeds) < 2.0:
        print("  ⚠️  PROBLEM: Car is barely moving!")
    if np.std(speeds) > 15:
        print("  ⚠️  PROBLEM: Speed wildly varying (might be oscillating)")
    if np.mean(distances) < 0.05:
        print("  ⚠️  PROBLEM: Not moving forward consistently")
    
    print()

env.close()
