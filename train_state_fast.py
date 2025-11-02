# train_state_fast.py - 100k steps with auto-resume and Ctrl+C save

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from carla_env_state import CarlaEnvState
import numpy as np
import os
import signal
import sys

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])
        
        if self.num_timesteps % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                print(f"\n[Step {self.num_timesteps}] Episodes: {len(self.episode_rewards)} | "
                      f"Avg Reward (last 100): {mean_reward:.1f} | "
                      f"Avg Length: {mean_length:.0f} steps")
        
        return True

model_global = None
env_global = None

def signal_handler(sig, frame):
    print('\n\n⚠️  Interrupted! Saving model...')
    if model_global is not None:
        model_global.save('ppo_carla_state_fixed.zip')
        print(f'✅ Model saved to: ppo_carla_state_fixed.zip')
    if env_global is not None:
        try:
            env_global.save('carla_vec_normalize.pkl')
            print(f'✅ Normalization stats saved')
        except:
            pass
        env_global.close()
    print('\nTo resume training, just run this script again!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("="*70)
print("TRAINING STATE-BASED PPO (100k STEPS MAX)")
print("="*70)

# CHECK IF MODEL ALREADY EXISTS (RESUME)
if os.path.exists('ppo_carla_state_fixed.zip'):
    print("\n✅ Found existing model! RESUMING from checkpoint...\n")
    resume = True
else:
    print("\n🆕 No existing model found. STARTING NEW TRAINING...\n")
    resume = False

env = DummyVecEnv([lambda: CarlaEnvState()])
env_global = env

# Normalize rewards
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=100.0)

# Load normalization stats if resuming
if resume and os.path.exists('carla_vec_normalize.pkl'):
    print("📊 Loading normalization stats...")
    try:
        env = VecNormalize.load('carla_vec_normalize.pkl', env)
        print("✅ Normalization stats loaded!\n")
    except Exception as e:
        print(f"⚠️  Could not load norm stats: {e}\n")

# LOAD or CREATE model
if resume:
    print("🔄 Loading model from checkpoint...")
    try:
        model = PPO.load('ppo_carla_state_fixed', env=env)
        print(f"✅ Model loaded! Current steps: {model.num_timesteps}")
        print("Resuming training from this point...\n")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        print("Starting fresh instead...\n")
        resume = False
        model = None

if not resume or model is None:
    model = PPO(
        'MlpPolicy', 
        env, 
        learning_rate=3e-4,
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1, 
        device='cuda'
    )
    print("✅ New model created!\n")

model_global = model

callback = ProgressCallback(check_freq=2048)

print("Press Ctrl+C anytime to pause and save!\n")
print("="*70)

try:
    # Calculate remaining steps to reach 100k total
    TARGET_STEPS = 100_000
    remaining_steps = TARGET_STEPS - model.num_timesteps
    
    if remaining_steps <= 0:
        print("\n🎉 Training already complete at 100,000 steps!")
        print("Model is ready to use!")
        print("Run: python check_behavior.py")
    else:
        print(f"Training {remaining_steps:,} more steps (total will be {TARGET_STEPS:,})...\n")
        
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=False  # IMPORTANT: Continue from where we left off
        )
        
        model.save("ppo_carla_state_fixed")
        env.save('carla_vec_normalize.pkl')
        print("\n✅ Training complete!")
        print("✅ Model: ppo_carla_state_fixed.zip")
        print("✅ Test it with: python check_behavior.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    try:
        model.save("ppo_carla_state_error")
    except:
        pass
    raise

finally:
    env.close()
