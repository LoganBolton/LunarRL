import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env_wrapper import RewardWrapper
from gymnasium.wrappers import RecordVideo
import os

# Create videos directory if it doesn't exist
os.makedirs("./videos", exist_ok=True)

# Load your trained model
print("Loading trained PPO model...")
model = PPO.load("ppo_lunarlander")

# Create environment with video recording
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RewardWrapper(env)
env = RecordVideo(env, video_folder="./videos/", episode_trigger=lambda x: True, name_prefix="lunarlander_test")

print("Starting test episode with video recording...")

# Run one episode
obs, _ = env.reset()
done = False
total_reward = 0.0
step_count = 0

print("\n--- Episode Progress ---")

while not done:
    action, _ = model.predict(obs, deterministic=True)  # Use deterministic for consistent results
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += float(reward)
    step_count += 1
    
    # Print reward breakdown every 20 steps
    if step_count % 20 == 0 and 'reward_breakdown' in info:
        print(f"Step {step_count}: Total Reward So Far = {total_reward:.2f}")
        for component, value in info['reward_breakdown'].items():
            if abs(value) > 0.01:  # Only show significant components
                print(f"  {component}: {value:.2f}")

env.close()

# Results
print(f"\n--- Final Results ---")
print(f"Total Reward: {total_reward:.2f}")
print(f"Total Steps: {step_count}")

# Determine performance
if total_reward > 200:
    print("🎉 SUCCESSFUL LANDING! Excellent performance!")
elif total_reward > 100:
    print("✅ Good attempt! Close to successful landing.")
elif total_reward > 0:
    print("🔄 Decent try, but needs improvement.")
else:
    print("💥 Crashed. Model needs more training.")

print(f"\nVideo saved in ./videos/ directory")
print("You can play the video file to see your agent's performance!") 