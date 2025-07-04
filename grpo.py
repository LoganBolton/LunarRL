import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env_wrapper import RewardWrapper
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
import os

# Create environment with custom reward
env = gym.make("LunarLander-v3")
env = RewardWrapper(env)

class RewardLogger(BaseCallback):
    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'reward_breakdown' in info:
                breakdown = info['reward_breakdown']
                if self.locals['dones'][0]:  # Only log on episode end
                    for component, value in breakdown.items():
                        self.logger.record(f"reward_terminal/{component}", value)
        return True

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./ppo_lunar_tensorboard/"
)

callback = RewardLogger()
model.learn(total_timesteps=900_000, callback=callback)
model.save("ppo_lunarlander")

# Test the trained model
model = PPO.load("ppo_lunarlander")

# Create videos directory if it doesn't exist
os.makedirs("./videos", exist_ok=True)

# Create a FRESH environment for testing with video recording
test_env = gym.make("LunarLander-v3", render_mode="rgb_array")
test_env = RewardWrapper(test_env)
test_env = RecordVideo(test_env, video_folder="./videos/", episode_trigger=lambda x: True, name_prefix="lunarlander_training_test")

print("Testing trained model with video recording...")
obs, _ = test_env.reset()
done = False
total_reward = 0.0  # Initialize as float to avoid type issues
step_count = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)  # Use deterministic for consistent results
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    total_reward += float(reward)  # Cast reward to float to avoid type issues
    step_count += 1
    
    # Print reward breakdown for analysis (less verbose)
    if step_count % 20 == 0 and 'reward_breakdown' in info:
        print(f"Step {step_count}: Total reward so far = {total_reward:.2f}")
        for component, value in info['reward_breakdown'].items():
            if abs(value) > 0.01:  # Only show significant components
                print(f"  {component}: {value:.2f}")

print(f"\n--- Final Test Results ---")
print(f"Total reward: {total_reward:.2f}")
print(f"Total steps: {step_count}")

# Determine success
if total_reward > 200:
    print("🎉 SUCCESSFUL LANDING!")
elif total_reward > 100:
    print("✅ Good landing attempt!")
elif total_reward > 0:
    print("🔄 Decent attempt, needs improvement")
else:
    print("💥 Crashed - needs more training")

test_env.close()
print(f"\n📹 Video saved in ./videos/ directory")
print("You can play the video file to see your agent's performance!")