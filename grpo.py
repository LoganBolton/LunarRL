import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env_wrapper import RewardWrapper

# Create environment with custom reward
env = gym.make("LunarLander-v3")
env = RewardWrapper(env)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./ppo_lunar_tensorboard/"
)

model.learn(total_timesteps=500_000)
model.save("ppo_lunarlander")

# Test the trained model
model = PPO.load("ppo_lunarlander")

obs, _ = env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    
    # Print reward breakdown for analysis
    if 'reward_breakdown' in info:
        print(f"Step reward: {reward:.2f}")
        for component, value in info['reward_breakdown'].items():
            if value != 0:
                print(f"  {component}: {value:.2f}")
    
    env.render()

print(f"Total reward: {total_reward:.2f}")
env.close()