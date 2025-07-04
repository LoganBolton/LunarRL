import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env_wrapper import RewardWrapper
from stable_baselines3.common.callbacks import BaseCallback

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