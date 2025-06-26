import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

model = PPO(
    policy="MlpPolicy",
    env = env,
    verbose=1,
    tensorboard_log="./ppo_lunar_tensorboard/"
)

model.learn(total_timesteps=400_000)

model.save("ppo_lunarlander")

model = PPO.load("ppo_lunarlander")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()

env.close()