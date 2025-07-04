# 🚀 LunarLander Custom Reward System

This project extends the LunarLander-v3 environment with a flexible custom reward system that allows you to experiment with different reward structures for reinforcement learning.

## 📁 Files Overview

- **`grpo.py`** - Main training script with custom reward wrapper
- **`reward_examples.py`** - Additional reward configuration examples
- **`analyze_rewards.py`** - Tools to analyze and compare different reward configurations

## 🎯 Quick Start

### 1. Basic Usage
```python
import gymnasium as gym
from grpo import CustomRewardWrapper, CONSERVATIVE_LANDING

# Create environment with custom rewards
env = gym.make("LunarLander-v3")
env = CustomRewardWrapper(env, CONSERVATIVE_LANDING)

# Train as usual
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=400_000)
```

### 2. Available Reward Configurations

| Configuration | Description |
|---------------|-------------|
| `CONSERVATIVE_LANDING` | Emphasizes safe, slow landings |
| `FUEL_EFFICIENT` | Heavy penalties for engine usage |
| `AGGRESSIVE_LANDING` | Fast approach to landing pad |
| `PRECISION_LANDING` | Prioritizes accuracy over speed |
| `SPEED_RUNNER` | Get there fast, ignore perfection |
| `BEGINNER_FRIENDLY` | Lower penalties, higher rewards |

### 3. Create Your Own Configuration

```python
MY_CUSTOM_REWARD = {
    'distance_to_pad': 2.0,      # Higher = more reward for being close
    'velocity_penalty': 1.5,     # Higher = more penalty for speed
    'angle_penalty': 1.0,        # Higher = more penalty for tilting
    'leg_contact': 1.0,          # Higher = more reward for ground contact
    'main_engine_penalty': 0.5,  # Higher = more penalty for fuel use
    'side_engine_penalty': 0.5,  
    'crash_penalty': 2.0,        # Higher = more penalty for crashing
    'landing_bonus': 1.5,        # Higher = more bonus for landing
}

env = CustomRewardWrapper(env, MY_CUSTOM_REWARD)
```

## 🔬 Reward Analysis

### Analyze Different Configurations
```bash
python analyze_rewards.py
```

This will:
- Compare all predefined configurations
- Generate visualization plots
- Show detailed component breakdowns
- Create an interactive custom configuration builder

### Understanding Reward Components

Each step, the agent receives rewards based on:

1. **Distance to Landing Pad**: Closer = better
2. **Velocity**: Slower = better (for safe landing)
3. **Angle**: More upright = better
4. **Leg Contact**: Both legs touching = +20 points
5. **Engine Usage**: Each engine use = small penalty
6. **Terminal Events**: 
   - Successful landing = +100 points
   - Crash = -100 points

## 📊 Real-time Debugging

The wrapper provides detailed reward breakdowns during testing:

```python
obs, reward, terminated, truncated, info = env.step(action)

print(f"Total reward: {reward:.2f}")
print("Breakdown:", info['reward_breakdown'])
# Output:
# Total reward: 2.15
# Breakdown: {
#   'distance_reward': -0.85,
#   'velocity_penalty': -0.32,
#   'leg_reward': 10.0,
#   'engine_penalty': -0.3
# }
```

## 🎮 Training Different Behaviors

### For Conservative Landing:
```python
env = CustomRewardWrapper(env, CONSERVATIVE_LANDING)
# Agent learns: slow, precise, upright landings
```

### For Fuel Efficiency:
```python
env = CustomRewardWrapper(env, FUEL_EFFICIENT)
# Agent learns: minimal engine use, gliding approaches
```

### For Speed:
```python
env = CustomRewardWrapper(env, AGGRESSIVE_LANDING)
# Agent learns: fast approaches, less concern for perfection
```

## 🛠️ Advanced Customization

You can add completely new reward components:

```python
def add_smoothness_reward(wrapper):
    """Penalize jerky movements"""
    original_calc = wrapper._calculate_custom_reward
    
    def new_calc(*args, **kwargs):
        reward = original_calc(*args, **kwargs)
        # Add your custom logic here
        return reward
    
    wrapper._calculate_custom_reward = new_calc
    return wrapper

env = CustomRewardWrapper(env, MY_CONFIG)
env = add_smoothness_reward(env)
```

## 📈 Monitoring Training

The system is compatible with TensorBoard logging:

```python
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    tensorboard_log="./custom_reward_logs/"
)
```

## 🤝 Contributing

Feel free to:
- Add new reward configurations
- Suggest additional reward components
- Improve the analysis tools
- Share interesting training results

## 📝 Notes

- Reward weights can be any positive or negative value
- Higher absolute values = stronger influence
- Zero values = disable that component entirely
- The system preserves original environment dynamics while customizing rewards 