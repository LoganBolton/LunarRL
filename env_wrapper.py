import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class RewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper for LunarLander that breaks out reward components
    """
    def __init__(self, env, reward_weights=None):
        super().__init__(env)
        
        # Default reward weights - modify these to customize behavior
        self.reward_weights = reward_weights or {
            'leg_contact': 2.0,           # Reward for legs touching ground
            'landing_bonus': 1.0,        # Bonus for successful landing (increased)
            'distance_to_pad': 0.5,       # penalty for being far from landing pad (reduced)
            'velocity_penalty': 5,      # Penalty for high velocity (increased)
            'angle_penalty': 0.5,         # Penalty for being tilted (increased)
            'main_engine_penalty': 0.3,   # Penalty for using main engine (reduced)
            'side_engine_penalty': 0.3,   # Penalty for using side engines (reduced)
            'crash_penalty': 5.0,         # Penalty for crashing (increased)
            'precision_landing_bonus': 5.0,  # Bonus for landing very close to center (increased)
            'progress_reward': 1.0,       # New: reward for getting closer to pad
        }
        
        # Store previous position to calculate movement
        self.prev_x = 0
        self.prev_y = 0
        self.prev_distance = 0  # Track previous distance for progress reward
        
        # Track episode step count for exponential penalty
        self.step_count = 0
        
        # Track if already landed to penalize continued engine use
        self.successfully_landed = False
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = obs[0]
        self.prev_y = obs[1]
        self.prev_distance = np.sqrt(obs[0]**2 + obs[1]**2)  # Initialize previous distance
        self.step_count = 0  # Reset step counter
        self.successfully_landed = False  # Reset landing state
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1  # Increment step counter
        
        # Extract state variables
        x, y = obs[0], obs[1]           # Position
        vx, vy = obs[2], obs[3]         # Velocity
        angle = obs[4]                  # Lander angle
        angular_vel = obs[5]            # Angular velocity
        leg1_contact = obs[6]           # Left leg contact
        leg2_contact = obs[7]           # Right leg contact
        
        # Calculate reward breakdown (this will be our single source of truth)
        reward_breakdown = self._get_reward_breakdown(
            x, y, vx, vy, angle, angular_vel,
            leg1_contact, leg2_contact, action,
            terminated, truncated
        )
        
        # Calculate total custom reward from breakdown components
        custom_reward = sum(reward_breakdown.values())
        
        # Update previous position
        self.prev_x, self.prev_y = x, y
        self.prev_distance = np.sqrt(x**2 + y**2)  # Update previous distance for progress tracking
        
        # Add debug info
        info['original_reward'] = original_reward
        info['custom_reward'] = custom_reward
        info['step_count'] = self.step_count
        info['reward_breakdown'] = reward_breakdown
        
        return obs, custom_reward, terminated, truncated, info
    

    
    def _get_reward_breakdown(self, x, y, vx, vy, angle, angular_vel,
                             leg1_contact, leg2_contact, action,
                             terminated, truncated):
        """Return detailed breakdown of reward components for analysis"""
        breakdown = {}
        legs_in_contact = int(leg1_contact) + int(leg2_contact)
        is_currently_landed = (legs_in_contact == 2 and abs(vx) < 0.5 and 
                              abs(vy) < 0.5 and abs(angle) < 0.3)
        
        # Reduced time penalty to discourage hovering but not dominate reward
        breakdown['time_penalty'] = -0.0001 * self.step_count
        
        distance_to_pad = np.sqrt(x**2 + y**2)
        
        # Progress reward - positive feedback for getting closer to landing pad
        distance_progress = self.prev_distance - distance_to_pad
        breakdown['progress_reward'] = distance_progress * self.reward_weights['progress_reward']
        
        # Distance penalty with diminishing returns (less harsh for being far away)
        # Use a softer penalty that caps at reasonable values
        distance_penalty = -np.tanh(distance_to_pad) * self.reward_weights['distance_to_pad']
        breakdown['distance_reward'] = distance_penalty
        
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        if self.step_count > 80:
            breakdown['velocity_penalty'] = -velocity_magnitude * self.reward_weights['velocity_penalty']
        else:
            breakdown['velocity_penalty'] = 0
        
        breakdown['angle_penalty'] = -abs(angle) * self.reward_weights['angle_penalty']
        
        # Leg contact reward - encourage touching ground with legs
        breakdown['leg_reward'] = legs_in_contact * 0.5 * self.reward_weights['leg_contact']
        
        breakdown['engine_penalty'] = 0
        if action == 2:  # Main engine
            engine_penalty = -0.1 * self.reward_weights['main_engine_penalty']
            if self.successfully_landed or is_currently_landed:
                engine_penalty *= 3
            breakdown['engine_penalty'] = engine_penalty
        elif action == 1 or action == 3:  # Side engines
            engine_penalty = -0.02 * self.reward_weights['side_engine_penalty']
            if self.successfully_landed or is_currently_landed:
                engine_penalty *= 3
            breakdown['engine_penalty'] = engine_penalty

        # Track successful landing state
        if is_currently_landed:
            self.successfully_landed = True

        breakdown['terminal_reward'] = 0
        breakdown['precision_bonus'] = 0
        if terminated:
            if is_currently_landed:
                breakdown['terminal_reward'] = 100 * self.reward_weights['landing_bonus']
                # Add precision bonus breakdown
                if distance_to_pad < 0.1:
                    breakdown['precision_bonus'] = 100 * self.reward_weights['precision_landing_bonus']
                elif distance_to_pad < 0.2:
                    breakdown['precision_bonus'] = 50 * self.reward_weights['precision_landing_bonus']
                elif distance_to_pad < 0.3:
                    breakdown['precision_bonus'] = 25 * self.reward_weights['precision_landing_bonus']
            else:
                breakdown['terminal_reward'] = -50 * self.reward_weights['crash_penalty']
        
        return breakdown

class RewardLogger(BaseCallback):
    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'reward_breakdown' in info:
                breakdown = info['reward_breakdown']
                for component, value in breakdown.items():
                    self.logger.record(f"reward/{component}", value)
        return True