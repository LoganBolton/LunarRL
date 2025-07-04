import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class RewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper for LunarLander that breaks out reward components
    """
    def __init__(self, env, reward_weights=None):
        super().__init__(env)
        
        # Default reward weights - modify these to customize behavior
        self.reward_weights = reward_weights or {
            'distance_to_pad': 3.0,      # Reward for being close to landing pad
            'leg_contact': 2.0,           # Reward for legs touching ground
            'landing_bonus': 2.0,         # Bonus for successful landing
            'velocity_penalty': 0.5,      # Penalty for high velocity
            'angle_penalty': 0.5,         # Penalty for being tilted
            'main_engine_penalty': 1.0,   # Penalty for using main engine
            'side_engine_penalty': 1.0,   # Penalty for using side engines
            'crash_penalty': 4.0,         # Penalty for crashing
        }
        
        # Store previous position to calculate movement
        self.prev_x = 0
        self.prev_y = 0
        
        # Track episode step count for exponential penalty
        self.step_count = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = obs[0]
        self.prev_y = obs[1]
        self.step_count = 0  # Reset step counter
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
        
        # Calculate custom reward components
        custom_reward = self._calculate_custom_reward(
            x, y, vx, vy, angle, angular_vel, 
            leg1_contact, leg2_contact, action, 
            terminated, truncated
        )
        
        # Update previous position
        self.prev_x, self.prev_y = x, y
        
        # Add debug info
        info['original_reward'] = original_reward
        info['custom_reward'] = custom_reward
        info['step_count'] = self.step_count
        info['reward_breakdown'] = self._get_reward_breakdown(
            x, y, vx, vy, angle, angular_vel,
            leg1_contact, leg2_contact, action,
            terminated, truncated
        )
        
        return obs, custom_reward, terminated, truncated, info
    
    def _calculate_custom_reward(self, x, y, vx, vy, angle, angular_vel, 
                                leg1_contact, leg2_contact, action, 
                                terminated, truncated):
        reward = 0
        
        # Add time penalty to discourage hovering
        time_penalty = -0.000 * self.step_count  # Penalty per step to encourage quick landing
        reward += time_penalty
        
        # 1. Distance to landing pad (0, 0)
        distance_to_pad = np.sqrt(x**2 + y**2)
        distance_reward = -distance_to_pad * self.reward_weights['distance_to_pad']
        reward += distance_reward
        
        # 2. Velocity penalty (encourage slower landing)
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        velocity_penalty = -velocity_magnitude * self.reward_weights['velocity_penalty']
        reward += velocity_penalty
        
        # 3. Angle penalty (encourage upright landing)
        angle_penalty = -abs(angle) * self.reward_weights['angle_penalty']
        reward += angle_penalty
        
        # 4. Leg contact reward
        legs_in_contact = int(leg1_contact) + int(leg2_contact)
        # leg_reward = legs_in_contact * 10 * self.reward_weights['leg_contact']
        # reward += leg_reward
        
        # 5. Engine usage penalties
        if action == 2:  # Main engine
            engine_penalty = -0.3 * self.reward_weights['main_engine_penalty']
            # reward += engine_penalty
        elif action == 1 or action == 3:  # Side engines
            engine_penalty = -0.03 * self.reward_weights['side_engine_penalty']
            # reward += engine_penalty
        
        # 6. Terminal rewards
        if terminated:
            # Check if crashed (body touching ground) vs successful landing
            if legs_in_contact == 2 and abs(vx) < 0.5 and abs(vy) < 0.5 and abs(angle) < 0.3:
                # Successful landing
                landing_bonus = 100 * self.reward_weights['landing_bonus']
                reward += landing_bonus
            else:
                # Crashed
                crash_penalty = -100 * self.reward_weights['crash_penalty']
                reward += crash_penalty
        
        return reward
    
    def _get_reward_breakdown(self, x, y, vx, vy, angle, angular_vel,
                             leg1_contact, leg2_contact, action,
                             terminated, truncated):
        """Return detailed breakdown of reward components for analysis"""
        breakdown = {}
        
        distance_to_pad = np.sqrt(x**2 + y**2)
        breakdown['distance_reward'] = -distance_to_pad * self.reward_weights['distance_to_pad']
        
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        breakdown['velocity_penalty'] = -velocity_magnitude * self.reward_weights['velocity_penalty']
        
        breakdown['angle_penalty'] = -abs(angle) * self.reward_weights['angle_penalty']
        
        legs_in_contact = int(leg1_contact) + int(leg2_contact)
        breakdown['leg_reward'] = legs_in_contact * 10 * self.reward_weights['leg_contact']
        
        breakdown['engine_penalty'] = 0
        if action == 2:  # Main engine
            breakdown['engine_penalty'] = -0.3 * self.reward_weights['main_engine_penalty']
        elif action == 1 or action == 3:  # Side engines
            breakdown['engine_penalty'] = -0.03 * self.reward_weights['side_engine_penalty']
        
        breakdown['terminal_reward'] = 0
        if terminated:
            if legs_in_contact == 2 and abs(vx) < 0.5 and abs(vy) < 0.5 and abs(angle) < 0.3:
                breakdown['terminal_reward'] = 100 * self.reward_weights['landing_bonus']
            else:
                breakdown['terminal_reward'] = -100 * self.reward_weights['crash_penalty']
        
        return breakdown