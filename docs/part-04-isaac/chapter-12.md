---
title: "Chapter 12: Reinforcement Learning for Robot Control"
description: "Implementing RL algorithms for robot control using Isaac Gym"
sidebar_position: 3
---

# Chapter 12: Reinforcement Learning for Robot Control

## Learning Objectives

After completing this chapter, you should be able to:

- Understand reinforcement learning fundamentals in robotics
- Implement RL algorithms using Isaac Gym for training
- Train policies for robot manipulation and navigation tasks
- Transfer learned policies from simulation to real robots (Sim-to-Real)

## Introduction to Reinforcement Learning in Robotics

### Overview of Reinforcement Learning

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives rewards based on its actions and aims to maximize cumulative reward over time.

In robotics, RL is particularly valuable because it can learn complex behaviors that are difficult to program explicitly, such as:
- Adaptive control strategies
- Complex manipulation tasks
- Dynamic navigation in unknown environments
- Multi-agent coordination

### Key RL Components in Robotics

1. **Agent**: The robot learning the task
2. **Environment**: The physical or simulated world the robot interacts with
3. **State Space**: The set of possible robot configurations (positions, velocities, sensor readings)
4. **Action Space**: The set of possible robot actions (torques, velocities, discrete commands)
5. **Reward Function**: Numerical feedback that guides learning

### Why RL in Robotics?

- **Adaptation**: Learns to adapt to environmental changes
- **Optimization**: Finds near-optimal solutions for complex tasks
- **Generalization**: Can handle variations in tasks and environments
- **Robustness**: Learns to handle unexpected situations

## Isaac Gym: GPU-Accelerated RL

### Overview of Isaac Gym

Isaac Gym is NVIDIA's GPU-accelerated physics simulation environment specifically designed for RL training. Key advantages include:

1. **Parallel Simulation**: Run thousands of robot environments simultaneously
2. **GPU Physics**: Accelerated physics computation using PhysX
3. **Integrated RL**: Direct integration with PyTorch for neural networks
4. **Contact Sensors**: High-fidelity contact detection and forces
5. **Realistic Simulation**: Accurate physics for sim-to-real transfer

### Simulation Parallelization

Isaac Gym enables massive parallelization by simulating thousands of environments in parallel:

```python
import isaacgym
import torch
import numpy as np

# Import Isaac Gym environments
from isaacgym import gymapi
from isaacgym import gymtorch

# Create gym
gym = gymapi.acquire_gym()

# Configure simulation
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.use_gpu_pipeline = True  # Use GPU physics

# Create simulation
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# Create environment
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# Create multiple environments in parallel
num_envs = 4096  # Simulate 4096 environments in parallel
envs = []
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, 1)
    envs.append(env)
    
    # Create robot and objects in each environment
    # (Implementation details would go here)
```

### Environment Design for RL

Creating effective RL environments requires careful design of:

1. **Observation Space**: What information the agent receives
2. **Action Space**: What actions the agent can take
3. **Reward Function**: How success is measured
4. **Episode Termination**: When an episode ends

```python
import torch
import numpy as np

class RobotRLTask:
    def __init__(self, num_envs, device, headless):
        self.device = device
        self.num_envs = num_envs
        
        # Define observation space
        self.num_obs = 11  # Example: pos, vel, goal_direction
        self.num_actions = 4  # Example: joint torques
        
        # Robot properties
        self.max_episode_length = 1000
        self.reset_dist = 3.0
        self.reach_goal_bonus = 10.0
        
        # Initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Robot state variables
        self.robot_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_orientations = torch.zeros((self.num_envs, 4), device=self.device)  # quaternion
        self.robot_lin_vels = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_ang_vels = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Goal positions
        self.goal_positions = torch.zeros((self.num_envs, 3), device=self.device)
        
    def compute_observations(self):
        """Compute observations for all environments"""
        # Robot position
        robot_pos = self.robot_positions
        
        # Goal direction
        goal_dir = self.goal_positions - robot_pos
        
        # Robot orientation and velocity
        robot_orn = self.robot_orientations
        robot_lin_vel = self.robot_lin_vels
        robot_ang_vel = self.robot_ang_vels
        
        # Concatenate all observation components
        self.obs_buf = torch.cat([
            robot_pos,          # 3
            robot_orn,          # 4
            robot_lin_vel,      # 3
            robot_ang_vel,      # 3
            goal_dir            # 3
        ], dim=-1)              # Total: 16 (adjust accordingly)
        
        return self.obs_buf
    
    def compute_rewards(self):
        """Compute rewards for all environments"""
        # Calculate distance to goal
        goal_dist = torch.norm(self.goal_positions - self.robot_positions, dim=-1)
        
        # Sparse reward: bonus for reaching goal
        goal_reached = goal_dist < 0.5  # Threshold for reaching goal
        goal_reward = goal_reached.float() * self.reach_goal_bonus
        
        # Dense reward: inverse distance (encourage moving toward goal)
        distance_reward = 1.0 / (1.0 + goal_dist)
        
        # Velocity penalty to encourage smooth motion
        linear_vel_penalty = -0.01 * torch.sum(self.robot_lin_vels ** 2, dim=-1)
        
        # Combine rewards
        self.rew_buf = goal_reward + distance_reward + linear_vel_penalty
        
        return self.rew_buf
    
    def reset_idx(self, env_ids):
        """Reset specific environments"""
        # Randomize robot positions
        rand_positions = torch.rand((len(env_ids), 3), device=self.device) * 2.0 - 1.0
        self.robot_positions[env_ids] = rand_positions
        
        # Randomize goal positions
        rand_goal_offsets = torch.rand((len(env_ids), 3), device=self.device) * 4.0 - 2.0
        self.goal_positions[env_ids] = rand_positions + rand_goal_offsets
        
        # Reset other state variables
        self.robot_orientations[env_ids] = torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float).repeat(len(env_ids), 1)
        self.robot_lin_vels[env_ids] = 0.0
        self.robot_ang_vels[env_ids] = 0.0
        
        # Reset progress and episode counters
        self.progress_buf[env_ids] = 0
        
    def pre_physics_step(self, actions):
        """Apply actions to the robot before physics simulation"""
        # Scale and apply actions
        scaled_actions = actions * 0.5  # Scale for stability
        
        # Apply actions based on your robot's control interface
        # This would involve setting joint torques, positions, or velocities
        
    def post_physics_step(self):
        """Process simulation results after physics step"""
        # Update progress
        self.progress_buf += 1
        
        # Get robot state after simulation step
        # self.robot_positions, self.robot_orientations, etc. would be updated here
        
        # Compute observations
        self.compute_observations()
        
        # Compute rewards
        self.compute_rewards()
        
        # Check for episode termination
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminated_env_ids = (self.progress_buf >= self.max_episode_length).nonzero(as_tuple=False).flatten()
        reset_env_ids = torch.cat([reset_env_ids, terminated_env_ids])
        
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        # Reset buffers for next step
        self.reset_buf = torch.zeros_like(self.reset_buf)
```

## Common RL Algorithms for Robotics

### Deep Deterministic Policy Gradient (DDPG)

DDPG is well-suited for continuous control tasks in robotics:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = deque(maxlen=1000000)
        self.max_action = max_action
        
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2
        
        self.total_it = 0
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, batch_size=100):
        self.total_it += 1
        
        # Sample replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, not_done = map(torch.FloatTensor, zip(*batch))
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device).unsqueeze(1)
        not_done = not_done.to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```

### Proximal Policy Optimization (PPO)

PPO is a policy gradient method that's stable and sample efficient:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, is_continuous=False):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.is_continuous = is_continuous
        if is_continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        features = self.shared_layers(state)
        
        value = self.critic(features)
        
        action_logits = self.actor(features)
        
        if self.is_continuous:
            # For continuous action spaces
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(action_logits, std)
        else:
            # For discrete action spaces
            dist = torch.distributions.Categorical(logits=action_logits)
        
        return dist, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.old_actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.update_old_policy()
        
    def update_old_policy(self):
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist, _ = self.actor_critic(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]
    
    def evaluate(self, state, action):
        dist, value = self.actor_critic(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, value
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """
        Compute Generalized Advantage Estimation
        """
        advantages = []
        gae = 0
        
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Calculate advantages
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[i + 1] * (1 - dones[i + 1])  # Don't bootstrap if episode ended
            
            # Calculate TD error
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            
            # Update GAE
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        return advantages
    
    def update(self, states, actions, rewards, dones, log_probs):
        """
        Update PPO policy using collected experiences
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Calculate values for states
        with torch.no_grad():
            _, values = self.old_actor_critic(states)
            values = values.squeeze(1).cpu().numpy()
        
        # Compute GAE
        advantages = self.compute_gae(rewards.cpu().numpy(), values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate returns
        returns = advantages + torch.tensor(values, dtype=torch.float32).to(self.device)
        
        # Optimize policy with multiple epochs
        for _ in range(4):  # 4 PPO epochs
            # Get current policy's probabilities
            log_probs, entropy, values = self.evaluate(states, actions)
            
            # Calculate ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogates
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            # Calculate total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
        
        # Update old policy
        self.update_old_policy()
```

## Training Policies in Isaac Gym

### Setting up the Training Environment

```python
import isaacgym
import torch
import torch.nn as nn
from rl_games.common.runner import Runner
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv

class IsaacGymVecEnv(vecenv.IVecEnv):
    def __init__(self, task, sim_device, rl_device, clip_observations=18.0, clip_actions=18.0):
        self.task = task
        
        self.num_envs = task.num_envs
        self.num_obs = task.num_obs
        self.num_actions = task.num_actions
        self.obs_space = torch_ext.DictSpace({'obs': spaces.Box(np.ones(self.num_obs) * -np.inf, 
                                                                np.ones(self.num_obs) * np.inf)})
        self.action_space = spaces.Box(np.ones(self.num_actions) * task.actions_low, 
                                       np.ones(self.num_actions) * task.actions_high)
        
        self.sim_device = sim_device
        self.rl_device = rl_device
        
        # normalization
        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
    
    def step(self, actions):
        # Convert actions to tensor if needed
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.sim_device, dtype=torch.float32)
        
        # Apply actions to task
        self.task.pre_physics_step(actions)
        
        # Simulate physics
        self.task.gym.simulate(self.task.sim)
        self.task.gym.fetch_results(self.task.sim, True)
        
        # Compute observations, rewards, etc.
        obs = self.task.compute_observations()
        rewards = self.task.compute_rewards()
        dones = self.task.reset_buf.bool()
        info = {}  # Additional info can be added here
        
        # Clamp observations and rewards
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        
        return obs, rewards, dones, info
    
    def reset(self):
        # Reset the environments
        self.task.reset_idx(torch.arange(self.num_envs, device=self.task.device))
        
        # Get initial observations
        obs = self.task.compute_observations()
        
        # Clamp observations
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        
        return obs
    
    def get_number_of_agents(self):
        return 1  # Single agent in each environment
    
    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.obs_space
        info['num_envs'] = self.num_envs
        info['num_obs'] = self.num_obs
        info['num_actions'] = self.num_actions
        return info
```

### Complete Training Pipeline

```python
import yaml
import os
from rl_games.common import tr_helpers
from rl_games.algos_torch import a2c_continuous

def create_isaac_gym_env(task, sim_device, rl_device):
    """Create IsaacGym environment wrapper for RL training"""
    return IsaacGymVecEnv(task, sim_device, rl_device)

def train_rl_policy():
    """Train an RL policy using Isaac Gym and RL Games"""
    
    # Set up device
    sim_device = "cuda:0"  # GPU for physics simulation
    rl_device = "cuda:0"   # GPU for RL training
    
    # Create task (robot environment)
    task = RobotRLTask(num_envs=4096, device=sim_device, headless=False)
    
    # Create environment wrapper
    env = create_isaac_gym_env(task, sim_device, rl_device)
    
    # RL Games configuration
    config = {
        'seed': 42,
        'network': {
            'name': 'actor_critic',
            'separate': False,
            'space': {
                'continuous': {
                    'mu_activation': 'None',
                    'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': 0.2},
                    'fixed_sigma': True
                }
            }
        },
        'algo': {
            'name': 'a2c_continuous'
        },
        'model': {
            'name': 'continuous_a2c_logstd'
        },
        'learning': {
            'gamma': 0.99,
            'tau': 0.95,
            'learning_rate': 3e-4,
            'critic_coef': 2,
            'clip_value': True,
            'bound_loss_coef': 0.001,
        },
        'params': {
            'ppo': {
                'clip_range': 0.2,
                'clip_range_eps': 0.02,
                'normalize_advantage': True,
                'extended_horizon': False,
                'kl_lambda': 0.9,
                'minibatch_size': 64,
                'epoch': 10,
                'dual_clip': None,
                'rewards_shaper': {'name': 'default', 'scale_value': 0.01},
                'value_bootstrap': True,
                'normalize_input': True,
                'normalize_value': True,
            }
        }
    }
    
    # Create runner
    runner = Runner()
    
    # Prepare experiment
    runner.algo = 'a2c_continuous'
    runner.config = config
    runner.device = rl_device
    runner.num_actors = task.num_envs
    
    # Set up networks
    obs_space = env.obs_space
    act_space = env.action_space
    
    # Create model
    model = a2c_continuous.ModelA2CContinuous(network_dict=config['network'],
                                              actions_num=act_space.shape[0],
                                              obs_space=obs_space)
    
    # Create agent
    agent = a2c_continuous.A2CAgent(
        base_name='rl_games',
        observation_space=obs_space,
        action_space=act_space,
        is_discrete=False,
        model=model,
        config=config['learning'],
        device=rl_device
    )
    
    # Create policy
    runner.model = model
    runner.agent = agent
    runner.env = env
    
    # Train the policy
    runner.run({
        'train': True,
        'load_checkpoint': False,
        'play': False
    })
    
    return runner
```

## Sim-to-Real Transfer Techniques

### Domain Randomization

Domain randomization helps policies generalize by training with randomized simulation parameters:

```python
class DomainRandomizedEnv:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        
        # Define ranges for randomization
        self.mass_range = [0.8, 1.2]  # Randomize 80-120% of base mass
        self.friction_range = [0.5, 1.5]  # Randomize friction coefficients
        self.restitution_range = [0.0, 0.2]  # Randomize bounciness
        self.torque_range = [0.8, 1.2]  # Randomize motor torques
        
        # Store original parameters for randomization
        self.original_masses = torch.ones(num_envs, device=device)
        self.original_frictions = torch.ones(num_envs, device=device)
        
    def randomize_parameters(self, env_ids=None):
        """Randomize physics parameters"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        # Randomize masses
        mass_multipliers = torch.rand(len(env_ids), device=self.device) * \
                          (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
        randomized_masses = self.original_masses[env_ids] * mass_multipliers
        
        # Randomize frictions
        friction_multipliers = torch.rand(len(env_ids), device=self.device) * \
                              (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]
        randomized_frictions = self.original_frictions[env_ids] * friction_multipliers
        
        # Apply to simulation
        self.apply_randomized_parameters(env_ids, randomized_masses, randomized_frictions)
    
    def apply_randomized_parameters(self, env_ids, masses, frictions):
        """Apply randomized parameters to simulation"""
        # This would interface with Isaac Gym physics to update parameters
        # In practice, you'd use Isaac Gym's domain randomization API
        pass
```

### System Identification

Identifying real-world system parameters to match simulation:

```python
class SystemIdentification:
    def __init__(self):
        self.sim_params = {}
        self.real_params = {}
        
    def excite_system(self, robot, signal_type="step", amplitude=1.0):
        """Excite the system with a known input to identify parameters"""
        # Apply excitation signal
        if signal_type == "step":
            u = torch.full((1000,), amplitude)
        elif signal_type == "sine":
            t = torch.linspace(0, 10, 1000)
            u = amplitude * torch.sin(2 * torch.pi * 0.5 * t)
        elif signal_type == "prbs":
            # Pseudo-random binary sequence
            u = torch.rand(1000) * amplitude * 2 - amplitude
        
        # Record response
        y = self.record_response(robot, u)
        
        # Estimate system parameters
        estimated_params = self.estimate_params(u, y)
        
        return estimated_params
    
    def estimate_params(self, input_signal, output_signal):
        """Estimate system parameters from I/O data"""
        # Use system identification techniques like:
        # - Least squares estimation
        # - ARX models
        # - State-space model estimation
        pass
    
    def match_sim_to_real(self):
        """Adjust simulation parameters to match real system"""
        # Optimize simulation parameters to minimize sim-to-real gap
        pass
```

### Robust RL

Training policies that are robust to parameter variations:

```python
class RobustPPO:
    def __init__(self, base_agent, uncertainty_model=None):
        self.base_agent = base_agent
        self.uncertainty_model = uncertainty_model or RobustnessAugmentation()
        
    def train_with_robustness(self, env, num_iterations=1000):
        """Train policy with robustness to uncertainties"""
        for iteration in range(num_iterations):
            # Sample uncertain parameters
            uncertain_params = self.uncertainty_model.sample_uncertainty()
            
            # Update environment with sampled parameters
            env.update_dynamics(uncertain_params)
            
            # Collect trajectories with these parameters
            trajectories = self.collect_trajectories(env)
            
            # Update policy using trajectories
            self.update_policy(trajectories)
            
            # Periodically evaluate on nominal parameters
            if iteration % 10 == 0:
                robustness_metrics = self.evaluate_robustness(env)
                print(f"Iteration {iteration}, Robustness: {robustness_metrics}")
    
    def collect_trajectories(self, env):
        """Collect trajectories from environment"""
        # Implementation of trajectory collection
        pass
    
    def update_policy(self, trajectories):
        """Update policy based on collected trajectories"""
        # Use PPO or other RL algorithm
        pass
```

## Practical Implementation Example: Robot Manipulation

Here's a complete example of training a manipulation policy:

```python
import torch
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
import torch.nn as nn

class ManipulationTask:
    def __init__(self, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        
        # Define action and observation space
        self.num_actions = 7  # Joint position actions for 7-DOF arm
        self.num_obs = 14 + 7 + 3  # Robot state + goal + object
        
        # Training parameters
        self.dist_reward_scale = 2.0
        self.reach_bonus = 0.5
        self.fall_dist = 0.2
        self.fall_penalty = -0.5
        
        # Initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Robot state
        self.arm_positions = torch.zeros((self.num_envs, 7), device=self.device)
        self.arm_velocities = torch.zeros((self.num_envs, 7), device=self.device)
        self.ee_positions = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Object state
        self.object_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        
        # Goal state
        self.goal_positions = torch.zeros((self.num_envs, 3), device=self.device)
        
    def compute_observations(self):
        """Compute observations for manipulation task"""
        # End-effector position relative to object
        ee_to_obj = self.object_positions - self.ee_positions
        
        # End-effector position relative to goal
        ee_to_goal = self.goal_positions - self.ee_positions
        
        # Combine observations
        self.obs_buf = torch.cat([
            self.arm_positions,      # 7: Joint positions
            self.arm_velocities,     # 7: Joint velocities
            ee_to_obj,               # 3: EE to object vector
            ee_to_goal,              # 3: EE to goal vector
            self.object_positions,   # 3: Object position
            self.goal_positions      # 3: Goal position
        ], dim=-1)                  # Total: 26
        
        return self.obs_buf
    
    def compute_rewards(self):
        """Compute rewards for manipulation task"""
        # Distance to goal
        goal_dist = torch.norm(self.goal_positions - self.ee_positions, dim=-1)
        
        # Distance to object
        obj_dist = torch.norm(self.object_positions - self.ee_positions, dim=-1)
        
        # Reward based on how close EE is to goal
        reward = -self.dist_reward_scale * goal_dist
        
        # Bonus for reaching the goal
        reward = torch.where(obj_dist < 0.1, reward + self.reach_bonus, reward)
        
        # Penalty if object falls
        reward = torch.where(self.object_positions[:, 2] < self.fall_dist, 
                            reward + self.fall_penalty, reward)
        
        self.rew_buf = reward
        return self.rew_buf
    
    def reset_idx(self, env_ids):
        """Reset specific environments"""
        # Randomize arm positions
        rand_arm = torch.rand((len(env_ids), 7), device=self.device) * 1.0 - 0.5
        self.arm_positions[env_ids] = rand_arm
        
        # Randomize object positions
        rand_obj = torch.rand((len(env_ids), 3), device=self.device)
        rand_obj[:, 0] = rand_obj[:, 0] * 0.5 + 0.5   # x: 0.5-1.0
        rand_obj[:, 1] = rand_obj[:, 1] * 0.6 - 0.3   # y: -0.3-0.3
        rand_obj[:, 2] = rand_obj[:, 2] * 0.2 + 0.4   # z: 0.4-0.6
        
        self.object_positions[env_ids] = rand_obj
        self.object_orientations[env_ids] = torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float).repeat(len(env_ids), 1)
        
        # Randomize goal positions
        rand_goal = torch.rand((len(env_ids), 3), device=self.device)
        rand_goal[:, 0] = rand_goal[:, 0] * 0.5 + 0.5
        rand_goal[:, 1] = rand_goal[:, 1] * 0.6 - 0.3
        rand_goal[:, 2] = rand_goal[:, 2] * 0.2 + 0.6
        
        self.goal_positions[env_ids] = rand_goal
        
        # Reset velocities
        self.arm_velocities[env_ids] = 0.0
        self.ee_positions[env_ids] = self.calculate_ee_position(env_ids)
        
        # Reset episode counters
        self.progress_buf[env_ids] = 0
        
    def calculate_ee_position(self, env_ids):
        """Calculate end-effector position based on arm configuration"""
        # This would implement forward kinematics
        # For simplicity, returning a placeholder
        return torch.zeros((len(env_ids), 3), device=self.device)
    
    def pre_physics_step(self, actions):
        """Apply actions to robot"""
        # Scale actions for stability
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Apply to simulation (implementation would interface with physics engine)
        # This is where you'd send actions to the actual robot or simulation
        
    def post_physics_step(self):
        """Process after physics step"""
        self.progress_buf += 1
        
        # Update robot and object states from simulation
        # (In practice, this would interface with Isaac Gym)
        
        # Compute observations
        self.compute_observations()
        
        # Compute rewards
        self.compute_rewards()
        
        # Reset environments that have reached max episode length
        reset_env_ids = (self.progress_buf >= 1000).nonzero(as_tuple=False).flatten()
        
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
```

## Exercises

1. **RL Algorithm Implementation**: Implement a PPO agent for a simple navigation task and train it in Isaac Gym.

2. **Policy Transfer Exercise**: Train a manipulation policy in simulation and implement domain randomization techniques to improve sim-to-real transfer.

3. **Advanced Algorithm Exercise**: Implement and compare DDPG and PPO algorithms on a robot control task, analyzing the trade-offs between them.

## Summary

This chapter covered reinforcement learning techniques for robot control using NVIDIA Isaac Gym. We explored the fundamentals of RL in robotics, implemented common algorithms like DDPG and PPO, and learned how to leverage Isaac Gym's parallel simulation capabilities. We also covered sim-to-real transfer techniques including domain randomization and system identification.

The key takeaways include:
- Isaac Gym enables massive parallelization for RL training with thousands of environments
- RL algorithms like PPO and DDPG can learn complex robot behaviors
- Domain randomization and system identification help with sim-to-real transfer
- Proper reward function design is crucial for successful RL training
- Robust RL techniques improve policy performance across parameter variations

## Cross-references

For foundational concepts about Physical AI, see [Chapter 1: Introduction to Physical AI](../part-01-foundations/chapter-1). For Isaac platform basics, see [Chapter 10: NVIDIA Isaac Platform Overview](../part-04-isaac/chapter-10). For perception systems that can support RL tasks, see [Chapter 11: Advanced Perception with Isaac](../part-04-isaac/chapter-11).