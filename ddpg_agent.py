import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import copy


buffer_size = int(1e5)  # replay buffer size
batch_size = 128       # minibatch size
discount = 0.99            # discount factor
tau = 1e-3              # for soft update of target parameters
lr_actor = 1e-4         # learning rate of the actor 
lr_critic = 1e-3        # learning rate of the critic


class ReplayBuffer:
    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
        """
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, n):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=n)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def size(self):
        return len(self.memory)

    
class Agent:
    def __init__(self, create_actor, create_critic, replay_buffer, noise, state_dim, action_dim, seed,
                 device="cpu"):
        torch.manual_seed(seed)

        self.actor_local = create_actor(state_dim=state_dim, action_dim=action_dim).to(device)
        self.actor_target = create_actor(state_dim=state_dim, action_dim=action_dim).to(device)
        self.actor_optimizer = optim.Adam(params=self.actor_local.parameters(), lr=lr_actor)

        self.critic_local = create_critic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.critic_target = create_critic(state_dim=state_dim, action_dim=action_dim).to(device)
        self.critic_optimizer = optim.Adam(params=self.critic_local.parameters(), lr=lr_critic)

        self.buffer = replay_buffer
        self.noise = noise
        self.device = device
        self.batch_size = batch_size
        self.discount = discount

        self.tau = tau

        Agent.hard_update(model_local=self.actor_local, model_target=self.actor_target)
        Agent.hard_update(model_local=self.critic_local, model_target=self.critic_target)

    def step(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)

        if self.buffer.size() >= self.batch_size:
            experiences = self.buffer.sample(self.batch_size)

            self.learn(self.to_tensor(experiences))

    def to_tensor(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(device=self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).data.numpy()
        self.actor_local.train()

        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        next_actions = self.actor_target(next_states)
        q_target_next = self.critic_target(next_states, next_actions)
        q_target = rewards + self.discount * q_target_next * (1.0 - dones)
        q_local = self.critic_local(states, actions)
        critic_loss = F.mse_loss(input=q_local, target=q_target)

        self.critic_local.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor policy
        actor_objective = self.critic_local(states, self.actor_local(states)).mean()
        self.actor_local.zero_grad()
        (-actor_objective).backward()
        self.actor_optimizer.step()

        # Update target networks
        Agent.soft_update(model_local=self.critic_local, model_target=self.critic_target, tau=self.tau)
        Agent.soft_update(model_local=self.actor_local, model_target=self.actor_target, tau=self.tau)

    @staticmethod
    def soft_update(model_local, model_target, tau):
        for local_param, target_param in zip(model_local.parameters(), model_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(model_local, model_target):
        Agent.soft_update(model_local=model_local, model_target=model_target, tau=1.0)

    def reset(self):
        self.noise.reset()




class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
