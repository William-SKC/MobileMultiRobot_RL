from copy import deepcopy
from typing import List
import numpy as np


import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent_vis:
    """Agent that take in 2d images and can interact with environment from pettingzoo"""

    def __init__(self, vis_width, act_dim, num_agents, actor_lr, critic_lr, device = "cpu"):
        
        self.AC_NNs = ActorCriticNetworks(vis_width, act_dim, num_agents)
    
        self.actor_optimizer = Adam(self.AC_NNs.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.AC_NNs.critic.parameters(), lr=critic_lr)

        #target networks
        self.target_AC_NNs = deepcopy(self.AC_NNs)
        self.num_agents = num_agents
        self.device = device

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])
        # print('action: obs', obs.shape)
        logits = self.AC_NNs.actor_forward(obs)  # torch.Size([batch_size, action_size])
        action = F.gumbel_softmax(logits, hard=True)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])
        logits = self.target_AC_NNs.actor_forward(obs)  # torch.Size([batch_size, action_size])
        action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        features = torch.cat(state_list, dim=0)
        actions = torch.cat(act_list, dim=0)
        return self.AC_NNs.critic_forward(features, actions).squeeze(1)

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        features = torch.cat(state_list, dim=0)
        actions = torch.cat(act_list, dim=0)
        return self.target_AC_NNs.critic_forward(features, actions).squeeze(1) # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.AC_NNs.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.AC_NNs.critic.parameters(), 0.5)
        self.critic_optimizer.step()

# def batchify_obs(obs, device):
#     """Converts PZ style observations to batch of torch arrays."""
#     # Converts observations to a batch of torch tensors.
#     # convert to list of np arrays 
#     # obs[a].shape: (height, width, channel)
#     obs = np.stack([obs[a] for a in obs], axis=0) 
#     # transpose to be (batch, channel, height, width)
#     obs = obs.transpose(0, -1, 1, 2)
#     # convert to torch
#     obs = torch.tensor(obs).to(device)
#     return obs

class ActorCriticNetworks(nn.Module):
    def __init__(self, vis_width, act_dim, num_agents, non_linear=nn.ReLU()):
        super(ActorCriticNetworks, self).__init__()
        self.vis_width = vis_width
        self.num_agents = num_agents

        self.feature_extractor = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 16, 3, padding=1)), 
            # Conv2d(in_channels, out_channels, kernel_size) 
            nn.MaxPool2d(2), 
            non_linear,
            self._layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)),
            nn.MaxPool2d(2), 
            non_linear,
            self._layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
            nn.MaxPool2d(2), 
            non_linear,
            nn.Flatten()
        )
        self.fc_input_dim = self.calculate_fc_input_dim()
        # Fully connected layers for the Actor-Critic heads
        # Shared layer
        self.shared_fc = nn.Linear(self.fc_input_dim, 512)
        
        # Actor head (Policy network)
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Softmax(dim=-1)  # Outputs a probability distribution over actions
        ).apply(self.init)
        
        # Critic head for MADDPG (Value network)
        # Accepts concatenated features and actions of all agents
        self.critic = nn.Sequential(
            nn.Linear((self.fc_input_dim + act_dim) * num_agents, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Outputs a single value representing the joint state-action value
        ).apply(self.init)

    def calculate_fc_input_dim(self):
        # Create a dummy input to pass through the feature extractor to determine the output size
        dummy_input = torch.zeros(1, 4, self.vis_width, self.vis_width)
        features = self.feature_extractor(dummy_input)
        return features.view(-1).size(0)

    def actor_forward(self, x):
        # Pass input through the feature extractor
        features = self.feature_extractor(x)
        # Pass features through the shared fully connected layer
        shared = F.relu(self.shared_fc(features))
        
        # Actor head (each agent gets its own policy)
        policy = self.actor(shared)
        
        return policy
    
    def critic_forward(self, features, actions):
        features = self.feature_extractor(features)
        # Concatenate features and actions of all agents
        # critic_input = torch.cat([features.view(self.num_agents, -1), actions.view(self.num_agents, -1)], dim=-1)
        critic_input = torch.cat((features, actions), dim=-1)
        # print('critic_forward: ', features.shape, actions.shape, critic_input.shape)
        # Pass concatenated input through the critic network
        value = self.critic(critic_input)
        
        return value

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    # A custom method to initialize layers.
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std) # initializes weights Tensor as an orthogonal matrix, scaled by std.
        torch.nn.init.constant_(layer.bias, bias_const) # Fill the bias Tensor with the bias_const value 
        return layer
