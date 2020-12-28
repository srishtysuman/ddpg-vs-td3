import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
import torch.optim as optim
from torch.autograd import Variable
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
#import roboschool
import sys
from collections import deque
import random

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

class Critic(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Critic,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.linear2=nn.Linear(hidden_size+1,hidden_size)
        self.linear3=nn.Linear(hidden_size,output_size)

    def forward(self,state,action):
        #print(state.shape)
        #print(action.shape)
        #x=torch.cat([state,action],-1)
        #print(x.shape)
        x=F.relu(self.linear1(state))
        x = torch.cat((x, action), dim=1)
        x=F.relu(self.linear2(x))
        x=self.linear3(x)

        return x

  # https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

class Replay_Buffer():
    def __init__(self,max_size):
        self.max_size=max_size
        self.buffer=deque(maxlen=max_size)

    def store(self,state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state, action, reward, next_state, done=[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s,a,r,n_s,d=experience
            state.append(s)
            action.append(a)
            reward.append(r)
            next_state.append(n_s)
            done.append(d)
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class DDPG_Agent():
    def __init__(self, env, hidden_size=256, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-2, memory_size=10000):
        self.gamma=gamma
        self.tau=tau
        self.num_states=env.observation_space.shape[0]
        self.num_actions=env.action_space.shape[0]
        self.hidden_size=hidden_size
        self.env=env

        #initialise actor and critic network
        self.actor=Actor(self.num_states,self.hidden_size,self.num_actions)
        self.critic=Critic(self.num_states,self.hidden_size,self.num_actions)

        #initialize target actor and target critic network with weighta similar to actor and critic
        self.target_actor=Actor(self.num_states,self.hidden_size,self.num_actions)
        self.target_critic=Critic(self.num_states,self.hidden_size,self.num_actions)

        #initialize actor and target with random weights
        for target_parameter, parameter in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_parameter.data.copy_(parameter.data)

        for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_parameter.data.copy_(parameter.data)

        #initialize optimizer for loss
        self.critic_loss  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        #initialize replay buffer with maximum size
        self.replay_buffer = Replay_Buffer(memory_size)       
        
        #initialize noise
        self.noise = OUNoise(self.env.action_space)

        #function for selecting action
    def select_action(self,state):
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            action = self.actor.forward(state)
            action = action.detach().numpy()[0,0]
            return action

    def train(self,max_episode, max_step, batch_size):
            rewards=np.zeros(shape=(max_episode*max_step,1))
            episode_list=np.zeros(shape=(max_episode*max_step,1))
            for k in range(max_episode*max_step):
                episode_list[k]=k
            for i in range(max_episode):
                state = self.env.reset()
                env.render()
                self.noise.reset()
                episode_reward = 0
                for j in range(max_step):
                    #get noisy action according to current policy and exploration noise
                    action = self.select_action(state)
                    action = self.noise.get_action(action, j)
         
                    #execute action at, observe reward rt and next state st+1
                    next_state, reward, done, _ = self.env.step(action) 

                    #store transition in replay buffer
                    self.replay_buffer.store(state,action,reward,next_state,done)
                    
                    if len(self.replay_buffer)>batch_size:
                        #sample a random minibatch of n(n=batch_size) transition from replay buffer 
                        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
                        states = torch.FloatTensor(states)
                        actions = torch.FloatTensor(actions)
                        rewards = torch.FloatTensor(rewards)
                        next_states = torch.FloatTensor(next_states)

                        #set Q'=ri+ gamma*next_q(si+1, ai+1)
                        Qvals = self.critic.forward(states, actions)
                        next_actions = self.target_actor.forward(next_states)
                        next_Q = self.target_critic.forward(next_states, next_actions.detach())
                        Qprime = rewards + self.gamma * next_Q

                        #update critic by minimizing mean squared error loss
                        critic_loss = self.critic_loss(Qvals, Qprime)

                        #update actor by policy gradient loss
                        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

                        #optimize actor and critic by backpropogating the loss
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        actor_loss.backward()
                        critic_loss.backward()
                        self.actor_optimizer.step()
                        self.critic_optimizer.step()

                        #update target and policy network
                        #target_p=tau*actor+(1-tau)*actor
                        for target_parameter, parameter in zip(self.target_actor.parameters(), self.actor.parameters()):
                            target_parameter.data.copy_(parameter.data)
                        for target_parameter, parameter in zip(self.target_actor.parameters(), self.actor.parameters()):
                            target_parameter.data.copy_(parameter.data)

                    state = next_state
                    episode_reward += reward

                    if done:
                         print("episode " + str(i) + ": " + str(episode_reward))
                         break

                    
              
            plt.plot(rewards)
            plt.show()    
            return rewards

class TD3():
    def __init__()
 

env = NormalizedEnv(gym.make("Pendulum-v0"))
print(env)
agent = DDPG_Agent(env)
#noise = OUNoise(env.action_space)
batch_size = 128
#rewards = []
#avg_rewards = []

agent.train(max_episode=50, max_step=500, batch_size=batch_size)

