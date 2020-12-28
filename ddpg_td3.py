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
from numpy import var
from numpy import mean
from numpy import savetxt
from numpy import load
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
    def __init__(self, env, gamma, tau, buffer_maxlen, delay_step, noise_std, critic_lr, actor_lr):
        self.gamma=gamma
        self.tau=tau
        self.num_states=env.observation_space.shape[0]
        self.num_actions=env.action_space.shape[0]
        self.hidden_size=256
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
        self.replay_buffer = Replay_Buffer(buffer_maxlen)       
        
        #initialize noise
        self.noise = OUNoise(self.env.action_space)

        #function for selecting action
    def get_action(self,state):
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))
            action = self.actor.forward(state)
            action = action.detach().numpy()[0,0]
            return action
    def update(self, batch_size):
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

   

class TD3Agent:

    def __init__(self, env, gamma, tau, buffer_maxlen, delay_step, noise_std, noise_bound, critic_lr, actor_lr):

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.hidden_size=256
        # hyperparameters    
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_bound = noise_bound
        self.update_step = 0 
        self.delay_step = delay_step
        
        # initialize actor and critic networks
        self.critic1 = Critic(self.obs_dim, self.hidden_size, self.action_dim)
        self.critic2 = Critic(self.obs_dim, self.hidden_size, self.action_dim)
        self.critic1_target = Critic(self.obs_dim, self.hidden_size, self.action_dim)
        self.critic2_target = Critic(self.obs_dim, self.hidden_size, self.action_dim)
        self.actor = Actor(self.obs_dim, self.hidden_size, self.action_dim)
        self.actor_target = Actor(self.obs_dim, self.hidden_size, self.action_dim)
        # Copy critic target parameters
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        # initialize optimizers        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr) 
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.replay_buffer = Replay_Buffer(buffer_maxlen)        

    def get_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def update(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        masks = torch.FloatTensor(masks)
        
        action_space_noise = self.generate_action_space_noise(action_batch)
        next_actions = self.actor.forward(state_batch) + action_space_noise
        next_Q1 = self.critic1_target.forward(next_state_batch, next_actions)
        next_Q2 = self.critic2_target.forward(next_state_batch, next_actions)
        expected_Q = reward_batch + self.gamma * torch.min(next_Q1, next_Q2)

        # critic loss
        curr_Q1 = self.critic1.forward(state_batch, action_batch)
        curr_Q2 = self.critic2.forward(state_batch, action_batch)
        critic1_loss = F.mse_loss(curr_Q1, expected_Q.detach())
        critic2_loss = F.mse_loss(curr_Q2, expected_Q.detach())
        
        # update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # delyaed update for actor & target networks  
        if(self.update_step % self.delay_step == 0):
            # actor
            self.actor_optimizer.zero_grad()
            policy_gradient = -self.critic1(state_batch, self.actor(state_batch)).mean()
            policy_gradient.backward()
            self.actor_optimizer.step()

            # target networks
            self.update_targets()

        self.update_step += 1

    def generate_action_space_noise(self, action_batch):
        noise = torch.normal(torch.zeros(action_batch.size()), self.noise_std).clamp(-self.noise_bound, self.noise_bound)
        return noise

    def update_targets(self):
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size,indicator):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        env.render()
        if indicator==1:
            noise = OUNoise(env.action_space)
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            if indicator==1:
                action = noise.get_action(action, step)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.store(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards



gamma = 0.99
tau = 1e-2
noise_std = 0.2
bound = 0.5
delay_step = 2
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-3
max_episodes = 100
max_steps = 500
batch_size = 32

#env2 = NormalizedEnv(gym.make("Pendulum-v0"))
#agent = DDPG_Agent(env2, gamma, tau, buffer_maxlen, delay_step, noise_std, critic_lr, actor_lr)

#er=np.zeros(shape=(10,100))
#for i in range(10):
#   print('ddpg run: ',i)
#   er[i]=mini_batch_train(env2, agent, 100, 500, 64,1)


#col_mean1 = mean(er, axis=0)
#col_var1 = var(er, ddof=1, axis=0)
#savetxt('ddpg_mean.csv',col_mean1,delimiter=',')
#savetxt('ddpg.csv', col_var1, delimiter=',')
#episode_rewards_ddpg=mini_batch_train(env2, agent, 100, 500, 64,1)
#savetxt('ddpg_reward.csv',episode_rewards_ddpg,delimiter=',')

#print(col_mean)

env = gym.make("Pendulum-v0")
agent2 = TD3Agent(env, gamma, tau, buffer_maxlen, delay_step, noise_std, bound, critic_lr, actor_lr)
episode_rewards_td3 = mini_batch_train(env, agent2, 100, 500, 64,2)
savetxt('td3_reward.csv',episode_rewards_td3,delimiter=',')
er2=np.zeros(shape=(10,100))
for i in range(10):
   print('td3 run : ', i)
   er2[i]=mini_batch_train(env, agent2, 100, 500, 64,1)


col_mean2 = mean(er2, axis=0)
savetxt('td3_mean.csv',col_mean2,delimiter=',')
col_var2 = var(er2, ddof=1, axis=0)
savetxt('td3.csv', col_var2, delimiter=',')

#


#plt.plot(episode_rewards_td3,'g',label='td3') 
#data_ddpg = load('ddpg.csv',delimiter=',')
#data_td3 = load('td3.csv',delimiter=',')
#plt.plot(data_ddpg,'r',label='ddpg')
#plt.plot(data_td3,'g',label='td3')
#plt.plot()
#plt.xlabel('Episode')
#plt.ylabel('Reward')
#plt.show()
# for episode in range(50):
#     state = env.reset()
#     noise.reset()
#     episode_reward = 0
    
#     for step in range(500):
#         action = agent.get_action(state)
#         action = noise.get_action(action, step)
#         new_state, reward, done, _ = env.step(action) 
#         agent.memory.push(state, action, reward, new_state, done)
        
#         if len(agent.memory) > batch_size:
#             agent.update(batch_size)        
        
#         state = new_state
#         episode_reward += reward

#         if done:
#             sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
#             break

#     rewards.append(episode_reward)
#     avg_rewards.append(np.mean(rewards[-10:]))


