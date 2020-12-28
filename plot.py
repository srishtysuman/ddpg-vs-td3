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
from numpy import savetxt
from numpy import loadtxt



data_td3 = loadtxt('td3_reward.csv',delimiter=',')
data_ddpg = loadtxt('ddpg_reward.csv',delimiter=',')
#data_td3 = loadtxt('td3.csv',delimiter=',')
plt.plot(data_ddpg,'r',label='ddpg')
plt.plot(data_td3,'g',label='td3')
plt.legend(loc='upper left')
plt.title('reward over single run of 100 episodes')
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
