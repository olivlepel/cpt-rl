import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm, trange
import statistics
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from random import random, choice, choices
from math import exp
import copy
from math import sin, pi
import numpy as np
class MLP_thin(nn.Module):
    def __init__(self,dropout=False,alpha=7,markovian=True):
        super(MLP_thin, self).__init__()
        self.fc1 = nn.Linear(2, 20)  if markovian else nn.Linear(3,12)
        self.fc2 = nn.Linear(20, 4)
        self.m = nn.Dropout(p=0.2)
        self.dropout = dropout
        self.alpha = alpha
    def setAlpha(self,alpha):
        self.alpha = alpha
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        #x = F.leaky_relu(self.m(self.fc2(x))) if self.dropout else F.leaky_relu(self.fc2(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return F.softmax(self.alpha*F.tanh(x/self.alpha))


class MLP(nn.Module):
    def __init__(self,dropout=False,alpha=4,markovian=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 12)  if markovian else nn.Linear(3,12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 4)
        self.m = nn.Dropout(p=0.2)
        self.dropout = dropout
        self.alpha = alpha
    def setAlpha(self,alpha):
        self.alpha = alpha
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.m(self.fc2(x))) if self.dropout else F.leaky_relu(self.fc2(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(self.alpha*F.tanh(x/self.alpha))

def select_action(state,policy,past=None, return_probs=False):
    probs = policy(torch.Tensor([state])) if past is None else policy(torch.Tensor([state+tuple([past])]))
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    if not return_probs:
        return action.item(), m.log_prob(action)
    else:
        return action.item(), m.log_prob(action), probs

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10))

class MicroMLP(nn.Module):
    def __init__(self,alpha=3,markovian=True):
        super(MicroMLP, self).__init__()
        self.fc1 = nn.Linear(2, 3)  if not markovian else nn.Linear(1,3)
        self.alpha = alpha
    def setAlpha(self,alpha):
        self.alpha = alpha
    def forward(self, x):
        x = self.fc1(x)
        return F.softmax(x)

class NanoMLP(nn.Module):
    def __init__(self,alpha=3,markovian=True):
        super(NanoMLP, self).__init__()
        self.fc1 = nn.Linear(2, 2)  if not markovian else nn.Linear(1,2)
        self.alpha = alpha
    def setAlpha(self,alpha):
        self.alpha = alpha
    def forward(self, x):
        x = self.fc1(x)
        return F.softmax(x)


class Electric_MLP(nn.Module):
    def __init__(self,dropout=True,alpha=2,markovian=True):
        super(Electric_MLP, self).__init__()
        self.fc1 = nn.Linear(5, 64)  if markovian else nn.Linear(6,10)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.m = nn.Dropout(p=0.2)
        self.dropout = dropout
        self.alpha = alpha
    def setAlpha(self,alpha):
        self.alpha = alpha
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc3(x))
        #x = F.leaky_relu(self.m(self.fc2(x))) if self.dropout else F.leaky_relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.fc3(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.cat((x[:, :1],(x[:, 1:])),dim=1)
        #return F.relu(self.fc3(x))

def select_action_continuous(state,policy,past=None,s=0.1):
    output= policy(torch.Tensor([state])) if past is None else policy(torch.Tensor([state+tuple([past])]))
    mu, sigma = output[:, :1],output[:, 1:]
    m = torch.distributions.Normal(mu[:, 0],s)#  *(1+5*F.sigmoid(sigma[:, 0])) torch.abs(sigma[:, 0])
    action = m.sample()
    return action.item(), m.log_prob(action),torch.log(sigma[:, 0])