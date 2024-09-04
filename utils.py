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

def trial(policy,display=True):
    env = Electric()
    observation, info = env.reset()
    past = 0
    Lsun,Lcharge,Lt,Lconsumption,Laction, Lprice= [],[],[],[],[],[]
    observation,info = env.reset()
    t, charge,sun, cons,price = observation
    Lsun.append(sun)
    Lt.append(t)
    Lconsumption.append(cons)
    Lcharge.append(charge)
    Lprice.append(price)
    for t in range(100):  # Run for a max of 100 timesteps
        action, log_prob,_ = select_action_continuous(observation,policy,s=0.001)
        observation, reward, terminated, truncated, info = env.step(action)
        past += reward
        t, charge,sun, cons,price = observation
        Lsun.append(sun)
        Lt.append(t)
        Laction.append(action)
        Lconsumption.append(cons)
        Lcharge.append(charge)
        Lprice.append(price)
        if terminated or truncated:
            observation, info = env.reset()
            break
    Laction.append(0)
    if display:
        plt.plot(Lt,Lsun)
        plt.plot(Lt,Laction,color="red")
        plt.plot(Lt,Lconsumption)
        plt.plot(Lt,Lcharge)
        plt.plot(Lt,[100*elt for elt in Lprice],color="pink")
        plt.show()
    return past,Laction

def w_approx_aux(L,x):
    n = (len(L)+1)//3
    breaking_points = L[2*n:]
    for i, elt in enumerate(breaking_points):
        if x < elt:
            return L[2*i]*x + L[2*i+1]
    i = n-1
    return L[2*i]*x + L[2*i+1]
def w_approx(L):
    return (lambda x:w_approx_aux(L,x))

def smooth(L,x):
  try:
    return [ sum(L[i:i+x])/x for i in range(len(L)-x)]
  except Exception:
    return None

LINEAR = 0
wp_three_segments = (LINEAR, [ 1.94671279,  0.        ,  0.69678392,  0.11132902,  2.70977837,
       -1.70977837,  0.08906833,  0.90467621])
wp_counterexample = (LINEAR, [5,0, 0.5555555555555555, 0.44444444444, 0.1])
w_neutral = (0, (1, 0, 1, 0, 0.5))
wp_new = (0, [0.5, 0, 5.5, -4.5, 0.9])
risk_averse_w = wp_new
def KT(z0=0,l=2.25,alpha=0.8):
    return (lambda x: (x-z0)**alpha if x>= z0 else -l*(z0-x)**alpha)

def select_action(state,policy,past=None, return_probs=False,action_direct=False):
    if action_direct:
        probs = policy(state)
    else:
        probs = policy(torch.Tensor([state])) if past is None else policy(torch.Tensor([state+tuple([past])]))
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    if not return_probs:
        return action.item(), m.log_prob(action)
    else:
        return action.item(), m.log_prob(action), probs

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10))