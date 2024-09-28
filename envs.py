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

#Constants used for Grid Worlds
UP, DOWN, LEFT, RIGHT = 0,1,2,3
SIZE = 4
arrows = {UP:"↑",DOWN:"↓",LEFT:"←",RIGHT:"→"}
class GridWorld:
    def __init__(self, height =SIZE, width =SIZE, starting_cell = None, goals = None,gaussian=False):
        self.height = height
        self.width = width
        self.x,self.y = starting_cell if starting_cell is not None else (0,0)
        self.goals = [(0,self.height-1),(self.width-1,self.height-1)] if goals is None else goals
        self.gaussian = gaussian
    def accessible(self,x,y):
        if x<0 or y<0 or x>=self.width or y>=self.height:
            return False
        return True
    def result(self):
        if self.x ==0 and self.y == self.height-1:
            return (self.x,self.y), 5, True,False,None
        if self.x ==self.width-1 and self.y == self.height-1:
            return ((self.x,self.y), 6, True, False, None) if not self.gaussian else ((self.x,self.y), np.random.normal(10,10), True, False, None)
        return ((self.x,self.y), choice([-1,0.8]), False, False, None) if not self.gaussian else ((self.x,self.y), np.random.normal(-0.2,1), False, False, None)
    def step(self,action):
        x,y = self.x,self.y
        if action ==UP:
            y-=1
        if action ==DOWN:
            y+=1
        if action == LEFT:
            x-=1
        if action == RIGHT:
            x+=1

        if self.accessible(x,y):
            self.x,self.y = x,y
            return self.result()
        else :
            #return (self.x,self.y), -1, False, False, None
            return (self.x,self.y),-1,True,False,None
    def reset(self,starting_cell=None):
        self.x,self.y = starting_cell if starting_cell is not None else (0,0)
        return (0,0), None
    def display(self,policy=None,threshold=0.9):
        # First row
        print(f"  ", end='')
        for j in range(self.width):
            print(f"| {j} ", end='')
        print("| ")
        print((self.width*4+4)*"-")

        # Other rows
        for i in range(self.height):
            print(f"{i} ", end='')
            for j in range(self.width):
                if (j,i) in self.goals:
                    if (j,i)==(0,self.height-1):
                        print("|+5 ", end='')
                    elif (j,i)==(self.width-1,self.height-1):
                        print("|+6 ", end='')
                    else: print("| G ", end='')
                elif policy is not None:
                    probs = policy(torch.tensor([(j+0.,i+0.)]))
                    probs= probs.tolist()[0]
                    symbol = "?"
                    for k in range(len(probs)):
                        if probs[k]>threshold:
                            symbol = arrows[k]
                    print(f"| {symbol} ", end='')
                elif j==self.x and i==self.y:
                    print("| X ", end='')
                else:
                    print("|   ", end='')
            print("| ")
            print((self.width*4+4)*"-")


class ToyWorld:
    def __init__(self):
        self.state = 0
    def reset(self):
        self.state = 0
        return tuple([0]),None
    def step(self,action):
        if self.state ==0:
            self.state = 1
            return tuple([1]), choice([-1,1]), False, False, None
        elif action==0:
            return tuple([2]), 1,True,False,None
        elif action==2:
            return tuple([2]), 0,True,False,None
        else:
            return tuple([2]), choice([0,3]), True, False, None

class NanoWorld:
    def __init__(self):
        self.state = 0
    def reset(self):
        self.state = 0
        return tuple([0]),None
    def step(self,action):
        if action==0:
            return tuple([2]), 1, True, False, None
        else:
            return tuple([1]), choice([0,1.5]), True, False, None

def randomSquareCity():
    x = choice([0,1,2,3])
    y = choice([0,1,2,3])
    if x==3 and y==2:
        return randomSquareCity()
    return (x,y)

class BusyCity(GridWorld):
    def __init__(self, height =SIZE, width =SIZE, starting_cell = None, goals = None,gaussian=False, random_start=True):
        self.height = height
        self.width = width
        self.random_start = random_start
        self.x,self.y = randomSquareCity() if random_start else (0,1)
        self.goals = [(3,2)]
        self.gaussian = False

    def reset(self,starting_cell=None):
        self.x,self.y = randomSquareCity() if self.random_start else (0,1)
        return (self.x,self.y), None

    def step(self,action):
        x,y = self.x,self.y
        if action ==UP:
            y-=1
        if action ==DOWN:
            y+=1
        if action == LEFT:
            x-=1
        if action == RIGHT:
            x+=1

        if self.accessible(x,y):
            self.x,self.y = x,y
            if x ==3 and y == 2:
                return (self.x,self.y), 10-1, True,False,None
            if (x==1 and y==1 and action==LEFT) or (x==1 and y==1 and action==UP) or (x==2 and y==1 and action==RIGHT) or (x==2 and y==1 and action==UP) or (x==1 and y==2 and action==LEFT) or (x==1 and y==2 and action==DOWN) or (x==2 and y==2 and action==RIGHT) or (x==2 and y==2 and action==DOWN):
                return (self.x,self.y), -16 if random()<0.1 else -1, False,False,None
            return ((self.x,self.y), -1, False, False, None)
        else :
            return (self.x,self.y),-20,True,False,None

def randomSquareTown():
    x = choice([0,1,2])
    y = choice([0,1,2])
    if x==2 and y==1:
        return randomSquareTown()
    return (x,y)

class BusyTown(GridWorld):
    def __init__(self, height =3, width =3,random_start=True, starting_cell = None, goals = None,gaussian=False):
        self.height = height
        self.width = width
        self.random_start = random_start
        self.x,self.y = randomSquareTown() if random_start else (0,1)
        self.goals = [(2,1)]
        self.gaussian = False

    def reset(self,starting_cell=None):
        self.x,self.y = randomSquareTown() if self.random_start else (0,1)
        return (self.x,self.y), None

    def step(self,action):
        x,y = self.x,self.y
        if action ==UP:
            y-=1
        if action ==DOWN:
            y+=1
        if action == LEFT:
            x-=1
        if action == RIGHT:
            x+=1

        if self.accessible(x,y):
            self.x,self.y = x,y
            if x ==2 and y == 1:
                return (self.x,self.y), 6, True,False,None
            if 1==y and x==1:
                return (self.x,self.y), -15 if random()<0.1 else -1, False,False,None
            return ((self.x,self.y), -1, False, False, None)
        else :
            return (self.x,self.y),-20,True,False,None #!!

selling_price_mwh = [81.53,70.19,56.12,46.93,46.16,42.96,44.48,47.76,36.95,34.65,12.54,3.59,2.68,0.03,0,0.8,5.36,14.93,47.06,87.04,66.72,76.79,101.53,84.18]
selling_price = [0.001*elt for elt in selling_price_mwh]
buying_price = 0.253

def sun_intensity(time):
    if time<=6 or time>=18 :
        return 0
    else:
        return sin((time-6)*pi/12)

#%%
def sun_intensity_random(time): #We take a deterministic one here
    x = 5*sun_intensity(time)
    return choice([x])

def consumption(time):
    return max(0, np.random.normal(0.5,0.5))

class Electric:
    def __init__(self,battery_capacity=15):
        self.t = 0
        self.capacity = battery_capacity
        self.charge = choice([0])
        self.sun_intensity = sun_intensity_random((self.t+6)%24)
        self.consumption = consumption((self.t+6)%24)
    def state(self):
        return (self.t,self.charge,self.sun_intensity,self.consumption,selling_price[(self.t+6)%24])
    def reset(self):
        self.t = 0
        self.charge = choice([0])
        self.sun_intensity = sun_intensity_random((self.t+6)%24)
        self.comsumption = consumption((self.t+6)%24)
        return self.state(),None
    def step(self,action):
        self.net_production = -self.consumption+self.sun_intensity
        reward = 0

        if action>self.charge+self.net_production:
            reward -= 0.1*(action-(self.charge+self.net_production))
            action = self.charge+self.net_production


        if action>0:
            reward += action*selling_price[(self.t+6)%24] #!!
        else:
            reward += action*buying_price
        self.charge = min(self.charge+self.net_production-action,self.capacity)
        self.t += 2
        self.sun_intensity = sun_intensity_random((self.t+6)%24)
        self.consumption = 2*consumption((self.t+6)%24)
        state = self.t,self.charge
        return self.state(), reward, (self.t==24), False, None


class DiagonalTown(GridWorld):
    """The scalable environment used for the algorithm comparison example"""
    def __init__(self, size,random_start=False,format=True):
        self.height = size
        self.width = size
        self.random_start = random_start
        self.x,self.y = (0,0)
        self.goals = [(size-1-i,i) for i in range(self.height)]
        self.format = (lambda x,y:(x,y)) if format else (lambda x,y: self.height*x+y)

    def reset(self,starting_cell=None):
        self.x,self.y = (0,0)
        return self.format(self.x,self.y), None

    def display(self,x):
        pass

    def step(self,action):
        x,y = self.x,self.y
        if action ==UP:
            y-=1
        if action ==DOWN:
            y+=1
        if action == LEFT:
            x-=1
        if action == RIGHT:
            x+=1

        if self.accessible(x,y):
            self.x,self.y = x,y
            if x+y==self.height-1:
                return self.format(self.x,self.y), choice([4*x/self.height,4*y/self.height]), True,False,None
            return (self.format(self.x,self.y), -1/self.height, False, False, None)
        else :
            return self.format(self.x,self.y),-2/self.height,False,False,None #!!