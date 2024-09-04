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

def train(policy, optimizer, adaptive_baseline=False,phi_baseline=False, entropy_reg=None,use_dict=False,num_episodes=100000, draw_phi=False,batch_size = 10, gamma=1,baseline = 0,log_interval=1, utility=None, draw_policy=False,random_start=False, p_exp = 0.3,display_threshold=None,changing_alpha=False,changing_entropy=False, entropy_max=3, entropy_min=0.5,markovian=True, env=None,gaussian=False, continuous=False, changing_s=False,s=0.1, return_best=True, w=None, reinforce=False):
    Rsum =0
    Usum = 0
    Ssum = 0
    L = []
    Lu = []
    Lx = []
    Lcpt = []
    best_u = float("-inf")
    best_dict = None
    T = 0
    if utility is None:
        utility = lambda x:x
    if env is None:
        env = GridWorld(gaussian=gaussian) if not random_start else GridWorld(gaussian=gaussian,starting_cell=(choice(list(range(SIZE))),choice(list(range(SIZE-1)))))
    exploration = int(0.9*num_episodes)
    for episode in range(num_episodes):
        if changing_alpha and episode<= exploration:
            policy.setAlpha(0.5+4*(episode+1)/exploration)
        if changing_s and episode<=exploration:
            s = 0.5*exp(-3*episode/exploration)
        if changing_entropy:
            entropy_reg = episode*(entropy_min-entropy_max)/num_episodes + entropy_max
        policy_loss = []
        utility_list = []
        trajectory_utilities = []
        entropy_list = []
        for j in range(batch_size):
            log_probs = []
            rewards = []
            observation, info = env.reset() if not random_start else env.reset(starting_cell=(choice(list(range(SIZE))),choice(list(range(SIZE-1)))))
            past = 0
            for t in range(100):  # Run for a max of 100 timesteps
                if (not continuous) and (entropy_reg is not None):
                    action, log_prob, probs = select_action(observation,policy,return_probs=True) if markovian else select_action(observation,policy,past,return_probs=True)
                    entropy_list.append(compute_entropy(probs))
                if not continuous:
                    action, log_prob = select_action(observation,policy) if markovian else select_action(observation,policy,past)
                else:
                    action, log_prob, sigma = select_action_continuous(observation,policy,s=s) if markovian else select_action(observation,policy,past)
                    entropy_list.append(sigma)
                observation, reward, terminated, truncated, info = env.step(action)
                past += reward
                rewards.append(reward)
                log_probs.append(log_prob)
                if terminated or truncated:
                    observation, info = env.reset()
                    break

            T +=t+1
            # Compute the cumulative discounted rewards
            R = 0
            returns = []
            for r in rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
            #Rsum += 1-exp(-0.1*R)
            Rsum += R
            Usum += utility(R)
            total_loss = exp(-0.01*R)
            returns = torch.tensor(returns)


            if adaptive_baseline:
                baseline = Lu[-1] if len(Lu)>0 else 0
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob)
                if not reinforce:
                  utility_list.append((utility(returns[0])))
                else:
                  utility_list.append(R)
            trajectory_utilities.append(utility(returns[0]))
                #policy_loss.append(-log_prob * (utility(returns[0])-baseline))#returns[0]


        trajectory_utilities = [(elt.item() if isinstance(elt,torch.Tensor) else elt )for elt in trajectory_utilities]
        phi = None
        beta = 1
        if w is None:
            phi = lambda x:x
        else:
            function_type, parameters = w
            if function_type==LINEAR:
                n =(len(parameters)+1) //3
                breaking_points = parameters[2*n:]

                distribution = sorted(trajectory_utilities)[::-1]
                distribution_losses = sorted([-elt for elt in trajectory_utilities])[::-1]

                qtilde = [distribution[int(batch_size*elt)] for elt in breaking_points]
                #print(f"distribution[0] is {distribution[0]}")
                qtilde.insert(0,distribution[0]+1)
                qtilde = [0 if elt<0 else elt for elt in qtilde]

                breaking_points_negative = [1-elt for elt in breaking_points]
                qtilde_losses = [distribution_losses[int(batch_size*elt)] for elt in breaking_points_negative]
                qtilde_losses.insert(0,distribution_losses[0]+1)
                qtilde_losses = [0 if elt<0 else elt for elt in qtilde_losses]
                memo = {}
                def phi(arg):
                  if not use_dict: return(phi_aux(arg))
                  if arg not in memo:
                    memo[arg] = phi_aux(arg)
                  return memo[arg]
                def phi_aux(arg):
                    res = 0
                    if arg>=0:
                        previous = 0
                        i = 1
                        while previous != -1:
                            beta = parameters[2*(n-i)]
                            #print(i)
                            #print(f"Arg is {arg}, qtilde is {qtilde}, i is {i}")
                            if arg> qtilde[-i]:
                                res, previous = res+beta*(qtilde[-i]-previous), qtilde[-i]
                            else:
                                return res+beta*(arg-previous)
                            i +=1
                    else :
                        arg = -arg
                        previous = 0
                        i = 1
                        while True:
                            beta = parameters[2*(i-1)]
                            #print(f"Arg is {-arg}, qtilde_losses is {qtilde_losses}, i is {i}")
                            if arg> qtilde_losses[-i]:
                                res, previous = res+beta*(qtilde_losses[-i]-previous), qtilde_losses[-i]
                                #print(f" res is {res}, previous is {previous}")
                            else:
                                #print(f" res is {res}, previous is {previous}")
                                #print(f"We return {-(res+beta*(arg-previous))}")
                                return -(res+beta*(arg-previous))
                            i +=1


            else:
                raise Exception("Unknown w function")
        phi_list = [phi(b) for b in utility_list]

        if phi_baseline:
            zL = [phi(elt) for elt in trajectory_utilities]
            x = sum(zL)/len(zL)
            phi_list = [elt-x for elt in phi_list]

        policy_loss = [a*phib for a,phib in zip(policy_loss,phi_list)]

        if entropy_reg is not None:
            policy_loss = [ a - entropy_reg*b for a,b in zip(policy_loss,entropy_list)]

        policy_loss = torch.sum(torch.stack(policy_loss), dim=0)




        if (episode+1) % log_interval == 0 :
            if w is not None:
                #trajectory_utilities = [elt.item() for elt in trajectory_utilities]
                segments = sorted(list(set(trajectory_utilities+[0])))
                negative_segments = sorted([-elt for elt in segments if elt<= 0])
                positive_segments = sorted([elt for elt in segments if elt>=0])
                #print(segments)
                res = 0
                if len(segments)==0:
                    break
                #Integral on gains
                for i in range(len(positive_segments)-1):
                    res+= w_approx(parameters)(sum([elt>positive_segments[i] for elt in trajectory_utilities])/batch_size)*(positive_segments[i+1]-positive_segments[i])
                #Integral on losses
                for i in range(len(negative_segments)-1):
                    res-= (1-w_approx(parameters)(1-sum([(-elt)>negative_segments[i] for elt in trajectory_utilities])/batch_size))*(negative_segments[i+1]-negative_segments[i])

                Lcpt.append(res)
            if draw_policy :
                clear_output(wait=True)
                if w is not None:
                    print(f'Batch n°{episode}\tAverage length: {T/batch_size/log_interval}\tMean return: {Rsum/batch_size/log_interval} \tMean utility: {Usum/batch_size/log_interval}\tMean CPT value: {Lcpt[-1]}\tEntropy coefficient: {entropy_reg}')
                else:
                    print(f'Batch n°{episode}\tAverage length: {T/batch_size/log_interval}\tMean return: {Rsum/batch_size/log_interval} \tMean utility: {Usum/batch_size/log_interval}')
                if isinstance(env,type(GridWorld())):
                    env.display(policy) if display_threshold is None else env.display(policy,threshold=display_threshold)
            Lx.append(episode*batch_size)
            L.append(Rsum/batch_size/log_interval)
            Lu.append(Usum/batch_size/log_interval)
            if w is None:
              if return_best and (Usum/batch_size/log_interval)>=best_u:
                best_u = Usum/batch_size/log_interval
                best_dict = copy.deepcopy(policy.state_dict())
                if draw_policy: print("New best!")
            else:
              if return_best and (Lcpt[-1])>=best_u:
                best_u = Lcpt[-1]
                best_dict = copy.deepcopy(policy.state_dict())
                if draw_policy: print("New best!")
            Rsum = 0
            Usum = 0
            T = 0

            if draw_phi:
                X = [0.0015*i for i in range(1000)]
                plt.plot(X, [phi(x) for x in X])
                plt.title("$\phi$")
                plt.show()
            if draw_policy:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                # Plot data on the first subplot
                ax1.plot(Lx,L)  # 'r-' means red solid line
                ax1.set_title(f'Mean return, calculated per {log_interval*batch_size} episodes')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Mean return')

                # Plot data on the second subplot
                ax2.plot(Lx,Lu,label="Mean utility")  # 'b-' means blue solid line
                if w is not None:
                    ax2.plot(Lx,Lcpt, label="Mean CPT value")  # 'b-' means blue solid line      w_approx(parameters)
                ax2.set_title(f'Mean utility, calculated per {log_interval*batch_size} episodes')
                ax2.set_xlabel('Episode')
                ax2.legend()



                # Adjust layout to prevent overlap
                plt.tight_layout()

                # Show the plots
                plt.show()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    if return_best:
        policy.load_state_dict(best_dict)
        print("Loading best state")
    return L,Lu,Lx,Lcpt


def array_sum(a,b):
  return [x+y for x,y in zip(a,b)]


def rademacher(n):
  return [choice([-1,1]) for _ in range(n)]

def estimate_cpt_value(env,policy,batch_size,gamma,utility,w):
    if utility is None:
        utility = lambda x:x
    _, parameters = w
    trajectory_utilities = []
    for j in range(batch_size):
        rewards = []
        observation, info = env.reset()
        past = 0
        for t in range(10):
            action =  policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)

            if terminated or truncated:
                observation, info = env.reset()
                break
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
        trajectory_utilities.append(utility(R))

    trajectory_utilities = [(elt.item() if isinstance(elt,torch.Tensor) else elt )for elt in trajectory_utilities]
    segments = sorted(list(set(trajectory_utilities+[0])))
    negative_segments = sorted([-elt for elt in segments if elt<= 0])
    positive_segments = sorted([elt for elt in segments if elt>=0])
    res = 0
    if len(segments)==0:
        return sum(trajectory_utilities)/len(trajectory_utilities)
    #Integral on gains
    for i in range(len(positive_segments)-1):
        res+= w_approx(parameters)(sum([elt>positive_segments[i] for elt in trajectory_utilities])/batch_size)*(positive_segments[i+1]-positive_segments[i])
    #Integral on losses
    for i in range(len(negative_segments)-1):
        res-= (1-w_approx(parameters)(1-sum([(-elt)>negative_segments[i] for elt in trajectory_utilities])/batch_size))*(negative_segments[i+1]-negative_segments[i])
    return res, sum(trajectory_utilities)/len(trajectory_utilities)

def tabular_policy_aux(theta,state):
    x,y = state
    state = x+3*y
    return choices([UP,DOWN,LEFT,RIGHT],weights=[exp(theta[state]),exp(theta[state+1]),exp(theta[state+2]),exp(theta[state+3])])[0]
def tabular_policy(theta):
    return (lambda x:tabular_policy_aux(theta,x))


def clip(n,t):
    return max(min(n,t),-t)


def train_spsa(policy_family,env, num_episodes=100000, batch_size = 10, gamma=1, utility=None, w=None,eta = 0.2,alpha=0.1):
    Rsum =0
    Usum = 0
    Ssum = 0
    L = []
    Lu = []
    Lx = []
    Lcpt = []


    if utility is None:
        utility = lambda x:x

    theta = [0 for i in range(36)]
    for episode in range(num_episodes):
        print(episode)
        delta = rademacher(36)
        theta_plus = [x + eta*y for x,y in zip(theta,delta)]
        theta_minus = [x - eta*y for x,y in zip(theta,delta)]

        cpt_plus, mean_plus = estimate_cpt_value(env,policy_family(theta_plus),batch_size,gamma,utility,w)
        cpt_minus, mean_minus = estimate_cpt_value(env,policy_family(theta_minus),batch_size,gamma,utility,w)
        nabla = [(cpt_plus-cpt_minus)/(2*eta)*elt for elt in delta] #Dividing by delta and multiplying by delta is the same thing
        nabla_clipped = [max(min(elt,2),-2) for elt in nabla]
        theta = [x + alpha*y for x,y in zip(theta,nabla_clipped)]

        theta = [clip(elt,5) for elt in theta]


        cpt,mean = estimate_cpt_value(env,policy_family(theta),batch_size,gamma,utility,w)
        Lcpt.append(cpt)
        Lu.append(mean)
        X = [2*batch_size*i for i in range(len(Lcpt))]
        clear_output(wait=True)
        plt.plot(X,Lu,label="Mean utility")
        plt.plot(X,Lcpt, label="Mean CPT value")


        plt.legend()
        plt.show()

    return Lcpt
