from utils import *
from training import *
from policies import *
from envs import *
def trial(policy,display=True):
    """Doing one test run of a policy on the Electricity environnement and returning the relevant data"""
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
def distribution(policy_arg):
    L_score = []
    for j in range(10000):
      score,L_action = trial(policy_arg,display=False)
      L_score.append(score)
    return L_score