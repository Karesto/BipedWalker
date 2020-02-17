import gym
import random
import numpy as np
import matplotlib.pyplot as plt

def noise(n,f):
    return np.random.rand(n)*2*f-f

def app_pol(obs,pol):
    return 0 if np.matmul(pol,obs)>0 else 1

def essai(pol):
    obs=env.reset()
    for i in range(200):
        act=app_pol(obs,pol)
        obs,rew,done,info=env.step(act)
        if done==True:
            return i

def essai_render(pol):
    obs=env.reset()
    for i in range(200):
        env.render()
        act=app_pol(obs,pol)
        obs,rew,done,info=env.step(act)
        if done==True:
            return i

env = gym.make('CartPole-v0')

s=6
pool=[noise(4,1) for i in range(3*s)]
avgs=[]
for i in range(55):
    res=[]
    avg=0
    for j in range(3*s):
        sc=essai(pool[j])
        avg+=sc
        res.append((sc,j))
    avg/=(3*s)
    print(i,avg)
    avgs.append(avg)
    res.sort()
    poolTmp=[]
    for j in range(s,3*s):
        poolTmp.append(pool[res[j][1]])
    for j in range(2*s,3*s):
        poolTmp.append(pool[res[j][1]]+noise(4,0.25))
    pool=poolTmp
    
plt.plot(avgs)
plt.show()

env.close()