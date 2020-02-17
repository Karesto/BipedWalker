import gym
import random
import numpy as np

def essai(pol):
    env.reset()
    for i in range(200):
        env.render()
        obs=env.env.state
        act=0 if np.matmul(pol,obs)>0 else 1
        obs,rew,done,info=env.step(act)
        if done==True:
            return i

def test(pol):
    env.reset()
    for i in range(200):
        env.render()
        obs=env.env.state
        act=0 if np.matmul(pol,obs)>0 else 1
        obs,rew,done,info=env.step(act)
        if done==True:
            return

env = gym.make('CartPole-v0')
pool=[np.random.rand(4)*2-1 for i in range(10)]
avg=0
for i in range(100):
    ch=np.random.randint(len(pool))
    pol=pool[ch]
    rew=essai(pol)
    avg=(avg*i+rew)/(i+1)
    tar=avg
    if len(pool)>10:
        tar*=len(pool)/10
    print(rew,avg,len(pool),tar)
    if rew>tar:
        pool.append(pol+np.random.rand(4)*0.5-0.25)
    else:
        pool.pop(ch)
        #pool.append(np.random.rand(4)*2-1)

env.close()