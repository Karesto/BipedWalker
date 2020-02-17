import gym
import random
import numpy as np
import matplotlib.pyplot as plt

h=3

def sigmoid(logp):
    return 1/(1+np.exp(-logp))

def back_sigmoid(logp):
    return logp*(1-logp)

def pol_forward(pol,obs):
    n1=np.dot(pol['W1'],obs)
    n1[n1<0]=0
    logp=np.dot(pol['W2'],n1)
    return logp,n1

def pol_backward(pol,obs,n1,logp):
    dlogp=back_sigmoid(logp)
    dW2=dlogp*n1
    dn1=pol['W2']*dlogp
    dn1[n1<0]=0
    dn1=dn1.reshape(3,1)
    obs=obs.reshape(4,1)
    dW1=np.dot(dn1,obs.T)
    g={}
    g['W1']=dW1
    g['W2']=dW2
    return g

def pol_step(pol,grad,lr):
    pol['W1']+=grad['W1']*lr
    pol['W2']+=grad['W2']*lr

def train(lr,pol,avg):
    obs=env.reset()
    actions=[]
    grads=[]
    for i in range(200):
        #env.render()
        logp,n1=pol_forward(pol,obs)
        p=sigmoid(logp)
        if np.random.random()>p:
            act=0
        else:
            act=1
        actions.append(act)
        grads.append(pol_backward(pol,obs,n1,logp))
        obs,rew,done,info=env.step(act)
        if done==True:
            score=(i-avg)/avg
            if score>1:
                score=1
            #score=-score
            for s in range(i+1):
                if actions[s]==0:
                    pol_step(pol,grads[s],-lr*score*0.98**(i-s))
                else:
                    pol_step(pol,grads[s],lr*score*0.98**(i-s))
            return i


env = gym.make('CartPole-v0')
pol={}
pol['W1']=np.random.randn(h,4)
pol['W2']=np.random.randn(h)
avg=10
avg100=10
scs=[]
avgs=[]
for i in range(100000):
    sc=train(0.003/avg100,pol,avg100)
    avg=(avg*i+sc)/(i+1)
    scs.append(sc)
    if i<100:
        avg100=(avg100*i+sc)/(i+1)
    else:
        avg100+=(sc-scs[-100])/100
    avgs.append(avg100)
    print(i,sc,avg)
plt.plot(avgs)
plt.show()

env.close()