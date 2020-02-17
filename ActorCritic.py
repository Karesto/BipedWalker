import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from gym.wrappers import Monitor

#Actor + Critic.
#In this method, we train a Neural Network to output the result (Actor)
#and a second NN (Critic), to assess it's potential

env = gym.make('BipedalWalker-v2')


h_act = [64,128]
h_cri = [64,128,64]
state_size  = env.observation_space.shape[0]
action_size = 4

class Actor(nn.Module):
    def __init__(self,h):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, h[0])
        self.fc2 = nn.Linear(h[0], h[1])
        self.fc3 = nn.Linear(h[1], action_size)
        self.dr = nn.Dropout(p = 0.05)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu((self.fc1(state)))
        x = F.relu((self.dr(self.fc2(x))))
        return F.torch.tanh(self.fc3(x))



class Critic(nn.Module):
    def __init__(self,h):
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_size, h[0])
        self.fcs2 = nn.Linear(h[0], h[1])
        self.fca1 = nn.Linear(state_size, h[2])
        self.fc1  = nn.LSTM(h[1]+h[2], 1)
        self.h = h
    def forward(self, state, action,k):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        xs = F.relu((self.fcs1(state)))
        xs = self.fcs2(xs)
        xa = self.fca1(action)

        x = F.relu(torch.cat([xs, xa], dim = 1))
        return self.fc1(x.view(1,-1,self.h[1]+self.h[2]),k)



actor  = Actor(h_act)
critic = Critic(h_cri)

gamma = 0.99

lr_act = 0.1
lr_crit= 4
act_optimizer = optim.Adam(actor.parameters(),  lr = lr_act)
cri_optimizer = optim.Adam(critic.parameters(), lr = lr_crit)

def actandcriticize(rewards,states,actions,h):
    #For thy Acting

    #print(states,actions,rewards)

    loss_actor = -torch.sum(critic.forward(states, states,h)[0])
    #loss_actor = - torch.sum(critic.forward(states , actions)) + (5)*torch.sum(critic.forward(states[1:] , actions[1:]))
    act_optimizer.zero_grad()
    loss_actor.backward(retain_graph= True)
    act_optimizer.step()
    R = 0
    rew = []
    for r in rewards.numpy()[::-1]:
        R = r + gamma * R
        rew.insert(0,R)

    #For thy critic
    rew = torch.tensor(rew)
    pred_y = critic.forward(states, states,h)[0]
    loss_critic = torch.abs(torch.sum(pred_y - torch.sum(rew)))**2
    cri_optimizer.zero_grad()
    loss_critic.backward()
    cri_optimizer.step()


scs=[]

def Train(episodes):
    timing = 350
    env = gym.make('BipedalWalkerHardcore-v2')
    # Does Literally everything
    h = None

    for episode in range(episodes):
        if episode%50 == 0:
            env = Monitor(env, 'Video13/'+ str(episode), force = True)
        state = env.reset()

        state = torch.from_numpy(state).type(torch.FloatTensor)
        done = False
        # Saving rewards, states, and actions to compute later
        rewards = []
        states  = [state.view(-1,24)]
        actions = []
        for time in range(timing):
            action = actor(state)
            #if episode%30 == 0:
                #env.render()
            state, reward, done, _ = env.step(action.detach().numpy())
            state = torch.from_numpy(state).type(torch.FloatTensor)
            #Saving
            if (state[2].item()) < 0.005 or np.abs(state[3].item()) < 0.005:
                rewards.append(torch.tensor(-20*1.0).view(-1))
            if (state[2].item()) > 0.5:
                rewards.append(torch.tensor(11*1.0).view(-1))

            elif (reward) > 0 or np.abs(reward) > 4.5:
                rewards.append(torch.tensor(reward*1.0*12).view(-1))
            else: rewards.append(torch.tensor(reward*1.0).view(-1))
            states.append(state.view(-1,24))
            actions.append(action.view(-1,4))

            if done:
                rewards[-1] = rewards[-1]*(350-time)
                break
        #for i in range(len(rewards)):
        #    rewards[i] /= time
        env.close()
        states.pop()

        states = torch.cat(states)
        rewards= torch.cat(rewards)
        actions= torch.cat(actions)

        h = actandcriticize(rewards,states,actions,h)
        loss_actor = -torch.sum(critic.forward(states, states,h)[0])

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tLast reward: {:.2f}'.format(episode, time, loss_actor))
            global scs
            scs.append(torch.sum(rewards))

        if episode % 100 == 0:
            with torch.no_grad():
                for param in actor.parameters():
                    param.add_(torch.randn(param.size())*0.8)
                for param in critic.parameters():
                    param.add_(torch.randn(param.size())*0.41)


episodes = 25001
Train(episodes)

plt.plot(scs)
plt.show()



with torch.no_grad():
    obs=env.reset()
    for i in range(500):
        h=None
        env.render()
        act,h=select_action(obs,h)
        obs,rew,done,info=env.step(act)
        if done==True:
            break
env.close()
