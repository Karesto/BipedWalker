import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical



env = gym.make('BipedalWalker-v2')


#Hyperparameters
learning_rate = 0.2
gamma = 0.99
h = [32,32,32,32]
class Policy(nn.Module):
    def __init__(self,h):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = 4

        self.l1 = nn.Linear(self.state_space, h[0])
        self.l2 = nn.Linear(h[0], h[1])
        self.l3 = nn.Linear(h[1], h[2])
        self.l4 = nn.Linear(h[2], h[3])
        self.ls = nn.LSTM(h[2] ,self.action_space)
        self.re = nn.ReLU()
        self.sm = nn.Softmax()
        self.th = nn.Tanh()
        self.gamma = gamma
        self.dr = nn.Dropout(p = 0.05)

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x,h):
        res1 = self.dr(self.re(self.l2(self.re(self.l1(x).view(1,1,-1)))))
        res, h = self.ls(res1,h)
        return self.th(res.view(self.action_space)), h


policy = Policy(h)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action(state,h):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    action,h = policy(Variable(state),h)
    #c = Categorical(state)
    #action = c.sample()
    # Add log probability of our chosen action to our history
    if policy.policy_history.nelement() != 0:

        policy.policy_history = torch.cat([policy.policy_history, action.view(-1,1)], dim = 1)
    else:
        policy.policy_history = action.view(-1,1)

    return action,h


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)

    # Scale rewards
    rewards = torch.tensor(rewards)

    #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    # Calculate loss
    #rew = np.array(policy.reward_episode)
    rew = rewards.numpy()
    for i in range(len(rew)):
        if    rew[i] > 0 : rew[i] = torch.tensor(np.random.randint(2)*2-1) * min(0.1,1/rew[i]) *0
        elif  rew[i] < 0 : rew[i] = torch.tensor(np.random.randint(2)*2-1) * (rew[i])*4
        else: rew[i] = 0

    loss = -torch.sum(policy.policy_history.double() * torch.tensor(rew).double())
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode= []


def essai():
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

scs=[0]
def main(episodes):
    running_reward = 0

    rewlist = []
    true_r = []
    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        h = None
        done = False
        corr_r = []
        for time in range(250):
            if episode%30 == 0:
                env.render()
            action,h = select_action(state,h)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.detach().numpy())
            corr_r.append(reward)
            # Save reward
            if not done:
                policy.reward_episode.append(reward*time)
            if done:
                policy.reward_episode.append(reward)
                env.close()
                break
        env.close()


        true_r.append(np.sum(corr_r))
        rewlist.append(np.sum(policy.reward_episode))
        if len(rewlist) > 50:
            rewlist.pop(0)
            true_r.pop(0)
        # Used to determine when the environment is solved.
        running_reward = np.sum(rewlist)/ len(rewlist)
        trunning = np.sum(true_r)/len(true_r)
        update_policy()
        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage reward: {:.2f}'.format(episode, time, trunning))
            global scs
            oldrn = scs[-1]
            scs.append(running_reward)
            #if np.abs(running_reward - oldrn) < 30 and False:
            if False:
                with torch.no_grad():
                    for param in policy.parameters():
                        param.add_(torch.randn(param.size())*5)

        if trunning > 600:
            print("Solved baby")
            break



episodes = 3001
main(episodes)

plt.plot(scs)
plt.show()


plt.plot(policy.loss_history)
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
