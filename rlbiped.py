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
env.seed(1); torch.manual_seed(1);


#Hyperparameters
learning_rate = 0.001
gamma = 0.99
h = [128,64]
class Policy(nn.Module):
    def __init__(self,h):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = 4

        self.l1 = nn.Linear(self.state_space, h[0])
        self.l2 = nn.Linear(h[0],h[1])
        self.l3 = nn.Linear(h[1], self.action_space)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3,
        )
        return model(x)


policy = Policy(h)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action(state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    action = policy(Variable(state))
    #c = Categorical(state)
    #action = c.sample()
    # Add log probability of our chosen action to our history
    if policy.policy_history.nelement() != 0:

        policy.policy_history = torch.cat([policy.policy_history, action.view(-1,1)], dim = 1)
    else:
        policy.policy_history = action.view(-1,1)

    return action


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)

    # Scale rewards
    rewards = torch.tensor(rewards)

    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    # Calculate loss
    loss = -torch.sum( policy.policy_history * Variable(rewards) )
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode= []

scs=[]
def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        done = False

        for time in range(1000):
            action = select_action(state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.detach().numpy())
            # Save reward
            if not done:
                policy.reward_episode.append(reward)
            if done:
                policy.reward_episode.append(time)
                break


        #for k in range(time):
            #policy.reward_episode.append(0)
        #policy.reward_episode.append(time-running_reward)

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))
            global scs
            scs.append(running_reward)

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break


episodes = 1001
main(episodes)

def essai():
    obs=env.reset()
    for i in range(200):
        env.render()
        act=policy.forward(obs)
        obs,rew,done,info=env.step(act.item())
        if done==True:
            return i

plt.plot(scs)
plt.show()

with torch.no_grad():
    obs=env.reset()
    for i in range(200):
        env.render()
        act=select_action(obs)
        obs,rew,done,info=env.step(act.detach().numpy())
        if done==True:
            break
env.close()
