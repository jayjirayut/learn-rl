import argparse  # for command line arguments
import gym
import numpy as np
from itertools import count  # for counting iterations
from collections import namedtuple  # for creating named tuples

import torch
import torch.nn as nn
import torch.nn.functional as F  # for F.relu and F.softmax
import torch.optim as optim  # for optim.Adam
from torch.distributions import Categorical  # for Categorical distribution

# Cart Pole Environment

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')

parser.add_argument('--render', action='store_true',
                    help='render the environment')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()


env = gym.make('CartPole-v1')
env.seed(args.seed)  # for reproducibility of results
torch.manual_seed(args.seed)  # for reproducibility in pytorch


# for storing actions and their log probabilities and values in the replay buffer for training the policy network and
# the critic network respectively in the next step of training the actor-critic network
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    Implements both actor and critic in one model. The policy network is a feedforward neural network with two hidden
    layers. The input layer has 4 nodes and the output layer has 2 nodes. The policy network is used to select
    actions given states. The policy network is trained using the actor-critic algorithm.
    """
    def __init__(self):
        super(Policy, self).__init__()  # call the parent class's constructor
        self.affine1 = nn.Linear(4, 128)  # 4 input nodes to 128 hidden nodes in the first hidden layer

        # actor's layer to choose action from state
        self.action_head = nn.Linear(128, 2)  # 2 actions in CartPole-v1 environment (left and right)

        # critic's layer to evaluate being in the state
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer to store actions and rewards for training the policy network and the critic network
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        Forward of both actor and critic. Takes the state as input and returns the log probabilities of each action
        and the value of the state. The output of the policy network is used to select actions. The output of the
        critic network is used to evaluate the value of the state.

        1. First, the input is passed through the first hidden layer.
        2. The output of the first hidden layer is passed through the second hidden layer.
        3. The output of the second hidden layer is passed through the action head.
        4. The output of the action head is passed through the value head.

        Args:
            x: the state of the environment
        Returns:
            log_prob: the log probabilities of each action given the state x
            value: the value of the state
        """
        x = F.relu(self.affine1(x))  # first hidden layer with relu activation function

        # actor: chooses action to take from state s_t
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()  # epsilon for avoiding zero division


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)  # call forward method of the model to get the probabilities of each action

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    """ Main training loop """
    running_reward = 10

    # run infinitely many episodes until the environment is solved
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't infinitely loop while learning (just in case)
        for t in range(1, 10000):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
