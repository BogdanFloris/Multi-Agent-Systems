import numpy as np
from policy import GreedyWithOptimisticInitPolicy


class Agent:
    """
    Agent class
    """
    def __init__(self, k, policy):
        # the number of bandits
        self.k = k
        # time
        self.t = 0
        # the reward list
        if isinstance(policy, GreedyWithOptimisticInitPolicy):
            self.Q = policy.init_val + np.zeros(k, dtype=np.float)
        else:
            self.Q = np.zeros(k, dtype=np.float)
        # how many times each action has been chosen
        self.action_choices = np.zeros(k, dtype=np.int)
        # the policy to be used
        self.policy = policy

    def choose_action(self):
        """
        Chooses an action based on the policy
        :return: the action to be player
        """
        action = self.policy.choose(self)
        return action

    def play_action(self, action, reward):
        """
        Updates the Q list, action_choices list and
        the time after an action has been played.
        :param action: the action that has been player (an index)
        :param reward: the reward that was received
        """
        self.action_choices[action] += 1
        self.Q[action] += + 1 / self.action_choices[action] * (
                reward - self.Q[action])
        self.t += 1

    @property
    def get_q(self):
        """
        :return: the Q array
        """
        return self.Q

    def reset(self):
        """
        Resets the agent.
        """
        self.t = 0
        if self.policy is GreedyWithOptimisticInitPolicy:
            self.Q = self.policy.init_val + np.zeros(self.k, dtype=np.float)
        else:
            self.Q = np.zeros(self.k, dtype=np.float)
        self.action_choices = np.zeros(self.k, dtype=np.int)
