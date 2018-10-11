import numpy as np


class Policy:
    """
    Represents a policy that the agent can use
    to choose his next action. This class is used
    as an interface and should be subclassed in order
    to implement a policy.
    """
    def __str__(self):
        return 'Generic policy'

    def choose(self, agent):
        pass


class GreedyWithOptimisticInitPolicy(Policy):
    """
    The Greedy policy chooses the best action that the
    agent knows about. If there is more than one best
    action, a random one will be chosen. It also initializes
    the agent's Q list with the value :var init_val.
    """
    def __init__(self, init_val):
        self.init_val = init_val

    def __str__(self):
        return 'Greedy With Optimistic Initialization: {}'.format(self.init_val)

    def choose(self, agent):
        action = np.argmax(agent.Q)
        all_max_actions = [i for i in range(len(agent.Q))
                           if agent.Q[i] == agent.Q[action]]
        return np.random.choice(all_max_actions)


class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon Greedy Policy chooses a random action
    with probability :var epsilon and the best action
    with probability 1 - :var epsilon. If there is more
    than one best action, a random one will be chosen.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.Q))
        else:
            action = np.argmax(agent.Q)
            all_max_actions = [i for i in range(len(agent.Q))
                               if agent.Q[i] == agent.Q[action]]
            return np.random.choice(all_max_actions)


class UCBPolicy(Policy):
    """
    The Upper Confidence Bound (UCB1) algorithm. It applies
    an exploration factor to the expected value of each arm
    which can influence a greedy selection strategy to more
    intelligently explore less confident options.
    """
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def choose(self, agent):
        exploration = np.log(agent.t + 1) / (2 * agent.action_choices)
        exploration[np.isnan(exploration)] = 0
        exploration = self.c * np.sqrt(exploration)
        q = agent.Q + exploration
        action = np.argmax(q)
        all_max_actions = [i for i in range(len(q))
                           if q[i] == q[action]]
        return np.random.choice(all_max_actions)
