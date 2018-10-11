import numpy.random as rand


class Bandit:
    """
    Represents an action that our agent can choose.
    It returns random rewards samples from a normal
    distribution with a random mean and standard dev.
    The mean
    """
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.reward = None
        self.reset()

    def reset(self):
        self.reward = rand.normal(self.mu, self.sigma)

    def get_reward(self):
        return self.reward
