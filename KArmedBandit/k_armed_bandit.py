"""
Main file for the K Armed Bandit Problem
"""
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from bandit import Bandit
from policy import Policy, GreedyWithOptimisticInitPolicy, EpsilonGreedyPolicy, UCBPolicy


def main():
    """
    Constants
    """
    k: int = 20
    epsilon: float = 0.01
    init_val: int = 10
    c: float = 1.0
    max_time: int = 1000
    rounds: int = 1000
    policy: Policy = EpsilonGreedyPolicy(epsilon)
    agent = Agent(k, policy)
    bandits = [Bandit() for _ in range(k)]

    def play_round():
        """
        Simulates one round of the game.
        """
        # get the next action
        action = agent.choose_action()
        # get a reward from the bandit
        reward = bandits[action].get_reward()
        # play the action
        agent.play_action(action, reward)
        return reward

    def reset():
        agent.reset()
        for bandit in bandits:
            bandit.reset()
        optimal_bandit = np.argmax([bandit.get_reward() for bandit in bandits])

    def print_bandits():
        for i, bandit in enumerate(bandits):
            print('Bandit {} reward={}'.format(i, bandit.get_reward()))

    def experiment():
        scores = np.zeros(max_time, dtype=float)
        for _ in range(rounds):
            for t in range(max_time):
                scores[t] += play_round()
            reset()

        return scores / rounds

    def plot(label):
        print_bandits()
        scores = experiment()
        time = range(max_time)
        plt.title(label + " for k = " + str(k))
        plt.ylim([0.0, 2.0])
        plt.xlabel('Steps')
        plt.ylabel('Avg. Reward')
        plt.scatter(x=time, y=scores, s=0.5)
        plt.show()

    plot(policy.__str__())


main()
