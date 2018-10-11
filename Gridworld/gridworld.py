"""
GridWorld problem HW5
"""
import numpy as np
from enum import Enum


class Actions(Enum):
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class GridWorld:
    """
    GridWorld
    """
    def __init__(self, n=5, gamma=0.9):
        self.n = n
        self.gamma = gamma
        self.v = np.zeros(n*n, dtype=float)
        self.P = np.zeros((n * n, n * n), dtype=float)
        self.r = np.zeros(n, dtype=float)
        self.policy = [{
            Actions.NORTH: 0.25,
            Actions.SOUTH: 0.25,
            Actions.EAST: 0.25,
            Actions.WEST: 0.25
        } for _ in range(n*n)]

    def policy_improvement(self):
        """
        Constructs the policy based on v.
        """
        # TODO: implement
        pass

    def policy_iteration(self, theta=0.001):
        """
        Calculates V using the algorithm on pg. 61, Sutton
        :param theta: checks when to break out of the while true loop
        """
        while True:
            delta = 0
            for i in range(self.n):
                for j in range(self.n):
                    v = self.v[i * self.n + j]
                    # get policy for self.grid[i * n, j]
                    policy = self.policy[i * self.n + j]
                    if i == 0 and j == 1:
                        # handle A
                        self.v[i * self.n + j] = 10 + self.gamma * self.v[(i + 4) * self.n + j]
                        continue
                    if i == 0 and j == 3:
                        # handle B
                        self.v[i * self.n + j] = 5 + self.gamma * self.v[(i + 2) * self.n + j]
                        continue
                    # calculate self.grid[i * n, j] using formula in book
                    actions_sum = 0
                    for action, prob in policy.items():
                        reward = 0
                        if action == Actions.NORTH:
                            # handle north action
                            next_cell = i - 1
                            if i == 0:
                                reward = -1
                                next_cell = 0
                            actions_sum += prob * (reward + self.gamma * self.v[next_cell * self.n + j])
                        elif action == Actions.SOUTH:
                            # handle south action
                            next_cell = i + 1
                            if i == self.n - 1:
                                reward = -1
                                next_cell = self.n - 1
                            actions_sum += prob * (reward + self.gamma * self.v[next_cell * self.n + j])
                        elif action == Actions.EAST:
                            # handle east action
                            next_cell = j + 1
                            if j == self.n - 1:
                                reward = -1
                                next_cell = self.n - 1
                            actions_sum += prob * (reward + self.gamma * self.v[i * self.n + next_cell])
                        elif action == Actions.WEST:
                            # handle west action
                            next_cell = j - 1
                            if j == 0:
                                reward = -1
                                next_cell = 0
                            actions_sum += prob * (reward + self.gamma * self.v[i * self.n + next_cell])
                    # update v
                    self.v[i * self.n + j] = actions_sum
                    delta = max(delta, abs(v - self.v[i * self.n + j]))
            if delta < theta:
                break

    def print_grid(self):
        for i in range(self.n):
            for j in range(self.n):
                print(round(self.v[i * self.n + j], 1), end=' ')
            print()


grid = GridWorld()
grid.policy_iteration()
grid.print_grid()
