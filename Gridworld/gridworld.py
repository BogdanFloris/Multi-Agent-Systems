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
        for i in range(self.n):
            for j in range(self.n):
                state_values = {}
                for action in Actions:
                    if action == Actions.NORTH:
                        if not i == 0:
                            state_values[Actions.NORTH] = self.v[(i - 1) * self.n + j]
                        else:
                            state_values[Actions.NORTH] = -np.infty
                    elif action == Actions.SOUTH:
                        if not i == self.n - 1:
                            state_values[Actions.SOUTH] = self.v[(i + 1) * self.n + j]
                        else:
                            state_values[Actions.SOUTH] = -np.infty
                    elif action == Actions.EAST:
                        if not j == self.n - 1:
                            state_values[Actions.EAST] = self.v[i * self.n + j + 1]
                        else:
                            state_values[Actions.EAST] = -np.infty
                    elif action == Actions.WEST:
                        if not j == 0:
                            state_values[Actions.WEST] = self.v[i * self.n + j - 1]
                        else:
                            state_values[Actions.WEST] = -np.infty
                # make probabilities
                best_action_value = state_values[max(state_values, key=state_values.get)]
                all_best_actions = [key for key in state_values.keys() if state_values[key] == best_action_value]
                for action in Actions:
                    if state_values[action] == best_action_value:
                        self.policy[i * self.n + j][action] = 1 / len(all_best_actions)
                    else:
                        self.policy[i * self.n + j][action] = 0

    def policy_evaluation(self, theta=0.001):
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
                self.v = np.round(self.v, 1)
                break

    def temporal_difference_0(self, num_episodes=200, alpha=2/3):
        """
        TD(0) algorithm as described on page 98 of the Sutton book.
        It computes the values for V but for an episodic game, where
        A and B are terminal states.
        :param alpha learning rate
        :param num_episodes the number of episodes to generate.
        """
        for _ in range(num_episodes):
            # generate random starting state
            i, j = self._generate_starting_state()
            while True:
                # generate random action based on policy
                action = np.random.choice(self.policy[i * self.n + j].keys(), p=self.policy[i * self.n + j].values())
                # determine the next state and the reward
                new_i, new_j, reward = 0, 0, 0
                if action == Actions.NORTH:
                    pass
                elif action == Actions.SOUTH:
                    pass
                elif action == Actions.EAST:
                    pass
                elif action == Actions.WEST:
                    pass
                # update state value
                self.v[i * self.n + j] += alpha * (reward + self.gamma * self.v[
                    new_i * self.n + new_j] - self.v[i * self.n + j])
                # update i and j
                i, j = new_j, new_j
                # check to see if we are in a terminal state
                if (i == 0 and j == 1) or (i == 0 and j == 3):
                    break

    @staticmethod
    def _generate_starting_state():
        i, j = np.random.randint(low=0, high=5, size=2)
        while (i == 0 and j == 1) or (i == 0 and j == 3):
            i, j = np.random.randint(low=0, high=5, size=2)
        return i, j

    def print_grid(self):
        for i in range(self.n):
            for j in range(self.n):
                print(self.v[i * self.n + j], end=' ')
            print()


# continuous grid world
grid = GridWorld()
grid.policy_evaluation()
grid.policy_improvement()
grid.policy_evaluation()
grid.print_grid()

# episodic grid world
grid2 = GridWorld(gamma=1)
grid2.print_grid()
